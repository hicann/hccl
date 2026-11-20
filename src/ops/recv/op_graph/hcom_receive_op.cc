/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 调用的 hcom.h (pkg_inc/hccl/hcom.h) 接口:
//   HcomGetCommHandleByGroup — 按 group 获取通信域 handle (基类 GetCommunicator)
//   HcomGetOutCCLbuffer + HcomCreateCommCCLbuffer — output CCL buffer 查询/创建
//                              (基类 GetOrCreateOutCCLbuf, CalcResourcesStatic 与 LaunchRefresh 使用)
//   HcomGetRankId        — Receive tag 的本端 destRank 补全
//   HcomGenerateCclOpTag — SendRecv baseTag 展开
//   HcomGet/SetWorkflowMode + HcomSetLaunchKernelMode — 动态路径相位包裹 (对照专用链)
// 调用的 hccl.h 接口:
//   HcclRecv — P2P 接收 (动态路径两段: shape 元数据 + 数据)
//   HcclRecvGraphMode (hccl.map 导出, recv.cc) — 静态路径直传/刷新通信
//
// 静态路径对照 (InfoStore hcom_ops_kernel_info_store.cc):
//   HcomReceiveOpKernel — 参数提取(无输入, 全来自 attrs) + needRefresh 分叉
//   HcomReceiveLoop     — HcceRecvGraphMode(commOutputPtr) → ⑤ RefreshOutputAddr, 先通信后拷回
//   Builder GetHcomReceiveOpOutputSize — shape attr → 512B 对齐 output size (标量 shapeSize=1)
// 动态路径对照 (lowering hcom_launch_kernel.cc):
//   HcomGetRecvBeforeKernelV2 — 收 shape 元数据 (HcclRecv 100×int64 + 同步 + D2H 回读)
//   LaunchRecvKernelV2         — 收 shape → 写 StorageShape → 分配 output → 收数据
//   WorkflowMode 相位 (专用链): OP_BASE + LaunchKernelMode(true) 进入, 元数据协商与分配阶段保持
//   true, 数据段 HcclRecv 前置 false, 末尾恢复 WorkflowMode
//   recvCount = RoundUp(dataSize, 32) / unitSize — 与 Send 侧严格对称, padding 无关

#include "hcom_receive_op.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "hccl/hcom.h"
#include "hccl_op_utils.h"
#include "hccl_rt_utils.h"
#include "log.h"

extern "C" {
HcclResult
HcclRecv(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream);
HcclResult HcclRecvGraphMode(
    void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, const char* group, aclrtStream stream,
    const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize);
}

namespace {
constexpr int64_t UNKNOWN_DIM = -1;

// 元数据 device buffer RAII 释放 (对照 HcomGetRecvBeforeKernelV2 的 unique_ptr deleter 模式)
struct AclrtMemDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            (void)aclrtFree(ptr);
        }
    }
};

// WorkflowMode 相位复刻的 RAII 守卫 (对照专用链): 构造前已设 OP_BASE + LaunchKernelMode(true),
// 数据段前由调用方显式置 false, 析构时兜底 false + 恢复原 WorkflowMode (异常路径不残留线程本地状态)
class WorkflowModeGuard {
public:
    explicit WorkflowModeGuard(HcclWorkflowMode lastMode) : lastMode_(lastMode) {}
    ~WorkflowModeGuard()
    {
        HcomSetLaunchKernelMode(false);
        (void)HcomSetWorkflowMode(lastMode_);
    }
    WorkflowModeGuard(const WorkflowModeGuard&) = delete;
    WorkflowModeGuard& operator=(const WorkflowModeGuard&) = delete;

private:
    HcclWorkflowMode lastMode_;
};

// 从 shape attr (ListInt) 构造 gert::Shape; 标量 (空 list) 按对照语义 shapeSize=1
ge::graphStatus
BuildShapeFromAttr(const gert::TypedContinuousVector<int64_t>* shapeAttr, gert::Shape& shape, int64_t& shapeSize)
{
    shapeSize = 1; // 标量: shapeDims 为空时 shapeSize=1 (Builder GetHcomReceiveOpOutputSize 同语义)
    if (shapeAttr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t* dims = shapeAttr->GetData();
    const size_t dimCount = shapeAttr->GetSize();
    // 与 InferShape 侧同款防御: attr 维数超 kMaxDimNum 时拒绝, 防御异常 attr 越界写 shape 内部
    if (dimCount > gert::Shape::kMaxDimNum) {
        HCCL_ERROR("[BuildShapeFromAttr] invalid shape attr size %zu (> %zu).", dimCount, gert::Shape::kMaxDimNum);
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < dimCount; i++) {
        shape.AppendDim(dims[i]);
    }
    if (dimCount != 0) {
        shapeSize = shape.GetShapeSize(); // 溢出/负维返回负值 (kInvalidDimValue)
        if (shapeSize < 0) {
            HCCL_ERROR("[BuildShapeFromAttr] invalid shape size [%ld] from shape attr.", shapeSize);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

HcomReceiveOp::~HcomReceiveOp() = default;

ge::graphStatus HcomReceiveOp::ExtractParamsStatic(hccl::HcclOpState& st)
{
    // Receive 无输入张量, 参数全部来自节点 attrs (按 IR 原型 index 读取)
    const gert::RuntimeAttrs* attrs = st.ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomReceiveOp][ExtractParamsStatic] attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const char* group = attrs->GetStr(ATTR_GROUP);
    const int64_t* srTag = attrs->GetInt(ATTR_SR_TAG);
    const int64_t* srcRank = attrs->GetInt(ATTR_SRC_RANK);
    const gert::TypedContinuousVector<int64_t>* shapeAttr = attrs->GetListInt(ATTR_SHAPE);
    const int64_t* dtype = attrs->GetInt(ATTR_DTYPE);
    if (group == nullptr || srTag == nullptr || srcRank == nullptr || shapeAttr == nullptr || dtype == nullptr) {
        HCCL_ERROR(
            "[HcomReceiveOp][ExtractParamsStatic] attrs missing, group=%p srTag=%p srcRank=%p "
            "shape=%p dtype=%p.",
            group, srTag, srcRank, shapeAttr, dtype);
        return ge::GRAPH_FAILED;
    }

    st.group = group;
    st.srTag = static_cast<uint32_t>(*srTag);
    st.srcRank = static_cast<uint32_t>(*srcRank);
    st.dataType = hccl::GeDataTypeToHccl(static_cast<ge::DataType>(*dtype));
    if (st.dataType == HCCL_DATA_TYPE_RESERVED) {
        // 早期校验: RESERVED 的 unitSize 为 0, 放行会在 count 公式除零
        HCCL_ERROR("[HcomReceiveOp][ExtractParamsStatic] unsupported dtype attr [%ld].", *dtype);
        return ge::GRAPH_FAILED;
    }

    gert::Shape outputShape;
    int64_t shapeSize = 1;
    HCCL_GE_CHK_RET(BuildShapeFromAttr(shapeAttr, outputShape, shapeSize));
    const uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    const uint64_t dataSize = static_cast<uint64_t>(shapeSize) * unitSize;
    // 静态 count: 精确逻辑字节数。静态路径 output TensorData 按 shape 逻辑尺寸上报(GE 内存分配不再
    // 512B 对齐), 512B 对齐口径仅适用于旧 InfoStore 路径(其 Builder 会将 output 分配同步覆写为
    // 512 对齐, 故 count 与分配自洽); 沿用到 custom_op 路径会导致 output 尺寸校验恒失败。
    st.count = dataSize / unitSize;

    // 静态场景 output 内存由 GE 预分配, MallocOutputTensor 只覆写 desc 并返回地址 (含分配粒度防御)
    size_t outputSize = 0;
    HCCL_GE_CHK_RET(AllocateStaticOutput(st, outputShape, *dtype, outputSize));

    HCCL_INFO(
        "[HcomReceiveOp][ExtractParamsStatic] group=%s srTag=%u srcRank=%u count=%lu dataType=%d "
        "output=%p outputSize=%zu.",
        st.group, st.srTag, st.srcRank, st.count, static_cast<int>(st.dataType), st.outputPtrs[0], outputSize);
    return ge::GRAPH_SUCCESS;
}

// 静态路径拆分: output 分配段 (原 ExtractParamsStatic 内联段, 行为与日志保序)
ge::graphStatus HcomReceiveOp::AllocateStaticOutput(
    hccl::HcclOpState& st, const gert::Shape& outputShape, int64_t dtype, size_t& outputSize)
{
    gert::StorageShape storageShape;
    storageShape.MutableOriginShape() = outputShape;
    storageShape.MutableStorageShape() = outputShape;
    const gert::StorageFormat storageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType());
    gert::Tensor* outputTensor
        = st.ctx->MallocOutputTensor(OUTPUT_INDEX, storageShape, storageFormat, static_cast<ge::DataType>(dtype));
    if (outputTensor == nullptr) {
        HCCL_ERROR("[HcomReceiveOp][ExtractParamsStatic] MallocOutputTensor failed.");
        return ge::GRAPH_FAILED;
    }
    // 防御: 框架静态分支无越界检查, output 上报尺寸小于接收字节数时报错而非越界写
    const uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    if (static_cast<uint64_t>(outputTensor->GetSize()) < st.count * unitSize) {
        HCCL_ERROR(
            "[HcomReceiveOp][ExtractParamsStatic] output allocated size %zu < aligned recv bytes %lu "
            "(allocation granularity insufficient).",
            outputTensor->GetSize(), st.count * unitSize);
        return ge::GRAPH_FAILED;
    }
    outputSize = outputTensor->GetSize();
    st.outputPtrs.push_back(outputTensor->GetAddr());
    st.stream = st.ctx->GetStream();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReceiveOp::CalcResourcesStatic(hccl::HcclOpState& st)
{
    // 简化版资源准备: 仅刷新路径需要 output CCL buffer 大小 (GenerateTagVec 的 loopMaxTime 用)。
    // 不做在线资源计算与 MallocWorkSpace (从流恒空、scratch 零消费、Selector 在 GraphMode 入口内部)。
    // Direct 路径不触碰 CCL buffer (对齐 InfoStore 直传分支), 直接返回。
    if (!st.needRefresh) {
        return ge::GRAPH_SUCCESS;
    }
    // 注意: 不能用 HcomGetCommCCLBufferSize — 其 950 V2 分支是空 stub (返回 SUCCESS 但不写 size)。
    // GetOrCreateOutCCLbuf (HcomGetOutCCLbuffer + 不存在则创建) 与 InfoStore GetHcomOutCCLbufferSize 同语义。
    void* commOutputPtr = nullptr;
    HCCL_GE_CHK_RET(GetOrCreateOutCCLbuf(st.group, commOutputPtr, st.cclBuffSize));
    if (st.cclBuffSize == 0) {
        HCCL_ERROR("[HcomReceiveOp][CalcResourcesStatic] output CCL buf size is 0 for group '%s'.", st.group);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReceiveOp::GenerateTagVec(hccl::HcclOpState& st)
{
    // SendRecv tag: srTag/srcRank 来自 attrs, destRank 为本端 rank (InfoStore GenerateOpTagFromTaskInfo 同源;
    // 与 Send 侧生成相同字符串以保证配对)
    u32 destRank = 0;
    HcclResult ret = HcomGetRankId(st.group, &destRank);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReceiveOp][GenerateTagVec] HcomGetRankId failed for group '%s', ret=%d.", st.group,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    const std::string baseTag = hccl::BuildSendRecvBaseTag(st.srTag, st.srcRank, destRank);

    // loopMaxTime 仅刷新路径计算: Receive 用 output CCL buf 公式 (组 1, rankSize 不消费, 惯例传 1)
    uint32_t loopMaxTime = 0;
    if (st.needRefresh) {
        HCCL_GE_CHK_RET(hccl::CalcLoopMaxTime(
            HCCL_KERNEL_OP_TYPE_RECEIVE, st.count, hccl::GetHcclDataTypeSize(st.dataType), st.cclBuffSize, 1,
            loopMaxTime));
    }
    // BuildTagVec 首参为传给 HcomGenerateCclOpTag 的种子串: 集合通信类传 opType, P2P 传 SendRecv baseTag
    HCCL_GE_CHK_RET(hccl::BuildTagVec(baseTag, st.group, loopMaxTime, st.tags));

    HCCL_INFO(
        "[HcomReceiveOp][GenerateTagVec] baseTag=%s loopMaxTime=%u tags=%zu tag[0]=%s.", baseTag.c_str(), loopMaxTime,
        st.tags.size(), st.tags.empty() ? "(none)" : st.tags[0].c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReceiveOp::LaunchHcclOpStatic(hccl::HcclOpState& st)
{
    HCCL_GE_CHK_RET(GenerateTagVec(st));

    // attach 流申请 + 交付 (双流: graph/group 粒度, 对照旧 InfoStore SetAttachedStream 每 op 一次;
    // AIV 跳过)
    // follow 流申请 (按 st.streamNum, 复刻旧 main_stream_id 池化; AIV/流数 0 自动跳过)
    HCCL_GE_CHK_RET(RequestFollowStreams(st));
    HCCL_GE_CHK_RET(RequestAttachStream(st));
    HCCL_GE_CHK_RET(RegisterUnfoldStreamToModel(st));
    if (!st.needRefresh) {
        return LaunchDirect(st);
    }
    return LaunchRefresh(st);
}

ge::graphStatus HcomReceiveOp::LaunchDirect(hccl::HcclOpState& st)
{
    // 对齐 InfoStore HcomReceiveOpKernel 直传分支: 无超限守卫, 直传用户 output buffer
    HcclResult ret = HcclRecvGraphMode(
        st.outputPtrs[0], st.count, st.dataType, st.srcRank, st.group, static_cast<aclrtStream>(st.stream),
        st.tags[0].c_str(), st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(),
        nullptr, 0);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReceiveOp][LaunchDirect] HcclRecvGraphMode failed, count=%lu, ret=%d.", st.count,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    HCCL_INFO(
        "[HcomReceiveOp][LaunchDirect] output=%p count=%lu srcRank=%u tag=%s.", st.outputPtrs[0], st.count, st.srcRank,
        st.tags[0].c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReceiveOp::LaunchRefresh(hccl::HcclOpState& st)
{
    // 地址刷新路径 (对应 InfoStore HcomReceiveLoop 的 ③→⑤, 950 单迭代简化; 先通信后拷回):
    //   ①④ (H2D 写 CCL buf 地址) 由 MallocReadOnlyDevArgs 统一管理
    //   ③  HcclRecvGraphMode(commOutputPtr, ...)
    //   ⑤  D2D ADDR: CCL output → 用户 output (间接寻址)
    void* commOutputPtr = nullptr;
    uint64_t commOutputSize = 0;
    HCCL_GE_CHK_RET(GetOrCreateOutCCLbuf(st.group, commOutputPtr, commOutputSize));

    const uint64_t curSize = st.count * hccl::GetHcclDataTypeSize(st.dataType);
    // 超限守卫: 严格大于报错 (Receive 只判 output 侧; 多迭代 loop 属 legacy 设备, 不在本切片范围)
    if (curSize > commOutputSize) {
        HCCL_ERROR(
            "[HcomReceiveOp][LaunchRefresh] curSize=%lu exceeds output CCL buf size=%lu; "
            "multi-iteration loop not supported.",
            curSize, commOutputSize);
        return ge::GRAPH_FAILED;
    }

    HcomRecvArgs args = {commOutputPtr, st.outputPtrs[0]};
    const gert::KernelArgs* devArgs = st.ctx->MallocReadOnlyDevArgs(&args, sizeof(args));
    if (devArgs == nullptr || devArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomReceiveOp][LaunchRefresh] MallocReadOnlyDevArgs failed.");
        return ge::GRAPH_FAILED;
    }
    char* devArgsBase = static_cast<char*>(devArgs->args_data);
    void* devCclOutputPtr = devArgsBase + offsetof(HcomRecvArgs, cclOutputPtr);
    void* devOutputAddr = devArgsBase + offsetof(HcomRecvArgs, outputAddr);

    // ③ HcclRecvGraphMode (收到 CCL output buf; 单迭代 tag 带 _looptime_0 后缀)
    HcclResult ret = HcclRecvGraphMode(
        commOutputPtr, st.count, st.dataType, st.srcRank, st.group, static_cast<aclrtStream>(st.stream),
        st.tags[0].c_str(), st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(),
        nullptr, 0);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReceiveOp][LaunchRefresh] HcclRecvGraphMode failed, count=%lu, ret=%d.", st.count,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    // ⑤ D2D ADDR: CCL output → 用户 output (间接寻址; src/dst 均为二级指针, 硬件 deref)
    ret = hccl::HcclRtMemAsyncCopy(
        devOutputAddr, curSize, devCclOutputPtr, curSize, hccl::HcclRtMemcpyKind::HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE,
        static_cast<aclrtStream>(st.stream));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReceiveOp][LaunchRefresh] D2D ADDR CCL->output failed, curSize=%lu, ret=%d.", curSize,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    HCCL_INFO(
        "[HcomReceiveOp][LaunchRefresh] devArgs=%p curSize=%lu cclOutput=%p outputAddr=%p tag=%s.", devArgs->args_data,
        curSize, commOutputPtr, st.outputPtrs[0], st.tags[0].c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReceiveOp::ExtractParamsDynamic(hccl::HcclOpState& st)
{
    // Receive 无输入张量, 参数全部来自节点 attrs (group/sr_tag/src_rank/dtype)
    const gert::RuntimeAttrs* attrs = st.ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomReceiveOp][ExtractParamsDynamic] attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const char* group = attrs->GetStr(ATTR_GROUP);
    const int64_t* srTag = attrs->GetInt(ATTR_SR_TAG);
    const int64_t* srcRank = attrs->GetInt(ATTR_SRC_RANK);
    const int64_t* dtype = attrs->GetInt(ATTR_DTYPE);
    if (group == nullptr || srTag == nullptr || srcRank == nullptr || dtype == nullptr) {
        HCCL_ERROR(
            "[HcomReceiveOp][ExtractParamsDynamic] attrs missing, group=%p srTag=%p "
            "srcRank=%p dtype=%p.",
            group, srTag, srcRank, dtype);
        return ge::GRAPH_FAILED;
    }

    st.group = group;
    st.srTag = static_cast<uint32_t>(*srTag);
    st.srcRank = static_cast<uint32_t>(*srcRank);
    st.dataType = hccl::GeDataTypeToHccl(static_cast<ge::DataType>(*dtype));
    if (st.dataType == HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR("[HcomReceiveOp][ExtractParamsDynamic] unsupported dtype attr [%ld].", *dtype);
        return ge::GRAPH_FAILED;
    }
    st.stream = st.ctx->GetStream();

    HCCL_INFO(
        "[HcomReceiveOp][ExtractParamsDynamic] group=%s srTag=%u srcRank=%u dtype=%ld unitSize=%lu.", st.group,
        st.srTag, st.srcRank, *dtype, hccl::GetHcclDataTypeSize(st.dataType));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReceiveOp::LaunchHcclOpDynamic(hccl::HcclOpState& st)
{
    const uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);

    // WorkflowMode 相位复刻对照专用链: OP_BASE + LaunchKernelMode(true) 进入,
    // 元数据协商与分配阶段保持 true, 数据段 HcclRecv 前置 false, 末尾恢复
    const HcclWorkflowMode lastMode = HcomGetWorkflowMode();
    HcclResult ret = HcomSetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReceiveOp][LaunchHcclOpDynamic] HcomSetWorkflowMode(OP_BASE) failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    HcomSetLaunchKernelMode(true);
    WorkflowModeGuard modeGuard(lastMode);

    // ---- Step 1: 阻塞接收 shape 元数据 (对照 HcomGetRecvBeforeKernelV2, true 相位) ----
    std::vector<int64_t> recvShape(hccl::SHAPE_INFO_COUNT, 0);
    HCCL_GE_CHK_RET(ReceiveShapeMetadata(st, recvShape));

    // ---- Step 2: dimNum 防御 + 解析 shape (对照语义: dimNum <= 0 或 > kMaxDimNum 拒绝) ----
    gert::Shape negotiatedShape;
    int64_t shapeSize = 0;
    uint64_t dataSize = 0;
    uint64_t count = 0;
    HCCL_GE_CHK_RET(ParseNegotiatedShape(recvShape, unitSize, negotiatedShape, shapeSize, dataSize, count));

    // ---- Step 3: 按协商 shape 分配 output (对照顺序: 收 shape → 分配 → 收数据; true 相位) ----
    size_t outputSize = 0;
    HCCL_GE_CHK_RET(AllocateNegotiatedOutput(st, negotiatedShape, outputSize));

    // ---- Step 4: 接收数据 (对照相位: 数据段 HcclRecv 前 LaunchKernelMode 置 false) ----
    HcomSetLaunchKernelMode(false);
    ret = HcclRecv(st.outputPtrs[0], count, st.dataType, st.srcRank, st.comm, static_cast<aclrtStream>(st.stream));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReceiveOp][LaunchHcclOpDynamic] HcclRecv data failed, count=%lu, ret=%d.", count,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    // 动态两段接收: 协商 shape 分配成功, output 实际分配 >= 对齐后字节数
    HCCL_INFO(
        "[HcomReceiveOp][LaunchHcclOpDynamic] dimNum=%ld shapeSize=%ld dataSize=%lu count=%lu "
        "output=%p outputSize=%zu.",
        recvShape[0], shapeSize, dataSize, count, st.outputPtrs[0], outputSize);
    return ge::GRAPH_SUCCESS;
}

// 动态路径拆分 Step 1: 阻塞接收 shape 元数据 (malloc + HcclRecv + 同步 + D2H 回读)
ge::graphStatus HcomReceiveOp::ReceiveShapeMetadata(const hccl::HcclOpState& st, std::vector<int64_t>& recvShape)
{
    const uint64_t shapeBufSize = hccl::SHAPE_INFO_COUNT * sizeof(int64_t);
    void* devShapeBuf = nullptr;
    aclError aclRet = aclrtMalloc(&devShapeBuf, shapeBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReceiveOp][LaunchHcclOpDynamic] aclrtMalloc shape buf failed, size=%lu, ret=%d.", shapeBufSize,
            static_cast<int>(aclRet));
        return ge::GRAPH_FAILED;
    }
    std::unique_ptr<void, AclrtMemDeleter> devShapeBufGuard(devShapeBuf);

    HcclResult ret = HcclRecv(
        devShapeBuf, hccl::SHAPE_INFO_COUNT, HCCL_DATA_TYPE_INT64, st.srcRank, st.comm,
        static_cast<aclrtStream>(st.stream));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReceiveOp][LaunchHcclOpDynamic] HcclRecv shape metadata failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    aclRet = aclrtSynchronizeStream(static_cast<aclrtStream>(st.stream));
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReceiveOp][LaunchHcclOpDynamic] aclrtSynchronizeStream failed, ret=%d.", static_cast<int>(aclRet));
        return ge::GRAPH_FAILED;
    }
    ret = hccl::HcclRtMemSyncCopy(
        recvShape.data(), shapeBufSize, devShapeBuf, shapeBufSize,
        hccl::HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomReceiveOp][LaunchHcclOpDynamic] D2H shape copy failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 动态路径拆分 Step 2: dimNum 防御 + 解析协商 shape + 对称 count (纯函数, 无副作用)
ge::graphStatus HcomReceiveOp::ParseNegotiatedShape(
    const std::vector<int64_t>& recvShape, uint64_t unitSize, gert::Shape& negotiatedShape, int64_t& shapeSize,
    uint64_t& dataSize, uint64_t& count)
{
    const int64_t dimNum = recvShape[0];
    if (dimNum < 1 || dimNum > static_cast<int64_t>(gert::Shape::kMaxDimNum)) {
        HCCL_ERROR(
            "[HcomReceiveOp][LaunchHcclOpDynamic] invalid dimNum [%ld] (expected [1, %zu]); wire protocol "
            "may be out of sync, session rebuild required.",
            dimNum, gert::Shape::kMaxDimNum);
        return ge::GRAPH_FAILED;
    }
    for (int64_t i = 1; i <= dimNum; i++) {
        negotiatedShape.AppendDim(recvShape[static_cast<size_t>(i)]);
    }
    shapeSize = negotiatedShape.GetShapeSize(); // 溢出/负维返回负值 (kInvalidDimValue)
    if (shapeSize < 0) {
        HCCL_ERROR("[HcomReceiveOp][LaunchHcclOpDynamic] invalid shape size [%ld] from negotiated shape.", shapeSize);
        return ge::GRAPH_FAILED;
    }
    dataSize = static_cast<uint64_t>(shapeSize) * unitSize;
    // count 与 Send 侧 GetSendCountByShapeAlign 严格对称: RoundUp(dataSize, 32) / unitSize (padding 无关)
    count = hccl::CalcAlignedCount(dataSize, unitSize, ALIGN_SIZE_DYNAMIC);
    return ge::GRAPH_SUCCESS;
}

// 动态路径拆分 Step 3: 按协商 shape 分配 output (dtype attr + MallocOutputTensor)
ge::graphStatus
HcomReceiveOp::AllocateNegotiatedOutput(hccl::HcclOpState& st, const gert::Shape& negotiatedShape, size_t& outputSize)
{
    const gert::RuntimeAttrs* attrs = st.ctx->GetAttrs();
    const int64_t* dtypeAttr = (attrs != nullptr) ? attrs->GetInt(ATTR_DTYPE) : nullptr;
    if (dtypeAttr == nullptr) {
        HCCL_ERROR("[HcomReceiveOp][LaunchHcclOpDynamic] dtype attr unreadable at output alloc stage.");
        return ge::GRAPH_FAILED;
    }
    gert::StorageShape storageShape;
    storageShape.MutableOriginShape() = negotiatedShape;
    storageShape.MutableStorageShape() = negotiatedShape;
    const gert::StorageFormat storageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType());
    gert::Tensor* outputTensor
        = st.ctx->MallocOutputTensor(OUTPUT_INDEX, storageShape, storageFormat, static_cast<ge::DataType>(*dtypeAttr));
    if (outputTensor == nullptr) {
        // 两段协议无回滚原语: 分配失败时对端数据消息滞留信道, 只能干净报错并提示重建会话
        HCCL_ERROR("[HcomReceiveOp][LaunchHcclOpDynamic] MallocOutputTensor failed; two-phase protocol desynced "
                   "(data message pending on channel), session rebuild required.");
        return ge::GRAPH_FAILED;
    }
    outputSize = outputTensor->GetSize();
    st.outputPtrs.push_back(outputTensor->GetAddr());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReceiveOp::UpdateHostArgs(gert::UpdateArgsContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomReceiveOp][UpdateHostArgs] update args context is null.");
        return ge::GRAPH_FAILED;
    }
    // sink 模型每次执行前 GE 回调: 只刷新 outputAddr。
    // Receive 无输入节点, GetInputTensor 恒 nullptr — 只读 GetOutputTensor
    const gert::Tensor* outputTensor = ctx->GetOutputTensor(OUTPUT_INDEX);
    if (outputTensor == nullptr) {
        HCCL_ERROR("[HcomReceiveOp][UpdateHostArgs] output tensor is null.");
        return ge::GRAPH_FAILED;
    }
    const gert::KernelArgs* hostArgs = ctx->GetKernelArgs(gert::Placement::kPlacementHost, 0);
    if (hostArgs == nullptr || hostArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomReceiveOp][UpdateHostArgs] hostArgs is null.");
        return ge::GRAPH_FAILED;
    }
    auto* args = static_cast<HcomRecvArgs*>(hostArgs->args_data);
    args->outputAddr = const_cast<void*>(outputTensor->GetAddr());

    HCCL_DEBUG("[HcomReceiveOp][UpdateHostArgs] outputAddr=%p.", args->outputAddr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReceiveOp::InferShape(gert::InferShapeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomReceiveOp][InferShape] infer shape context is null.");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outputShape = ctx->GetOutputShape(OUTPUT_INDEX);
    if (outputShape == nullptr) {
        HCCL_ERROR("[HcomReceiveOp][InferShape] output shape is null.");
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs* attrs = ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomReceiveOp][InferShape] attrs is null.");
        return ge::GRAPH_FAILED;
    }

    const gert::TypedContinuousVector<int64_t>* shapeAttr = attrs->GetListInt(ATTR_SHAPE);

    // 动态/静态判定 (attr 依据; 与基类 Execute 的 kStaticModel 口径相互独立 —
    // InferShape 发生在编译期, 无 EagerOpExecutionContext 可查):
    //   任一维为非正值 (含 -1 未知维) → 动态占位, output 置 -1 未知维, 真实 shape 由
    //     Execute 运行时取得 (实测 InferShape 期 attr 即以含 -1 占位形态到达);
    //   维数 > 0 且全部维为正 → 静态, 从 attr 推导 output shape;
    //   attr 缺失 → 报错 (HcomReceive IR 中 shape 为必选属性, 缺失属异常形态,
    //     不静默猜测动态 — 恢复原始实现行为)。实测未观测到该形态。
    if (shapeAttr == nullptr) {
        HCCL_ERROR("[HcomReceiveOp][InferShape] shape attr is null.");
        return ge::GRAPH_FAILED;
    }
    if (shapeAttr->GetSize() > gert::Shape::kMaxDimNum) {
        HCCL_ERROR(
            "[HcomReceiveOp][InferShape] invalid shape attr size %zu (> %zu).", shapeAttr->GetSize(),
            gert::Shape::kMaxDimNum);
        return ge::GRAPH_FAILED;
    }
    const int64_t* dims = shapeAttr->GetData();
    const size_t dimCount = shapeAttr->GetSize();
    bool hasNonPositiveDim = false;
    for (size_t i = 0; i < dimCount; i++) {
        if (dims[i] <= 0) {
            hasNonPositiveDim = true;
            break;
        }
    }
    if (hasNonPositiveDim) {
        HCCL_INFO("[HcomReceiveOp][InferShape] shape attr has non-positive dim, output dims=[-1].");
        outputShape->SetDimNum(0U);
        outputShape->AppendDim(UNKNOWN_DIM);
        return ge::GRAPH_SUCCESS;
    }

    // 静态分支: 从 shape attr 逐维推导 (空 list = 标量, 对照 GetHcomReceiveOpOutputSize 同语义)
    HCCL_INFO("[HcomReceiveOp][InferShape] infer output shape from attr, dimCount=%zu.", dimCount);
    outputShape->SetDimNum(0U);
    for (size_t i = 0; i < dimCount; i++) {
        outputShape->AppendDim(dims[i]);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReceiveOp::InferDataType(gert::InferDataTypeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomReceiveOp][InferDataType] infer datatype context is null.");
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs* attrs = ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomReceiveOp][InferDataType] attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const int64_t* dtype = attrs->GetInt(ATTR_DTYPE);
    if (dtype == nullptr) {
        HCCL_ERROR("[HcomReceiveOp][InferDataType] dtype attr is null.");
        return ge::GRAPH_FAILED;
    }
    return ctx->SetOutputDataType(OUTPUT_INDEX, static_cast<ge::DataType>(*dtype));
}

const char* HcomReceiveOp::GetOpTypeName() const { return HCCL_KERNEL_OP_TYPE_RECEIVE.c_str(); }

static ge::BaseCustomOp* CreateHcomReceiveOp() { return new HcomReceiveOp(); }
// 条件注册: 目标平台 (950/960) 且 GE/RTS 新接口在位才注册接管, 否则回退原 hccl_engine 路径
// (hccl_custom_op.h CreateSocGatedRegister: SoC 判定 + GE/RTS 接口在位判定)
static ge::CustomOpCreatorRegister* g_hcom_receive_register
    = hccl::CreateSocGatedRegister(HCCL_KERNEL_OP_TYPE_RECEIVE.c_str(), CreateHcomReceiveOp);
