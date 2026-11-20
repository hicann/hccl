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
//   HcomGetInCCLbuffer + HcomCreateCommCCLbuffer — input CCL buffer 查询/创建
//                              (基类 GetOrCreateInCCLbuf, CalcResourcesStatic 与 LaunchRefresh 使用)
//   HcomGetRankId        — Send tag 的本端 srcRank 补全
//   HcomGenerateCclOpTag — SendRecv baseTag 展开
//   HcomGet/SetWorkflowMode + HcomSetLaunchKernelMode — 动态路径相位包裹 (对照通用链)
// 调用的 hccl.h 接口:
//   HcclSend — P2P 发送 (动态路径两段: shape 元数据 + 数据)
//   HcclSendGraphMode (hccl.map 导出, send.cc) — 静态路径直传/刷新通信
//
// 静态路径对照 (InfoStore hcom_ops_kernel_info_store.cc):
//   HcomSendOpKernel — 参数提取 + needRefresh 分叉; 直传分支无超限守卫
//   HcomSendLoop     — ② RefreshInputAddr → HcceSendGraphMode(commInputPtr), 仅 input 侧无 ⑤
// 动态路径对照 (lowering hcom_launch_kernel.cc):
//   HcomSendKernelV2 — 元数据 [维数, 各维] H2D → HcceSend(100×int64) → 同步 → 数据
//   GetSendCountByShapeAlign — count = RoundUp(dataSize, 32) / unitSize
//   WorkflowMode 包裹位于外层 PrepareHcomKernel/LaunchHcomKernel 通用链:
//   OP_BASE + LaunchKernelMode(true) 贯穿 launch 至完成后 false + 恢复 (RAII 复刻)

#include "hcom_send_op.h"

#include <cstddef>
#include <cstdint>
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
HcclSend(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream);
HcclResult HcclSendGraphMode(
    void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, const char* group, aclrtStream stream,
    const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize);
}

namespace {
// 元数据 device buffer RAII 释放 (对照 HcomSendKernelV2 的 unique_ptr deleter 模式)
struct AclrtMemDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            (void)aclrtFree(ptr);
        }
    }
};

// WorkflowMode 相位复刻的 RAII 守卫: 构造时已由调用方设置 OP_BASE + LaunchKernelMode(true),
// 析构时 false + 恢复原 WorkflowMode (异常路径不残留线程本地状态)
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
} // namespace

HcomSendOp::~HcomSendOp() = default;

ge::graphStatus HcomSendOp::ExtractParamsStatic(hccl::HcclOpState& st)
{
    const gert::Tensor* inputTensor = st.ctx->GetInputTensor(INPUT_INDEX);
    if (inputTensor == nullptr) {
        HCCL_ERROR("[HcomSendOp][ExtractParamsStatic] input tensor is null.");
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs* attrs = st.ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomSendOp][ExtractParamsStatic] attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const char* group = attrs->GetStr(ATTR_GROUP);
    const int64_t* srTag = attrs->GetInt(ATTR_SR_TAG);
    const int64_t* destRank = attrs->GetInt(ATTR_DEST_RANK);
    if (group == nullptr || srTag == nullptr || destRank == nullptr) {
        HCCL_ERROR(
            "[HcomSendOp][ExtractParamsStatic] attrs missing, group=%p srTag=%p destRank=%p.", group, srTag, destRank);
        return ge::GRAPH_FAILED;
    }

    st.dataType = hccl::GeDataTypeToHccl(inputTensor->GetDataType());
    if (st.dataType == HCCL_DATA_TYPE_RESERVED) {
        // 早期校验: RESERVED 的 unitSize 为 0, 放行会在 count 公式除零
        HCCL_ERROR(
            "[HcomSendOp][ExtractParamsStatic] unsupported input dtype %d.",
            static_cast<int>(inputTensor->GetDataType()));
        return ge::GRAPH_FAILED;
    }

    st.group = group;
    st.srTag = static_cast<uint32_t>(*srTag);
    st.destRank = static_cast<uint32_t>(*destRank);
    const uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    // 静态 count: 精确逻辑字节数 (静态 input TensorData 按 shape 逻辑尺寸上报, 无对齐;
    // 与 Recv 侧严格对称, 双侧须同步变更)
    st.count = static_cast<uint64_t>(inputTensor->GetSize()) / unitSize;
    st.inputPtrs.push_back(const_cast<void*>(inputTensor->GetAddr()));
    st.stream = st.ctx->GetStream();
    // Send IR 原型无 OUTPUT 锚点, 不做 MallocOutputTensor (lowering converter 以 input 兼作 output 属其内部机制)

    HCCL_INFO(
        "[HcomSendOp][ExtractParamsStatic] group=%s srTag=%u destRank=%u count=%lu dataType=%d input=%p.", st.group,
        st.srTag, st.destRank, st.count, static_cast<int>(st.dataType), st.inputPtrs[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomSendOp::CalcResourcesStatic(hccl::HcclOpState& st)
{
    // 简化版资源准备: 仅刷新路径需要 input CCL buffer 大小 (GenerateTagVec 的 loopMaxTime 用)。
    // 不做在线资源计算与 MallocWorkSpace: InfoStore Builder 对 SEND/RECEIVE 强制从流数 0,
    // scratchMemAddr 在 HCCL 全仓零消费, 算法选择在 HcclSendGraphMode 入口内部执行。
    // Direct 路径不触碰 CCL buffer (对齐 InfoStore 直传分支), 直接返回。
    if (!st.needRefresh) {
        return ge::GRAPH_SUCCESS;
    }
    // 注意: 不能用 HcomGetCommCCLBufferSize — 其 950 V2 分支是空 stub (返回 SUCCESS 但不写 size)。
    // GetOrCreateInCCLbuf (HcomGetInCCLbuffer + 不存在则创建) 与 InfoStore GetHcomInCCLbufferSize 同语义。
    void* commInputPtr = nullptr;
    HCCL_GE_CHK_RET(GetOrCreateInCCLbuf(st.group, commInputPtr, st.cclBuffSize));
    if (st.cclBuffSize == 0) {
        HCCL_ERROR("[HcomSendOp][CalcResourcesStatic] input CCL buf size is 0 for group '%s'.", st.group);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomSendOp::GenerateTagVec(hccl::HcclOpState& st)
{
    // SendRecv tag: srTag/destRank 来自 attrs, srcRank 为本端 rank (InfoStore GenerateOpTagFromTaskInfo 同源)
    u32 srcRank = 0;
    HcclResult ret = HcomGetRankId(st.group, &srcRank);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomSendOp][GenerateTagVec] HcomGetRankId failed for group '%s', ret=%d.", st.group,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    const std::string baseTag = hccl::BuildSendRecvBaseTag(st.srTag, srcRank, st.destRank);

    // loopMaxTime 仅刷新路径计算: Send 用 input CCL buf 公式 (组 1, rankSize 不消费, 惯例传 1)
    uint32_t loopMaxTime = 0;
    if (st.needRefresh) {
        HCCL_GE_CHK_RET(hccl::CalcLoopMaxTime(
            HCCL_KERNEL_OP_TYPE_SEND, st.count, hccl::GetHcclDataTypeSize(st.dataType), st.cclBuffSize, 1,
            loopMaxTime));
    }
    // BuildTagVec 首参为传给 HcomGenerateCclOpTag 的种子串: 集合通信类传 opType, P2P 传 SendRecv baseTag
    HCCL_GE_CHK_RET(hccl::BuildTagVec(baseTag, st.group, loopMaxTime, st.tags));

    HCCL_INFO(
        "[HcomSendOp][GenerateTagVec] baseTag=%s loopMaxTime=%u tags=%zu tag[0]=%s.", baseTag.c_str(), loopMaxTime,
        st.tags.size(), st.tags.empty() ? "(none)" : st.tags[0].c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomSendOp::LaunchHcclOpStatic(hccl::HcclOpState& st)
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

ge::graphStatus HcomSendOp::LaunchDirect(hccl::HcclOpState& st)
{
    // 对齐 InfoStore HcomSendOpKernel 直传分支: 无超限守卫, 直传用户 buffer
    HcclResult ret = HcclSendGraphMode(
        st.inputPtrs[0], st.count, st.dataType, st.destRank, st.group, static_cast<aclrtStream>(st.stream),
        st.tags[0].c_str(), st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(),
        nullptr, 0);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomSendOp][LaunchDirect] HcclSendGraphMode failed, count=%lu, ret=%d.", st.count, static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    HCCL_INFO(
        "[HcomSendOp][LaunchDirect] input=%p count=%lu destRank=%u tag=%s.", st.inputPtrs[0], st.count, st.destRank,
        st.tags[0].c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomSendOp::LaunchRefresh(hccl::HcclOpState& st)
{
    // 地址刷新路径 (对应 InfoStore HcomSendLoop 的 ②→③, 950 单迭代简化):
    //   ①④ (H2D 写 CCL buf 地址) 由 MallocReadOnlyDevArgs 统一管理
    //   ②  D2D ADDR: 用户 input → CCL input (间接寻址)
    //   ③  HcclSendGraphMode(commInputPtr, ...)
    void* commInputPtr = nullptr;
    uint64_t commInputSize = 0;
    HCCL_GE_CHK_RET(GetOrCreateInCCLbuf(st.group, commInputPtr, commInputSize));

    const uint64_t curSize = st.count * hccl::GetHcclDataTypeSize(st.dataType);
    // 超限守卫: 严格大于报错 (Send 只判 input 侧; 多迭代 loop 属 legacy 设备, 不在本切片范围)
    if (curSize > commInputSize) {
        HCCL_ERROR(
            "[HcomSendOp][LaunchRefresh] curSize=%lu exceeds input CCL buf size=%lu; multi-iteration "
            "loop not supported.",
            curSize, commInputSize);
        return ge::GRAPH_FAILED;
    }

    HcomSendArgs args = {commInputPtr, st.inputPtrs[0]};
    const gert::KernelArgs* devArgs = st.ctx->MallocReadOnlyDevArgs(&args, sizeof(args));
    if (devArgs == nullptr || devArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomSendOp][LaunchRefresh] MallocReadOnlyDevArgs failed.");
        return ge::GRAPH_FAILED;
    }
    // devArgs->args_data 为 device 侧 args 起始地址, 字段偏移做二级指针 (同 AllReduce LaunchRefresh)
    char* devArgsBase = static_cast<char*>(devArgs->args_data);
    void* devCclInputPtr = devArgsBase + offsetof(HcomSendArgs, cclInputPtr);
    void* devInputAddr = devArgsBase + offsetof(HcomSendArgs, inputAddr);

    // ② D2D ADDR: 用户 input → CCL input (间接寻址; src/dst 均为二级指针, 硬件 deref)
    HcclResult ret = hccl::HcclRtMemAsyncCopy(
        devCclInputPtr, curSize, devInputAddr, curSize, hccl::HcclRtMemcpyKind::HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE,
        static_cast<aclrtStream>(st.stream));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomSendOp][LaunchRefresh] D2D ADDR input->CCL failed, curSize=%lu, ret=%d.", curSize,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    // ③ HcclSendGraphMode (从 CCL input buf 发送; 单迭代 tag 带 _looptime_0 后缀)
    ret = HcclSendGraphMode(
        commInputPtr, st.count, st.dataType, st.destRank, st.group, static_cast<aclrtStream>(st.stream),
        st.tags[0].c_str(), st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(),
        nullptr, 0);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomSendOp][LaunchRefresh] HcclSendGraphMode failed, count=%lu, ret=%d.", st.count,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    HCCL_INFO(
        "[HcomSendOp][LaunchRefresh] devArgs=%p curSize=%lu cclInput=%p inputAddr=%p tag=%s.", devArgs->args_data,
        curSize, commInputPtr, st.inputPtrs[0], st.tags[0].c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomSendOp::ExtractParamsDynamic(hccl::HcclOpState& st)
{
    const gert::Tensor* inputTensor = st.ctx->GetInputTensor(INPUT_INDEX);
    if (inputTensor == nullptr) {
        HCCL_ERROR("[HcomSendOp][ExtractParamsDynamic] input tensor is null.");
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs* attrs = st.ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomSendOp][ExtractParamsDynamic] attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const char* group = attrs->GetStr(ATTR_GROUP);
    const int64_t* srTag = attrs->GetInt(ATTR_SR_TAG);
    const int64_t* destRank = attrs->GetInt(ATTR_DEST_RANK);
    if (group == nullptr || srTag == nullptr || destRank == nullptr) {
        HCCL_ERROR(
            "[HcomSendOp][ExtractParamsDynamic] attrs missing, group=%p srTag=%p destRank=%p.", group, srTag, destRank);
        return ge::GRAPH_FAILED;
    }

    st.dataType = hccl::GeDataTypeToHccl(inputTensor->GetDataType());
    if (st.dataType == HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR(
            "[HcomSendOp][ExtractParamsDynamic] unsupported input dtype %d.",
            static_cast<int>(inputTensor->GetDataType()));
        return ge::GRAPH_FAILED;
    }

    st.group = group;
    st.srTag = static_cast<uint32_t>(*srTag);
    st.destRank = static_cast<uint32_t>(*destRank);
    // 动态 count: 32B 对齐 (GetSendCountByShapeAlign 同公式, 与 Recv 侧严格对称)。
    // dataSize 须与 shape 元数据同源 (GetOriginShape): 对照蓝本 HcomSendKernelV2 以同一
    // inputShapes[0] 同时生成元数据与 count; origin != storage (格式转换) 时 count 走 storage
    // 会与 Recv 侧按元数据重建 shape 计算的 count 失配, ND 下两者相等行为不变
    const uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    const uint64_t dataSize = static_cast<uint64_t>(inputTensor->GetOriginShape().GetShapeSize()) * unitSize;
    st.count = hccl::CalcAlignedCount(dataSize, unitSize, ALIGN_SIZE_DYNAMIC);
    st.inputPtrs.push_back(const_cast<void*>(inputTensor->GetAddr()));
    st.stream = st.ctx->GetStream();

    HCCL_INFO(
        "[HcomSendOp][ExtractParamsDynamic] group=%s destRank=%u count=%lu dataType=%d input=%p.", st.group,
        st.destRank, st.count, static_cast<int>(st.dataType), st.inputPtrs[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomSendOp::LaunchHcclOpDynamic(hccl::HcclOpState& st)
{
    // WorkflowMode 相位复刻通用链: OP_BASE + LaunchKernelMode(true) 贯穿两次 HcclSend 至 launch 完成
    const HcclWorkflowMode lastMode = HcomGetWorkflowMode();
    HcclResult ret = HcomSetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomSendOp][LaunchHcclOpDynamic] HcomSetWorkflowMode(OP_BASE) failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    HcomSetLaunchKernelMode(true);
    WorkflowModeGuard modeGuard(lastMode);

    // ---- 第一段: shape 元数据 [维数, 各维...] (固定 100×int64, zero-init 尾部) ----
    const gert::Tensor* inputTensor = st.ctx->GetInputTensor(INPUT_INDEX);
    if (inputTensor == nullptr) {
        HCCL_ERROR("[HcomSendOp][LaunchHcclOpDynamic] input tensor is null.");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& inputShape = inputTensor->GetOriginShape();
    const int64_t dimNum = static_cast<int64_t>(inputShape.GetDimNum());
    // sendInfo 固定 SHAPE_INFO_COUNT 容量, 维数 + 1 超限即越界写; GE shape 上界 kMaxDimNum=25
    // 正常不可达, 校验将常量关系显式化 (Recv 侧对线协议数据有同款防御)
    if (static_cast<uint64_t>(dimNum) + 1U > hccl::SHAPE_INFO_COUNT) {
        HCCL_ERROR(
            "[HcomSendOp][LaunchHcclOpDynamic] dimNum %ld + 1 exceeds SHAPE_INFO_COUNT %lu.", dimNum,
            hccl::SHAPE_INFO_COUNT);
        return ge::GRAPH_FAILED;
    }
    const std::vector<int64_t> sendInfo = PackShapeMetadata(inputShape);
    HCCL_GE_CHK_RET(SendShapeMetadata(st, sendInfo));

    // ---- 第二段: 数据 (32B 对齐 count) ----
    ret = HcclSend(st.inputPtrs[0], st.count, st.dataType, st.destRank, st.comm, static_cast<aclrtStream>(st.stream));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomSendOp][LaunchHcclOpDynamic] HcclSend data failed, count=%lu, ret=%d.", st.count,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    HCCL_INFO(
        "[HcomSendOp][LaunchHcclOpDynamic] dimNum=%ld count=%lu destRank=%u (two-phase send done).", dimNum, st.count,
        st.destRank);
    return ge::GRAPH_SUCCESS;
}

// 动态路径拆分: shape 元数据组包 (纯函数; [0]=维数, [1..]=各维, 尾部 zero-init)
std::vector<int64_t> HcomSendOp::PackShapeMetadata(const gert::Shape& inputShape)
{
    const int64_t dimNum = static_cast<int64_t>(inputShape.GetDimNum());
    std::vector<int64_t> sendInfo(hccl::SHAPE_INFO_COUNT, 0);
    sendInfo[0] = dimNum;
    for (int64_t i = 0; i < dimNum; i++) {
        sendInfo[static_cast<size_t>(i) + 1] = inputShape.GetDim(static_cast<size_t>(i));
    }
    return sendInfo;
}

// 动态路径拆分: 元数据发送 (malloc + H2D + HcclSend + 同步)
ge::graphStatus HcomSendOp::SendShapeMetadata(const hccl::HcclOpState& st, const std::vector<int64_t>& sendInfo)
{
    const uint64_t shapeBufSize = hccl::SHAPE_INFO_COUNT * sizeof(int64_t);
    void* devShapeBuf = nullptr;
    aclError aclRet = aclrtMalloc(&devShapeBuf, shapeBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR(
            "[HcomSendOp][LaunchHcclOpDynamic] aclrtMalloc shape buf failed, size=%lu, ret=%d.", shapeBufSize,
            static_cast<int>(aclRet));
        return ge::GRAPH_FAILED;
    }
    std::unique_ptr<void, AclrtMemDeleter> devShapeBufGuard(devShapeBuf);
    HcclResult ret = hccl::HcclRtMemSyncCopy(
        devShapeBuf, shapeBufSize, sendInfo.data(), shapeBufSize,
        hccl::HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomSendOp][LaunchHcclOpDynamic] H2D shape copy failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    ret = HcclSend(
        devShapeBuf, hccl::SHAPE_INFO_COUNT, HCCL_DATA_TYPE_INT64, st.destRank, st.comm,
        static_cast<aclrtStream>(st.stream));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomSendOp][LaunchHcclOpDynamic] HcclSend shape metadata failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    aclRet = aclrtSynchronizeStream(static_cast<aclrtStream>(st.stream));
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR(
            "[HcomSendOp][LaunchHcclOpDynamic] aclrtSynchronizeStream failed, ret=%d.", static_cast<int>(aclRet));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomSendOp::UpdateHostArgs(gert::UpdateArgsContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomSendOp][UpdateHostArgs] update args context is null.");
        return ge::GRAPH_FAILED;
    }
    // sink 模型每次执行前 GE 回调: 只刷新 inputAddr (Send 无 output, output 侧恒 nullptr)
    const gert::Tensor* inputTensor = ctx->GetInputTensor(INPUT_INDEX);
    if (inputTensor == nullptr) {
        HCCL_ERROR("[HcomSendOp][UpdateHostArgs] input tensor is null.");
        return ge::GRAPH_FAILED;
    }
    const gert::KernelArgs* hostArgs = ctx->GetKernelArgs(gert::Placement::kPlacementHost, 0);
    if (hostArgs == nullptr || hostArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomSendOp][UpdateHostArgs] hostArgs is null.");
        return ge::GRAPH_FAILED;
    }
    auto* args = static_cast<HcomSendArgs*>(hostArgs->args_data);
    args->inputAddr = const_cast<void*>(inputTensor->GetAddr());

    HCCL_DEBUG("[HcomSendOp][UpdateHostArgs] inputAddr=%p.", args->inputAddr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomSendOp::InferShape(gert::InferShapeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomSendOp][InferShape] infer shape context is null.");
        return ge::GRAPH_FAILED;
    }
    // HcomSend IR 原型无 OUTPUT 锚点; 若出现 output 节点属 IR 变更, 显式报错防止静默错配
    if (ctx->GetComputeNodeOutputNum() != 0) {
        HCCL_ERROR(
            "[HcomSendOp][InferShape] unexpected output num %zu (HcomSend IR has no output anchor).",
            ctx->GetComputeNodeOutputNum());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomSendOp::InferDataType(gert::InferDataTypeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomSendOp][InferDataType] infer datatype context is null.");
        return ge::GRAPH_FAILED;
    }
    if (ctx->GetComputeNodeOutputNum() != 0) {
        HCCL_ERROR(
            "[HcomSendOp][InferDataType] unexpected output num %zu (HcomSend IR has no output anchor).",
            ctx->GetComputeNodeOutputNum());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

const char* HcomSendOp::GetOpTypeName() const { return HCCL_KERNEL_OP_TYPE_SEND.c_str(); }

static ge::BaseCustomOp* CreateHcomSendOp() { return new HcomSendOp(); }
// 条件注册: 目标平台 (950/960) 且 GE/RTS 新接口在位才注册接管, 否则回退原 hccl_engine 路径
// (hccl_custom_op.h CreateSocGatedRegister: SoC 判定 + GE/RTS 接口在位判定)
static ge::CustomOpCreatorRegister* g_hcom_send_register
    = hccl::CreateSocGatedRegister(HCCL_KERNEL_OP_TYPE_SEND.c_str(), CreateHcomSendOp);
