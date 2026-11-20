/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 调用的 hcom.h (pkg_inc/hccl/hcom.h) 接口：
//   HcomCreateCommCCLbuffer  — 创建通信域 CCL buffer（hccl_custom_op.cc: GetOrCreateInCCLbuf）
//   HcomGetInCCLbuffer       — 获取输入 CCL buffer（hccl_custom_op.cc: GetOrCreateInCCLbuf）
// Broadcast 只用 InCCLbuf（对照 InfoStore RefreshOutputAddr 的 BROADCAST 专用重载, 从 InCCLbuf 取）
// 另调用 hccl_comm.h: HcclGetRankSize（CalcResourcesStatic）
// CCL buffer 尺寸获取（CalcResourcesStatic）: GetOrCreateInCCLbuf 实际尺寸 / GetCCLBufferAvailableSize 环境推导
// 另使用常量: HCCL_KERNEL_OP_TYPE_BROADCAST (hcom.h), HCCL_CMD_BROADCAST (hccl_types.h)
// 通信调用: 动态路径 HcclBroadcast (单算子), 静态/Direct/Refresh 路径 HcclBroadcastGraphMode —
//           均本仓 src/ops/broadcast/broadcast.h, hccl.map 导出, extern "C" 前向声明,
//           单 buf in-place 带 root 参数 (区别于 AG 的双 buf 无 root)
// 公共工具 (CalcLoopMaxTime/BuildTagVec/CalcOnlineResources/HcclRtMemAddrAsyncCopyWithOffset) 见
//           hccl_op_utils.h/.cc 与 hccl_rt_utils.h/.cc

#include "hcom_broadcast_op.h"

#include <cstddef>

#include "acl/acl_rt.h"
#include "hccl/hccl_comm.h"
#include "hccl/hcom.h"
#include "hccl_op_utils.h"
#include "hccl_rt_utils.h"
#include "log.h"

// 前向声明, 避免引入内部头文件依赖链 (签名对照本仓 src/ops/broadcast/broadcast.h)
extern "C" {
HcclResult HcclSetAivCoreLimitGraphMode(const char* group, uint32_t aivCoreLimit);
HcclResult
HcclBroadcast(void* buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream);
HcclResult HcclBroadcastGraphMode(
    void* buf, uint64_t count, HcclDataType dataType, uint32_t root, const char* group, aclrtStream stream,
    const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize);
}

// ExtractParams 公共校验: 首 input 判空、attrs 判空、root_rank 校验、group/stream 落 st、dataType 转换
// (静态/动态路径共用); attr root_rank (index 0, REQUIRED): 广播源 rank, 越界校验由 HCCL 库内部负责
static ge::graphStatus ExtractCommonParams(hccl::HcclOpState& st, const char* fnName, const gert::RuntimeAttrs*& attrs)
{
    const gert::Tensor* firstInput = st.ctx->GetInputTensor(HcomBroadcastOp::INPUT_INDEX);
    if (firstInput == nullptr) {
        HCCL_ERROR("[HcomBroadcastOp][%s] input tensor is null.", fnName);
        return ge::GRAPH_FAILED;
    }
    attrs = st.ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomBroadcastOp][%s] attrs is null.", fnName);
        return ge::GRAPH_FAILED;
    }
    const int64_t* attrRootRank = attrs->GetInt(HcomBroadcastOp::ATTR_ROOT_RANK);
    if (attrRootRank == nullptr) {
        HCCL_ERROR("[HcomBroadcastOp][%s] root_rank attr is missing.", fnName);
        return ge::GRAPH_FAILED;
    }
    if (*attrRootRank < 0 || *attrRootRank > UINT32_MAX) {
        HCCL_ERROR("[HcomBroadcastOp][%s] invalid root_rank %ld.", fnName, *attrRootRank);
        return ge::GRAPH_FAILED;
    }
    st.root = static_cast<uint32_t>(*attrRootRank);
    st.group = attrs->GetStr(HcomBroadcastOp::ATTR_GROUP);
    if (st.group == nullptr) {
        HCCL_ERROR("[HcomBroadcastOp][%s] group attr is null.", fnName);
        return ge::GRAPH_FAILED;
    }
    st.stream = st.ctx->GetStream();

    st.dataType = hccl::GeDataTypeToHccl(firstInput->GetDataType());
    if (st.dataType == HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR(
            "[HcomBroadcastOp][%s] unsupported data type %d.", fnName, static_cast<int>(firstInput->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 512B 对齐尺寸计算 + 溢出防御 (对照引擎 CalcCountForAlignedOp 的 INVALID_U64 guard):
// GetSize() 返回 size_t (无负值), 接近 UINT64_MAX 时对齐加法回绕
static ge::graphStatus CalcAlignedSize(size_t inputIdx, uint64_t tensorSize, uint64_t& alignedSize)
{
    constexpr uint64_t ALIGNED_SIZE = 512;
    if (tensorSize > UINT64_MAX - ALIGNED_SIZE) {
        HCCL_ERROR(
            "[HcomBroadcastOp][ExtractParamsStatic] input[%zu] size %lu is invalid (negative or overflow).", inputIdx,
            tensorSize);
        return ge::GRAPH_FAILED;
    }
    alignedSize = (tensorSize + ALIGNED_SIZE - 1) / ALIGNED_SIZE * ALIGNED_SIZE;
    return ge::GRAPH_SUCCESS;
}

// 静态 input 遍历: output[i] 引用 input[i] + 512B 对齐累加 + 溢出防御 (对照 CalcCountForAlignedOp)
static ge::graphStatus AllocRefOutputsAndCalcCount(hccl::HcclOpState& st, size_t& inputCount)
{
    uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    if (unitSize == 0) {
        HCCL_ERROR(
            "[HcomBroadcastOp][ExtractParamsStatic] unitSize is 0 (dataType=%d).", static_cast<int>(st.dataType));
        return ge::GRAPH_FAILED;
    }
    uint64_t totalAlignedSize = 0;
    inputCount = 0;

    for (size_t i = 0;; i++) {
        const gert::Tensor* inTensor = st.ctx->GetInputTensor(i);
        if (inTensor == nullptr) {
            break;
        }
        inputCount++;

        gert::Tensor* outTensor = st.ctx->MakeOutputRefInput(i, i);
        if (outTensor == nullptr) {
            HCCL_ERROR("[HcomBroadcastOp][ExtractParamsStatic] MakeOutputRefInput(%zu, %zu) failed.", i, i);
            return ge::GRAPH_FAILED;
        }

        uint64_t alignedSize = 0;
        HCCL_GE_CHK_RET(CalcAlignedSize(i, inTensor->GetSize(), alignedSize));
        if (totalAlignedSize > UINT64_MAX - alignedSize) {
            HCCL_ERROR("[HcomBroadcastOp][ExtractParamsStatic] total aligned size overflow at input[%zu].", i);
            return ge::GRAPH_FAILED;
        }
        totalAlignedSize += alignedSize;

        st.inputPtrs.push_back(const_cast<void*>(inTensor->GetAddr()));
        st.outputPtrs.push_back(outTensor->GetAddr());

        HCCL_INFO(
            "[HcomBroadcastOp][ExtractParamsStatic] input[%zu] addr=%p tensorSize=%lu alignedSize=%lu (in-place refs).",
            i, inTensor->GetAddr(), inTensor->GetSize(), alignedSize);
    }
    if (inputCount == 0) {
        HCCL_ERROR("[HcomBroadcastOp][ExtractParamsStatic] no input tensors.");
        return ge::GRAPH_FAILED;
    }
    st.count = totalAlignedSize / unitSize;
    if (st.count == 0) {
        HCCL_ERROR("[HcomBroadcastOp][ExtractParamsStatic] input count is 0 (totalAlignedSize=%lu).", totalAlignedSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 动态路径 input 遍历: output[i] 引用 input[i] (MakeOutputRefInput), 记录地址 (count 由首输入单独结算)
static ge::graphStatus AllocRefOutputsDynamic(hccl::HcclOpState& st, size_t& inputCount)
{
    inputCount = 0;
    for (size_t i = 0;; i++) {
        const gert::Tensor* inTensor = st.ctx->GetInputTensor(i);
        if (inTensor == nullptr) {
            break;
        }
        inputCount++;

        gert::Tensor* outTensor = st.ctx->MakeOutputRefInput(i, i);
        if (outTensor == nullptr) {
            HCCL_ERROR("[HcomBroadcastOp][ExtractParamsDynamic] MakeOutputRefInput(%zu, %zu) failed.", i, i);
            return ge::GRAPH_FAILED;
        }

        HCCL_INFO(
            "[HcomBroadcastOp][ExtractParamsDynamic] input[%zu] addr=%p output[%zu] addr=%p "
            "outShapeSize=%ld (output references input, addr should be identical).",
            i, inTensor->GetAddr(), i, outTensor->GetAddr(), outTensor->GetShapeSize());
        st.inputPtrs.push_back(const_cast<void*>(inTensor->GetAddr()));
        st.outputPtrs.push_back(outTensor->GetAddr());
    }
    if (inputCount == 0) {
        HCCL_ERROR("[HcomBroadcastOp][ExtractParamsDynamic] no input tensors.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomBroadcastOp::ExtractParamsStatic(hccl::HcclOpState& st)
{
    // 静态路径 (InfoStore-style): count 对照引擎 CalcCountForAlignedOp —
    // ALLREDUCE/BROADCAST 专用 512B 对齐口径: 逐 input GetSize() 向上对齐 512B 累加, count = totalSize / unitSize
    // (GetSize 是 TensorDesc 分配字节数, 区别于动态路径的 shapeSize*unitSize 口径)
    // 通信地址只用 inputPtrs[0] (InfoStore kernel 层 hcclInfos==1, 融合场景假设地址连续)
    // Broadcast 无 crack 清零 (GetTensorNum 清零列表仅 RS/AllReduce/Reduce, 区别于 AllReduce 的 crack 追踪)
    const gert::RuntimeAttrs* attrs = nullptr;
    HCCL_GE_CHK_RET(ExtractCommonParams(st, "ExtractParamsStatic", attrs));
    const int64_t* fusion = attrs->GetInt(ATTR_FUSION);
    const int64_t* fusionId = attrs->GetInt(ATTR_FUSION_ID);

    size_t inputCount = 0;
    HCCL_GE_CHK_RET(AllocRefOutputsAndCalcCount(st, inputCount));

    HCCL_INFO(
        "[HcomBroadcastOp][ExtractParamsStatic] inputCount=%zu input[0]=%p output[0]=%p root=%u count=%lu "
        "dataType=%d group=%s stream=%p fusion=%ld fusionId=%ld.",
        inputCount, st.inputPtrs[0], st.outputPtrs[0], st.root, st.count, st.dataType, st.group ? st.group : "(null)",
        st.stream, fusion ? *fusion : -1, fusionId ? *fusionId : -1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomBroadcastOp::ExtractParamsDynamic(hccl::HcclOpState& st)
{
    // 动态路径 (lowering-style): 对照 lowering HcomBroadcastKernelV2
    // - 只校验 inputNum >= 1 (区别于 RS 动态路径强制 ==1): 多 input 不报错, count 与通信地址只用第一个
    // - count 用 GetCountByShapeAlign 512B 对齐口径:
    //   count = RoundUp(shapeSize * unitSize, 512) / unitSize (区别于 AG 的非对齐 GetCountByShape)
    // - root 从 IR attr root_rank (index 0) 读入 st.root, 传通信调用第 4 参数
    //   (root < rankSize 校验在静态路径 CalcResourcesStatic, 对照 Reduce 同款)
    const gert::RuntimeAttrs* attrs = nullptr;
    HCCL_GE_CHK_RET(ExtractCommonParams(st, "ExtractParamsDynamic", attrs));

    // 512B 对齐口径 (对照 lowering GetCountByShapeAlign, memAlignSize=512):
    // count = RoundUp(shapeSize * unitSize, 512) / unitSize
    const uint64_t memAlignSize = 512;
    const gert::Tensor* firstInput = st.ctx->GetInputTensor(INPUT_INDEX);
    uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    if (unitSize == 0) {
        HCCL_ERROR(
            "[HcomBroadcastOp][ExtractParamsDynamic] unitSize is 0 (dataType=%d).", static_cast<int>(st.dataType));
        return ge::GRAPH_FAILED;
    }
    uint64_t shapeSize = static_cast<uint64_t>(firstInput->GetShapeSize());
    // 溢出/负值防御: shapeSize 为负时 cast 成巨数, shapeSize*unitSize+512 对齐回绕时拒绝
    if (shapeSize > (UINT64_MAX - (memAlignSize - 1)) / unitSize) {
        HCCL_ERROR(
            "[HcomBroadcastOp][ExtractParamsDynamic] input size overflow: shapeSize=%lu unitSize=%lu.", shapeSize,
            unitSize);
        return ge::GRAPH_FAILED;
    }
    uint64_t alignedBytes = (shapeSize * unitSize + memAlignSize - 1) / memAlignSize * memAlignSize;
    st.count = alignedBytes / unitSize;
    if (st.count == 0) {
        HCCL_ERROR("[HcomBroadcastOp][ExtractParamsDynamic] input count is 0 (shapeSize=%lu).", shapeSize);
        return ge::GRAPH_FAILED;
    }

    // 遍历所有 input, 输出与输入逐对: output[i] 经 MakeOutputRefInput(i, i) 引用 input[i]
    size_t inputCount = 0;
    HCCL_GE_CHK_RET(AllocRefOutputsDynamic(st, inputCount));

    HCCL_INFO(
        "[HcomBroadcastOp][ExtractParamsDynamic] inputCount=%zu root=%u count=%lu (512B aligned) "
        "dataType=%d group=%s stream=%p.",
        inputCount, st.root, st.count, st.dataType, st.group ? st.group : "(null)", st.stream);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomBroadcastOp::CalcResourcesStatic(hccl::HcclOpState& st)
{
    // 对照 AllReduce/AG CalcResourcesStatic: rankSize 入 st.rankSize
    // (Broadcast 组 1 loop 公式不消费 rankSize, CalcOnlineResources 资源计算仍需要)
    uint32_t rankSize = 0;
    HcclResult ret = HcclGetRankSize(st.comm, &rankSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomBroadcastOp][CalcResourcesStatic] HcclGetRankSize failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    // root 上界校验: 旧图路径由图编译上游保证 root < rankSize, eager 路径无该环节,
    // 在 rankSize 可得处显式拦截 (对照 Reduce 同款; root 越界时广播源 rank 非法)
    if (st.root >= rankSize) {
        HCCL_ERROR(
            "[HcomBroadcastOp][CalcResourcesStatic] root %u out of range [0, %u), group '%s'.", st.root, rankSize,
            st.group ? st.group : "(null)");
        return ge::GRAPH_FAILED;
    }
    st.rankSize = rankSize;

    // st.cclBuffSize: 实际 in-CCL buffer 尺寸 — 仅 Refresh 路径消费 (GenerateTagVec 的 loopMaxTime 与
    // LaunchRefresh 的 maxCountPerLoop, 两处均门控 needRefresh); Direct 直传用户 buffer 不经 CCL 中转,
    // 无需创建 (get-or-create 语义对照 InfoStore GetHcomInCCLbufferSize, 口径同 AllReduce)
    if (st.needRefresh) {
        void* cclBufPtr = nullptr;
        HCCL_GE_CHK_RET(GetOrCreateInCCLbuf(st.group, cclBufPtr, st.cclBuffSize));
    }

    // 在线计算输入: 环境推导 available size, 与实际 buffer 尺寸是两个来源 (口径同 AllReduce)
    uint64_t cclBufferAvailSize = 0;
    HCCL_GE_CHK_RET(hccl::GetCCLBufferAvailableSize(cclBufferAvailSize));

    return hccl::CalcOnlineResources(
        st, rankSize, cclBufferAvailSize, HCCL_KERNEL_OP_TYPE_BROADCAST, HCCL_CMD_BROADCAST);
}

ge::graphStatus HcomBroadcastOp::GenerateTagVec(hccl::HcclOpState& st)
{
    // 对应 InfoStore GenerateOpTagFromTaskInfo: needRefresh 时算 loopMaxTime
    // Broadcast 组 1 公式 (对照 InfoStore GetOpKernelLoopTime: Broadcast 与 AllReduce/Reduce/Send 同组):
    // maxCountPerLoop = cclBuffSize / unitSize (无 rankSize 因子, 区别于 AG 的组 2)
    // 守卫: cclBuffSize < unitSize 时 maxCountPerLoop==0 (tag 数为 0 且循环死循环), 报错拒绝
    uint32_t loopMaxTime = 0;
    if (st.needRefresh) {
        uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
        if (unitSize == 0) {
            HCCL_ERROR("[HcomBroadcastOp][GenerateTagVec] unitSize is 0 (dataType=%d).", static_cast<int>(st.dataType));
            return ge::GRAPH_FAILED;
        }
        if (st.cclBuffSize / unitSize == 0) {
            HCCL_ERROR(
                "[HcomBroadcastOp][GenerateTagVec] cclBuffSize=%lu cannot hold one element (unitSize=%lu).",
                st.cclBuffSize, unitSize);
            return ge::GRAPH_FAILED;
        }
        // 组 1 公式不消费 rankSize, 惯例传 1 (对齐 AllReduce)
        HCCL_GE_CHK_RET(
            hccl::CalcLoopMaxTime(HCCL_KERNEL_OP_TYPE_BROADCAST, st.count, unitSize, st.cclBuffSize, 1, loopMaxTime));
    }

    // 对应 InfoStore GetTagVectorInfo: loopMaxTime==0 → {tag}; loopMaxTime>0 → {tag + "_looptime_" + i}
    return hccl::BuildTagVec(HCCL_KERNEL_OP_TYPE_BROADCAST, st.group, loopMaxTime, st.tags);
}

ge::graphStatus HcomBroadcastOp::LaunchHcclOpStatic(hccl::HcclOpState& st)
{
    // 静态路径 (InfoStore-style): needRefresh 判定已收编基类 Execute, 此处只消费 st.needRefresh
    // scratch 创建流程在此 (区别于动态路径传 nullptr/0):
    // CalcResourcesStatic 已算 st.scratchMemSize (AIV 算法时 512B), 此处 MallocWorkSpace 实际分配
    if (st.scratchMemSize > 0) {
        st.scratchMem = st.ctx->MallocWorkSpace(st.scratchMemSize);
        if (st.scratchMem == nullptr) {
            HCCL_ERROR("[HcomBroadcastOp][LaunchHcclOpStatic] MallocWorkSpace failed, size=%lu.", st.scratchMemSize);
            return ge::GRAPH_FAILED;
        }
    }

    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomBroadcastOp][LaunchHcclOpStatic] HcclSetAivCoreLimitGraphMode failed, ret=%d.",
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    // needRefresh 判定已收编基类 Execute
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

ge::graphStatus HcomBroadcastOp::LaunchHcclOpDynamic(hccl::HcclOpState& st)
{
    // 动态路径 (lowering-style): 对照 lowering HcomLaunchBroadcastKernelV2 —
    // SetAivCoreLimit 后单次 HcclBroadcast 直传用户 buffer, 不用 CCL buffer, 无 crack
    // 单算子接口 (hccl.map 公共导出): 免 tag 构造/group 串解析; GraphMode 为烘焙态接口, engine 动态路径不使用
    // 单 buf in-place: 通信写 input 地址即写 output (output 经 MakeOutputRefInput 与 input 同址)
    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomBroadcastOp][LaunchHcclOpDynamic] HcclSetAivCoreLimitGraphMode failed, ret=%d.",
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    HCCL_INFO(
        "[HcomBroadcastOp][LaunchHcclOpDynamic] input=%p output=%p root=%u count=%lu comm=%p (direct, user "
        "buffer, in-place, single-op).",
        st.inputPtrs[0], st.outputPtrs[0], st.root, st.count, st.comm);

    // 第 4 参数 root: 广播源 rank (区别于 AG 无 root); 单 buf: 只传 inputPtrs[0], 无独立 output buf 参数
    // (lowering 同款: outputAddrs 参数 [[maybe_unused]], 通信只用 inputAddrs[0])
    // comm 由基类 GetCommunicator 按 group 预解析 (hccl_custom_op.cc), 对照 lowering
    // HcomBroadcastKernelV2 的 HcomGetCommHandleByGroup 预解析后单算子直传
    ret = HcclBroadcast(st.inputPtrs[0], st.count, st.dataType, st.root, st.comm, static_cast<aclrtStream>(st.stream));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomBroadcastOp][LaunchHcclOpDynamic] HcclBroadcast failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomBroadcastOp::LaunchDirect(hccl::HcclOpState& st)
{
    // Direct 路径: GE HCCL engine 的 needRefresh=false 路径不做 CCL buffer 检查
    // (HCCL 库内部 HcclExecOp 能自行处理大数据分块), 保留此行为
    // Broadcast 无 crack 清零 (GetTensorNum 清零列表不含 Broadcast)
    HCCL_INFO(
        "[HcomBroadcastOp][LaunchDirect] input=%p output=%p root=%u count=%lu (non-refresh, user buffer "
        "directly, in-place).",
        st.inputPtrs[0], st.outputPtrs[0], st.root, st.count);

    // 单 buf 直传 + 第 4 参数 root (区别于 AG 双 buf 无 root); scratch 由 LaunchHcclOpStatic 分配
    HcclResult ret = HcclBroadcastGraphMode(
        st.inputPtrs[0], st.count, st.dataType, st.root, st.group, static_cast<aclrtStream>(st.stream),
        st.tags[0].c_str(), st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(),
        st.scratchMem, st.scratchMemSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomBroadcastOp][LaunchDirect] HcclBroadcastGraphMode failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// LaunchRefresh 循环上下文: InCCLbuf 与 args 字段 device 地址
// (devXxx 为 args struct 各字段的 device 侧地址, D2D ADDR 经其二级指针间接寻址;
//  Broadcast 只用 InCCLbuf — 原地执行后结果留在原处, 无 OutCCLbuf)
struct HcomBroadcastRefreshCtx {
    void* commInputPtr = nullptr;
    uint64_t commInputSize = 0;
    void* devCclInputPtr = nullptr;
    void* devInputAddr = nullptr;
    void* devOutputAddr = nullptr;
    void* devArgsData = nullptr;
    uint64_t outputMaxSize = 0;
    uint64_t maxCountPerLoop = 0;
};

// 刷新上下文的 args 部分: maxCountPerLoop 计算、args struct 分配与字段地址换算
// (args 经 MallocReadOnlyDevArgs 分配 host+device 双份, GE 统一 H2D 同步 — 对应 InfoStore
// RefreshInputAddr/RefreshOutputAddr(BROADCAST 重载) 的 ①④ 次拷贝; InCCLbuf 由调用方经 GetOrCreate 获取后填入 ctx)
static ge::graphStatus InitRefreshArgs(hccl::HcclOpState& st, HcomBroadcastRefreshCtx& ctx)
{
    const uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    if (unitSize == 0) {
        HCCL_ERROR("[HcomBroadcastOp][LaunchRefresh] unitSize is 0 (dataType=%d).", static_cast<int>(st.dataType));
        return ge::GRAPH_FAILED;
    }
    // maxCountPerLoop 与 GenerateTagVec 的 loopMaxTime 同源 (st.cclBuffSize 单一来源), 组 1 公式无 rankSize
    ctx.maxCountPerLoop = st.cclBuffSize / unitSize;
    if (ctx.maxCountPerLoop == 0) {
        HCCL_ERROR(
            "[HcomBroadcastOp][LaunchRefresh] cclBuffSize=%lu cannot hold one element (unitSize=%lu).", st.cclBuffSize,
            unitSize);
        return ge::GRAPH_FAILED;
    }
    // ⑤ 侧 dstMax = count*unitSize (无膨胀, 区别于 AG)
    ctx.outputMaxSize = st.count * unitSize;

    // args struct: InCCLbuf 地址 (固定) + 用户地址 (UpdateHostArgs 刷新)
    // 对照 AllReduce/Reduce/ReduceScatter 的 LaunchRefresh: 每次 Execute 构造并注册。
    // GetKernelArgs 的 index 按 per-ctx deque 计, 每个 sink 的 ctx 首次注册即其 index 0,
    // UpdateHostArgs 固定刷 index 0 亦即本 ctx 的注册; 若跨执行复用成员缓存, 后续 sink 的
    // D2D ADDR task 会解引用首个注册的旧地址 (多轮执行静默失效)。
    HcomBroadcastArgs args = {ctx.commInputPtr, st.inputPtrs[0], st.outputPtrs[0]};
    const gert::KernelArgs* devArgs = st.ctx->MallocReadOnlyDevArgs(&args, sizeof(args));
    if (devArgs == nullptr || devArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomBroadcastOp][LaunchRefresh] MallocReadOnlyDevArgs failed.");
        return ge::GRAPH_FAILED;
    }
    // devArgs->args_data 是 device 侧 args 起始地址
    // 字段偏移: cclInputPtr=0, inputAddr=8, outputAddr=16
    char* devArgsBase = static_cast<char*>(devArgs->args_data);
    ctx.devArgsData = devArgs->args_data;
    ctx.devCclInputPtr = devArgsBase + offsetof(HcomBroadcastArgs, cclInputPtr);
    ctx.devInputAddr = devArgsBase + offsetof(HcomBroadcastArgs, inputAddr);
    ctx.devOutputAddr = devArgsBase + offsetof(HcomBroadcastArgs, outputAddr);
    return ge::GRAPH_SUCCESS;
}

// LaunchRefresh 单迭代游标
struct HcomBroadcastLoopCursor {
    uint64_t loopTime = 0;
    uint64_t inputOffset = 0;
    uint64_t outputOffset = 0;
    uint64_t curCount = 0;
};

// ⑤ 等长拷出 — InCCLbuf → 用户 output (统一带偏移, 单一形态, 单迭代 outputOffset=0)
static ge::graphStatus
CopyCclToOutput(hccl::HcclOpState& st, const HcomBroadcastRefreshCtx& ctx, const HcomBroadcastLoopCursor& cursor)
{
    // ⑤ D2D ADDR 带偏移: InCCLbuf → 用户 output (等长回写, 统一带偏移, 单迭代 outputOffset=0)
    //   src = devCclInputPtr  (二级指针 → 解引用得 InCCLbuf 地址; 结果留在原处)
    //   dst = devOutputAddr   (二级指针 → 解引用得用户 output 地址; 经 MakeOutputRefInput 与 input 同址)
    const uint64_t curSize = cursor.curCount * hccl::GetHcclDataTypeSize(st.dataType);
    HcclResult hrtRet = hccl::HcclRtMemAddrAsyncCopyWithOffset(
        ctx.devOutputAddr, ctx.outputMaxSize, cursor.outputOffset, ctx.devCclInputPtr, curSize, 0,
        static_cast<aclrtStream>(st.stream));
    if (hrtRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomBroadcastOp][RefreshOneLoop] D2D ADDR CCL to output failed, loopTime=%lu, ret=%d.", cursor.loopTime,
            static_cast<int>(hrtRet));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// LaunchRefresh 单迭代: ② D2D ADDR 搬入 → ③ 单 buf Broadcast 带 root → ⑤ 等长拷出 (每迭代独立 tag: _looptime_N)
static ge::graphStatus
RefreshOneLoop(hccl::HcclOpState& st, const HcomBroadcastRefreshCtx& ctx, HcomBroadcastLoopCursor& cursor)
{
    const uint64_t curSize = cursor.curCount * hccl::GetHcclDataTypeSize(st.dataType);

    // tag 越界守卫, 对照 InfoStore HcomBroadcastLoop
    if (cursor.loopTime >= st.tags.size()) {
        HCCL_ERROR(
            "[HcomBroadcastOp][RefreshOneLoop] tagVec access out-of-bounds, loopTime=%lu tags.size=%zu.",
            cursor.loopTime, st.tags.size());
        return ge::GRAPH_FAILED;
    }

    // ② D2D ADDR 带偏移: 用户 input → InCCLbuf (等长, 无膨胀)
    //   src = devInputAddr    (二级指针 → 解引用得用户 input 地址)
    //   dst = devCclInputPtr  (二级指针 → 解引用得 InCCLbuf 地址)
    HcclResult hrtRet = hccl::HcclRtMemAddrAsyncCopyWithOffset(
        ctx.devCclInputPtr, ctx.commInputSize, 0, ctx.devInputAddr, curSize, cursor.inputOffset,
        static_cast<aclrtStream>(st.stream));
    if (hrtRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomBroadcastOp][RefreshOneLoop] D2D ADDR input to CCL failed, loopTime=%lu, ret=%d.", cursor.loopTime,
            static_cast<int>(hrtRet));
        return ge::GRAPH_FAILED;
    }

    // ③ HcclBroadcastGraphMode: 在 InCCLbuf 上原地执行, root 的数据广播到所有 rank 的 InCCLbuf
    //   单 buf: 只传 commInputPtr (无独立 output buf 参数), 带 root (每迭代独立 tag: _looptime_N)
    HcclResult ret = HcclBroadcastGraphMode(
        ctx.commInputPtr, cursor.curCount, st.dataType, st.root, st.group, static_cast<aclrtStream>(st.stream),
        st.tags[cursor.loopTime].c_str(), st.followStreams.empty() ? nullptr : st.followStreams.data(),
        st.followStreams.size(), st.scratchMem, st.scratchMemSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomBroadcastOp][RefreshOneLoop] HcclBroadcastGraphMode failed, loopTime=%lu, ret=%d.", cursor.loopTime,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    return CopyCclToOutput(st, ctx, cursor);
}

ge::graphStatus HcomBroadcastOp::LaunchRefresh(hccl::HcclOpState& st)
{
    // 地址刷新路径: args struct + ArgsUpdater, GE 统一 H2D
    // 多迭代 loop: 结构对照 InfoStore HcomBroadcastLoop, 组 1 公式无 rankSize
    // (Broadcast 只用 InCCLbuf — 原地执行后结果留在原处, 对照 InfoStore RefreshOutputAddr 的
    //  BROADCAST 专用重载; ② 等长拷入 / ③ 单 buf Broadcast 带 root / ⑤ 等长拷出, 无膨胀无散射)
    HcomBroadcastRefreshCtx ctx;
    HCCL_GE_CHK_RET(GetOrCreateInCCLbuf(st.group, ctx.commInputPtr, ctx.commInputSize));
    HCCL_GE_CHK_RET(InitRefreshArgs(st, ctx));

    // 大于 cclbuffer 时 count 取最大支持 count, 否则取剩余 count (无膨胀系数)
    HcomBroadcastLoopCursor cursor;
    for (uint64_t countLeft = st.count; countLeft > 0; countLeft -= cursor.curCount) {
        cursor.curCount = (countLeft > ctx.maxCountPerLoop) ? ctx.maxCountPerLoop : countLeft;
        HCCL_INFO(
            "[HcomBroadcastOp][LaunchRefresh] loopTime=%lu inputOffset=%lu outputOffset=%lu countLeft=%lu "
            "maxCountPerLoop=%lu.",
            cursor.loopTime, cursor.inputOffset, cursor.outputOffset, countLeft, ctx.maxCountPerLoop);

        HCCL_GE_CHK_RET(RefreshOneLoop(st, ctx, cursor));

        const uint64_t curSize = cursor.curCount * hccl::GetHcclDataTypeSize(st.dataType);
        cursor.inputOffset += curSize;
        cursor.outputOffset += curSize;
        cursor.loopTime++;
    }

    HCCL_INFO(
        "[HcomBroadcastOp][LaunchRefresh] devArgs=%p count=%lu maxCountPerLoop=%lu cclInput=%p "
        "inputAddr=%p outputAddr=%p root=%u.",
        ctx.devArgsData, st.count, ctx.maxCountPerLoop, ctx.commInputPtr, st.inputPtrs[0], st.outputPtrs[0], st.root);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomBroadcastOp::UpdateHostArgs(gert::UpdateArgsContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomBroadcastOp][UpdateHostArgs] update args context is null.");
        return ge::GRAPH_FAILED;
    }
    // GE 在 sink 模型每次执行前调: 刷新 args struct 中的 inputAddr/outputAddr
    // GE 统一 H2D 同步 host → device, D2D ADDR task 自动 dereference 新地址
    // (output 经 MakeOutputRefInput 与 input 同址, 两字段值相同, 仍分别刷新保持字段语义独立)
    const gert::Tensor* inputTensor = ctx->GetInputTensor(INPUT_INDEX);
    const gert::Tensor* outputTensor = ctx->GetOutputTensor(OUTPUT_INDEX);
    if (inputTensor == nullptr || outputTensor == nullptr) {
        HCCL_ERROR("[HcomBroadcastOp][UpdateHostArgs] input=%p output=%p.", inputTensor, outputTensor);
        return ge::GRAPH_FAILED;
    }

    const gert::KernelArgs* hostArgs = ctx->GetKernelArgs(gert::Placement::kPlacementHost, 0);
    if (hostArgs == nullptr || hostArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomBroadcastOp][UpdateHostArgs] hostArgs is null.");
        return ge::GRAPH_FAILED;
    }

    auto* args = static_cast<HcomBroadcastArgs*>(hostArgs->args_data);
    args->inputAddr = const_cast<void*>(inputTensor->GetAddr());
    args->outputAddr = const_cast<void*>(outputTensor->GetAddr());

    HCCL_INFO("[HcomBroadcastOp][UpdateHostArgs] inputAddr=%p outputAddr=%p.", args->inputAddr, args->outputAddr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomBroadcastOp::InferShape(gert::InferShapeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomBroadcastOp][InferShape] infer shape context is null.");
        return ge::GRAPH_FAILED;
    }
    // Broadcast 输出 shape = 输入 shape 逐对直通, 无 dim0 变换
    // (广播语义下所有 rank 输出相同, 区别于 AG 的 dim0×rankSize / RS 的 dim0/rankSize)
    // 不读 attrs: shape 不依赖 root_rank (root 只决定数据来源, 不决定形状)
    // DYNAMIC_INPUT/DYNAMIC_OUTPUT 一一对应: inputNum != outputNum 报错
    // 整 shape 拷贝, UNKNOWN_DIM 自然直通, 无需分支
    size_t inputNum = ctx->GetComputeNodeInputNum();
    size_t outputNum = ctx->GetComputeNodeOutputNum();
    if (inputNum != outputNum) {
        HCCL_ERROR("[HcomBroadcastOp][InferShape] inputNum=%zu != outputNum=%zu.", inputNum, outputNum);
        return ge::GRAPH_FAILED;
    }
    if (inputNum == 0) {
        HCCL_ERROR("[HcomBroadcastOp][InferShape] inputNum is 0.");
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < inputNum; i++) {
        const gert::Shape* inputShape = ctx->GetInputShape(i);
        if (inputShape == nullptr) {
            HCCL_ERROR("[HcomBroadcastOp][InferShape] input shape[%zu] is null.", i);
            return ge::GRAPH_FAILED;
        }
        gert::Shape* outputShape = ctx->GetOutputShape(i);
        if (outputShape == nullptr) {
            HCCL_ERROR("[HcomBroadcastOp][InferShape] output shape[%zu] is null.", i);
            return ge::GRAPH_FAILED;
        }
        *outputShape = *inputShape;
        HCCL_INFO("[HcomBroadcastOp][InferShape] [%zu] dimNum=%zu (passthrough).", i, outputShape->GetDimNum());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomBroadcastOp::InferDataType(gert::InferDataTypeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomBroadcastOp][InferDataType] infer datatype context is null.");
        return ge::GRAPH_FAILED;
    }
    // 数据类型逐对直通, DYNAMIC_INPUT/DYNAMIC_OUTPUT 一一对应
    size_t inputNum = ctx->GetComputeNodeInputNum();
    size_t outputNum = ctx->GetComputeNodeOutputNum();
    if (inputNum != outputNum) {
        HCCL_ERROR("[HcomBroadcastOp][InferDataType] inputNum=%zu != outputNum=%zu.", inputNum, outputNum);
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < inputNum; i++) {
        ge::DataType dataType = ctx->GetInputDataType(i);
        if (dataType == ge::DT_UNDEFINED) {
            HCCL_ERROR("[HcomBroadcastOp][InferDataType] input dtype[%zu] is undefined.", i);
            return ge::GRAPH_FAILED;
        }
        ge::graphStatus ret = ctx->SetOutputDataType(i, dataType);
        if (ret != ge::GRAPH_SUCCESS) {
            HCCL_ERROR("[HcomBroadcastOp][InferDataType] SetOutputDataType[%zu] failed.", i);
            return ret;
        }
    }
    return ge::GRAPH_SUCCESS;
}

const char* HcomBroadcastOp::GetOpTypeName() const { return HCCL_KERNEL_OP_TYPE_BROADCAST.c_str(); }

static ge::BaseCustomOp* CreateHcomBroadcastOp() { return new HcomBroadcastOp(); }
// 条件注册: 目标平台 (950/960) 且 GE/RTS 新接口在位才注册接管, 否则回退原 hccl_engine 路径
// (hccl_custom_op.h CreateSocGatedRegister: SoC 判定 + GE/RTS 接口在位判定)
static ge::CustomOpCreatorRegister* g_hcom_broadcast_register
    = hccl::CreateSocGatedRegister(HCCL_KERNEL_OP_TYPE_BROADCAST.c_str(), CreateHcomBroadcastOp);
