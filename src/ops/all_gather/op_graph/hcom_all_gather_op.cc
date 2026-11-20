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
//   HcomCreateCommCCLbuffer  — 创建通信域 CCL buffer（hccl_custom_op.cc: GetOrCreateInCCLbuf/OutCCLbuf）
//   HcomGetInCCLbuffer       — 获取输入 CCL buffer（hccl_custom_op.cc: GetOrCreateInCCLbuf）
//   HcomGetOutCCLbuffer      — 获取输出 CCL buffer（hccl_custom_op.cc: GetOrCreateOutCCLbuf）
// 另调用 hccl_comm.h: HcclGetRankSize（CalcResourcesStatic）
// CCL buffer 尺寸获取（CalcResourcesStatic）: GetOrCreateInCCLbuf 实际尺寸 / GetCCLBufferAvailableSize 环境推导
// 另使用 hcom.h 常量: HCCL_KERNEL_OP_TYPE_ALLGATHER, HCCL_CMD_ALLGATHER
// 通信调用: 动态路径 HcclAllGather (单算子), 静态/Direct/Refresh 路径 HcclAllGatherGraphMode —
//           均本仓 src/ops/all_gather/all_gather.h, hccl.map 导出, extern "C" 前向声明,
//           无 reduction 参数 (区别于 AllReduce/Reduce/RS)
// 公共工具 (CalcLoopMaxTime/BuildTagVec/CalcOnlineResources/HcclRtMemAddrAsyncCopyWithOffset) 见
//           hccl_op_utils.h/.cc 与 hccl_rt_utils.h/.cc

#include "hcom_all_gather_op.h"

#include <cstddef>

#include "acl/acl_rt.h"
#include "hccl/hccl_comm.h"
#include "hccl/hcom.h"
#include "hccl_op_utils.h"
#include "hccl_rt_utils.h"
#include "log.h"

// 前向声明, 避免引入内部头文件依赖链 (签名对照本仓 src/ops/all_gather/all_gather.h)
extern "C" {
HcclResult HcclSetAivCoreLimitGraphMode(const char* group, uint32_t aivCoreLimit);
HcclResult HcclAllGather(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream);
HcclResult HcclAllGatherGraphMode(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, const char* group, aclrtStream stream,
    const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize);
}

// 输出 shape 构造: 输入第 0 维 × rankSize, 其余维度不变 (与 InferShape 契约一致)
// UNKNOWN_DIM 直通 (不乘, 保持未知标记); dimNum==0 (标量) 输出 [rankSize]
// origin/storage 双写 (eager MallocOutputTensor 消费 StorageShape)
static gert::StorageShape BuildAllGatherOutShape(const gert::Shape& inShape, uint64_t rankSize)
{
    gert::StorageShape outShape;
    if (inShape.GetDimNum() == 0) {
        outShape.MutableOriginShape().AppendDim(static_cast<int64_t>(rankSize));
        outShape.MutableStorageShape().AppendDim(static_cast<int64_t>(rankSize));
        return outShape;
    }
    for (size_t i = 0; i < inShape.GetDimNum(); i++) {
        int64_t dim = inShape.GetDim(i);
        if (i == 0 && dim != ge::UNKNOWN_DIM) {
            dim = dim * static_cast<int64_t>(rankSize);
        }
        outShape.MutableOriginShape().AppendDim(dim);
        outShape.MutableStorageShape().AppendDim(dim);
    }
    return outShape;
}

// ExtractParams 公共校验: 首 input 判空、attrs 判空、rank_size 校验、dataType 转换 (静态/动态路径共用)
// attr rank_size (index 0, REQUIRED): 输出 shape 变换用; 通信域 rankSize 在 CalcResourcesStatic 校验一致性
static ge::graphStatus ExtractCommonParams(
    hccl::HcclOpState& st, const char* fnName, const gert::RuntimeAttrs*& attrs, uint64_t& attrRankSizeU,
    const char*& group)
{
    const gert::Tensor* firstInput = st.ctx->GetInputTensor(HcomAllGatherOp::INPUT_INDEX);
    if (firstInput == nullptr) {
        HCCL_ERROR("[HcomAllGatherOp][%s] input tensor is null.", fnName);
        return ge::GRAPH_FAILED;
    }
    attrs = st.ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomAllGatherOp][%s] attrs is null.", fnName);
        return ge::GRAPH_FAILED;
    }
    const int64_t* attrRankSize = attrs->GetInt(HcomAllGatherOp::ATTR_RANK_SIZE);
    if (attrRankSize == nullptr) {
        HCCL_ERROR("[HcomAllGatherOp][%s] rank_size attr is missing.", fnName);
        return ge::GRAPH_FAILED;
    }
    if (*attrRankSize <= 0) {
        HCCL_ERROR("[HcomAllGatherOp][%s] invalid rank_size %ld.", fnName, *attrRankSize);
        return ge::GRAPH_FAILED;
    }
    attrRankSizeU = static_cast<uint64_t>(*attrRankSize);
    group = attrs->GetStr(HcomAllGatherOp::ATTR_GROUP);
    if (group == nullptr) {
        HCCL_ERROR("[HcomAllGatherOp][%s] group attr is null.", fnName);
        return ge::GRAPH_FAILED;
    }

    st.dataType = hccl::GeDataTypeToHccl(firstInput->GetDataType());
    if (st.dataType == HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR(
            "[HcomAllGatherOp][%s] unsupported data type %d.", fnName, static_cast<int>(firstInput->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 单输入输出分配: BuildAllGatherOutShape + MallocOutputTensor + input/output 地址与 group/stream 落 st
// (静态/动态路径共用)
static ge::graphStatus
AllocOutputByRankSize(hccl::HcclOpState& st, const char* fnName, uint64_t attrRankSizeU, const char* group)
{
    const gert::Tensor* firstInput = st.ctx->GetInputTensor(HcomAllGatherOp::INPUT_INDEX);
    gert::StorageShape outShape = BuildAllGatherOutShape(firstInput->GetShape().GetOriginShape(), attrRankSizeU);
    gert::Tensor* outTensor = st.ctx->MallocOutputTensor(
        HcomAllGatherOp::OUTPUT_INDEX, outShape, firstInput->GetFormat(), firstInput->GetDataType());
    if (outTensor == nullptr) {
        HCCL_ERROR("[HcomAllGatherOp][%s] failed to allocate output tensor.", fnName);
        return ge::GRAPH_FAILED;
    }
    st.inputPtrs.push_back(const_cast<void*>(firstInput->GetAddr()));
    st.outputPtrs.push_back(outTensor->GetAddr());
    st.group = group;
    st.stream = st.ctx->GetStream();
    return ge::GRAPH_SUCCESS;
}

// 计数输入: inputCount > 1 即融合节点 (GE 融合 pass 产出, 判据与 AR 一致)
static size_t CountInputs(const hccl::HcclOpState& st)
{
    size_t inputCount = 0;
    for (size_t i = 0;; i++) {
        if (st.ctx->GetInputTensor(i) == nullptr) {
            break;
        }
        inputCount++;
    }
    return inputCount;
}

ge::graphStatus HcomAllGatherOp::ExtractParamsStatic(hccl::HcclOpState& st)
{
    const gert::RuntimeAttrs* attrs = nullptr;
    uint64_t attrRankSizeU = 0;
    const char* group = nullptr;
    HCCL_GE_CHK_RET(ExtractCommonParams(st, "ExtractParamsStatic", attrs, attrRankSizeU, group));
    const int64_t* fusion = attrs->GetInt(ATTR_FUSION);
    const int64_t* fusionId = attrs->GetInt(ATTR_FUSION_ID);

    // 计数输入: inputCount > 1 即融合节点 (GE 融合 pass 产出, 判据与 AR 一致), 分派融合分支
    const size_t inputCount = CountInputs(st);
    if (inputCount > 1) {
        return ExtractParamsStaticFused(st, inputCount, attrRankSizeU, group, fusion, fusionId);
    }

    // 单输入路径 (非融合): count = shapeSize (GetCountByShape 口径, 不 512B 对齐)
    // 引擎融合优化器只对多输入的 AG/RS 设 CONTINUOUS_INPUT/OUTPUT (HcomOptimizeSetAttr),
    // 单输入节点无跨输入连续性语义, 不做 512B 对齐;
    // 融合场景的连续性假设见 ExtractParamsStaticFused 的连续布局约束注释
    // GetShapeSize() 返回 int64_t: 负值 (维度乘积溢出/UNKNOWN_DIM 混入) cast 后
    // 变巨数会绕过零值检查, 先判原值再 cast
    const gert::Tensor* firstInput = st.ctx->GetInputTensor(INPUT_INDEX);
    int64_t shapeSize = firstInput->GetShapeSize();
    if (shapeSize <= 0) {
        HCCL_ERROR("[HcomAllGatherOp][ExtractParamsStatic] invalid input shape size %ld.", shapeSize);
        return ge::GRAPH_FAILED;
    }
    st.count = static_cast<uint64_t>(shapeSize);

    HCCL_GE_CHK_RET(AllocOutputByRankSize(st, "ExtractParamsStatic", attrRankSizeU, group));

    HCCL_INFO(
        "[HcomAllGatherOp][ExtractParamsStatic] input=%p output=%p count=%lu dataType=%d group=%s stream=%p "
        "attrRankSize=%lu fusion=%ld fusionId=%ld.",
        st.inputPtrs[0], st.outputPtrs[0], st.count, st.dataType, st.group ? st.group : "(null)", st.stream,
        attrRankSizeU, fusion ? *fusion : -1, fusionId ? *fusionId : -1);
    return ge::GRAPH_SUCCESS;
}

// 融合输入解析 — dtype 一致性 + 512B 对齐累加 (口径对照 CalcCommonCount) + 溢出防御, 写 st.inputPtrs/st.count
static ge::graphStatus
ParseFusedInputs(hccl::HcclOpState& st, size_t inputCount, uint64_t& unitSize, uint64_t& totalAlignedSize)
{
    const gert::Tensor* firstInput = st.ctx->GetInputTensor(HcomAllGatherOp::INPUT_INDEX);
    constexpr uint64_t ALIGNED_SIZE = 512;
    unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    if (unitSize == 0) {
        HCCL_ERROR(
            "[HcomAllGatherOp][ExtractParamsStaticFused] unitSize is 0 (dataType=%d).", static_cast<int>(st.dataType));
        return ge::GRAPH_FAILED;
    }
    totalAlignedSize = 0;
    for (size_t i = 0; i < inputCount; i++) {
        const gert::Tensor* inTensor = st.ctx->GetInputTensor(i);
        // dtype 一致性防御: 融合 label 要求同 dtype, 混入异 dtype 会使 count 公式失真
        if (inTensor->GetDataType() != firstInput->GetDataType()) {
            HCCL_ERROR(
                "[HcomAllGatherOp][ExtractParamsStaticFused] input[%zu] dtype %d != input[0] dtype %d.", i,
                static_cast<int>(inTensor->GetDataType()), static_cast<int>(firstInput->GetDataType()));
            return ge::GRAPH_FAILED;
        }
        uint64_t tensorSize = static_cast<uint64_t>(inTensor->GetSize());
        // 溢出防御 (对照引擎 CalcCountForAlignedOp): GetSize() 无负值, 接近 UINT64_MAX 时对齐加法回绕
        if (tensorSize > UINT64_MAX - ALIGNED_SIZE) {
            HCCL_ERROR(
                "[HcomAllGatherOp][ExtractParamsStaticFused] input[%zu] size %lu is invalid "
                "(negative or overflow).",
                i, tensorSize);
            return ge::GRAPH_FAILED;
        }
        uint64_t alignedSize = (tensorSize + ALIGNED_SIZE - 1) / ALIGNED_SIZE * ALIGNED_SIZE;
        if (totalAlignedSize > UINT64_MAX - alignedSize) {
            HCCL_ERROR("[HcomAllGatherOp][ExtractParamsStaticFused] total aligned size overflow at input[%zu].", i);
            return ge::GRAPH_FAILED;
        }
        totalAlignedSize += alignedSize;
        // inputPtrs 仅 [0] 被 AG 消费 (HCCL 首地址调用); 全量记录用于输出地址日志与布局排查
        st.inputPtrs.push_back(const_cast<void*>(inTensor->GetAddr()));
        HCCL_INFO(
            "[HcomAllGatherOp][ExtractParamsStaticFused] input[%zu] addr=%p tensorSize=%lu alignedSize=%lu.", i,
            inTensor->GetAddr(), tensorSize, alignedSize);
    }
    st.count = totalAlignedSize / unitSize;
    if (st.count == 0) {
        HCCL_ERROR("[HcomAllGatherOp][ExtractParamsStaticFused] total count is 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 融合输出分配 — anchor idx = r*N+i, shape = input[idx%N] 原样
// 不乘 rankSize (对照融合 pass AddAllGatherNode, 乘 R 会毁掉 pass 已设的输出 desc; rankSize 倍
// 体现在 anchor 个数, 每 op 的 ×R 由下游 Concat 完成), 写 st.outputPtrs
static ge::graphStatus AllocateFusedOutputs(hccl::HcclOpState& st, size_t inputCount, size_t outputCount)
{
    for (size_t idx = 0; idx < outputCount; idx++) {
        const gert::Tensor* inTensor = st.ctx->GetInputTensor(idx % inputCount);
        gert::Tensor* outTensor
            = st.ctx->MallocOutputTensor(idx, inTensor->GetShape(), inTensor->GetFormat(), inTensor->GetDataType());
        if (outTensor == nullptr) {
            HCCL_ERROR("[HcomAllGatherOp][ExtractParamsStaticFused] failed to allocate output tensor %zu.", idx);
            return ge::GRAPH_FAILED;
        }
        st.outputPtrs.push_back(outTensor->GetAddr());
        HCCL_INFO("[HcomAllGatherOp][ExtractParamsStaticFused] output[%zu] addr=%p.", idx, outTensor->GetAddr());
    }
    return ge::GRAPH_SUCCESS;
}

// ==== 融合分支 (inputCount > 1): N 输入 / N*rankSize 输出 ====
// (输入解析见 ParseFusedInputs, 输出分配见 AllocateFusedOutputs)
// HCCL 调用仍用首输入/首输出地址 (布局等价性: 融合 AG ≡ 连续块单次大 AG)
// AG 无 crack 清零 (区别于 AR/RS/Reduce), 不计算 crack
//
// 连续布局约束: 引擎融合优化器对多输入 AG 设 CONTINUOUS_INPUT/CONTINUOUS_OUTPUT
// (HcomOptimizeSetAttr), 图规划器据此将输入/输出按下标序连续分配、每段 512B 对齐;
// 本分支以首地址 + 总 count 通信, 依赖该连续布局。属性为 OpDesc 级, eager 上下文
// 不可读, 运行期不做显式校验
// 提示, 下方地址日志用于排查布局问题。
ge::graphStatus HcomAllGatherOp::ExtractParamsStaticFused(
    hccl::HcclOpState& st, size_t inputCount, uint64_t attrRankSizeU, const char* group, const int64_t* fusion,
    const int64_t* fusionId)
{
    uint64_t unitSize = 0;
    uint64_t totalAlignedSize = 0;
    HCCL_GE_CHK_RET(ParseFusedInputs(st, inputCount, unitSize, totalAlignedSize));

    // 溢出防御: outputCount = inputCount * attrRankSize, 超过 SIZE_MAX 拒绝 (attrRankSize 仅校验了 >0)
    if (attrRankSizeU > SIZE_MAX / inputCount) {
        HCCL_ERROR(
            "[HcomAllGatherOp][ExtractParamsStaticFused] output count overflow: inputCount=%zu "
            "attrRankSize=%lu.",
            inputCount, attrRankSizeU);
        return ge::GRAPH_FAILED;
    }
    size_t outputCount = inputCount * static_cast<size_t>(attrRankSizeU);
    HCCL_GE_CHK_RET(AllocateFusedOutputs(st, inputCount, outputCount));

    st.group = group;
    st.stream = st.ctx->GetStream();

    HCCL_INFO(
        "[HcomAllGatherOp][ExtractParamsStaticFused] inputCount=%zu outputCount=%zu count=%lu "
        "(totalAlignedSize=%lu(B) unitSize=%lu(B)) dataType=%d group=%s stream=%p attrRankSize=%lu "
        "fusion=%ld fusionId=%ld.",
        inputCount, outputCount, st.count, totalAlignedSize, unitSize, st.dataType, st.group ? st.group : "(null)",
        st.stream, attrRankSizeU, fusion ? *fusion : -1, fusionId ? *fusionId : -1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllGatherOp::ExtractParamsDynamic(hccl::HcclOpState& st)
{
    // 动态路径 (lowering-style): count 取 inputShape (GetCountByShape 口径, 不 512B 对齐), 不计算 crack
    // 对照 lowering HcomAllGatherKernelV2: 单 input 直传, count 从 inputShape
    const gert::RuntimeAttrs* attrs = nullptr;
    uint64_t attrRankSizeU = 0;
    const char* group = nullptr;
    HCCL_GE_CHK_RET(ExtractCommonParams(st, "ExtractParamsDynamic", attrs, attrRankSizeU, group));

    // 防御: 融合多输入节点不支持动态路径 — 引擎动态下发 (lowering) 强制单输入,
    // 到达即异常拓扑, 响亮拒绝 (融合的正路是静态路径 ExtractParamsStaticFused)
    const size_t inputCount = CountInputs(st);
    if (inputCount > 1) {
        HCCL_ERROR(
            "[HcomAllGatherOp][ExtractParamsDynamic] fused multi-input (inputNum=%zu) not supported on "
            "dynamic path (engine enforces single input).",
            inputCount);
        return ge::GRAPH_FAILED;
    }

    // GetShapeSize() 返回 int64_t: 负值 (维度乘积溢出/UNKNOWN_DIM 混入) cast 后变巨数, 先判原值再 cast
    const gert::Tensor* firstInput = st.ctx->GetInputTensor(INPUT_INDEX);
    int64_t shapeSize = firstInput->GetShapeSize();
    if (shapeSize <= 0) {
        HCCL_ERROR("[HcomAllGatherOp][ExtractParamsDynamic] invalid input shape size %ld.", shapeSize);
        return ge::GRAPH_FAILED;
    }
    st.count = static_cast<uint64_t>(shapeSize);

    HCCL_GE_CHK_RET(AllocOutputByRankSize(st, "ExtractParamsDynamic", attrRankSizeU, group));

    HCCL_INFO(
        "[HcomAllGatherOp][ExtractParamsDynamic] input=%p output=%p count=%lu dataType=%d group=%s stream=%p.",
        st.inputPtrs[0], st.outputPtrs[0], st.count, st.dataType, st.group ? st.group : "(null)", st.stream);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllGatherOp::CalcResourcesStatic(hccl::HcclOpState& st)
{
    uint32_t rankSize = 0;
    HcclResult ret = HcclGetRankSize(st.comm, &rankSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomAllGatherOp][CalcResourcesStatic] HcclGetRankSize failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    st.rankSize = rankSize;

    // attr rank_size 与通信域 rankSize 一致性校验: 输出按 attr 分配 (MallocOutputTensor), 散射按通信域
    // (outputMaxSize=count*unitSize*rankSize), commRank > attrRank 时写越界 → 图配置错误, 报错拒绝
    const gert::RuntimeAttrs* attrs = st.ctx->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* attrRankSize = attrs->GetInt(ATTR_RANK_SIZE);
        if (attrRankSize != nullptr && static_cast<uint64_t>(*attrRankSize) != static_cast<uint64_t>(rankSize)) {
            HCCL_ERROR(
                "[HcomAllGatherOp][CalcResourcesStatic] attr rank_size=%ld != communicator rankSize=%u "
                "(output allocated by attr, scatter by communicator; overflow when comm is larger), reject.",
                *attrRankSize, rankSize);
            return ge::GRAPH_FAILED;
        }
    }

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
        st, rankSize, cclBufferAvailSize, HCCL_KERNEL_OP_TYPE_ALLGATHER, HCCL_CMD_ALLGATHER);
}

ge::graphStatus HcomAllGatherOp::GenerateTagVec(hccl::HcclOpState& st)
{
    // 对应 InfoStore GenerateOpTagFromTaskInfo: needRefresh 时算 loopMaxTime
    // AG 属组 2 公式 (膨胀, maxCountPerLoop = cclBuffSize / (rankSize*unitSize)), rankSize 用通信域值
    // (st.rankSize, CalcResourcesStatic 已存), 与 LaunchRefresh 循环同源, tag 数恒等于实际迭代数
    uint32_t loopMaxTime = 0;
    if (st.needRefresh) {
        uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
        if (unitSize == 0) {
            HCCL_ERROR("[HcomAllGatherOp][GenerateTagVec] unitSize is 0 (dataType=%d).", static_cast<int>(st.dataType));
            return ge::GRAPH_FAILED;
        }
        if (st.cclBuffSize / (st.rankSize * unitSize) == 0) {
            HCCL_ERROR(
                "[HcomAllGatherOp][GenerateTagVec] cclBuffSize=%lu cannot hold one inflated element "
                "(rankSize=%u, unitSize=%lu).",
                st.cclBuffSize, st.rankSize, unitSize);
            return ge::GRAPH_FAILED;
        }
        HCCL_GE_CHK_RET(hccl::CalcLoopMaxTime(
            HCCL_KERNEL_OP_TYPE_ALLGATHER, st.count, unitSize, st.cclBuffSize, st.rankSize, loopMaxTime));
    }

    // 对应 InfoStore GetTagVectorInfo: loopMaxTime==0 → {tag}; loopMaxTime>0 → {tag + "_looptime_" + i}
    return hccl::BuildTagVec(HCCL_KERNEL_OP_TYPE_ALLGATHER, st.group, loopMaxTime, st.tags);
}

ge::graphStatus HcomAllGatherOp::LaunchHcclOpStatic(hccl::HcclOpState& st)
{
    // 静态路径 (InfoStore-style): needRefresh 判定已收编基类 Execute, 此处只消费 st.needRefresh
    if (st.scratchMemSize > 0) {
        st.scratchMem = st.ctx->MallocWorkSpace(st.scratchMemSize);
        if (st.scratchMem == nullptr) {
            HCCL_ERROR("[HcomAllGatherOp][LaunchHcclOpStatic] MallocWorkSpace failed, size=%lu.", st.scratchMemSize);
            return ge::GRAPH_FAILED;
        }
    }

    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllGatherOp][LaunchHcclOpStatic] HcclSetAivCoreLimitGraphMode failed, ret=%d.",
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

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

ge::graphStatus HcomAllGatherOp::LaunchHcclOpDynamic(hccl::HcclOpState& st)
{
    // 动态路径 (lowering-style): 对照 lowering HcomAllGatherKernelV2 —
    // 单 input (ExtractParamsDynamic 已强制), count 取 inputShape, 直传用户 buffer, 无 crack
    // 单算子接口 (hccl.map 公共导出): 免 tag 构造/group 串解析; GraphMode 为烘焙态接口, engine 动态路径不使用
    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllGatherOp][LaunchHcclOpDynamic] HcclSetAivCoreLimitGraphMode failed, ret=%d.",
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    HCCL_INFO(
        "[HcomAllGatherOp][LaunchHcclOpDynamic] input=%p output=%p count=%lu comm=%p (direct, user buffers, "
        "single-op).",
        st.inputPtrs[0], st.outputPtrs[0], st.count, st.comm);

    // comm 由基类 GetCommunicator 按 group 预解析 (hccl_custom_op.cc), 对照 lowering
    // HcomAllGatherKernelV2 的 HcomGetCommHandleByGroup 预解析后单算子直传
    ret = HcclAllGather(
        st.inputPtrs[0], st.outputPtrs[0], st.count, st.dataType, st.comm, static_cast<aclrtStream>(st.stream));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomAllGatherOp][LaunchHcclOpDynamic] HcclAllGather failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllGatherOp::LaunchDirect(hccl::HcclOpState& st)
{
    // Direct 路径: GE HCCL engine 的 needRefresh=false 路径不做 CCL buffer 检查
    // (HCCL 库内部 HcclExecOp 能自行处理大数据分块), 保留此行为
    // AllGather 无 crack 清零 (InfoStore GenerateTask 的 crack tensor 列表不含 AG, tensorNum=0)
    HCCL_INFO(
        "[HcomAllGatherOp][LaunchDirect] input=%p output=%p count=%lu (non-refresh, user buffers directly).",
        st.inputPtrs[0], st.outputPtrs[0], st.count);

    HcclResult ret = HcclAllGatherGraphMode(
        st.inputPtrs[0], st.outputPtrs[0], st.count, st.dataType, st.group, static_cast<aclrtStream>(st.stream),
        st.tags[0].c_str(), st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(),
        st.scratchMem, st.scratchMemSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomAllGatherOp][LaunchDirect] HcclAllGatherGraphMode failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// LaunchRefresh 循环上下文: CCL buf、args 字段 device 地址与循环参数
// (devXxx 为 args struct 各字段的 device 侧地址, D2D ADDR 经其二级指针间接寻址;
//  singleIteration 标记单迭代 (膨胀总量不超 CCL buf), 决定 ⑤ 的连续直拷形态)
struct HcomAllGatherRefreshCtx {
    void* commInputPtr = nullptr;
    uint64_t commInputSize = 0;
    void* commOutputPtr = nullptr;
    uint64_t commOutputSize = 0;
    void* devCclInputPtr = nullptr;
    void* devInputAddr = nullptr;
    void* devCclOutputPtr = nullptr;
    void* devOutputAddr = nullptr;
    void* devArgsData = nullptr;
    uint64_t outputMaxSize = 0;
    uint64_t maxCountPerLoop = 0;
    bool singleIteration = false;
};

// 刷新上下文的 args 部分: maxCountPerLoop 计算、args struct 分配与字段地址换算
// (args 经 MallocReadOnlyDevArgs 分配 host+device 双份, GE 统一 H2D 同步 — 对应 InfoStore
// RefreshInputAddr/RefreshAllgatherOutputAddr 的 ①④ 次拷贝; CCL buf 由调用方经 GetOrCreate 获取后填入 ctx)
static ge::graphStatus InitRefreshArgs(hccl::HcclOpState& st, HcomAllGatherRefreshCtx& ctx)
{
    const uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    if (unitSize == 0) {
        HCCL_ERROR("[HcomAllGatherOp][LaunchRefresh] unitSize is 0 (dataType=%d).", static_cast<int>(st.dataType));
        return ge::GRAPH_FAILED;
    }
    // maxCountPerLoop 与 GenerateTagVec 的 loopMaxTime 同源 (st.cclBuffSize + st.rankSize 单一来源)
    // AG 输出 = 输入 × rankSize, CCL buf 需容纳膨胀后数据: maxCountPerLoop = cclBuffSize / (rankSize*unitSize)
    ctx.maxCountPerLoop = st.cclBuffSize / (st.rankSize * unitSize);
    if (ctx.maxCountPerLoop == 0) {
        HCCL_ERROR(
            "[HcomAllGatherOp][LaunchRefresh] cclBuffSize=%lu cannot hold one inflated element "
            "(rankSize=%u, unitSize=%lu).",
            st.cclBuffSize, st.rankSize, unitSize);
        return ge::GRAPH_FAILED;
    }
    // ⑤ 侧 dstMax = 输出全量 count*unitSize*rankSize (AG 输出为输入 × rankSize)
    // 溢出防御: 膨胀总量乘法回绕会使后续偏移/散射计算全部失真, 超限拒绝
    if (st.count > UINT64_MAX / unitSize / st.rankSize) {
        HCCL_ERROR(
            "[HcomAllGatherOp][LaunchRefresh] output size overflow: count=%lu unitSize=%lu rankSize=%u.", st.count,
            unitSize, st.rankSize);
        return ge::GRAPH_FAILED;
    }
    ctx.outputMaxSize = st.count * unitSize * st.rankSize;
    // 单迭代判定: 膨胀总量不超 CCL buf (对照 InfoStore secAddrCopyWithoutOffset 尺寸判据)
    ctx.singleIteration = (ctx.outputMaxSize <= st.cclBuffSize);

    // args struct: CCL buf 地址 (固定) + 用户地址 (UpdateHostArgs 刷新)
    // 对照 AllReduce LaunchRefresh / 引擎 HcomAllGatherLoop: 每次 Execute 构造并注册。
    // GetKernelArgs 的 index 按 per-ctx deque 计, 每个 sink 的 ctx 首次注册即其 index 0,
    // UpdateHostArgs 固定刷 index 0 亦即本 ctx 的注册; 若跨执行复用成员缓存, 后续 sink 的
    // D2D ADDR task 会解引用首个注册的旧地址 (多轮执行静默失效)。
    HcomAllGatherArgs args = {ctx.commInputPtr, ctx.commOutputPtr, st.inputPtrs[0], st.outputPtrs[0]};
    const gert::KernelArgs* devArgs = st.ctx->MallocReadOnlyDevArgs(&args, sizeof(args));
    if (devArgs == nullptr || devArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomAllGatherOp][LaunchRefresh] MallocReadOnlyDevArgs failed.");
        return ge::GRAPH_FAILED;
    }
    // devArgs->args_data 是 device 侧 args 起始地址
    // 字段偏移: cclInputPtr=0, cclOutputPtr=8, inputAddr=16, outputAddr=24
    char* devArgsBase = static_cast<char*>(devArgs->args_data);
    ctx.devArgsData = devArgs->args_data;
    ctx.devCclInputPtr = devArgsBase + offsetof(HcomAllGatherArgs, cclInputPtr);
    ctx.devCclOutputPtr = devArgsBase + offsetof(HcomAllGatherArgs, cclOutputPtr);
    ctx.devInputAddr = devArgsBase + offsetof(HcomAllGatherArgs, inputAddr);
    ctx.devOutputAddr = devArgsBase + offsetof(HcomAllGatherArgs, outputAddr);
    return ge::GRAPH_SUCCESS;
}

// LaunchRefresh 单迭代游标
struct HcomAllGatherLoopCursor {
    uint64_t loopTime = 0;
    uint64_t inputOffset = 0;
    uint64_t outputOffset = 0;
    uint64_t curCount = 0;
};

// ⑤ 输出散射 (膨胀, 双形态: 单迭代 (膨胀总量不超 CCL buf) 一次连续拷贝 / 多迭代 per-rank 分段散射,
// 对照 InfoStore RefreshAllgatherOutputAddr 非 legacy / legacy 两分支)
static ge::graphStatus
ScatterCclBufToOutput(hccl::HcclOpState& st, const HcomAllGatherRefreshCtx& ctx, const HcomAllGatherLoopCursor& cursor)
{
    const uint64_t curSize = cursor.curCount * hccl::GetHcclDataTypeSize(st.dataType);
    // ⑤ D2D ADDR 带偏移: CCL output buf → 用户 output (膨胀散射, 双形态)
    //   src = devCclOutputPtr  (二级指针 → 解引用得 CCL buf 地址)
    //   dst = devOutputAddr    (二级指针 → 解引用得用户 output 地址)
    if (ctx.singleIteration) {
        // 单迭代: 一次连续拷贝 curSize*rankSize (对照 InfoStore 非 legacy 单次分支)
        HcclResult hrtRet = hccl::HcclRtMemAddrAsyncCopyWithOffset(
            ctx.devOutputAddr, curSize * st.rankSize, 0, ctx.devCclOutputPtr, curSize * st.rankSize, 0,
            static_cast<aclrtStream>(st.stream));
        if (hrtRet != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HcomAllGatherOp][RefreshOneLoop] D2D ADDR CCL to output (contiguous) failed, loopTime=%lu, "
                "ret=%d.",
                cursor.loopTime, static_cast<int>(hrtRet));
            return ge::GRAPH_FAILED;
        }
    } else {
        // 多迭代: per-rank 分段散射 — OutCCLbuf 的 rank r 段 [r*curSize, +curSize) →
        // 用户 output 的 [outputOffset + r*count*unitSize, +curSize) (rank r 段内当前迭代位置)
        // (对照 InfoStore RefreshAllgatherOutputAddr legacy 分支:
        //  userOutMemOffset = outputOffset + count*unitSize*i, indirectOutCCLBufOffset = curSize*i, size = curSize)
        const uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
        for (uint32_t r = 0; r < st.rankSize; r++) {
            HcclResult hrtRet = hccl::HcclRtMemAddrAsyncCopyWithOffset(
                ctx.devOutputAddr, ctx.outputMaxSize,
                cursor.outputOffset + static_cast<uint64_t>(r) * st.count * unitSize, ctx.devCclOutputPtr, curSize,
                static_cast<uint64_t>(r) * curSize, static_cast<aclrtStream>(st.stream));
            if (hrtRet != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[HcomAllGatherOp][RefreshOneLoop] D2D ADDR CCL to output (per-rank r=%u) failed, "
                    "loopTime=%lu, ret=%d.",
                    r, cursor.loopTime, static_cast<int>(hrtRet));
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

// LaunchRefresh 单迭代: ② D2D ADDR 搬入 → ③ AllGather → ⑤ 膨胀散射回写 (每迭代独立 tag: _looptime_N)
static ge::graphStatus
RefreshOneLoop(hccl::HcclOpState& st, const HcomAllGatherRefreshCtx& ctx, HcomAllGatherLoopCursor& cursor)
{
    const uint64_t curSize = cursor.curCount * hccl::GetHcclDataTypeSize(st.dataType);

    // tag 越界守卫, 对照 InfoStore HcomAllGatherLoop
    if (cursor.loopTime >= st.tags.size()) {
        HCCL_ERROR(
            "[HcomAllGatherOp][RefreshOneLoop] tagVec access out-of-bounds, loopTime=%lu tags.size=%zu.",
            cursor.loopTime, st.tags.size());
        return ge::GRAPH_FAILED;
    }

    // ② D2D ADDR 带偏移: 用户 input → CCL input buf (等长, 无膨胀)
    //   src = devInputAddr    (二级指针 → 解引用得用户 input 地址)
    //   dst = devCclInputPtr  (二级指针 → 解引用得 CCL buf 地址)
    HcclResult hrtRet = hccl::HcclRtMemAddrAsyncCopyWithOffset(
        ctx.devCclInputPtr, ctx.commInputSize, 0, ctx.devInputAddr, curSize, cursor.inputOffset,
        static_cast<aclrtStream>(st.stream));
    if (hrtRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllGatherOp][RefreshOneLoop] D2D ADDR input to CCL failed, loopTime=%lu, ret=%d.", cursor.loopTime,
            static_cast<int>(hrtRet));
        return ge::GRAPH_FAILED;
    }

    // ③ HcclAllGatherGraphMode (每迭代独立 tag: _looptime_N; 无 reduction 参数)
    HcclResult ret = HcclAllGatherGraphMode(
        ctx.commInputPtr, ctx.commOutputPtr, cursor.curCount, st.dataType, st.group,
        static_cast<aclrtStream>(st.stream), st.tags[cursor.loopTime].c_str(),
        st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(), st.scratchMem,
        st.scratchMemSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllGatherOp][RefreshOneLoop] HcclAllGatherGraphMode failed, loopTime=%lu, ret=%d.", cursor.loopTime,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    return ScatterCclBufToOutput(st, ctx, cursor);
}

ge::graphStatus HcomAllGatherOp::LaunchRefresh(hccl::HcclOpState& st)
{
    // 地址刷新路径: args struct + ArgsUpdater, GE 统一 H2D
    // 多迭代 loop: 结构对照 InfoStore HcomAllGatherLoop, 超限数据分迭代执行
    // (AG 膨胀系数 rankSize 在输出侧: ② 等长搬入, ⑤ 双形态散射见 ScatterCclBufToOutput;
    //  AllGather 无 crack 清零, 对应 InfoStore RefreshInputAddr/RefreshAllgatherOutputAddr 的 ②⑤ 拷贝)
    HcomAllGatherRefreshCtx ctx;
    HCCL_GE_CHK_RET(GetOrCreateInCCLbuf(st.group, ctx.commInputPtr, ctx.commInputSize));
    HCCL_GE_CHK_RET(GetOrCreateOutCCLbuf(st.group, ctx.commOutputPtr, ctx.commOutputSize));
    HCCL_GE_CHK_RET(InitRefreshArgs(st, ctx));

    // 大于 cclbuffer 时 count 取最大支持 count, 否则取剩余 count (膨胀系数 rankSize)
    HcomAllGatherLoopCursor cursor;
    for (uint64_t countLeft = st.count; countLeft > 0; countLeft -= cursor.curCount) {
        cursor.curCount = (countLeft > ctx.maxCountPerLoop) ? ctx.maxCountPerLoop : countLeft;
        HCCL_INFO(
            "[HcomAllGatherOp][LaunchRefresh] loopTime=%lu inputOffset=%lu outputOffset=%lu countLeft=%lu "
            "maxCountPerLoop=%lu.",
            cursor.loopTime, cursor.inputOffset, cursor.outputOffset, countLeft, ctx.maxCountPerLoop);

        HCCL_GE_CHK_RET(RefreshOneLoop(st, ctx, cursor));

        const uint64_t curSize = cursor.curCount * hccl::GetHcclDataTypeSize(st.dataType);
        cursor.inputOffset += curSize;
        cursor.outputOffset += curSize;
        cursor.loopTime++;
    }

    HCCL_INFO(
        "[HcomAllGatherOp][LaunchRefresh] devArgs=%p count=%lu maxCountPerLoop=%lu rankSize=%u "
        "cclInput=%p cclOutput=%p inputAddr=%p outputAddr=%p.",
        ctx.devArgsData, st.count, ctx.maxCountPerLoop, st.rankSize, ctx.commInputPtr, ctx.commOutputPtr,
        st.inputPtrs[0], st.outputPtrs[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllGatherOp::UpdateHostArgs(gert::UpdateArgsContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomAllGatherOp][UpdateHostArgs] update args context is null.");
        return ge::GRAPH_FAILED;
    }
    // GE 在 sink 模型每次执行前调: 刷新 args struct 中的 inputAddr/outputAddr
    // GE 统一 H2D 同步 host → device, D2D ADDR task 自动 dereference 新地址
    const gert::Tensor* inputTensor = ctx->GetInputTensor(INPUT_INDEX);
    const gert::Tensor* outputTensor = ctx->GetOutputTensor(OUTPUT_INDEX);
    if (inputTensor == nullptr || outputTensor == nullptr) {
        HCCL_ERROR("[HcomAllGatherOp][UpdateHostArgs] input=%p output=%p.", inputTensor, outputTensor);
        return ge::GRAPH_FAILED;
    }

    const gert::KernelArgs* hostArgs = ctx->GetKernelArgs(gert::Placement::kPlacementHost, 0);
    if (hostArgs == nullptr || hostArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomAllGatherOp][UpdateHostArgs] hostArgs is null.");
        return ge::GRAPH_FAILED;
    }

    auto* args = static_cast<HcomAllGatherArgs*>(hostArgs->args_data);
    args->inputAddr = const_cast<void*>(inputTensor->GetAddr());
    args->outputAddr = const_cast<void*>(outputTensor->GetAddr());

    HCCL_INFO("[HcomAllGatherOp][UpdateHostArgs] inputAddr=%p outputAddr=%p.", args->inputAddr, args->outputAddr);
    return ge::GRAPH_SUCCESS;
}

// 融合分支 shape 推导 — N 入 / N*rankSize 出,
// output[idx] = input[idx % N] 原样直通, 不乘 rankSize —— rankSize 倍体现在 anchor 个数,
// 每 op 的 ×R 由下游 Concat 完成 (对照融合 pass 契约, 乘 R 会毁 pass 已设 desc)
static ge::graphStatus InferShapeFused(gert::InferShapeContext* ctx, size_t inputNum, size_t outputNum)
{
    const gert::RuntimeAttrs* fusedAttrs = ctx->GetAttrs();
    if (fusedAttrs == nullptr) {
        HCCL_ERROR("[HcomAllGatherOp][InferShape] fused attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const int64_t* fusedRankSize = fusedAttrs->GetInt(HcomAllGatherOp::ATTR_RANK_SIZE);
    if (fusedRankSize == nullptr || *fusedRankSize <= 0) {
        HCCL_ERROR("[HcomAllGatherOp][InferShape] fused rank_size attr not available or invalid.");
        return ge::GRAPH_FAILED;
    }
    if (outputNum != inputNum * static_cast<size_t>(*fusedRankSize)) {
        HCCL_ERROR(
            "[HcomAllGatherOp][InferShape] fused topology mismatch: inputNum=%zu outputNum=%zu rankSize=%ld, "
            "expected outputNum=inputNum*rankSize.",
            inputNum, outputNum, *fusedRankSize);
        return ge::GRAPH_FAILED;
    }
    for (size_t idx = 0; idx < outputNum; idx++) {
        const gert::Shape* inShape = ctx->GetInputShape(idx % inputNum);
        gert::Shape* outShape = ctx->GetOutputShape(idx);
        if (inShape == nullptr || outShape == nullptr) {
            HCCL_ERROR("[HcomAllGatherOp][InferShape] fused shape is null at idx=%zu.", idx);
            return ge::GRAPH_FAILED;
        }
        *outShape = *inShape;
    }
    HCCL_INFO(
        "[HcomAllGatherOp][InferShape] FUSED inputNum=%zu outputNum=%zu rankSize=%ld, "
        "output[idx]=input[idx%%%zu] passthrough.",
        inputNum, outputNum, *fusedRankSize, inputNum);
    return ge::GRAPH_SUCCESS;
}

// 单输入分支 shape 推导: 输出 shape = 输入 dim0 × rank_size (rank_size 取自 IR attr index 0, 编译期无通信域)
static ge::graphStatus InferShapeSingle(gert::InferShapeContext* ctx, size_t inputNum, size_t outputNum)
{
    if (inputNum != 1 || outputNum != 1) {
        // 防御: 融合节点 (inputNum>1) 已由调用方分支处理, 此处只剩异常拓扑 (0 入 / 1 入多出),
        // 拒绝以避免静默部分推导 (只填 output[0], 其余输出 desc 无人设置)
        HCCL_ERROR(
            "[HcomAllGatherOp][InferShape] anomalous topology inputNum=%zu outputNum=%zu, expected 1/1 or "
            "N/N*rankSize.",
            inputNum, outputNum);
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* inputShape = ctx->GetInputShape(HcomAllGatherOp::INPUT_INDEX);
    if (inputShape == nullptr) {
        HCCL_ERROR("[HcomAllGatherOp][InferShape] input shape is null.");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outputShape = ctx->GetOutputShape(HcomAllGatherOp::OUTPUT_INDEX);
    if (outputShape == nullptr) {
        HCCL_ERROR("[HcomAllGatherOp][InferShape] output shape is null.");
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomAllGatherOp][InferShape] attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const int64_t* rankSizePtr = attrs->GetInt(HcomAllGatherOp::ATTR_RANK_SIZE);
    if (rankSizePtr == nullptr || *rankSizePtr <= 0) {
        HCCL_ERROR("[HcomAllGatherOp][InferShape] rank_size attr not available or invalid.");
        return ge::GRAPH_FAILED;
    }
    int64_t rankSize = *rankSizePtr;

    // 输出 shape: 输入第 0 维 × rankSize, 其余维度不变
    // 动态 shape 处理: 第 0 维为 UNKNOWN_DIM(-1) 时直接复制输入 shape 保持未知标记
    // 标量输入 (dimNum==0): 输出为 [rankSize]
    *outputShape = *inputShape;
    if (inputShape->GetDimNum() == 0) {
        outputShape->AppendDim(rankSize);
    } else if (inputShape->GetDim(0) != ge::UNKNOWN_DIM) {
        outputShape->SetDim(0, inputShape->GetDim(0) * rankSize);
    }

    HCCL_INFO(
        "[HcomAllGatherOp][InferShape] inputDimNum=%zu outputDim0=%ld rankSize=%ld.", inputShape->GetDimNum(),
        outputShape->GetDim(0), rankSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllGatherOp::InferShape(gert::InferShapeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomAllGatherOp][InferShape] infer shape context is null.");
        return ge::GRAPH_FAILED;
    }
    size_t inputNum = ctx->GetComputeNodeInputNum();
    size_t outputNum = ctx->GetComputeNodeOutputNum();

    // 融合分支 (GE 融合 pass 产出的节点): N 入 / N*rankSize 出, 见 InferShapeFused
    if (inputNum > 1) {
        return InferShapeFused(ctx, inputNum, outputNum);
    }
    return InferShapeSingle(ctx, inputNum, outputNum);
}

ge::graphStatus HcomAllGatherOp::InferDataType(gert::InferDataTypeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomAllGatherOp][InferDataType] infer datatype context is null.");
        return ge::GRAPH_FAILED;
    }
    size_t inputNum = ctx->GetComputeNodeInputNum();
    size_t outputNum = ctx->GetComputeNodeOutputNum();

    // 融合分支: N 入 / N*rankSize 出, output[idx] dtype = input[idx % N] 直通 (融合 label 同 dtype)
    // 拓扑严格校验在 InferShape 融合分支 (outputNum==inputNum*rankSize), 此处仅按锚点数透传
    if (inputNum > 1) {
        for (size_t idx = 0; idx < outputNum; idx++) {
            ge::DataType inDtype = ctx->GetInputDataType(idx % inputNum);
            if (inDtype == ge::DT_UNDEFINED) {
                HCCL_ERROR("[HcomAllGatherOp][InferDataType] fused input[%zu] dtype is undefined.", idx % inputNum);
                return ge::GRAPH_FAILED;
            }
            ge::graphStatus ret = ctx->SetOutputDataType(idx, inDtype);
            if (ret != ge::GRAPH_SUCCESS) {
                HCCL_ERROR("[HcomAllGatherOp][InferDataType] fused SetOutputDataType(%zu) failed.", idx);
                return ret;
            }
        }
        HCCL_INFO("[HcomAllGatherOp][InferDataType] FUSED inputNum=%zu outputNum=%zu.", inputNum, outputNum);
        return ge::GRAPH_SUCCESS;
    }

    if (inputNum != 1 || outputNum != 1) {
        HCCL_ERROR("[HcomAllGatherOp][InferDataType] inputNum=%zu outputNum=%zu, expected 1/1.", inputNum, outputNum);
        return ge::GRAPH_FAILED;
    }

    ge::DataType dataType = ctx->GetInputDataType(INPUT_INDEX);
    if (dataType == ge::DT_UNDEFINED) {
        HCCL_ERROR("[HcomAllGatherOp][InferDataType] input dtype is undefined.");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus ret = ctx->SetOutputDataType(OUTPUT_INDEX, dataType);
    if (ret != ge::GRAPH_SUCCESS) {
        HCCL_ERROR("[HcomAllGatherOp][InferDataType] SetOutputDataType failed.");
        return ret;
    }

    HCCL_INFO("[HcomAllGatherOp][InferDataType] dataType=%d.", static_cast<int>(dataType));
    return ge::GRAPH_SUCCESS;
}

const char* HcomAllGatherOp::GetOpTypeName() const { return HCCL_KERNEL_OP_TYPE_ALLGATHER.c_str(); }

static ge::BaseCustomOp* CreateHcomAllGatherOp() { return new HcomAllGatherOp(); }
// 条件注册: 目标平台 (950/960) 且 GE/RTS 新接口在位才注册接管, 否则回退原 hccl_engine 路径
// (hccl_custom_op.h CreateSocGatedRegister: SoC 判定 + GE/RTS 接口在位判定)
static ge::CustomOpCreatorRegister* g_hcom_all_gather_register
    = hccl::CreateSocGatedRegister(HCCL_KERNEL_OP_TYPE_ALLGATHER.c_str(), CreateHcomAllGatherOp);
