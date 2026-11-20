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
//   HcomGetRankSize — 按 group 获取 rank 数 (ExtractParamsStatic)
// 另使用 hcom.h 常量: HCCL_KERNEL_OP_TYPE_ALLGATHERV
// 公共工具 (CalcOnlineResources/GetCCLBufferAvailableSize/BuildTagVec/ReadU64VectorWithD2H 等) 见
// hccl_op_utils.h/.cc
//
// 前向声明的 libhccl.so 内部接口 (同包直调, 对应 InfoStore/lowering 的 HcceAllGatherV*
// dlsym 路径; 不 include src/ops/all_gather_v/all_gather_v.h 以避免内部头依赖链):
//   HcclAllGatherV               (all_gather_v.h) — comm 版, 动态路径
//   HcclAllGatherVGraphMode      (all_gather_v.h) — group+tag+streams+scratch 版, 静态路径
//   HcclSetAivCoreLimitGraphMode — AIV 核心数限制 (同 hcom_all_reduce_op.cc 的前向声明)

#include "hcom_all_gather_v_op.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "hccl/hcom.h"
#include "hccl_op_utils.h"
#include "log.h"

extern "C" {
HcclResult HcclAllGatherV(
    void* sendBuf, uint64_t sendCount, void* recvBuf, const void* recvCounts, const void* recvDispls,
    HcclDataType dataType, HcclComm comm, aclrtStream stream);
HcclResult HcclAllGatherVGraphMode(
    void* sendBuf, void* recvBuf, uint64_t sendCount, const void* recvCounts, const void* recvDispls,
    HcclDataType dataType, const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount,
    void* scratchMemAddr, uint64_t scratchMemSize);
HcclResult HcclSetAivCoreLimitGraphMode(const char* group, uint32_t aivCoreLimit);
}

ge::graphStatus HcomAllGatherVOp::ExtractVAttrs(hccl::HcclOpState& st, const char*& group)
{
    const gert::RuntimeAttrs* attrs = st.ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomAllGatherVOp][ExtractVAttrs] attrs is null.");
        return ge::GRAPH_FAILED;
    }
    group = attrs->GetStr(ATTR_GROUP);
    if (group == nullptr) {
        HCCL_ERROR("[HcomAllGatherVOp] group attr is null.");
        return ge::GRAPH_FAILED;
    }

    st.dataType = hccl::GeDataTypeToHccl(st.ctx->GetInputTensor(INPUT_X)->GetDataType());
    if (st.dataType == HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR(
            "[HcomAllGatherVOp][ExtractVAttrs] unsupported data type %d.",
            static_cast<int>(st.ctx->GetInputTensor(INPUT_X)->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllGatherVOp::CalcOutDim0(
    const std::vector<uint64_t>& dimDispls, const std::vector<uint64_t>& rawRecvCounts, int64_t& outDim0)
{
    // outDim0 = max(dimDispls[i] + rawRecvCounts[i]) (首维单位, 直取不换算);
    // counts/displs 来自调用方张量: 加法前做 uint64 溢出校验, 窄化 int64 前做范围校验, fail fast
    uint64_t maxEnd = 0;
    for (size_t i = 0; i < rawRecvCounts.size(); i++) {
        if (dimDispls[i] > std::numeric_limits<uint64_t>::max() - rawRecvCounts[i]) {
            HCCL_ERROR(
                "[HcomAllGatherVOp][CalcOutDim0] recv_displacements[%zu]=%lu + recv_counts[%zu]=%lu overflows.", i,
                dimDispls[i], i, rawRecvCounts[i]);
            return ge::GRAPH_FAILED;
        }
        const uint64_t end = dimDispls[i] + rawRecvCounts[i];
        if (maxEnd < end) {
            maxEnd = end;
        }
    }
    if (maxEnd > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        HCCL_ERROR("[HcomAllGatherVOp][CalcOutDim0] output dim0 size=%lu exceeds int64 range.", maxEnd);
        return ge::GRAPH_FAILED;
    }
    outDim0 = static_cast<int64_t>(maxEnd);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllGatherVOp::CalcSendCount(uint64_t rawSendCount, uint64_t otherDims, uint64_t& sendCount)
{
    // sendCount = send_count[0] × otherDims; counts 来自调用方张量: 乘法前做 uint64 溢出校验, fail fast
    // (与 CalcOutDim0 口径一致); otherDims==0 (x dims[1:] 含 0 维) 时积恒 0, 无溢出风险
    if (otherDims != 0 && rawSendCount > std::numeric_limits<uint64_t>::max() / otherDims) {
        HCCL_ERROR(
            "[HcomAllGatherVOp][CalcSendCount] send_count=%lu * otherDims=%lu overflows.", rawSendCount, otherDims);
        return ge::GRAPH_FAILED;
    }
    sendCount = rawSendCount * otherDims;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllGatherVOp::ResolveRecvDisplsForInferShape(
    gert::InferShapeContext* ctx, const std::vector<uint64_t>& recvCounts, std::vector<uint64_t>& recvDispls)
{
    // displs 为可选输入: 先判空再读, 避免 ReadHostU64Vector 的 null ERROR 噪声
    // 三态: 未接入/空张量(无数据地址) → 安静回退前缀和; 有效数据 → 长度相符直取;
    // 长度不符 → 回退前缀和 (对齐运行期 ReadOptionalVDispls 契约: displs 可选、长度不符忽略)
    const gert::Tensor* displsTensor = ctx->GetInputTensor(INPUT_RECV_DISPLS);
    if (displsTensor != nullptr && displsTensor->GetAddr() != nullptr
        && hccl::ReadHostU64Vector(displsTensor, recvDispls) == ge::GRAPH_SUCCESS && !recvDispls.empty()) {
        if (recvDispls.size() == recvCounts.size()) {
            return ge::GRAPH_SUCCESS;
        }
        HCCL_WARNING(
            "[HcomAllGatherVOp][ResolveRecvDisplsForInferShape] recv_displacements size=%zu != recv_counts "
            "size=%zu, fallback to prefix-sum.",
            recvDispls.size(), recvCounts.size());
    }
    // 未传入/不可读/空列表/长度不符: 按 recvCounts 前缀和 (对齐 proto 与 vinputvec pass CalcDisplacements)
    recvDispls.clear();
    uint64_t tempSum = 0;
    for (uint64_t c : recvCounts) {
        recvDispls.push_back(tempSum);
        tempSum += c;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllGatherVOp::ExtractParamsStatic(hccl::HcclOpState& st)
{
    // 参数获取与 counts/displacements 读取换算 (对应 InfoStore HcomAllGatherVOpKernel)
    const gert::Tensor* xTensor = st.ctx->GetInputTensor(INPUT_X);
    if (xTensor == nullptr) {
        HCCL_ERROR("[HcomAllGatherVOp][ExtractParamsStatic] x tensor is null.");
        return ge::GRAPH_FAILED;
    }

    const char* group = nullptr;
    HCCL_GE_CHK_RET(ExtractVAttrs(st, group));
    constexpr const char* opName = "HcomAllGatherVOp";

    // rankSize: counts 读取长度与 displs 计算都依赖; GetCommunicator 在模板后续才执行,
    // 此处用 group 版接口
    HCCL_GE_CHK_RET(hccl::GetVRankSizeByGroup(group, opName, st.rankSize));

    // 读 counts 二元组 (send_count / recv_counts) + 可选 displs 张量 (三态, 见 ReadOptionalVDispls)
    // custom op 读不到 vInputVec (RuntimeAttrs 仅 index-based 标量), 改为从输入张量 host 直读或 D2H
    std::vector<uint64_t> rawSendCount;
    std::vector<uint64_t> rawRecvCounts;
    std::vector<uint64_t> rawRecvDispls;
    HCCL_GE_CHK_RET(hccl::ReadVCountPairWithCheck(
        st.ctx->GetInputTensor(INPUT_SEND_COUNT), st.ctx->GetInputTensor(INPUT_RECV_COUNTS), opName, "send_count",
        "recv_counts", st.rankSize, rawSendCount, rawRecvCounts, static_cast<aclrtStream>(st.ctx->GetStream())));
    bool hasDispls = false;
    HCCL_GE_CHK_RET(hccl::ReadOptionalVDispls(
        st.ctx->GetInputTensor(INPUT_RECV_DISPLS), opName, "recv_displacements", st.rankSize, rawRecvDispls, hasDispls,
        static_cast<aclrtStream>(st.ctx->GetStream())));

    // otherDims 换算: x shape dims[1:] 乘积 (对应 vinputvec pass GetOtherDims)
    uint64_t otherDims = 0;
    HCCL_GE_CHK_RET(hccl::CalcOtherDimsFromShape(xTensor->GetShape().GetStorageShape(), opName, otherDims));

    // recvCounts/recvDispls ×otherDims → 元素单位 (对应 lowering recvcounts[i] *= multiplier)
    // dimDispls (首维单位前缀和或直取) 用于 outDim0 计算, 最后与 recvCounts 统一 ×otherDims
    std::vector<uint64_t> dimDispls;
    HCCL_GE_CHK_RET(hccl::ResolveVDispls(hasDispls, rawRecvDispls, rawRecvCounts, dimDispls));
    HCCL_GE_CHK_RET(
        hccl::ConvertVCountsToElementUnit(rawRecvCounts, dimDispls, otherDims, st.recvCounts, st.recvDispls));

    // 输出 shape: outDim0 = max(dimDispls[i] + rawRecvCounts[i]) (首维单位, 与上游 proto 行为一致),
    // 不再除 otherDims; 空输出 (各 rank 接收量全 0) 只定形不分配; HCCL 侧 recvCounts 全 0 为合法
    // 空操作、允许 recvBuf 为空 (CheckAllGatherVRecvAndGetRank), 输出地址为空合法
    int64_t outDim0 = 0;
    HCCL_GE_CHK_RET(CalcOutDim0(dimDispls, rawRecvCounts, outDim0));
    gert::Tensor* outTensor = hccl::AllocVOutputTensor(st, xTensor, outDim0);
    if (outTensor == nullptr) {
        HCCL_ERROR("[HcomAllGatherVOp][ExtractParamsStatic] failed to allocate output tensor.");
        return ge::GRAPH_FAILED;
    }

    HCCL_GE_CHK_RET(CalcSendCount(rawSendCount[0], otherDims, st.sendCount));
    st.inputPtrs.push_back(const_cast<void*>(xTensor->GetAddr()));
    st.outputPtrs.push_back(outTensor->GetAddr());
    st.group = group;
    st.stream = st.ctx->GetStream();

    // rankSize 大时逐条 INFO 会刷爆日志, 合并为单条摘要 (前 MAX_LOG_RANK_NUM 个 + total 提示)
    HCCL_INFO(
        "[HcomAllGatherVOp][ExtractParamsStatic] input=%p output=%p sendCount=%lu rankSize=%u "
        "otherDims=%lu hasDispls=%d outDim0=%ld dataType=%d group=%s stream=%p recvCounts/displs: %s.",
        st.inputPtrs[0], st.outputPtrs[0], st.sendCount, st.rankSize, otherDims, static_cast<int>(hasDispls), outDim0,
        st.dataType, st.group ? st.group : "(null)", st.stream,
        hccl::SummarizeVCounts(st.recvCounts, st.recvDispls).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllGatherVOp::CalcResourcesStatic(hccl::HcclOpState& st)
{
    // 对应 Builder CalcOpRunningResources 的 AGV 分支 (hcom_ops_kernel_builder.cc):
    // count = max(recvCounts); AGV 无归约属性, reduceOp 传默认 SUM (对齐 builder 的初始化值)
    uint64_t count = 0;
    if (!st.recvCounts.empty()) {
        count = *std::max_element(st.recvCounts.begin(), st.recvCounts.end());
    }
    // CalcOnlineResources 从 st.count/st.reduceOp 取值; rankSize 由 ExtractParamsStatic 存入 st.rankSize
    st.count = count;
    st.reduceOp = HCCL_REDUCE_SUM;

    // 在线计算输入: 环境推导 available size (CalcOnlineResources 契约, 对照旧路径 kernel_builder 的
    // GetCCLBufferAvailableSize, 与 AllReduce/Reduce/RS 同源), 与实际 buffer 尺寸是两个来源;
    // V 算子无 Refresh 路径 (loopMaxTime 恒 0), 不消费 st.cclBuffSize, 不查询通信域实际 buffer
    uint64_t cclBufferAvailSize = 0;
    HCCL_GE_CHK_RET(hccl::GetCCLBufferAvailableSize(cclBufferAvailSize));
    return hccl::CalcOnlineResources(
        st, st.rankSize, cclBufferAvailSize, HCCL_KERNEL_OP_TYPE_ALLGATHERV, HcclCMDType::HCCL_CMD_ALLGATHER_V);
}

ge::graphStatus HcomAllGatherVOp::LaunchHcclOpStatic(hccl::HcclOpState& st)
{
    // 静态直传: 用户地址 + 单 tag, 无 loop/refresh/crack, 不判 needRefresh
    if (st.scratchMemSize > 0) {
        st.scratchMem = st.ctx->MallocWorkSpace(st.scratchMemSize);
        if (st.scratchMem == nullptr) {
            HCCL_ERROR("[HcomAllGatherVOp][LaunchHcclOpStatic] MallocWorkSpace failed, size=%lu.", st.scratchMemSize);
            return ge::GRAPH_FAILED;
        }
    }

    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllGatherVOp][LaunchHcclOpStatic] HcclSetAivCoreLimitGraphMode failed, ret=%d.",
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    // attach 流申请 + 交付 (双流: graph/group 粒度, 对照旧 InfoStore SetAttachedStream 每 op 一次;
    // AIV 跳过)
    // follow 流申请 (按 st.streamNum, 复刻旧 main_stream_id 池化; AIV/流数 0 自动跳过)
    HCCL_GE_CHK_RET(RequestFollowStreams(st));
    HCCL_GE_CHK_RET(RequestAttachStream(st));
    HCCL_GE_CHK_RET(RegisterUnfoldStreamToModel(st));

    // tag: V 算子不走 CCL buf 分迭代 (loopMaxTime 恒 0) → 单 tag, 无 _looptime_ 后缀;
    // 公共化为 BuildTagVec (对应 InfoStore GenerateOpTagFromTaskInfo)
    HCCL_GE_CHK_RET(hccl::BuildTagVec(HCCL_KERNEL_OP_TYPE_ALLGATHERV, st.group, 0, st.tags));

    // 直传用户地址 (组 4); recvCounts/recvDispls 传 host 向量数据 (InfoStore 同样传 host 指针)
    ret = HcclAllGatherVGraphMode(
        st.inputPtrs[0], st.outputPtrs[0], st.sendCount, st.recvCounts.data(), st.recvDispls.data(), st.dataType,
        st.group, static_cast<aclrtStream>(st.stream), st.tags[0].c_str(),
        st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(), st.scratchMem,
        st.scratchMemSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllGatherVOp][LaunchHcclOpStatic] HcclAllGatherVGraphMode failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllGatherVOp::LaunchHcclOpDynamic(hccl::HcclOpState& st)
{
    // 对应 lowering HcomLaunchAllGatherVKernel: 直传用户地址, comm 版接口,
    // 无 tag/streams/scratch; 参数提取已由基类默认 ExtractParamsDynamic → ExtractParamsStatic 完成
    // AIV 算力上限用 GraphMode 版 HcclSetAivCoreLimitGraphMode (与 AllReduce/Reduce 动态路径同款)
    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllGatherVOp][LaunchHcclOpDynamic] HcclSetAivCoreLimitGraphMode failed, ret=%d.",
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    ret = HcclAllGatherV(
        st.inputPtrs[0], st.sendCount, st.outputPtrs[0], st.recvCounts.data(), st.recvDispls.data(), st.dataType,
        st.comm, static_cast<aclrtStream>(st.stream));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomAllGatherVOp][LaunchHcclOpDynamic] HcclAllGatherV failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllGatherVOp::InferShape(gert::InferShapeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomAllGatherVOp][InferShape] infer shape context is null.");
        return ge::GRAPH_FAILED;
    }
    // 移植 all_gather_v_proto.cc HcomAllGatherVInferShapeV2 (含上游首维单位修复):
    // recv_counts/recv_displacements 为首维单位; 输出首维 = max(displ[i]+counts[i]) 直取,
    // 不再除 otherDims (旧元素口径对多维输入首维推导偏小)
    const gert::Shape* inputShape = ctx->GetInputShape(INPUT_X);
    if (inputShape == nullptr) {
        HCCL_ERROR("[HcomAllGatherVOp][InferShape] input shape is null.");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outputShape = ctx->GetOutputShape(0);
    if (outputShape == nullptr) {
        HCCL_ERROR("[HcomAllGatherVOp][InferShape] output shape is null.");
        return ge::GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() == 0) {
        HCCL_ERROR("[HcomAllGatherVOp][InferShape] input tensor's first dim is illegal, expected: > 0, actual: 0.");
        return ge::GRAPH_FAILED;
    }

    // const 数据不可用 → 拷贝 input shape 且首维未知 (对应 proto HcomIsConstData 失败分支;
    // eager 路径下真实 shape 由 Execute 的 MallocOutputTensor 决定)
    std::vector<uint64_t> sendCount;
    std::vector<uint64_t> recvCounts;
    if (hccl::ReadHostU64Vector(ctx->GetInputTensor(INPUT_SEND_COUNT), sendCount) != ge::GRAPH_SUCCESS
        || hccl::ReadHostU64Vector(ctx->GetInputTensor(INPUT_RECV_COUNTS), recvCounts) != ge::GRAPH_SUCCESS) {
        *outputShape = *inputShape;
        outputShape->SetDim(0, ge::UNKNOWN_DIM);
        return ge::GRAPH_SUCCESS;
    }
    if (recvCounts.empty()) {
        HCCL_ERROR("[HcomAllGatherVOp][InferShape] recv_counts is empty.");
        return ge::GRAPH_FAILED;
    }

    // recvDispls: 有非空 const 位移用真实值, 否则按 recvCounts 前缀和
    std::vector<uint64_t> recvDispls;
    HCCL_GE_CHK_RET(ResolveRecvDisplsForInferShape(ctx, recvCounts, recvDispls));

    // outDim = max(recvDispls[i] + recvCounts[i]): 复用 CalcOutDim0 (uint64 加法溢出 + int64 窄化
    // fail fast), 与 ExtractParamsStatic 同一实现, 消除编译期/运行期校验分叉
    int64_t outDim0 = 0;
    HCCL_GE_CHK_RET(CalcOutDim0(recvDispls, recvCounts, outDim0));

    *outputShape = *inputShape;
    outputShape->SetDim(0, outDim0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllGatherVOp::InferDataType(gert::InferDataTypeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomAllGatherVOp][InferDataType] infer datatype context is null.");
        return ge::GRAPH_FAILED;
    }
    // 移植 all_gather_v_proto.cc HcomAllGatherVInferDataTypeV2: out dtype = in[0] dtype
    ge::DataType inputType = ctx->GetInputDataType(INPUT_X);
    if (inputType == ge::DT_UNDEFINED) {
        HCCL_ERROR("[HcomAllGatherVOp][InferDataType] input dtype is undefined.");
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus ret = ctx->SetOutputDataType(0, inputType);
    if (ret != ge::GRAPH_SUCCESS) {
        HCCL_ERROR("[HcomAllGatherVOp][InferDataType] SetOutputDataType failed.");
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

const char* HcomAllGatherVOp::GetOpTypeName() const { return HCCL_KERNEL_OP_TYPE_ALLGATHERV.c_str(); }

static ge::BaseCustomOp* CreateHcomAllGatherVOp() { return new HcomAllGatherVOp(); }
// 条件注册: 目标平台 (950/960) 且 GE/RTS 新接口在位才注册接管, 否则回退原 hccl_engine 路径
// (hccl_custom_op.h CreateSocGatedRegister: SoC 判定 + GE/RTS 接口在位判定)
static ge::CustomOpCreatorRegister* g_hcom_all_gather_v_register
    = hccl::CreateSocGatedRegister(HCCL_KERNEL_OP_TYPE_ALLGATHERV.c_str(), CreateHcomAllGatherVOp);
