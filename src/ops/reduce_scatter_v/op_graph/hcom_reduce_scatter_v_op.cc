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
// 另使用 hcom.h 常量: HCCL_KERNEL_OP_TYPE_REDUCESCATTERV
// 公共工具 (CalcOnlineResources/GetCCLBufferAvailableSize/BuildTagVec/ReadU64VectorWithD2H 等) 见
// hccl_op_utils.h/.cc
//
// 前向声明的 libhccl.so 内部接口 (同包直调, 对应 InfoStore/lowering 的 HcceReduceScatterV*
// dlsym 路径; 不 include src/ops/reduce_scatter_v/reduce_scatter_v.h 以避免内部头依赖链):
//   HcclReduceScatterV            (reduce_scatter_v.h) — comm 版, 动态路径
//   HcclReduceScatterVGraphMode   (reduce_scatter_v.h) — group+tag+streams+scratch 版, 静态路径
//   HcclSetAivCoreLimitGraphMode  — AIV 核心数限制 (同 hcom_all_reduce_op.cc 的前向声明)

#include "hcom_reduce_scatter_v_op.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "hccl/hcom.h"
#include "hccl_op_utils.h"
#include "log.h"

extern "C" {
HcclResult HcclReduceScatterV(
    void* sendBuf, const void* sendCounts, const void* sendDispls, void* recvBuf, uint64_t recvCount,
    HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream);
HcclResult HcclReduceScatterVGraphMode(
    void* sendBuf, const void* sendCounts, const void* sendDispls, void* recvBuf, uint64_t recvCount,
    HcclDataType dataType, HcclReduceOp op, const char* group, aclrtStream stream, const char* tag, void** streams,
    size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize);
HcclResult HcclSetAivCoreLimitGraphMode(const char* group, uint32_t aivCoreLimit);
}

ge::graphStatus HcomReduceScatterVOp::ExtractVAttrs(hccl::HcclOpState& st, const char*& group)
{
    const gert::RuntimeAttrs* attrs = st.ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomReduceScatterVOp][ExtractVAttrs] attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const char* reduction = attrs->GetStr(ATTR_REDUCTION);
    group = attrs->GetStr(ATTR_GROUP);
    if (group == nullptr) {
        HCCL_ERROR("[HcomReduceScatterVOp] group attr is null.");
        return ge::GRAPH_FAILED;
    }

    st.dataType = hccl::GeDataTypeToHccl(st.ctx->GetInputTensor(INPUT_X)->GetDataType());
    if (st.dataType == HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR(
            "[HcomReduceScatterVOp][ExtractVAttrs] unsupported data type %d.",
            static_cast<int>(st.ctx->GetInputTensor(INPUT_X)->GetDataType()));
        return ge::GRAPH_FAILED;
    }

    st.reduceOp = hccl::StringToReduceOp(reduction);
    if (st.reduceOp == HCCL_REDUCE_RESERVED) {
        HCCL_ERROR(
            "[HcomReduceScatterVOp][ExtractVAttrs] unsupported reduction '%s'.", reduction ? reduction : "(null)");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterVOp::CheckOutDim0(uint64_t rawOutDim0, int64_t& outDim0)
{
    // uint64 → int64 窄化校验: counts 来自调用方张量, 超 INT64_MAX 会回绕为负非法值
    if (rawOutDim0 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        HCCL_ERROR("[HcomReduceScatterVOp][CheckOutDim0] recv_count[0]=%lu exceeds int64 range.", rawOutDim0);
        return ge::GRAPH_FAILED;
    }
    outDim0 = static_cast<int64_t>(rawOutDim0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterVOp::ExtractParamsStatic(hccl::HcclOpState& st)
{
    // 参数获取与 counts/displacements 读取换算 (对应 InfoStore HcomReduceScatterVOpKernel)
    const gert::Tensor* xTensor = st.ctx->GetInputTensor(INPUT_X);
    if (xTensor == nullptr) {
        HCCL_ERROR("[HcomReduceScatterVOp][ExtractParamsStatic] x tensor is null.");
        return ge::GRAPH_FAILED;
    }

    const char* group = nullptr;
    HCCL_GE_CHK_RET(ExtractVAttrs(st, group));
    constexpr const char* opName = "HcomReduceScatterVOp";

    // rankSize: counts 读取长度与 displs 计算都依赖; GetCommunicator 在模板后续才执行,
    // 此处用 group 版接口
    HCCL_GE_CHK_RET(hccl::GetVRankSizeByGroup(group, opName, st.rankSize));

    // 读 counts 二元组 (recv_count / send_counts) + 可选 displs 张量 (三态, 见 ReadOptionalVDispls)
    // custom op 读不到 vInputVec (RuntimeAttrs 仅 index-based 标量), 改为从输入张量 host 直读或 D2H
    std::vector<uint64_t> rawRecvCount;
    std::vector<uint64_t> rawSendCounts;
    std::vector<uint64_t> rawSendDispls;
    HCCL_GE_CHK_RET(hccl::ReadVCountPairWithCheck(
        st.ctx->GetInputTensor(INPUT_RECV_COUNT), st.ctx->GetInputTensor(INPUT_SEND_COUNTS), opName, "recv_count",
        "send_counts", st.rankSize, rawRecvCount, rawSendCounts, static_cast<aclrtStream>(st.ctx->GetStream())));
    bool hasDispls = false;
    HCCL_GE_CHK_RET(hccl::ReadOptionalVDispls(
        st.ctx->GetInputTensor(INPUT_SEND_DISPLS), opName, "send_displacements", st.rankSize, rawSendDispls, hasDispls,
        static_cast<aclrtStream>(st.ctx->GetStream())));

    // otherDims 换算: x shape dims[1:] 乘积 (对应 vinputvec pass GetOtherDims)
    uint64_t otherDims = 0;
    HCCL_GE_CHK_RET(hccl::CalcOtherDimsFromShape(xTensor->GetShape().GetStorageShape(), opName, otherDims));

    // sendCounts ×otherDims; sendDispls = rawSendDispls(或前缀和) ×otherDims → 元素单位
    // (对应 vinputvec pass MultiplyByOtherDims / CalcDisplacements)
    std::vector<uint64_t> dimDispls;
    HCCL_GE_CHK_RET(hccl::ResolveVDispls(hasDispls, rawSendDispls, rawSendCounts, dimDispls));
    HCCL_GE_CHK_RET(
        hccl::ConvertVCountsToElementUnit(rawSendCounts, dimDispls, otherDims, st.sendCounts, st.sendDispls));

    // 输出 shape: outDim0 = rawRecvCount[0] 直取 (recv_count 为首维单位,
    // 与上游 proto 行为一致, 不再除 otherDims); 空输出 (recv_count=0, 本 rank 只发不收) 只定形不分配,
    // HCCL 对 recvCount==0 不校验 recvBuf (CheckReduceScatterVInputParam), 输出地址为空合法
    int64_t outDim0 = 0;
    HCCL_GE_CHK_RET(CheckOutDim0(rawRecvCount[0], outDim0));
    gert::Tensor* outTensor = hccl::AllocVOutputTensor(st, xTensor, outDim0);
    if (outTensor == nullptr) {
        HCCL_ERROR("[HcomReduceScatterVOp][ExtractParamsStatic] failed to allocate output tensor.");
        return ge::GRAPH_FAILED;
    }

    st.inputPtrs.push_back(const_cast<void*>(xTensor->GetAddr()));
    st.outputPtrs.push_back(outTensor->GetAddr());
    // recvCount = output 元素总数 (lowering 语义: GetCountByShape(outputShapes[0]))
    st.recvCount = static_cast<uint64_t>(outTensor->GetShapeSize());
    st.group = group;
    st.stream = st.ctx->GetStream();

    // rankSize 大时逐条 INFO 会刷爆日志, 合并为单条摘要 (前 MAX_LOG_RANK_NUM 个 + total 提示)
    HCCL_INFO(
        "[HcomReduceScatterVOp][ExtractParamsStatic] input=%p output=%p recvCount=%lu rankSize=%u "
        "otherDims=%lu hasDispls=%d dataType=%d reduceOp=%d group=%s stream=%p sendCounts/displs: %s.",
        st.inputPtrs[0], st.outputPtrs[0], st.recvCount, st.rankSize, otherDims, static_cast<int>(hasDispls),
        st.dataType, st.reduceOp, st.group ? st.group : "(null)", st.stream,
        hccl::SummarizeVCounts(st.sendCounts, st.sendDispls).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterVOp::CalcResourcesStatic(hccl::HcclOpState& st)
{
    // 对应 Builder CalcOpRunningResources 的 RSV 分支 (hcom_ops_kernel_builder.cc):
    // count = max(sendCounts), 空 sendCounts 时 count=0
    uint64_t count = 0;
    if (!st.sendCounts.empty()) {
        count = *std::max_element(st.sendCounts.begin(), st.sendCounts.end());
    }
    // CalcOnlineResources 从 st.count/st.reduceOp 取值; rankSize 由 ExtractParamsStatic 存入 st.rankSize
    st.count = count;

    // 在线计算输入: 环境推导 available size (CalcOnlineResources 契约, 对照旧路径 kernel_builder 的
    // GetCCLBufferAvailableSize, 与 AllReduce/Reduce/RS 同源), 与实际 buffer 尺寸是两个来源;
    // V 算子无 Refresh 路径 (loopMaxTime 恒 0), 不消费 st.cclBuffSize, 不查询通信域实际 buffer
    uint64_t cclBufferAvailSize = 0;
    HCCL_GE_CHK_RET(hccl::GetCCLBufferAvailableSize(cclBufferAvailSize));
    return hccl::CalcOnlineResources(
        st, st.rankSize, cclBufferAvailSize, HCCL_KERNEL_OP_TYPE_REDUCESCATTERV,
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
}

ge::graphStatus HcomReduceScatterVOp::LaunchHcclOpStatic(hccl::HcclOpState& st)
{
    // 静态直传: 用户地址 + 单 tag, 无 loop/refresh/crack, 不判 needRefresh
    if (st.scratchMemSize > 0) {
        st.scratchMem = st.ctx->MallocWorkSpace(st.scratchMemSize);
        if (st.scratchMem == nullptr) {
            HCCL_ERROR(
                "[HcomReduceScatterVOp][LaunchHcclOpStatic] MallocWorkSpace failed, size=%lu.", st.scratchMemSize);
            return ge::GRAPH_FAILED;
        }
    }

    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceScatterVOp][LaunchHcclOpStatic] HcclSetAivCoreLimitGraphMode failed, ret=%d.",
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
    HCCL_GE_CHK_RET(hccl::BuildTagVec(HCCL_KERNEL_OP_TYPE_REDUCESCATTERV, st.group, 0, st.tags));

    // 直传用户地址 (组 4); sendCounts/sendDispls 传 host 向量数据 (InfoStore 同样传 host 指针)
    ret = HcclReduceScatterVGraphMode(
        st.inputPtrs[0], st.sendCounts.data(), st.sendDispls.data(), st.outputPtrs[0], st.recvCount, st.dataType,
        st.reduceOp, st.group, static_cast<aclrtStream>(st.stream), st.tags[0].c_str(),
        st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(), st.scratchMem,
        st.scratchMemSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceScatterVOp][LaunchHcclOpStatic] HcclReduceScatterVGraphMode failed, ret=%d.",
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterVOp::LaunchHcclOpDynamic(hccl::HcclOpState& st)
{
    // 对应 lowering HcomLaunchReduceScatterVKernel: 直传用户地址, comm 版接口,
    // 无 tag/streams/scratch; 参数提取已由基类默认 ExtractParamsDynamic → ExtractParamsStatic 完成
    // AIV 算力上限用 GraphMode 版 HcclSetAivCoreLimitGraphMode (与 AllReduce/Reduce 动态路径同款)
    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceScatterVOp][LaunchHcclOpDynamic] HcclSetAivCoreLimitGraphMode failed, ret=%d.",
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    ret = HcclReduceScatterV(
        st.inputPtrs[0], st.sendCounts.data(), st.sendDispls.data(), st.outputPtrs[0], st.recvCount, st.dataType,
        st.reduceOp, st.comm, static_cast<aclrtStream>(st.stream));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceScatterVOp][LaunchHcclOpDynamic] HcclReduceScatterV failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterVOp::InferShape(gert::InferShapeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomReduceScatterVOp][InferShape] infer shape context is null.");
        return ge::GRAPH_FAILED;
    }
    // 移植 reduce_scatter_v_proto.cc HcomReduceScatterVInferShapeV2 (含上游首维单位修复):
    // recv_count 为首维单位, 输出首维直取 recvCount[0], 不再除 otherDims
    // (旧元素口径对多维输入首维推导偏小)
    const gert::Shape* inputShape = ctx->GetInputShape(INPUT_X);
    if (inputShape == nullptr) {
        HCCL_ERROR("[HcomReduceScatterVOp][InferShape] input shape is null.");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outputShape = ctx->GetOutputShape(0);
    if (outputShape == nullptr) {
        HCCL_ERROR("[HcomReduceScatterVOp][InferShape] output shape is null.");
        return ge::GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() == 0) {
        HCCL_ERROR("[HcomReduceScatterVOp][InferShape] input tensor's first dim is illegal, expected: > 0, "
                   "actual: 0.");
        return ge::GRAPH_FAILED;
    }

    // const 数据不可用 → 拷贝 input shape 且首维未知 (对应 proto HcomIsConstData 失败分支;
    // eager 路径下真实 shape 由 Execute 的 MallocOutputTensor 决定)
    std::vector<uint64_t> recvCount;
    std::vector<uint64_t> sendCounts;
    if (hccl::ReadHostU64Vector(ctx->GetInputTensor(INPUT_RECV_COUNT), recvCount) != ge::GRAPH_SUCCESS
        || hccl::ReadHostU64Vector(ctx->GetInputTensor(INPUT_SEND_COUNTS), sendCounts) != ge::GRAPH_SUCCESS) {
        *outputShape = *inputShape;
        outputShape->SetDim(0, ge::UNKNOWN_DIM);
        return ge::GRAPH_SUCCESS;
    }
    if (recvCount.empty()) {
        HCCL_ERROR("[HcomReduceScatterVOp][InferShape] recv_count is empty.");
        return ge::GRAPH_FAILED;
    }

    *outputShape = *inputShape;
    // 受检窄化: 复用 CheckOutDim0, 超界报错而非静默回绕
    int64_t outDim0 = 0;
    HCCL_GE_CHK_RET(CheckOutDim0(recvCount[0], outDim0));
    outputShape->SetDim(0, outDim0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterVOp::InferDataType(gert::InferDataTypeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomReduceScatterVOp][InferDataType] infer datatype context is null.");
        return ge::GRAPH_FAILED;
    }
    // 移植 reduce_scatter_v_proto.cc HcomReduceScatterVInferDataTypeV2: out dtype = in[0] dtype
    ge::DataType inputType = ctx->GetInputDataType(INPUT_X);
    if (inputType == ge::DT_UNDEFINED) {
        HCCL_ERROR("[HcomReduceScatterVOp][InferDataType] input dtype is undefined.");
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus ret = ctx->SetOutputDataType(0, inputType);
    if (ret != ge::GRAPH_SUCCESS) {
        HCCL_ERROR("[HcomReduceScatterVOp][InferDataType] SetOutputDataType failed.");
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

const char* HcomReduceScatterVOp::GetOpTypeName() const { return HCCL_KERNEL_OP_TYPE_REDUCESCATTERV.c_str(); }

static ge::BaseCustomOp* CreateHcomReduceScatterVOp() { return new HcomReduceScatterVOp(); }
// 条件注册: 目标平台 (950/960) 且 GE/RTS 新接口在位才注册接管, 否则回退原 hccl_engine 路径
// (hccl_custom_op.h CreateSocGatedRegister: SoC 判定 + GE/RTS 接口在位判定)
static ge::CustomOpCreatorRegister* g_hcom_reduce_scatter_v_register
    = hccl::CreateSocGatedRegister(HCCL_KERNEL_OP_TYPE_REDUCESCATTERV.c_str(), CreateHcomReduceScatterVOp);
