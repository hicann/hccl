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
//   HcomGetRankSize / HcomGetRankId — 通信域 rank 数与实际 rank (CalcV/VCRecvDim)
//   HcomGenerateCclOpTag     — 生成 op tag: opType_hash(group)_autoIndex (GenerateTag)
// 另使用 hcom.h 常量: HCCL_KERNEL_OP_TYPE_ALLTOALLV, HCCL_KERNEL_OP_TYPE_ALLTOALLVC,
//                    HCCL_KERNEL_OP_TYPE_ALLTOALL, CCL_OP_TAG_MAX_LEN,
//                    ALG_NAME_MAX_LEN, CommNumHcom::COMM_VALUE_DEFAULT
// 另调用 hccl_comm.h: HcclGetRankSize (CalcResourcesStatic / CalcAllToAllPerRankCount)
// 另使用 hccl_types.h: HcclCMDType 枚举 — HCCL_CMD_ALLTOALLV/ALLTOALLVC/ALLTOALL

// 前向声明的 libhccl.so 内部接口 (同包直调, 签名对齐 src/ops/all_to_all_v/all_to_all_v.h):
//   HcclAlltoAllGraphMode / HcclAlltoAllVGraphMode / HcclAlltoAllVCGraphMode
//   HcclCreateOpParamGraphMode 等 OpParamGraphMode 系列 + HcclSelectAlgGraphMode
//   + HcclCalcOpResOnlineGraphMode + HcclSetAivCoreLimitGraphMode

#include "hcom_alltoallv_op.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "acl/acl_rt.h"
#include "hccl/hccl_comm.h"
#include "hccl/hcom.h"
#include "hccl_op_utils.h"
#include "hccl_rt_utils.h"
#include "log.h"
#include "graph_mode_c_api.h"

// ---- kind 工具函数函数 ----

const char* HcomAllToAllVOp::GetOpTypeStr() const
{
    switch (kind_) {
        case HcomAllToAllKind::kAllToAll:
            return HCCL_KERNEL_OP_TYPE_ALLTOALL.c_str();
        case HcomAllToAllKind::kAllToAllV:
            return HCCL_KERNEL_OP_TYPE_ALLTOALLV.c_str();
        case HcomAllToAllKind::kAllToAllVC:
        default:
            return HCCL_KERNEL_OP_TYPE_ALLTOALLVC.c_str();
    }
}

HcclCMDType HcomAllToAllVOp::GetCmdType() const
{
    switch (kind_) {
        case HcomAllToAllKind::kAllToAll:
            return HCCL_CMD_ALLTOALL;
        case HcomAllToAllKind::kAllToAllV:
            return HCCL_CMD_ALLTOALLV;
        case HcomAllToAllKind::kAllToAllVC:
        default:
            return HCCL_CMD_ALLTOALLVC;
    }
}

size_t HcomAllToAllVOp::GetExpectInputNum() const
{
    switch (kind_) {
        case HcomAllToAllKind::kAllToAll:
            return 1; // x
        case HcomAllToAllKind::kAllToAllV:
            return 5; // send_data + send_counts/sdispls/recv_counts/rdispls
        case HcomAllToAllKind::kAllToAllVC:
        default:
            return 2; // send_data + send_count_matrix
    }
}

// ---- 静态路径 (InfoStore-style) ----

// 公共参数校验: send tensor / attrs / group (VC 的 group attr 索引为 1) / dataType 转换 / 全部 input 存在性
static ge::graphStatus ExtractCommonParams(
    hccl::HcclOpState& st, HcomAllToAllKind kind, size_t expectInputNum, const gert::Tensor*& sendData,
    const gert::RuntimeAttrs*& attrs, const char*& group)
{
    sendData = st.ctx->GetInputTensor(HcomAllToAllVOp::INPUT_IDX_SEND_DATA);
    if (sendData == nullptr) {
        HCCL_ERROR("[HcomAllToAllVOp][ExtractParamsStatic] send_data tensor is null.");
        return ge::GRAPH_FAILED;
    }

    attrs = st.ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomAllToAllVOp][ExtractParamsStatic] attrs is null.");
        return ge::GRAPH_FAILED;
    }
    group = attrs->GetStr(
        kind == HcomAllToAllKind::kAllToAllVC ? HcomAllToAllVOp::ATTR_IDX_GROUP_VC : HcomAllToAllVOp::ATTR_IDX_GROUP);
    if (group == nullptr) {
        HCCL_ERROR("[HcomAllToAllVOp][ExtractParamsStatic] group attr is null.");
        return ge::GRAPH_FAILED;
    }

    // ConversionOpDataType: dataType 取 input[0] (AllToAll 系列无 HCOM_ATTR_DTYPE 属性)
    st.dataType = hccl::GeDataTypeToHccl(sendData->GetDataType());
    if (st.dataType == HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][ExtractParamsStatic] unsupported data type %d.",
            static_cast<int>(sendData->GetDataType()));
        return ge::GRAPH_FAILED;
    }

    // PrepareSelectAivParam: reduction 默认 HCCL_REDUCE_SUM, 仅 ALLREDUCE/REDUCE_SCATTER 才读 opDesc
    st.reduceOp = HCCL_REDUCE_SUM;

    // 校验全部 input 存在 (V: 5 个数组/数据 tensor; VC: 2 个; AllToAll: 1 个)
    for (size_t i = 0; i < expectInputNum; i++) {
        if (st.ctx->GetInputTensor(i) == nullptr) {
            HCCL_ERROR("[HcomAllToAllVOp][ExtractParamsStatic] input tensor[%zu] is null.", i);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// V 形态输出维度 (GE InfoStore GetHcomAlltoallVOpMemSize 口径): counts 四元组读成 host 副本
// (host placement 直读 / device placement 由 ReadU64VectorWithD2H 内部同步主流后 D2H: 图模式下
// counts tensor 地址是 device HBM 且未必 host 可读, 直接解引用会段错误), 校验长度 >= rankSize,
// outDim0 = max(rdispls[i]+recvCounts[i])
static ge::graphStatus CalcVRecvDim(hccl::HcclOpState& st, const char* group, int64_t& outDim0)
{
    HcclResult ret = HcomGetRankSize(group, &st.rankSize);
    if (ret != HCCL_SUCCESS || st.rankSize == 0) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][ExtractParamsStatic] HcomGetRankSize failed for group '%s', ret=%d.", group,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    // 四张 counts 同流产出: 入口同步一次后逐个 NoSync 读, 免去逐张量 4 次流同步
    const aclrtStream stream = static_cast<aclrtStream>(st.ctx->GetStream());
    HCCL_GE_CHK_RET(hccl::SyncStreamBeforeD2H(stream));
    if (hccl::ReadU64VectorWithD2HNoSync(st.ctx->GetInputTensor(HcomAllToAllVOp::INPUT_IDX_SEND_COUNTS), st.sendCounts)
            != ge::GRAPH_SUCCESS
        || hccl::ReadU64VectorWithD2HNoSync(
               st.ctx->GetInputTensor(HcomAllToAllVOp::INPUT_IDX_SEND_DISPLS), st.sendDispls)
               != ge::GRAPH_SUCCESS
        || hccl::ReadU64VectorWithD2HNoSync(
               st.ctx->GetInputTensor(HcomAllToAllVOp::INPUT_IDX_RECV_COUNTS), st.recvCounts)
               != ge::GRAPH_SUCCESS
        || hccl::ReadU64VectorWithD2HNoSync(
               st.ctx->GetInputTensor(HcomAllToAllVOp::INPUT_IDX_RECV_DISPLS), st.recvDispls)
               != ge::GRAPH_SUCCESS) {
        HCCL_ERROR("[HcomAllToAllVOp][ExtractParamsStatic] read counts arrays (sendCounts/sdispls/recvCounts/rdispls) "
                   "via D2H failed.");
        return ge::GRAPH_FAILED;
    }
    if (st.sendCounts.size() < st.rankSize || st.sendDispls.size() < st.rankSize || st.recvCounts.size() < st.rankSize
        || st.recvDispls.size() < st.rankSize) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][ExtractParamsStatic] sendCounts=%zu sdispls=%zu recvCounts=%zu rdispls=%zu, "
            "expect >= rankSize=%u.",
            st.sendCounts.size(), st.sendDispls.size(), st.recvCounts.size(), st.recvDispls.size(), st.rankSize);
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < st.rankSize; i++) {
        int64_t end = static_cast<int64_t>(st.recvDispls[i]) + static_cast<int64_t>(st.recvCounts[i]);
        if (outDim0 < end) {
            outDim0 = end;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// VC 形态输出维度 (GE InfoStore GetHcomAlltoallVCOpMemSize 口径): rankId 取通信域实际 rank
// (对齐 InfoStore 运行时口径, attr rank 仅日志), 矩阵读成 host 副本写回 sendCountMatrix (launch 消费),
// outDim0 = 本 rank 列和 Σ matrix[rankId + i*rankSize]
static ge::graphStatus CalcVCRecvDim(hccl::HcclOpState& st, const char* group, int64_t& outDim0)
{
    HcclResult ret = HcomGetRankSize(group, &st.rankSize);
    if (ret != HCCL_SUCCESS || st.rankSize == 0) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][ExtractParamsStatic] HcomGetRankSize failed for group '%s', ret=%d.", group,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    ret = HcomGetRankId(group, &st.rankId);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][ExtractParamsStatic] HcomGetRankId failed for group '%s', ret=%d.", group,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    if (hccl::ReadU64VectorWithD2H(
            st.ctx->GetInputTensor(HcomAllToAllVOp::INPUT_IDX_SEND_COUNT_MATRIX), st.sendCountMatrix,
            static_cast<aclrtStream>(st.ctx->GetStream()))
        != ge::GRAPH_SUCCESS) {
        HCCL_ERROR("[HcomAllToAllVOp][ExtractParamsStatic] read sendCountMatrix via D2H failed.");
        return ge::GRAPH_FAILED;
    }
    if (st.sendCountMatrix.size() < static_cast<uint64_t>(st.rankSize) * st.rankSize) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][ExtractParamsStatic] matrix.size=%zu < rankSize^2=%u.", st.sendCountMatrix.size(),
            st.rankSize * st.rankSize);
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < st.rankSize; i++) {
        const uint64_t cnt = st.sendCountMatrix[st.rankId + static_cast<uint64_t>(st.rankSize) * i];
        // 受检窄化+累加: u64 超出 int64 值域或累加溢出时报错, 不静默回绕成负维
        if (cnt > static_cast<uint64_t>(INT64_MAX) || outDim0 > INT64_MAX - static_cast<int64_t>(cnt)) {
            HCCL_ERROR("[HcomAllToAllVOp][CalcVCRecvDim] outDim0 accumulate overflow, cnt=%lu.", cnt);
            return ge::GRAPH_FAILED;
        }
        outDim0 += static_cast<int64_t>(cnt);
    }
    return ge::GRAPH_SUCCESS;
}

// 输出张量分配: AllToAll 镜像 send_data shape; V/VC 收发不等长, 镜像 send_data 会在 recv > send 时
// 写溢出 — 定形 1-D [outDim0] 按接收侧实际大小分配; 零接收 (V: recvCounts 全 0 / VC: 本 rank 列和为 0)
// 在 HCCL 语义内合法 (入口 CheckBufNullptr: counts 全零直接成功, recvBuf 可空), 但 GE eager
// MallocOutputTensor 断言 GetSize()>0 — 空输出改走公共 SetEmptyOutputTensor 只定形不分配
static ge::graphStatus AllocRecvTensor(
    hccl::HcclOpState& st, HcomAllToAllKind kind, const gert::Tensor* sendData, int64_t outDim0,
    gert::Tensor*& recvData)
{
    recvData = nullptr;
    if (kind == HcomAllToAllKind::kAllToAll) {
        recvData = st.ctx->MallocOutputTensor(
            HcomAllToAllVOp::OUTPUT_IDX_RECV_DATA, sendData->GetShape(), sendData->GetFormat(),
            sendData->GetDataType());
    } else {
        gert::StorageShape outShape = sendData->GetShape();
        outShape.MutableStorageShape().SetDimNum(1);
        outShape.MutableStorageShape().SetDim(0, outDim0);
        outShape.MutableOriginShape().SetDimNum(1);
        outShape.MutableOriginShape().SetDim(0, outDim0);
        if (outDim0 == 0) {
            recvData = hccl::SetEmptyOutputTensor(
                st.ctx, HcomAllToAllVOp::OUTPUT_IDX_RECV_DATA, outShape, sendData->GetFormat(),
                sendData->GetDataType());
        } else {
            recvData = st.ctx->MallocOutputTensor(
                HcomAllToAllVOp::OUTPUT_IDX_RECV_DATA, outShape, sendData->GetFormat(), sendData->GetDataType());
        }
    }
    if (recvData == nullptr) {
        HCCL_ERROR("[HcomAllToAllVOp][ExtractParamsStatic] failed to allocate output tensor.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// send/recv 数据 buffer 透传 tensor 原地址 (device 地址, HCCL 作为通信数据源/目的消费);
// counts 数组/矩阵不透传 device 地址 — 上游已读成 host 副本, launch 传 host .data()
// (HCCL 入口对 counts 仅 host 消费, host 指针在 device 内存 host 可读/不可读两类环境均安全)
static void CollectIoPtrs(hccl::HcclOpState& st, size_t inputNum, gert::Tensor* recvData)
{
    for (size_t i = 0; i < inputNum; i++) {
        const gert::Tensor* inTensor = st.ctx->GetInputTensor(i);
        st.inputPtrs.push_back(const_cast<void*>(inTensor->GetAddr()));
    }
    st.outputPtrs.push_back(recvData->GetAddr());
}

ge::graphStatus HcomAllToAllVOp::ExtractParamsStatic(hccl::HcclOpState& st)
{
    const gert::Tensor* sendData = nullptr;
    const gert::RuntimeAttrs* attrs = nullptr;
    const char* group = nullptr;
    size_t expectInputNum = GetExpectInputNum();
    HCCL_GE_CHK_RET(ExtractCommonParams(st, kind_, expectInputNum, sendData, attrs, group));

    gert::Tensor* recvData = nullptr;
    int64_t outDim0 = 0;
    if (kind_ == HcomAllToAllKind::kAllToAllV) {
        HCCL_GE_CHK_RET(CalcVRecvDim(st, group, outDim0));
    } else if (kind_ == HcomAllToAllKind::kAllToAllVC) {
        HCCL_GE_CHK_RET(CalcVCRecvDim(st, group, outDim0));
    }
    if (outDim0 < 0) {
        HCCL_ERROR("[HcomAllToAllVOp][ExtractParamsStatic] output dim0=%ld is negative.", outDim0);
        return ge::GRAPH_FAILED;
    }
    HCCL_GE_CHK_RET(AllocRecvTensor(st, kind_, sendData, outDim0, recvData));
    CollectIoPtrs(st, expectInputNum, recvData);

    // AllToAll per-rank count 素材 (InfoStore/lowering 语义: sendCount = 总元素数 / rankSize)
    st.totalElemCount = static_cast<uint64_t>(sendData->GetShapeSize());

    st.group = group;
    st.stream = st.ctx->GetStream();

    // AllToAllVC: attr rank 仅日志对照 (output 大小用通信域实际 rank, HcomGetRankId, 对齐 InfoStore 运行时口径)
    int64_t vcRank = -1;
    if (kind_ == HcomAllToAllKind::kAllToAllVC) {
        const int64_t* rankAttr = attrs->GetInt(HcomAllToAllVOp::ATTR_IDX_RANK_VC);
        vcRank = rankAttr ? *rankAttr : -1;
    }

    HCCL_INFO(
        "[HcomAllToAllVOp][ExtractParamsStatic] kind=%d inputNum=%zu send=%p recv=%p totalElemCount=%lu "
        "dataType=%d group=%s stream=%p rank=%ld.",
        static_cast<int>(kind_), expectInputNum, st.inputPtrs[HcomAllToAllVOp::INPUT_IDX_SEND_DATA], st.outputPtrs[0],
        st.totalElemCount, st.dataType, st.group ? st.group : "(null)", st.stream, vcRank);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllToAllVOp::CalcResourcesStatic(hccl::HcclOpState& st)
{
    uint32_t rankSize = 0;
    HcclResult ret = HcclGetRankSize(st.comm, &rankSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomAllToAllVOp][CalcResourcesStatic] HcclGetRankSize failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    // 在线计算输入: 环境推导 available size — 对照 GE builder CalcOpRunningParam 末尾
    // GetCCLBufferAvailableSize (hcom_ops_kernel_builder.cc, 对含 ALLTOALLV 的全部算子同源),
    // 与实际 buffer 尺寸是两个来源 (AllReduce 同款双尺寸保真口径)。
    // 旧实现 HcomGetCommCCLBufferSize 只查不建, buf 未创建时输入 0, 影响 opMemSize/算法选择
    uint64_t cclBuffSize = 0;
    HCCL_GE_CHK_RET(hccl::GetCCLBufferAvailableSize(cclBuffSize));

    // selector count: HcomOpUtils::CalcCommonCount 语义 — 遍历全部 inputs (V 的数组 tensor 也计入)
    //   ALLTOALL 分支:   blockSize = inputSize / rankSize (不对齐)
    //   默认分支 (V/VC): blockSize = inputSize 向上 512B 对齐
    //   count = totalSize / dataTypeSize (dataTypeSize 统一取 op dataType, 即 input[0])
    constexpr uint64_t ALIGNED_SIZE = 512;
    uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    uint64_t totalSize = 0;
    for (size_t i = 0; i < st.inputPtrs.size(); i++) {
        const gert::Tensor* inTensor = st.ctx->GetInputTensor(i);
        uint64_t inputSize = static_cast<uint64_t>(inTensor->GetShapeSize()) * unitSize;
        uint64_t blockSize = (kind_ == HcomAllToAllKind::kAllToAll) ?
                                 inputSize / rankSize :
                                 (inputSize + ALIGNED_SIZE - 1) / ALIGNED_SIZE * ALIGNED_SIZE;
        totalSize += blockSize;
    }
    st.count = totalSize / unitSize;

    return CalcOnlineResources(st, rankSize, cclBuffSize);
}

// opParam 组装: 公共 setter 组 (aivCoreLimit/opType/dataType/rankSize/count/cclBuffSize) +
// V 形态 dataCount 覆写 (CalcOpRunningParam ALLTOALLV 分支: max(sendCounts), GE builder L2114 同口径) —
// 消费 ExtractParamsStatic 经 D2H 读出的 host 副本 st.sendCounts (图模式下 counts tensor 是
// device 地址且未必 host 可读, host 直接解引用会段错误); 错误路径不 Destroy, 由调用方统一处理
static ge::graphStatus SetOpParams(
    OpParamGraphMode* opParam, const hccl::HcclOpState& st, const char* opType, HcomAllToAllKind kind,
    uint32_t rankSize, uint64_t cclBuffSize)
{
    // setter 返回码逐个检查: 失败静默继续会用缺参 opParam 计算资源
    HcclResult setRet = HcclSetAivSelectOpParamGraphMode(opParam, st.aivCoreLimit);
    if (setRet != HCCL_SUCCESS || (setRet = HcclSetOpParamGraphModeOpType(opParam, opType)) != HCCL_SUCCESS
        || (setRet = HcclSetOpParamGraphModeDataType(opParam, st.dataType)) != HCCL_SUCCESS
        || (setRet = HcclSetOpParamGraphModeRankSize(opParam, &rankSize)) != HCCL_SUCCESS
        || (setRet = HcclSetOpParamGraphModeDataCount(opParam, &st.count)) != HCCL_SUCCESS
        || (setRet = HcclSetOpParamGraphModeHCCLBufferSize(opParam, &cclBuffSize)) != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomAllToAllVOp][SetOpParams] HcclSet*OpParamGraphMode failed, ret=%d.", static_cast<int>(setRet));
        return ge::GRAPH_FAILED;
    }

    if (kind != HcomAllToAllKind::kAllToAllV) {
        return ge::GRAPH_SUCCESS;
    }
    if (st.sendCounts.size() < static_cast<size_t>(rankSize)) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][CalcOnlineResources] sendCounts.size=%zu < rankSize=%u.", st.sendCounts.size(),
            rankSize);
        return ge::GRAPH_FAILED;
    }
    uint64_t maxCount = 0;
    for (uint32_t i = 0; i < rankSize; i++) {
        maxCount = std::max(maxCount, st.sendCounts[i]);
    }
    HcclResult ret = HcclSetOpParamGraphModeDataCount(opParam, &maxCount);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][CalcOnlineResources] HcclSetOpParamGraphModeDataCount (max) failed, ret=%d.",
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    HCCL_INFO(
        "[HcomAllToAllVOp][CalcOnlineResources] ALLTOALLV dataCount overwritten to max(sendCounts)=%lu.", maxCount);
    return ge::GRAPH_SUCCESS;
}

// selector 算法选择 + AIV 资源覆写 (SetSuperKernelScopeAttr → PrepareSelectAivParam →
// HcceSelectAlgGraphMode: count 用 selector count, reduction 固定 SUM, 仅 ALLREDUCE/REDUCE_SCATTER
// 才读 opDesc); ifAiv 时离线结果覆写为 AIV 固定资源; 错误路径不 Destroy, 由调用方统一处理
static ge::graphStatus SelectAlgAndApplyAiv(hccl::HcclOpState& st, HcclCMDType cmdType, uint64_t& opMemSize)
{
    char algName[ALG_NAME_MAX_LEN] = {0};
    HcclResult ret = HcclSelectAlgGraphMode(
        st.group, st.count, st.dataType, HCCL_REDUCE_SUM, cmdType, st.aivCoreLimit, &st.ifAiv, algName);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][CalcOnlineResources] HcclSelectAlgGraphMode failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    if (st.ifAiv) {
        constexpr uint32_t AIV_WORKSPACE_MEM_SIZE = 512;
        constexpr uint32_t AIV_TASK_NUM = 3;
        st.streamNum = 0;
        opMemSize = AIV_WORKSPACE_MEM_SIZE;
        st.taskNum = AIV_TASK_NUM;
    }
    return ge::GRAPH_SUCCESS;
}

// 不复用公共 hccl::CalcOnlineResources: V 形态 dataCount 与 selector count 双口径
// (dataCount = max(sendCounts), selector count = CalcResourcesStatic 的 totalSize/unitSize),
// 公共版单一 st.count 无法表达
ge::graphStatus HcomAllToAllVOp::CalcOnlineResources(hccl::HcclOpState& st, uint32_t rankSize, uint64_t cclBuffSize)
{
    OpParamGraphMode* opParam = nullptr;
    HcclResult ret = HcclCreateOpParamGraphMode(&opParam);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][CalcOnlineResources] HcclCreateOpParamGraphMode failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    if (SetOpParams(opParam, st, GetOpTypeStr(), kind_, rankSize, cclBuffSize) != ge::GRAPH_SUCCESS) {
        HcclDestroyOpParamGraphMode(opParam);
        return ge::GRAPH_FAILED;
    }

    uint64_t opMemSize = 0;
    ret = HcclCalcOpResOnlineGraphMode(opParam, &opMemSize, &st.streamNum, &st.taskNum, &st.aivCoreNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][CalcOnlineResources] HcclCalcOpResOnlineGraphMode failed, ret=%d.",
            static_cast<int>(ret));
        HcclDestroyOpParamGraphMode(opParam);
        return ge::GRAPH_FAILED;
    }

    if (SelectAlgAndApplyAiv(st, GetCmdType(), opMemSize) != ge::GRAPH_SUCCESS) {
        HcclDestroyOpParamGraphMode(opParam);
        return ge::GRAPH_FAILED;
    }

    HcclDestroyOpParamGraphMode(opParam);

    st.scratchMemSize = opMemSize;
    st.cclBuffSize = cclBuffSize;

    HCCL_INFO(
        "[HcomAllToAllVOp][CalcOnlineResources] kind=%d rankSize=%u count=%lu scratchMemSize=%lu streamNum=%u "
        "taskNum=%u ifAiv=%d cclBuffSize=%lu.",
        static_cast<int>(kind_), rankSize, st.count, st.scratchMemSize, st.streamNum, st.taskNum, st.ifAiv,
        st.cclBuffSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllToAllVOp::GenerateTag(hccl::HcclOpState& st)
{
    // 对应 InfoStore GenerateOpTagFromTaskInfo + GetTagVectorInfo:
    // AllToAll 系列不在 GetOpKernelLoopTime 覆盖列表 → 永远单 tag, 无 "_looptime_N" 后缀
    char cTag[CCL_OP_TAG_MAX_LEN];
    HcclResult ret
        = HcomGenerateCclOpTag(GetOpTypeStr(), static_cast<s64>(CommNumHcom::COMM_VALUE_DEFAULT), st.group, cTag);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomAllToAllVOp][GenerateTag] HcomGenerateCclOpTag failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    st.tags.clear();
    st.tags.emplace_back(cTag);

    HCCL_INFO("[HcomAllToAllVOp][GenerateTag] tag=%s.", cTag);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllToAllVOp::LaunchHcclOpStatic(hccl::HcclOpState& st)
{
    if (st.scratchMemSize > 0) {
        st.scratchMem = st.ctx->MallocWorkSpace(st.scratchMemSize);
        if (st.scratchMem == nullptr) {
            HCCL_ERROR("[HcomAllToAllVOp][LaunchHcclOpStatic] MallocWorkSpace failed, size=%lu.", st.scratchMemSize);
            return ge::GRAPH_FAILED;
        }
    }

    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][LaunchHcclOpStatic] HcclSetAivCoreLimitGraphMode failed, ret=%d.",
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    HCCL_GE_CHK_RET(GenerateTag(st));

    // attach 流申请 + 交付 (双流: graph/group 粒度, 对照旧 InfoStore SetAttachedStream 每 op 一次;
    // AIV 跳过)
    // follow 流申请 (按 st.streamNum, 复刻旧 main_stream_id 池化; AIV/流数 0 自动跳过)
    HCCL_GE_CHK_RET(RequestFollowStreams(st));
    HCCL_GE_CHK_RET(RequestAttachStream(st));
    HCCL_GE_CHK_RET(RegisterUnfoldStreamToModel(st));

    // 刷新形态分化 (对照 InfoStore):
    //   AllToAll 等长: 原路径 HcomAllToAllOpKernel 不调 RefreshInputAddr/RefreshOutputAddr, 对齐保持直传
    //   V/VC: unknownShape 场景走单次整块刷新 (LaunchRefresh, 超容量报错不分批)
    if (st.needRefresh) {
        if (kind_ == HcomAllToAllKind::kAllToAll) {
            HCCL_WARNING(
                "[HcomAllToAllVOp][LaunchHcclOpStatic] needRefresh=true on AllToAll (equal-length): original path "
                "never refreshes this op, launch with user buffers directly.");
        } else {
            return LaunchRefresh(st);
        }
    }
    return LaunchGraphMode(st);
}

// ---- 动态路径 (lowering-style) ----

ge::graphStatus HcomAllToAllVOp::LaunchHcclOpDynamic(hccl::HcclOpState& st)
{
    // 对应 lowering HcomLaunchAllToAllV/V/VC Kernel(V2): comm 版入口, 无 tag/streams/scratch;
    // send/recv 数据 buffer 直传用户地址, counts/矩阵传 ExtractParamsStatic 读出的 host 副本
    // (HCCL 入口对 counts 仅 host 消费 — 传 device 原地址在 host 不可读环境下会段错误)
    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][LaunchHcclOpDynamic] HcclSetAivCoreLimitGraphMode failed, ret=%d.",
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    uint64_t perRankCount = 0;
    if (kind_ == HcomAllToAllKind::kAllToAll) {
        HCCL_GE_CHK_RET(CalcAllToAllPerRankCount(st, perRankCount));
    }

    switch (kind_) {
        case HcomAllToAllKind::kAllToAll:
            ret = HcclAlltoAll(
                st.inputPtrs[INPUT_IDX_SEND_DATA], perRankCount, st.dataType, st.outputPtrs[OUTPUT_IDX_RECV_DATA],
                perRankCount, st.dataType, st.comm, static_cast<aclrtStream>(st.stream));
            break;
        case HcomAllToAllKind::kAllToAllV:
            ret = HcclAlltoAllV(
                st.inputPtrs[INPUT_IDX_SEND_DATA], st.sendCounts.data(), st.sendDispls.data(), st.dataType,
                st.outputPtrs[OUTPUT_IDX_RECV_DATA], st.recvCounts.data(), st.recvDispls.data(), st.dataType, st.comm,
                static_cast<aclrtStream>(st.stream));
            break;
        case HcomAllToAllKind::kAllToAllVC:
        default:
            ret = HcclAlltoAllVC(
                st.inputPtrs[INPUT_IDX_SEND_DATA], st.sendCountMatrix.data(), st.dataType,
                st.outputPtrs[OUTPUT_IDX_RECV_DATA], st.dataType, st.comm, static_cast<aclrtStream>(st.stream));
            break;
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][LaunchHcclOpDynamic] comm entry failed, kind=%d ret=%d.", static_cast<int>(kind_),
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    HCCL_INFO(
        "[HcomAllToAllVOp][LaunchHcclOpDynamic] kind=%d send=%p recv=%p perRankCount=%lu.", static_cast<int>(kind_),
        st.inputPtrs[INPUT_IDX_SEND_DATA], st.outputPtrs[OUTPUT_IDX_RECV_DATA], perRankCount);
    return ge::GRAPH_SUCCESS;
}

// ---- 共享 launch: 按 kind 分发到三个 GraphMode 入口 ----

ge::graphStatus HcomAllToAllVOp::CalcAllToAllPerRankCount(const hccl::HcclOpState& st, uint64_t& perRankCount) const
{
    // lowering HcomLaunchAllToAllKernel L713: GetCountByShape 全量元素数 / rankSize;
    // InfoStore 等价 (v1 task 对齐总量 count / rankSize)
    uint32_t rankSize = 0;
    HcclResult ret = HcclGetRankSize(st.comm, &rankSize);
    if (ret != HCCL_SUCCESS || rankSize == 0) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][CalcAllToAllPerRankCount] HcclGetRankSize failed, ret=%d rankSize=%u.",
            static_cast<int>(ret), rankSize);
        return ge::GRAPH_FAILED;
    }
    perRankCount = st.totalElemCount / rankSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllToAllVOp::LaunchGraphMode(hccl::HcclOpState& st)
{
    if (st.tags.empty()) {
        HCCL_ERROR("[HcomAllToAllVOp][LaunchGraphMode] tags is empty.");
        return ge::GRAPH_FAILED;
    }
    const char* tag = st.tags[0].c_str();

    // InfoStore HcomAllToAllOpKernel: sendCount = recvCount = hcclInfo.count / rankSize (per-rank)
    uint64_t perRankCount = 0;
    if (kind_ == HcomAllToAllKind::kAllToAll) {
        HCCL_GE_CHK_RET(CalcAllToAllPerRankCount(st, perRankCount));
    }

    HcclResult ret;
    switch (kind_) {
        case HcomAllToAllKind::kAllToAll:
            ret = HcclAlltoAllGraphMode(
                st.inputPtrs[INPUT_IDX_SEND_DATA], perRankCount, st.dataType, st.outputPtrs[OUTPUT_IDX_RECV_DATA],
                perRankCount, st.dataType, st.group, static_cast<aclrtStream>(st.stream), tag,
                st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(), st.scratchMem,
                st.scratchMemSize);
            break;
        case HcomAllToAllKind::kAllToAllV:
            ret = HcclAlltoAllVGraphMode(
                st.inputPtrs[INPUT_IDX_SEND_DATA], st.sendCounts.data(), st.sendDispls.data(), st.dataType,
                st.outputPtrs[OUTPUT_IDX_RECV_DATA], st.recvCounts.data(), st.recvDispls.data(), st.dataType, st.group,
                static_cast<aclrtStream>(st.stream), tag, st.followStreams.empty() ? nullptr : st.followStreams.data(),
                st.followStreams.size(), st.scratchMem, st.scratchMemSize);
            break;
        case HcomAllToAllKind::kAllToAllVC:
        default:
            ret = HcclAlltoAllVCGraphMode(
                st.inputPtrs[INPUT_IDX_SEND_DATA], st.sendCountMatrix.data(), st.dataType,
                st.outputPtrs[OUTPUT_IDX_RECV_DATA], st.dataType, st.group, static_cast<aclrtStream>(st.stream), tag,
                st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(), st.scratchMem,
                st.scratchMemSize);
            break;
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][LaunchGraphMode] GraphMode entry failed, kind=%d ret=%d.", static_cast<int>(kind_),
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    HCCL_INFO(
        "[HcomAllToAllVOp][LaunchGraphMode] kind=%d send=%p recv=%p perRankCount=%lu dataType=%d tag=%s.",
        static_cast<int>(kind_), st.inputPtrs[INPUT_IDX_SEND_DATA], st.outputPtrs[OUTPUT_IDX_RECV_DATA], perRankCount,
        st.dataType, tag);
    return ge::GRAPH_SUCCESS;
}

// V/VC 刷新路径上下文: CCL buf 双侧 (GetOrCreate 获取) + 数据量口径 + args struct 的 device 字段地址
struct HcomAllToAllRefreshCtx {
    void* commInputPtr = nullptr;
    uint64_t commInputSize = 0;
    void* commOutputPtr = nullptr;
    uint64_t commOutputSize = 0;
    uint64_t sendCount = 0; // V: 发送侧末端偏移最大值 / VC: 本 rank 行和
    uint64_t recvCount = 0; // V: 接收侧末端偏移最大值 / VC: 本 rank 列和
    uint64_t inputMemSize = 0;
    uint64_t outputMemSize = 0;
    void* devArgsData = nullptr;    // args device 起始地址 (日志用)
    void* devCclInputPtr = nullptr; // cclInputPtr 字段 device 地址 (D2D ADDR 二级寻址)
    void* devCclOutputPtr = nullptr;
    void* devInputAddr = nullptr;
    void* devOutputAddr = nullptr;
};

// 数据量口径对照 InfoStore GetHcomAlltoallV/VCOpMemSize (host 副本上计算, 图模式 tensor 原地址不可解引用):
//   V:  发送/接收侧末端偏移最大值 max(counts[i] + displs[i])
//   VC: 本 rank 行和 (发送量) / 列和 (接收量), rank 取通信域实际值 (对齐 InfoStore 运行时口径)
// 容量守卫对照 InfoStore RefreshInputAddr (inputMemSize > CCLbufferSize → HCCL_E_MEMORY), 单次整块不分批
static ge::graphStatus CalcCommMemSize(const hccl::HcclOpState& st, HcomAllToAllKind kind, HcomAllToAllRefreshCtx& ctx)
{
    uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    if (kind == HcomAllToAllKind::kAllToAllV) {
        for (uint32_t i = 0; i < st.rankSize; i++) {
            ctx.sendCount = std::max(ctx.sendCount, st.sendCounts[i] + st.sendDispls[i]);
            ctx.recvCount = std::max(ctx.recvCount, st.recvCounts[i] + st.recvDispls[i]);
        }
    } else { // kAllToAllVC
        uint64_t rankSize = static_cast<uint64_t>(st.rankSize);
        for (uint32_t i = 0; i < st.rankSize; i++) {
            ctx.sendCount += st.sendCountMatrix[static_cast<uint64_t>(st.rankId) * rankSize + i];
            ctx.recvCount += st.sendCountMatrix[static_cast<uint64_t>(st.rankId) + rankSize * i];
        }
    }
    ctx.inputMemSize = ctx.sendCount * unitSize;
    ctx.outputMemSize = ctx.recvCount * unitSize;

    if (ctx.inputMemSize > ctx.commInputSize) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][LaunchRefresh] inputMemSize=%lu (sendCount=%lu) exceeds in-CCLbuf size=%lu: single "
            "shot only, batching not supported for AllToAllV/VC refresh path.",
            ctx.inputMemSize, ctx.sendCount, ctx.commInputSize);
        return ge::GRAPH_FAILED;
    }
    if (ctx.outputMemSize > ctx.commOutputSize) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][LaunchRefresh] outputMemSize=%lu (recvCount=%lu) exceeds out-CCLbuf size=%lu: single "
            "shot only, batching not supported for AllToAllV/VC refresh path.",
            ctx.outputMemSize, ctx.recvCount, ctx.commOutputSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// args struct 分配与字段 device 地址换算: MallocReadOnlyDevArgs 分配 host+device 双份, GE 统一 H2D 同步
// (对照 InfoStore 的 ①④ 次拷贝); devArgs->args_data 是 device 侧 args 起始地址,
// D2D ADDR task 用字段 device 地址二级寻址
static ge::graphStatus InitRefreshArgs(hccl::HcclOpState& st, HcomAllToAllRefreshCtx& ctx)
{
    HcomAllToAllArgs args
        = {ctx.commInputPtr, ctx.commOutputPtr, st.inputPtrs[HcomAllToAllVOp::INPUT_IDX_SEND_DATA],
           st.outputPtrs[HcomAllToAllVOp::OUTPUT_IDX_RECV_DATA]};
    const gert::KernelArgs* devArgs = st.ctx->MallocReadOnlyDevArgs(&args, sizeof(args));
    if (devArgs == nullptr || devArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomAllToAllVOp][LaunchRefresh] MallocReadOnlyDevArgs failed.");
        return ge::GRAPH_FAILED;
    }

    char* devArgsBase = static_cast<char*>(devArgs->args_data);
    ctx.devArgsData = devArgs->args_data;
    ctx.devCclInputPtr = devArgsBase + offsetof(HcomAllToAllArgs, cclInputPtr);
    ctx.devCclOutputPtr = devArgsBase + offsetof(HcomAllToAllArgs, cclOutputPtr);
    ctx.devInputAddr = devArgsBase + offsetof(HcomAllToAllArgs, inputAddr);
    ctx.devOutputAddr = devArgsBase + offsetof(HcomAllToAllArgs, outputAddr);
    return ge::GRAPH_SUCCESS;
}

// ③ GraphMode 通信: send/recv 用 CCL buf 地址 (对照 InfoStore GetCommCCLBuf), counts/矩阵传 host 副本;
// 单 tag 无 "_looptime_" 后缀 (AllToAll 系不在分批覆盖表)
static HcclResult LaunchRefreshComm(hccl::HcclOpState& st, HcomAllToAllKind kind, const HcomAllToAllRefreshCtx& ctx)
{
    if (kind == HcomAllToAllKind::kAllToAllV) {
        return HcclAlltoAllVGraphMode(
            ctx.commInputPtr, st.sendCounts.data(), st.sendDispls.data(), st.dataType, ctx.commOutputPtr,
            st.recvCounts.data(), st.recvDispls.data(), st.dataType, st.group, static_cast<aclrtStream>(st.stream),
            st.tags[0].c_str(), st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(),
            st.scratchMem, st.scratchMemSize);
    }
    return HcclAlltoAllVCGraphMode(
        ctx.commInputPtr, st.sendCountMatrix.data(), st.dataType, ctx.commOutputPtr, st.dataType, st.group,
        static_cast<aclrtStream>(st.stream), st.tags[0].c_str(),
        st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(), st.scratchMem,
        st.scratchMemSize);
}

// D2D ADDR 二级指针拷贝下发 (整块偏移 0): ② send_data→CCL input / ⑤ CCL output→recv_data 两处复用
static ge::graphStatus LaunchD2DAddrCopy(
    hccl::HcclOpState& st, void* devDst, uint64_t dstMax, void* devSrc, uint64_t srcMax, const char* errTag)
{
    HcclResult hrtRet = hccl::HcclRtMemAddrAsyncCopyWithOffset(
        devDst, dstMax, 0, devSrc, srcMax, 0, static_cast<aclrtStream>(st.stream));
    if (hrtRet != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomAllToAllVOp][LaunchRefresh] D2D ADDR %s failed, ret=%d.", errTag, static_cast<int>(hrtRet));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllToAllVOp::LaunchRefresh(hccl::HcclOpState& st)
{
    // V/VC 地址刷新路径 (unknownShape 场景): args struct + ArgsUpdater, GE 统一 H2D —
    // 结构对照 AllReduce LaunchRefresh, 差异为单次整块执行 (无多迭代 loop):
    // 对照 InfoStore HcomAlltoAllV/VCOpKernel 的 RefreshInputAddr/RefreshOutputAddr 整块版本 —
    // 数据量超 CCL buf 直接报错, 不按数据量分批 (变长描述的 counts/displs 随数据形态变化, 分批切分无意义)
    // 执行序: ② D2D ADDR (send_data → CCL input buf, 偏移 0 整块) → ③ 通信 (CCL buf 上执行, 对照
    // InfoStore unknownShape 分支 GetCommCCLBuf 替换 send/recv buf) → ⑤ D2D ADDR (CCL output buf → recv_data)

    if (st.tags.empty()) {
        HCCL_ERROR("[HcomAllToAllVOp][LaunchRefresh] tags is empty.");
        return ge::GRAPH_FAILED;
    }

    HcomAllToAllRefreshCtx ctx;
    HCCL_GE_CHK_RET(GetOrCreateInCCLbuf(st.group, ctx.commInputPtr, ctx.commInputSize));
    HCCL_GE_CHK_RET(GetOrCreateOutCCLbuf(st.group, ctx.commOutputPtr, ctx.commOutputSize));
    HCCL_GE_CHK_RET(CalcCommMemSize(st, kind_, ctx));
    HCCL_GE_CHK_RET(InitRefreshArgs(st, ctx));

    // ② D2D ADDR: send_data → CCL input buf (src=devInputAddr 二级指针, dst=devCclInputPtr 二级指针, 整块偏移 0)
    HCCL_GE_CHK_RET(LaunchD2DAddrCopy(
        st, ctx.devCclInputPtr, ctx.commInputSize, ctx.devInputAddr, ctx.inputMemSize, "send_data->CCL"));

    HcclResult ret = LaunchRefreshComm(st, kind_, ctx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][LaunchRefresh] GraphMode entry failed, kind=%d ret=%d.", static_cast<int>(kind_),
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    // ⑤ D2D ADDR: CCL output buf → recv_data (dst=devOutputAddr 二级指针, src=devCclOutputPtr 二级指针, 整块偏移 0)
    HCCL_GE_CHK_RET(LaunchD2DAddrCopy(
        st, ctx.devOutputAddr, ctx.outputMemSize, ctx.devCclOutputPtr, ctx.outputMemSize, "CCL->recv_data"));

    HCCL_INFO(
        "[HcomAllToAllVOp][LaunchRefresh] devArgs=%p kind=%d sendCount=%lu recvCount=%lu cclInput=%p cclOutput=%p "
        "inputAddr=%p outputAddr=%p tag=%s.",
        ctx.devArgsData, static_cast<int>(kind_), ctx.sendCount, ctx.recvCount, ctx.commInputPtr, ctx.commOutputPtr,
        st.inputPtrs[HcomAllToAllVOp::INPUT_IDX_SEND_DATA], st.outputPtrs[HcomAllToAllVOp::OUTPUT_IDX_RECV_DATA],
        st.tags[0].c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllToAllVOp::UpdateHostArgs(gert::UpdateArgsContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomAllToAllVOp][UpdateHostArgs] update args context is null.");
        return ge::GRAPH_FAILED;
    }
    // GE 在 sink 模型每次执行前调 (仅 MallocReadOnlyDevArgs 分配过 args 的算子进回调表 —
    // AllToAll 直传形态与 V/VC 直传形态均未分配 args, 不会被回调到此):
    // 刷新 send_data/recv_data 两个数据地址; counts/矩阵描述张量为图中常量, 不刷新 (对齐原路径)
    const gert::Tensor* inputTensor = ctx->GetInputTensor(INPUT_IDX_SEND_DATA);
    const gert::Tensor* outputTensor = ctx->GetOutputTensor(OUTPUT_IDX_RECV_DATA);
    if (inputTensor == nullptr || outputTensor == nullptr) {
        HCCL_ERROR("[HcomAllToAllVOp][UpdateHostArgs] input=%p output=%p.", inputTensor, outputTensor);
        return ge::GRAPH_FAILED;
    }

    // [DFX] 回调入口: 本次从 ctx 获取到的输入输出地址 (函数尾有 旧→新 对照)
    HCCL_INFO(
        "[HcomAllToAllVOp][UpdateHostArgs] enter input=%p output=%p.", inputTensor->GetAddr(), outputTensor->GetAddr());

    const gert::KernelArgs* hostArgs = ctx->GetKernelArgs(gert::Placement::kPlacementHost, 0);
    if (hostArgs == nullptr || hostArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomAllToAllVOp][UpdateHostArgs] hostArgs is null.");
        return ge::GRAPH_FAILED;
    }

    auto* args = static_cast<HcomAllToAllArgs*>(hostArgs->args_data);
    void* prevIn = args->inputAddr; // Execute 阶段填入或上次刷新的地址
    void* prevOut = args->outputAddr;
    args->inputAddr = const_cast<void*>(inputTensor->GetAddr());
    args->outputAddr = const_cast<void*>(outputTensor->GetAddr());

    // DFX: 地址刷新链路对照 — 旧值 = Execute 阶段获取的地址(或上次刷新值), 新值 = 本次回调从 ctx 获取的地址
    HCCL_INFO(
        "[HcomAllToAllVOp][UpdateHostArgs] input %p -> %p, output %p -> %p.", prevIn, args->inputAddr, prevOut,
        args->outputAddr);
    return ge::GRAPH_SUCCESS;
}

// ---- shape/type 推导 ----

// V 形态 shape 推导 (proto HcomAllToAllVInferShapeV2): input[3] recv_counts + input[4] recv_displacements,
// const counts 可读时 [max(rdispls[i]+recvCounts[i])], 否则 UNKNOWN_DIM_NUM
ge::graphStatus HcomAllToAllVOp::InferVRecvShape(gert::InferShapeContext* ctx, gert::Shape* outputShape)
{
    std::vector<uint64_t> recvCounts;
    std::vector<uint64_t> rdispls;
    if (hccl::ReadHostU64Vector(ctx->GetInputTensor(HcomAllToAllVOp::INPUT_IDX_RECV_COUNTS), recvCounts)
            != ge::GRAPH_SUCCESS
        || hccl::ReadHostU64Vector(ctx->GetInputTensor(HcomAllToAllVOp::INPUT_IDX_RECV_DISPLS), rdispls)
               != ge::GRAPH_SUCCESS
        || recvCounts.size() != rdispls.size()) {
        outputShape->SetDimNum(1);
        outputShape->SetDim(0, ge::UNKNOWN_DIM_NUM);
        return ge::GRAPH_SUCCESS;
    }
    int64_t recvShape = -1;
    for (size_t i = 0; i < recvCounts.size(); i++) {
        int64_t end = static_cast<int64_t>(rdispls[i]) + static_cast<int64_t>(recvCounts[i]);
        if (recvShape < end) {
            recvShape = end;
        }
    }
    outputShape->SetDimNum(1);
    outputShape->SetDim(0, recvShape);
    return ge::GRAPH_SUCCESS;
}

// VC 形态 shape 推导 (proto HcomAllToAllVCInferShapeV2): rank attr 校验 + 矩阵列和,
// const 矩阵可读时 [Σ matrix[rank+i*rankSize]], 否则 UNKNOWN_DIM_NUM
ge::graphStatus HcomAllToAllVOp::InferVCRecvShape(gert::InferShapeContext* ctx, gert::Shape* outputShape)
{
    const gert::RuntimeAttrs* attrs = ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomAllToAllVOp][InferShape] attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const int64_t* rankAttr = attrs->GetInt(HcomAllToAllVOp::ATTR_IDX_RANK_VC);
    if (rankAttr == nullptr) {
        HCCL_ERROR("[HcomAllToAllVOp][InferShape] rank attr is null.");
        return ge::GRAPH_FAILED;
    }
    std::vector<uint64_t> sendCountMatrix;
    if (hccl::ReadHostU64Vector(ctx->GetInputTensor(HcomAllToAllVOp::INPUT_IDX_SEND_COUNT_MATRIX), sendCountMatrix)
        != ge::GRAPH_SUCCESS) {
        outputShape->SetDimNum(1);
        outputShape->SetDim(0, ge::UNKNOWN_DIM_NUM);
        return ge::GRAPH_SUCCESS;
    }
    int64_t rankSize = static_cast<int64_t>(std::sqrt(static_cast<double>(sendCountMatrix.size())));
    if (rankSize <= 0) {
        HCCL_ERROR("[HcomAllToAllVOp][InferShape] rankSize is illegal, expected: > 0, actual: %ld.", rankSize);
        return ge::GRAPH_FAILED;
    }
    if (*rankAttr < 0 || *rankAttr >= rankSize) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][InferShape] attr rank: %ld is illegal, expected: [0 ~ %ld].", *rankAttr, rankSize - 1);
        return ge::GRAPH_FAILED;
    }
    int64_t recvCount = 0;
    for (int64_t i = 0; i < rankSize; i++) {
        const uint64_t cnt = sendCountMatrix[*rankAttr + i * rankSize];
        // 受检窄化+累加 (与执行侧 CalcVCRecvDim 同款): 超值域/溢出报错, 不静默回绕成负维
        if (cnt > static_cast<uint64_t>(INT64_MAX) || recvCount > INT64_MAX - static_cast<int64_t>(cnt)) {
            HCCL_ERROR("[HcomAllToAllVOp][InferShape] recvCount accumulate overflow, cnt=%lu.", cnt);
            return ge::GRAPH_FAILED;
        }
        recvCount += static_cast<int64_t>(cnt);
    }
    outputShape->SetDimNum(1);
    outputShape->SetDim(0, recvCount);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllToAllVOp::InferShape(gert::InferShapeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomAllToAllVOp][InferShape] infer shape context is null.");
        return ge::GRAPH_FAILED;
    }
    // IR 单 output, shape 按 kind (移植 all_to_all_v_proto.cc 三段 InferShapeV2):
    //   AllToAll:   y 镜像 x
    //   AllToAllV:  InferVRecvShape
    //   AllToAllVC: InferVCRecvShape
    size_t inputNum = ctx->GetComputeNodeInputNum();
    size_t outputNum = ctx->GetComputeNodeOutputNum();
    size_t expectInputNum = GetExpectInputNum();
    if (inputNum != expectInputNum || outputNum != 1) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][InferShape] inputNum=%zu (expect %zu) outputNum=%zu (expect 1).", inputNum,
            expectInputNum, outputNum);
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* inputShape = ctx->GetInputShape(INPUT_IDX_SEND_DATA);
    if (inputShape == nullptr) {
        HCCL_ERROR("[HcomAllToAllVOp][InferShape] input shape is null.");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outputShape = ctx->GetOutputShape(OUTPUT_IDX_RECV_DATA);
    if (outputShape == nullptr) {
        HCCL_ERROR("[HcomAllToAllVOp][InferShape] output shape is null.");
        return ge::GRAPH_FAILED;
    }

    if (kind_ == HcomAllToAllKind::kAllToAll) {
        *outputShape = *inputShape;
        return ge::GRAPH_SUCCESS;
    }
    if (kind_ == HcomAllToAllKind::kAllToAllV) {
        return InferVRecvShape(ctx, outputShape);
    }
    return InferVCRecvShape(ctx, outputShape);
}

ge::graphStatus HcomAllToAllVOp::InferDataType(gert::InferDataTypeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomAllToAllVOp][InferDataType] infer datatype context is null.");
        return ge::GRAPH_FAILED;
    }
    size_t inputNum = ctx->GetComputeNodeInputNum();
    size_t outputNum = ctx->GetComputeNodeOutputNum();
    size_t expectInputNum = GetExpectInputNum();
    if (inputNum != expectInputNum || outputNum != 1) {
        HCCL_ERROR(
            "[HcomAllToAllVOp][InferDataType] inputNum=%zu (expect %zu) outputNum=%zu (expect 1).", inputNum,
            expectInputNum, outputNum);
        return ge::GRAPH_FAILED;
    }

    ge::DataType dataType = ctx->GetInputDataType(INPUT_IDX_SEND_DATA);
    if (dataType == ge::DT_UNDEFINED) {
        HCCL_ERROR("[HcomAllToAllVOp][InferDataType] input dtype is undefined.");
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus ret = ctx->SetOutputDataType(OUTPUT_IDX_RECV_DATA, dataType);
    if (ret != ge::GRAPH_SUCCESS) {
        HCCL_ERROR("[HcomAllToAllVOp][InferDataType] SetOutputDataType failed.");
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

const char* HcomAllToAllVOp::GetOpTypeName() const
{
    return GetOpTypeStr(); // kind → ALLTOALL/ALLTOALLV/ALLTOALLVC
}

static ge::BaseCustomOp* CreateHcomAllToAllOp() { return new HcomAllToAllVOp(HcomAllToAllKind::kAllToAll); }
static ge::BaseCustomOp* CreateHcomAllToAllVOp() { return new HcomAllToAllVOp(HcomAllToAllKind::kAllToAllV); }
static ge::BaseCustomOp* CreateHcomAllToAllVCOp() { return new HcomAllToAllVOp(HcomAllToAllKind::kAllToAllVC); }
// 条件注册: 目标平台 (950/960) 且 GE/RTS 新接口在位才注册接管, 否则回退原 hccl_engine 路径
// (hccl_custom_op.h CreateSocGatedRegister: SoC 判定 + GE/RTS 接口在位判定)
static ge::CustomOpCreatorRegister* g_hcom_alltoall_register
    = hccl::CreateSocGatedRegister(HCCL_KERNEL_OP_TYPE_ALLTOALL.c_str(), CreateHcomAllToAllOp);
static ge::CustomOpCreatorRegister* g_hcom_alltoallv_register
    = hccl::CreateSocGatedRegister(HCCL_KERNEL_OP_TYPE_ALLTOALLV.c_str(), CreateHcomAllToAllVOp);
static ge::CustomOpCreatorRegister* g_hcom_alltoallvc_register
    = hccl::CreateSocGatedRegister(HCCL_KERNEL_OP_TYPE_ALLTOALLVC.c_str(), CreateHcomAllToAllVCOp);
