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
//   HcomGetCommHandleByGroup — 按 group 获取通信域 handle (hccl_custom_op.cc)
//   GetOrCreateInCCLbuf (基类) — get-or-create 取 in-CCL buffer 实际尺寸 (CalcResourcesStatic,
//                                对照 InfoStore GetHcomInCCLbufferSize; HcomGetInCCLbuffer/HcomCreateCommCCLbuffer)
//   HcomCreateCommCCLbuffer  — 创建通信域 CCL buffer (hccl_custom_op.cc GetOrCreateInCCLbuf/OutCCLbuf)
//   HcomGetInCCLbuffer       — 获取输入 CCL buffer (hccl_custom_op.cc GetOrCreateInCCLbuf)
//   HcomGetOutCCLbuffer      — 获取输出 CCL buffer (hccl_custom_op.cc GetOrCreateOutCCLbuf)
//   HcomGenerateCclOpTag     — 生成 op tag: opType_hash(group)_autoIndex (BuildTagVec / LaunchBatch)
//
// 另调用 hccl_comm.h: HcclGetRankSize (CalcResourcesStatic)
// 另使用 hcom.h 常量: HCCL_KERNEL_OP_TYPE_ALLREDUCE, HCCL_CMD_ALLREDUCE, rtStream_t,
//                    CCL_OP_TAG_MAX_LEN, CommNumHcom::COMM_VALUE_DEFAULT
// 公共工具 (CalcLoopMaxTime/BuildTagVec/CalcOnlineResources/CrackCleaner) 见 hccl_op_utils.h/.cc

#include "hcom_all_reduce_op.h"

#include <cstddef>

#include "acl/acl_rt.h"
#include "hccl.h" // 本仓源头 include/hccl.h (HcclAllReduce 单算子接口), 非 CANN 安装产物
#include "hccl/hccl_comm.h"
#include "hccl/hcom.h"
#include "hccl_op_utils.h"
#include "hccl_rt_utils.h"
#include "log.h"

// 前向声明 OpParamGraphMode, 避免引入内部头文件依赖链
extern "C" {
HcclResult HcclSetAivCoreLimitGraphMode(const char* group, uint32_t aivCoreLimit);
HcclResult HcclAllReduceGraphMode(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclReduceOp op, const char* group,
    aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,
    uint64_t scratchMemSize);
}

// ExtractParams 公共校验: 首 input 判空、attrs 判空、dataType/reduceOp 解析 (静态/动态路径共用)
static ge::graphStatus ExtractCommonParams(
    hccl::HcclOpState& st, const char* fnName, const gert::RuntimeAttrs*& attrs, const char*& reduction,
    const char*& group)
{
    const gert::Tensor* firstInput = st.ctx->GetInputTensor(HcomAllReduceOp::INPUT_INDEX);
    if (firstInput == nullptr) {
        HCCL_ERROR("[HcomAllReduceOp][%s] input tensor is null.", fnName);
        return ge::GRAPH_FAILED;
    }
    attrs = st.ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomAllReduceOp][%s] attrs is null.", fnName);
        return ge::GRAPH_FAILED;
    }
    reduction = attrs->GetStr(HcomAllReduceOp::ATTR_REDUCTION);
    group = attrs->GetStr(HcomAllReduceOp::ATTR_GROUP);
    if (reduction == nullptr) {
        HCCL_ERROR("[HcomAllReduceOp][%s] reduction attr is null.", fnName);
        return ge::GRAPH_FAILED;
    }
    if (group == nullptr) {
        HCCL_ERROR("[HcomAllReduceOp][%s] group attr is null.", fnName);
        return ge::GRAPH_FAILED;
    }

    st.dataType = hccl::GeDataTypeToHccl(firstInput->GetDataType());
    if (st.dataType == HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR(
            "[HcomAllReduceOp][%s] unsupported data type %d.", fnName, static_cast<int>(firstInput->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    st.reduceOp = hccl::StringToReduceOp(reduction);
    if (st.reduceOp == HCCL_REDUCE_RESERVED) {
        HCCL_ERROR("[HcomAllReduceOp][%s] unsupported reduction '%s'.", fnName, reduction ? reduction : "(null)");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 静态路径 input 遍历: 每 input 分配输出张量, 按 512B 对齐累加总 count (InfoStore CalcCountForAlignedOp
// 口径: GetSize() 向上对齐累加再除以 dataTypeSize), 并记录融合缝隙 (crack, Builder GetCrackParamsInfo 口径)
static ge::graphStatus AllocOutputsAndCalcCracks(hccl::HcclOpState& st, size_t& inputCount)
{
    constexpr uint64_t ALIGNED_SIZE = 512;
    uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    uint64_t totalAlignedSize = 0;
    inputCount = 0;

    for (size_t i = 0;; i++) {
        const gert::Tensor* inTensor = st.ctx->GetInputTensor(i);
        if (inTensor == nullptr) {
            break;
        }
        inputCount++;

        gert::Tensor* outTensor
            = st.ctx->MallocOutputTensor(i, inTensor->GetShape(), inTensor->GetFormat(), inTensor->GetDataType());
        if (outTensor == nullptr) {
            HCCL_ERROR("[HcomAllReduceOp][ExtractParamsStatic] failed to allocate output tensor %zu.", i);
            return ge::GRAPH_FAILED;
        }

        // crackOffset = 当前 tensor 在融合 buffer 中的偏移 + 实际数据大小
        // crackSize = 512B 对齐后大小 - 实际数据大小 (padding 缝隙)
        uint64_t tensorSize = static_cast<uint64_t>(inTensor->GetSize());
        uint64_t alignedSize = (tensorSize + ALIGNED_SIZE - 1) / ALIGNED_SIZE * ALIGNED_SIZE;
        if (alignedSize > tensorSize) {
            st.crackOffsets.push_back(totalAlignedSize + tensorSize);
            st.crackSizes.push_back(alignedSize - tensorSize);
        }

        totalAlignedSize += alignedSize;
        st.inputPtrs.push_back(const_cast<void*>(inTensor->GetAddr()));
        st.outputPtrs.push_back(outTensor->GetAddr());

        HCCL_INFO(
            "[HcomAllReduceOp][ExtractParamsStatic] input[%zu] addr=%p output addr=%p tensorSize=%lu(B) "
            "alignedSize=%lu(B).",
            i, inTensor->GetAddr(), outTensor->GetAddr(), tensorSize, alignedSize);
    }

    if (inputCount == 0) {
        HCCL_ERROR("[HcomAllReduceOp][ExtractParamsStatic] no input tensors.");
        return ge::GRAPH_FAILED;
    }
    st.count = totalAlignedSize / unitSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::ExtractParamsStatic(hccl::HcclOpState& st)
{
    const gert::RuntimeAttrs* attrs = nullptr;
    const char* reduction = nullptr;
    const char* group = nullptr;
    HCCL_GE_CHK_RET(ExtractCommonParams(st, "ExtractParamsStatic", attrs, reduction, group));

    size_t inputCount = 0;
    HCCL_GE_CHK_RET(AllocOutputsAndCalcCracks(st, inputCount));

    st.group = group;
    st.stream = st.ctx->GetStream();

    const int64_t* fusion = attrs->GetInt(ATTR_FUSION);
    const int64_t* fusionId = attrs->GetInt(ATTR_FUSION_ID);

    HCCL_INFO(
        "[HcomAllReduceOp][ExtractParamsStatic] inputCount=%zu input[0]=%p output[0]=%p count=%lu dataType=%d "
        "reduceOp=%d "
        "group=%s stream=%p fusion=%ld fusionId=%ld.",
        inputCount, st.inputPtrs[0], st.outputPtrs[0], st.count, st.dataType, st.reduceOp,
        st.group ? st.group : "(null)", st.stream, fusion ? *fusion : -1, fusionId ? *fusionId : -1);

    return ge::GRAPH_SUCCESS;
}

// 动态路径 input 遍历: 每 input 分配输出张量并取独立 count (不对齐, lowering GetCountByShape 口径),
// 校验 input/output shape 一致后存入 inputCounts
static ge::graphStatus AllocOutputsAndCalcCounts(hccl::HcclOpState& st, size_t& inputCount)
{
    inputCount = 0;
    for (size_t i = 0;; i++) {
        const gert::Tensor* inTensor = st.ctx->GetInputTensor(i);
        if (inTensor == nullptr) {
            break;
        }
        inputCount++;

        gert::Tensor* outTensor
            = st.ctx->MallocOutputTensor(i, inTensor->GetShape(), inTensor->GetFormat(), inTensor->GetDataType());
        if (outTensor == nullptr) {
            HCCL_ERROR("[HcomAllReduceOp][ExtractParamsDynamic] failed to allocate output tensor %zu.", i);
            return ge::GRAPH_FAILED;
        }

        uint64_t count = static_cast<uint64_t>(inTensor->GetShapeSize());
        uint64_t outCount = static_cast<uint64_t>(outTensor->GetShapeSize());
        if (count != outCount) {
            HCCL_ERROR(
                "[HcomAllReduceOp][ExtractParamsDynamic] input[%zu] count=%lu != output count=%lu.", i, count,
                outCount);
            return ge::GRAPH_FAILED;
        }
        st.inputCounts.push_back(count);
        st.count += count;
        st.inputPtrs.push_back(const_cast<void*>(inTensor->GetAddr()));
        st.outputPtrs.push_back(outTensor->GetAddr());

        HCCL_INFO(
            "[HcomAllReduceOp][ExtractParamsDynamic] input[%zu] addr=%p output[%zu] addr=%p count=%lu.", i,
            inTensor->GetAddr(), i, outTensor->GetAddr(), count);
    }

    if (inputCount == 0) {
        HCCL_ERROR("[HcomAllReduceOp][ExtractParamsDynamic] no input tensors.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::ExtractParamsDynamic(hccl::HcclOpState& st)
{
    // 动态路径 (lowering-style): 每 input 独立 count (不做 512B 对齐)，不计算 crack
    // 对应 lowering GetCountByShape: count = shape.GetShapeSize()
    const gert::RuntimeAttrs* attrs = nullptr;
    const char* reduction = nullptr;
    const char* group = nullptr;
    HCCL_GE_CHK_RET(ExtractCommonParams(st, "ExtractParamsDynamic", attrs, reduction, group));

    size_t inputCount = 0;
    HCCL_GE_CHK_RET(AllocOutputsAndCalcCounts(st, inputCount));

    st.group = group;
    st.stream = st.ctx->GetStream();

    HCCL_INFO(
        "[HcomAllReduceOp][ExtractParamsDynamic] inputCount=%zu totalCount=%lu dataType=%d reduceOp=%d "
        "group=%s stream=%p.",
        inputCount, st.count, st.dataType, st.reduceOp, st.group ? st.group : "(null)", st.stream);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::CalcResourcesStatic(hccl::HcclOpState& st)
{
    uint32_t rankSize = 0;
    HcclResult ret = HcclGetRankSize(st.comm, &rankSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomAllReduceOp][CalcResourcesStatic] HcclGetRankSize failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    // st.cclBuffSize: 实际 in-CCL buffer 尺寸 — 仅 Refresh 路径消费 (GenerateTagVec 的 loopMaxTime 与
    // LaunchRefresh 的 maxCountPerLoop, 两处均门控 needRefresh); Direct 直传用户 buffer 不经 CCL 中转,
    // 无需创建 (旧路径无 Direct 分支恒走中转循环, 拆分后收敛到 Refresh)。get-or-create 语义对照
    // InfoStore GetHcomInCCLbufferSize (查→建→重查)
    if (st.needRefresh) {
        void* cclBufPtr = nullptr;
        HCCL_GE_CHK_RET(GetOrCreateInCCLbuf(st.group, cclBufPtr, st.cclBuffSize));
    }

    // 在线计算输入: 环境推导 available size — 对照 InfoStore GetCCLBufferAvailableSize
    // (kernel_builder 与 graph_optimizer 的 SetOpParam 组装同源), 与实际 buffer 尺寸是两个来源
    uint64_t cclBufferAvailSize = 0;
    HCCL_GE_CHK_RET(hccl::GetCCLBufferAvailableSize(cclBufferAvailSize));

    return hccl::CalcOnlineResources(
        st, rankSize, cclBufferAvailSize, HCCL_KERNEL_OP_TYPE_ALLREDUCE, HCCL_CMD_ALLREDUCE);
}

ge::graphStatus HcomAllReduceOp::GenerateTagVec(hccl::HcclOpState& st)
{
    // 对应 InfoStore GenerateOpTagFromTaskInfo: needRefresh 时算 loopMaxTime
    // loopMaxTime 与 LaunchRefresh 循环同源推导 (公共 hccl::CalcLoopMaxTime): maxCountPerLoop =
    // st.cclBuffSize / unitSize (单一来源), loopMaxTime = ceil(count / maxCountPerLoop),
    // 保证 tags.size() 恒等于实际迭代数。
    // 旧公式 ceil(count*unitSize / cclBuffSize) 在 cclBuffSize 非 unitSize 整除时偏小 (tag 不足), 已弃用
    uint32_t loopMaxTime = 0;
    if (st.needRefresh) {
        uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
        // AllReduce 组 1 公式不消费 rankSize, 惯例传 1; 错误场景 (unitSize==0 / buf 装不下)
        // 由 CalcLoopMaxTime 内部报错并返回 GRAPH_FAILED
        HCCL_GE_CHK_RET(
            hccl::CalcLoopMaxTime(HCCL_KERNEL_OP_TYPE_ALLREDUCE, st.count, unitSize, st.cclBuffSize, 1, loopMaxTime));
    }

    // 对应 InfoStore GetTagVectorInfo: loopMaxTime==0 → {tag}; loopMaxTime>0 → {tag + "_looptime_" + i}
    return hccl::BuildTagVec(HCCL_KERNEL_OP_TYPE_ALLREDUCE, st.group, loopMaxTime, st.tags);
}

ge::graphStatus HcomAllReduceOp::LaunchHcclOpStatic(hccl::HcclOpState& st)
{
    // 静态路径 (InfoStore-style): needRefresh 判定已收编基类 Execute, 此处只消费 st.needRefresh
    if (st.scratchMemSize > 0) {
        st.scratchMem = st.ctx->MallocWorkSpace(st.scratchMemSize);
        if (st.scratchMem == nullptr) {
            HCCL_ERROR("[HcomAllReduceOp][LaunchHcclOpStatic] MallocWorkSpace failed, size=%lu.", st.scratchMemSize);
            return ge::GRAPH_FAILED;
        }
    }

    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllReduceOp][LaunchHcclOpStatic] HcclSetAivCoreLimitGraphMode failed, ret=%d.",
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

ge::graphStatus HcomAllReduceOp::LaunchHcclOpDynamic(hccl::HcclOpState& st)
{
    // 动态路径: 多 input 按各自独立 count, 经 CCL buffer 分段中转执行;
    // AIV 算力上限用 GraphMode 版 HcclSetAivCoreLimitGraphMode (对照 lowering V2 kernel)
    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllReduceOp][LaunchHcclOpDynamic] HcclSetAivCoreLimitGraphMode failed, ret=%d.",
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    return LaunchBatchDynamic(st);
}

ge::graphStatus HcomAllReduceOp::LaunchDirect(hccl::HcclOpState& st)
{
    HCCL_INFO(
        "[HcomAllReduceOp][LaunchDirect] input=%p output=%p count=%lu crackNum=%zu (non-refresh, user buffers "
        "directly).",
        st.inputPtrs[0], st.outputPtrs[0], st.count, st.crackOffsets.size());

    // 直连无 CCL buf 中转, 不做超限拦截 — 对照 InfoStore direct 分支 (INFO.cc) 同样无
    // CheckHcomOpMemSize (该检查仅挂 Loop 路径入口), 任意 size 由 HcclAllReduceGraphMode 底层处理

    // Direct 路径: crack 地址 = 用户 buffer 基址 + crackOffset（inputOffset=0）
    // 对应 InfoStore GetCrackParamsInfoFromTaskInfo: crackAddr[i] = crackOffset[i] - inputOffset + inputAddr
    std::vector<void*> crackAddrs;
    std::vector<uint64_t> crackSizes;
    for (size_t i = 0; i < st.crackOffsets.size(); i++) {
        crackAddrs.push_back(static_cast<char*>(st.inputPtrs[0]) + st.crackOffsets[i]);
        crackSizes.push_back(st.crackSizes[i]);
    }
    HCCL_GE_CHK_RET(crackCleaner_.Clean(st, crackAddrs, crackSizes));

    HcclResult ret = HcclAllReduceGraphMode(
        st.inputPtrs[0], st.outputPtrs[0], st.count, st.dataType, st.reduceOp, st.group,
        static_cast<aclrtStream>(st.stream), st.tags[0].c_str(),
        st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(), st.scratchMem,
        st.scratchMemSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomAllReduceOp][LaunchDirect] HcclAllReduceGraphMode failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus
HcomAllReduceOp::CalcBatchSlices(const hccl::HcclOpState& st, uint64_t maxCommCount, std::vector<uint32_t>& sliceIdxs)
{
    // 对应 lowering HcomGetBatchAllReduceInfo: 按 CCL buffer 容量将 inputs 分段
    uint32_t sliceIdx = 0;
    uint32_t sliceSum = 0;
    uint64_t commCount = 0;

    for (size_t i = 0; i < st.inputCounts.size(); i++) {
        if (st.inputCounts[i] > maxCommCount) {
            HCCL_ERROR(
                "[HcomAllReduceOp][CalcBatchSlices] input[%zu] count %lu exceeds CCL buffer capacity %lu.", i,
                st.inputCounts[i], maxCommCount);
            return ge::GRAPH_FAILED;
        }
        commCount += st.inputCounts[i];
        if (commCount <= maxCommCount) {
            sliceIdx++;
        } else {
            if (sliceIdx == 0) {
                sliceIdx++;
                sliceIdxs.push_back(sliceIdx);
                sliceSum += sliceIdx;
                commCount = 0;
                sliceIdx = 0;
            } else {
                sliceIdxs.push_back(sliceIdx);
                sliceSum += sliceIdx;
                commCount = st.inputCounts[i];
                sliceIdx = 1;
            }
        }
    }
    if (sliceSum < st.inputCounts.size()) {
        sliceIdxs.push_back(static_cast<uint32_t>(st.inputCounts.size() - sliceSum));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::CopyInputsToCCLbuff(
    const hccl::HcclOpState& st, uint32_t inputOffset, uint32_t inputsNum, void* cclBuff, uint64_t cclBuffSize,
    uint64_t& commCount)
{
    // 逐 input 异步拷贝用户数据入 CCL buffer (D2D, dst 按序前移), 累计本段通信元素数 commCount
    uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    uint8_t* dstBuff = static_cast<uint8_t*>(cclBuff);
    uint64_t maxDstSize = cclBuffSize;
    commCount = 0;

    for (uint32_t j = 0; j < inputsNum; j++) {
        uint32_t inputIdx = inputOffset + j;
        uint64_t copySize = st.inputCounts[inputIdx] * unitSize;
        aclError aclRet = aclrtMemcpyAsync(
            dstBuff, maxDstSize, st.inputPtrs[inputIdx], copySize, ACL_MEMCPY_DEVICE_TO_DEVICE,
            static_cast<aclrtStream>(st.stream));
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR(
                "[HcomAllReduceOp][CopyInputsToCCLbuff] aclrtMemcpyAsync input[%u]->CCL failed, ret=%d.", inputIdx,
                static_cast<int>(aclRet));
            return ge::GRAPH_FAILED;
        }
        dstBuff += copySize;
        maxDstSize -= copySize;
        commCount += st.inputCounts[inputIdx];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::CopyCCLbuffToOutputs(
    const hccl::HcclOpState& st, uint32_t outputOffset, uint32_t outputsNum, void* cclBuff)
{
    // 逐 output 异步拷贝 CCL buffer 数据回用户输出 (D2D, src 按序前移)
    uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    uint8_t* srcBuff = static_cast<uint8_t*>(cclBuff);

    for (uint32_t j = 0; j < outputsNum; j++) {
        uint32_t outputIdx = outputOffset + j;
        uint64_t copySize = st.inputCounts[outputIdx] * unitSize;
        aclError aclRet = aclrtMemcpyAsync(
            st.outputPtrs[outputIdx], copySize, srcBuff, copySize, ACL_MEMCPY_DEVICE_TO_DEVICE,
            static_cast<aclrtStream>(st.stream));
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR(
                "[HcomAllReduceOp][CopyCCLbuffToOutputs] aclrtMemcpyAsync CCL->output[%u] failed, ret=%d.", outputIdx,
                static_cast<int>(aclRet));
            return ge::GRAPH_FAILED;
        }
        srcBuff += copySize;
    }
    return ge::GRAPH_SUCCESS;
}

// 逐 slice 执行: 单 input slice 直传用户地址, 多 input slice 经 CCL buf 中转
// (对照 lowering HcomLaunchAllReduceKernelV2 的分段下发)
ge::graphStatus HcomAllReduceOp::ExecuteBatchSlices(
    hccl::HcclOpState& st, const std::vector<uint32_t>& sliceIdxs, void* commInputPtr, uint64_t commInputSize,
    void* commOutputPtr)
{
    HcclResult ret = HCCL_SUCCESS;
    uint32_t inputOffset = 0;
    for (size_t i = 0; i < sliceIdxs.size(); i++) {
        if (sliceIdxs[i] == 1) {
            ret = HcclAllReduce(
                st.inputPtrs[inputOffset], st.outputPtrs[inputOffset], st.inputCounts[inputOffset], st.dataType,
                st.reduceOp, st.comm, static_cast<aclrtStream>(st.stream));
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[HcomAllReduceOp][ExecuteBatchSlices] HcclAllReduce (direct) failed, slice=%zu, ret=%d.", i,
                    static_cast<int>(ret));
                return ge::GRAPH_FAILED;
            }
        } else {
            uint64_t sliceCommCount = 0;
            HCCL_GE_CHK_RET(
                CopyInputsToCCLbuff(st, inputOffset, sliceIdxs[i], commInputPtr, commInputSize, sliceCommCount));
            ret = HcclAllReduce(
                commInputPtr, commOutputPtr, sliceCommCount, st.dataType, st.reduceOp, st.comm,
                static_cast<aclrtStream>(st.stream));
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[HcomAllReduceOp][ExecuteBatchSlices] HcclAllReduce (CCL) failed, slice=%zu, ret=%d.", i,
                    static_cast<int>(ret));
                return ge::GRAPH_FAILED;
            }
            HCCL_GE_CHK_RET(CopyCCLbuffToOutputs(st, inputOffset, sliceIdxs[i], commOutputPtr));
        }
        inputOffset += sliceIdxs[i];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::LaunchBatchDynamic(hccl::HcclOpState& st)
{
    // 动态路径 (lowering-style): batch slicing through CCL buffer
    // 对应 lowering HcomLaunchAllReduceKernelV2
    uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);

    // AIV 算力上限: 对照 lowering HcomLaunchAllReduceKernelV2 — HcceSetAivCoreLimitGraphMode
    // 在分段循环之前下发 (与 Reduce 动态路径同款)
    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllReduceOp][LaunchBatchDynamic] HcclSetAivCoreLimitGraphMode failed, ret=%d.",
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    // Step 1: 获取 CCL buffer
    void* commInputPtr = nullptr;
    uint64_t commInputSize = 0;
    HCCL_GE_CHK_RET(GetOrCreateInCCLbuf(st.group, commInputPtr, commInputSize));
    void* commOutputPtr = nullptr;
    uint64_t commOutputSize = 0;
    HCCL_GE_CHK_RET(GetOrCreateOutCCLbuf(st.group, commOutputPtr, commOutputSize));

    // Step 2: 计算 slice 分段
    uint64_t maxCommCount = commInputSize / unitSize;
    std::vector<uint32_t> sliceIdxs;
    HCCL_GE_CHK_RET(CalcBatchSlices(st, maxCommCount, sliceIdxs));

    HCCL_INFO(
        "[HcomAllReduceOp][LaunchBatchDynamic] inputNum=%zu sliceNum=%zu maxCommCount=%lu totalCount=%lu.",
        st.inputCounts.size(), sliceIdxs.size(), maxCommCount, st.count);

    // Step 3: 逐 slice 执行 (单算子接口 HcclAllReduce 无 tag 参数, 对照 lowering
    // HcomAllReduceKernel: HcceAllReduce → dlsym "HcclAllReduce", 同样无 tag)
    HCCL_GE_CHK_RET(ExecuteBatchSlices(st, sliceIdxs, commInputPtr, commInputSize, commOutputPtr));

    HCCL_INFO(
        "[HcomAllReduceOp][LaunchBatchDynamic] success, inputNum=%zu sliceNum=%zu.", st.inputCounts.size(),
        sliceIdxs.size());
    return ge::GRAPH_SUCCESS;
}

// LaunchRefresh 循环上下文: CCL buf、args 字段 device 地址与 crack 清理器
// (devXxx 为 args struct 各字段的 device 侧地址, D2D ADDR 经其二级指针间接寻址)
struct HcomAllReduceRefreshCtx {
    hccl::CrackCleaner* crackCleaner = nullptr;
    void* commInputPtr = nullptr;
    uint64_t commInputSize = 0;
    void* commOutputPtr = nullptr;
    void* devCclInputPtr = nullptr;
    void* devInputAddr = nullptr;
    void* devCclOutputPtr = nullptr;
    void* devOutputAddr = nullptr;
    void* devArgsData = nullptr;
    uint64_t outputMaxSize = 0;
    uint64_t maxCountPerLoop = 0;
};

// 刷新上下文的 args 部分: maxCountPerLoop 计算、args struct 分配与字段地址换算
// (args 经 MallocReadOnlyDevArgs 分配 host+device 双份, GE 统一 H2D 同步 — 对应 InfoStore
// RefreshInputAddr/RefreshOutputAddr 的 ①④ 次拷贝; CCL buf 由调用方经 GetOrCreate 获取后填入 ctx)
static ge::graphStatus InitRefreshArgs(hccl::HcclOpState& st, HcomAllReduceRefreshCtx& ctx)
{
    uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    // maxCountPerLoop 与 GenerateTagVec 的 loopMaxTime 同源 (st.cclBuffSize 单一来源), tag 数恒等于迭代数
    ctx.maxCountPerLoop = st.cclBuffSize / unitSize;
    if (ctx.maxCountPerLoop == 0) {
        HCCL_ERROR(
            "[HcomAllReduceOp][LaunchRefresh] cclBuffSize=%lu cannot hold one element (unitSize=%lu).", st.cclBuffSize,
            unitSize);
        return ge::GRAPH_FAILED;
    }
    // ⑤ 侧 dstMax = 输出全量 count*unitSize
    ctx.outputMaxSize = st.count * unitSize;

    HcomAllReduceArgs args = {ctx.commInputPtr, ctx.commOutputPtr, st.inputPtrs[0], st.outputPtrs[0]};
    const gert::KernelArgs* devArgs = st.ctx->MallocReadOnlyDevArgs(&args, sizeof(args));
    if (devArgs == nullptr || devArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomAllReduceOp][LaunchRefresh] MallocReadOnlyDevArgs failed.");
        return ge::GRAPH_FAILED;
    }
    // devArgs->args_data 是 device 侧 args 起始地址; 字段偏移: cclInputPtr=0, cclOutputPtr=8, inputAddr=16,
    // outputAddr=24
    char* devArgsBase = static_cast<char*>(devArgs->args_data);
    ctx.devArgsData = devArgs->args_data;
    ctx.devCclInputPtr = devArgsBase + offsetof(HcomAllReduceArgs, cclInputPtr);
    ctx.devCclOutputPtr = devArgsBase + offsetof(HcomAllReduceArgs, cclOutputPtr);
    ctx.devInputAddr = devArgsBase + offsetof(HcomAllReduceArgs, inputAddr);
    ctx.devOutputAddr = devArgsBase + offsetof(HcomAllReduceArgs, outputAddr);
    return ge::GRAPH_SUCCESS;
}

// LaunchRefresh 单迭代游标
struct HcomAllReduceLoopCursor {
    uint64_t loopTime = 0;
    uint64_t inputOffset = 0;
    uint64_t outputOffset = 0;
    uint64_t curCount = 0;
    bool isLastLoop = false; // curCount == 剩余量, 末迭代才做缝隙清理
};

// ② D2D ADDR 带偏移: 用户 input → CCL input buf (srcOffset=inputOffset, dstOffset=0)
//   src = devInputAddr    (二级指针 → 解引用得用户 input 地址)
//   dst = devCclInputPtr  (二级指针 → 解引用得 CCL buf 地址)
static ge::graphStatus
CopyInputToCCL(hccl::HcclOpState& st, const HcomAllReduceRefreshCtx& ctx, const HcomAllReduceLoopCursor& cursor)
{
    const uint64_t curSize = cursor.curCount * hccl::GetHcclDataTypeSize(st.dataType);
    HcclResult hrtRet = hccl::HcclRtMemAddrAsyncCopyWithOffset(
        ctx.devCclInputPtr, ctx.commInputSize, 0, ctx.devInputAddr, curSize, cursor.inputOffset,
        static_cast<aclrtStream>(st.stream));
    if (hrtRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllReduceOp][RefreshOneLoop] D2D ADDR input->CCL failed, loopTime=%lu, ret=%d.", cursor.loopTime,
            static_cast<int>(hrtRet));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 末迭代窗口缝隙清理: RefreshOneLoop 仅末迭代调用 (isLastLoop 门控, 对照 InfoStore
// HcomAllReduceLoop 的 curCount == countLeft), 清与末窗口 [inputOffset, inputOffset + curSize)
// 有交集的 crack, 地址 = commInputPtr + (交集起点 - inputOffset)。
// 对照 InfoStore 差异: 其用末迭代 inputOffset 无过滤平移全部 crack — 非末窗口 crack 平移出
// 负地址, CleanInterMemory 不检查地址直接 D2D 清零 (缓冲区前越界); 本实现的交集过滤跳过
// 非末窗口 crack, 无该问题; 钳位把写入约束在窗口内属防御性 (末窗口缝隙按几何恒不越窗口尾:
// totalSize 为 512 对齐块之和, 末窗口尾恰为 totalSize)。
// 非末窗口缝隙两侧均不清 (其 padding 位结果含垃圾和, 用户不读 padding 位) — 与 InfoStore 一致
static ge::graphStatus
CleanCracksInWindow(hccl::HcclOpState& st, const HcomAllReduceRefreshCtx& ctx, uint64_t inputOffset, uint64_t curSize)
{
    std::vector<void*> crackAddrs;
    std::vector<uint64_t> crackSizes;
    const uint64_t windowEnd = inputOffset + curSize;
    for (size_t i = 0; i < st.crackOffsets.size(); i++) {
        const uint64_t gapStart = st.crackOffsets[i];
        const uint64_t gapEnd = gapStart + st.crackSizes[i];
        if (gapStart >= windowEnd || gapEnd <= inputOffset) {
            continue; // 与本窗口无交集
        }
        const uint64_t cleanStart = gapStart > inputOffset ? gapStart : inputOffset;
        const uint64_t cleanEnd = gapEnd < windowEnd ? gapEnd : windowEnd;
        crackAddrs.push_back(static_cast<char*>(ctx.commInputPtr) + (cleanStart - inputOffset));
        crackSizes.push_back(cleanEnd - cleanStart);
    }
    return ctx.crackCleaner->Clean(st, crackAddrs, crackSizes);
}

// LaunchRefresh 单迭代: ② D2D ADDR 搬入 → 窗口内 crack 交集清理 → ③ 通信 → ⑤ D2D ADDR 回写
static ge::graphStatus
RefreshOneLoop(hccl::HcclOpState& st, const HcomAllReduceRefreshCtx& ctx, HcomAllReduceLoopCursor& cursor)
{
    const uint64_t curSize = cursor.curCount * hccl::GetHcclDataTypeSize(st.dataType);

    // tag 越界守卫, 对照 InfoStore HcomAllReduceLoop
    if (cursor.loopTime >= st.tags.size()) {
        HCCL_ERROR(
            "[HcomAllReduceOp][RefreshOneLoop] tagVec access out-of-bounds, loopTime=%lu tags.size=%zu.",
            cursor.loopTime, st.tags.size());
        return ge::GRAPH_FAILED;
    }

    HCCL_GE_CHK_RET(CopyInputToCCL(st, ctx, cursor));

    // 缝隙清理: 仅末迭代, 窗口内存在 crack 才清
    if (cursor.isLastLoop && !st.crackOffsets.empty()) {
        HCCL_GE_CHK_RET(CleanCracksInWindow(st, ctx, cursor.inputOffset, curSize));
    }

    // ③ HcclAllReduceGraphMode (每迭代独立 tag: _looptime_N)
    HcclResult ret = HcclAllReduceGraphMode(
        ctx.commInputPtr, ctx.commOutputPtr, cursor.curCount, st.dataType, st.reduceOp, st.group,
        static_cast<aclrtStream>(st.stream), st.tags[cursor.loopTime].c_str(),
        st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(), st.scratchMem,
        st.scratchMemSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllReduceOp][RefreshOneLoop] HcclAllReduceGraphMode failed, loopTime=%lu, ret=%d.", cursor.loopTime,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    // ⑤ D2D ADDR 带偏移: CCL output buf → 用户 output (dstOffset=outputOffset, srcOffset=0)
    //   src = devCclOutputPtr  (二级指针 → 解引用得 CCL buf 地址)
    //   dst = devOutputAddr    (二级指针 → 解引用得用户 output 地址)
    HcclResult hrtRet = hccl::HcclRtMemAddrAsyncCopyWithOffset(
        ctx.devOutputAddr, ctx.outputMaxSize, cursor.outputOffset, ctx.devCclOutputPtr, curSize, 0,
        static_cast<aclrtStream>(st.stream));
    if (hrtRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomAllReduceOp][RefreshOneLoop] D2D ADDR CCL->output failed, loopTime=%lu, ret=%d.", cursor.loopTime,
            static_cast<int>(hrtRet));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::LaunchRefresh(hccl::HcclOpState& st)
{
    // 地址刷新路径: args struct + ArgsUpdater, GE 统一 H2D
    // 多迭代 loop: 结构对照 InfoStore HcomAllReduceLoop, 超限数据分迭代执行
    // 单迭代 (数据不超 CCL buf) 时 inputOffset=0、curSize=全量, ②③⑤ 参数与改造前逐字节等价 (回归底线)
    HcomAllReduceRefreshCtx ctx;
    ctx.crackCleaner = &crackCleaner_;
    HCCL_GE_CHK_RET(GetOrCreateInCCLbuf(st.group, ctx.commInputPtr, ctx.commInputSize));
    // buffer 尺寸用局部变量接, 不写入 ctx.outputMaxSize (其语义 = 用户输出全量, 由
    // InitRefreshArgs 赋值, 对照 InfoStore 同名变量的单一用途)
    uint64_t outCclBufCapacity = 0;
    HCCL_GE_CHK_RET(GetOrCreateOutCCLbuf(st.group, ctx.commOutputPtr, outCclBufCapacity));
    HCCL_GE_CHK_RET(InitRefreshArgs(st, ctx));

    // 大于 cclbuffer 时 count 取最大支持 count, 否则取剩余 count
    HcomAllReduceLoopCursor cursor;
    for (uint64_t countLeft = st.count; countLeft > 0; countLeft -= cursor.curCount) {
        cursor.curCount = (countLeft > ctx.maxCountPerLoop) ? ctx.maxCountPerLoop : countLeft;
        cursor.isLastLoop = (cursor.curCount == countLeft);
        HCCL_INFO(
            "[HcomAllReduceOp][LaunchRefresh] loopTime=%lu inputOffset=%lu outputOffset=%lu countLeft=%lu "
            "maxCountPerLoop=%lu.",
            cursor.loopTime, cursor.inputOffset, cursor.outputOffset, countLeft, ctx.maxCountPerLoop);

        HCCL_GE_CHK_RET(RefreshOneLoop(st, ctx, cursor));

        const uint64_t curSize = cursor.curCount * hccl::GetHcclDataTypeSize(st.dataType);
        cursor.inputOffset += curSize;
        cursor.outputOffset += curSize;
        cursor.loopTime++;
    }

    HCCL_INFO(
        "[HcomAllReduceOp][LaunchRefresh] devArgs=%p count=%lu maxCountPerLoop=%lu cclInput=%p cclOutput=%p "
        "inputAddr=%p outputAddr=%p.",
        ctx.devArgsData, st.count, ctx.maxCountPerLoop, ctx.commInputPtr, ctx.commOutputPtr, st.inputPtrs[0],
        st.outputPtrs[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::UpdateHostArgs(gert::UpdateArgsContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomAllReduceOp][UpdateHostArgs] update args context is null.");
        return ge::GRAPH_FAILED;
    }
    // GE 在 sink 模型每次执行前调: 刷新 args struct 中的 inputAddr/outputAddr
    // GE 统一 H2D 同步 host → device, D2D ADDR task 自动 dereference 新地址
    const gert::Tensor* inputTensor = ctx->GetInputTensor(INPUT_INDEX);
    const gert::Tensor* outputTensor = ctx->GetOutputTensor(OUTPUT_INDEX);
    if (inputTensor == nullptr || outputTensor == nullptr) {
        HCCL_ERROR("[HcomAllReduceOp][UpdateHostArgs] input=%p output=%p.", inputTensor, outputTensor);
        return ge::GRAPH_FAILED;
    }

    // [DFX] 回调入口: 本次从 ctx 获取到的输入输出地址 (函数尾有 旧→新 对照)
    HCCL_INFO(
        "[HcomAllReduceOp][UpdateHostArgs] enter input=%p output=%p.", inputTensor->GetAddr(), outputTensor->GetAddr());

    const gert::KernelArgs* hostArgs = ctx->GetKernelArgs(gert::Placement::kPlacementHost, 0);
    if (hostArgs == nullptr || hostArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomAllReduceOp][UpdateHostArgs] hostArgs is null.");
        return ge::GRAPH_FAILED;
    }

    auto* args = static_cast<HcomAllReduceArgs*>(hostArgs->args_data);
    void* prevIn = args->inputAddr; // Execute 阶段填入或上次刷新的地址
    void* prevOut = args->outputAddr;
    args->inputAddr = const_cast<void*>(inputTensor->GetAddr());
    args->outputAddr = const_cast<void*>(outputTensor->GetAddr());

    // DFX: 地址刷新链路对照 — 旧值 = Execute 阶段获取的地址(或上次刷新值), 新值 = 本次回调从 ctx 获取的地址
    HCCL_INFO(
        "[HcomAllReduceOp][UpdateHostArgs] input %p -> %p, output %p -> %p.", prevIn, args->inputAddr, prevOut,
        args->outputAddr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::InferShape(gert::InferShapeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomAllReduceOp][InferShape] infer shape context is null.");
        return ge::GRAPH_FAILED;
    }
    size_t inputNum = ctx->GetComputeNodeInputNum();
    size_t outputNum = ctx->GetComputeNodeOutputNum();
    if (inputNum != outputNum) {
        HCCL_ERROR("[HcomAllReduceOp][InferShape] inputNum=%zu != outputNum=%zu.", inputNum, outputNum);
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < inputNum; i++) {
        const gert::Shape* inputShape = ctx->GetInputShape(i);
        if (inputShape == nullptr) {
            HCCL_ERROR("[HcomAllReduceOp][InferShape] input shape[%zu] is null.", i);
            return ge::GRAPH_FAILED;
        }
        gert::Shape* outputShape = ctx->GetOutputShape(i);
        if (outputShape == nullptr) {
            HCCL_ERROR("[HcomAllReduceOp][InferShape] output shape[%zu] is null.", i);
            return ge::GRAPH_FAILED;
        }
        *outputShape = *inputShape;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomAllReduceOp::InferDataType(gert::InferDataTypeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomAllReduceOp][InferDataType] infer datatype context is null.");
        return ge::GRAPH_FAILED;
    }
    size_t inputNum = ctx->GetComputeNodeInputNum();
    size_t outputNum = ctx->GetComputeNodeOutputNum();
    if (inputNum != outputNum) {
        HCCL_ERROR("[HcomAllReduceOp][InferDataType] inputNum=%zu != outputNum=%zu.", inputNum, outputNum);
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < inputNum; i++) {
        ge::DataType dataType = ctx->GetInputDataType(i);
        if (dataType == ge::DT_UNDEFINED) {
            HCCL_ERROR("[HcomAllReduceOp][InferDataType] input dtype[%zu] is undefined.", i);
            return ge::GRAPH_FAILED;
        }
        ge::graphStatus ret = ctx->SetOutputDataType(i, dataType);
        if (ret != ge::GRAPH_SUCCESS) {
            HCCL_ERROR("[HcomAllReduceOp][InferDataType] SetOutputDataType[%zu] failed.", i);
            return ret;
        }
    }
    return ge::GRAPH_SUCCESS;
}

const char* HcomAllReduceOp::GetOpTypeName() const { return HCCL_KERNEL_OP_TYPE_ALLREDUCE.c_str(); }

static ge::BaseCustomOp* CreateHcomAllReduceOp() { return new HcomAllReduceOp(); }
// 条件注册: 目标平台 (950/960) 且 GE/RTS 新接口在位才注册接管, 否则回退原 hccl_engine 路径
// (hccl_custom_op.h CreateSocGatedRegister: SoC 判定 + GE/RTS 接口在位判定)
// 5 元接管注册 (kBottom/kHccl)
static ge::CustomOpCreatorRegister* g_hcom_all_reduce_register
    = hccl::CreateSocGatedRegister(HCCL_KERNEL_OP_TYPE_ALLREDUCE.c_str(), CreateHcomAllReduceOp);
