/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_reduce_experimental_sole_executor.h"
#include "topo_match_one_level.h"
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_all_reduce_experimental_mesh_1D.h"
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#endif

namespace ops_hccl_experimental {

using namespace ops_hccl;

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AllReduceExperimentalSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2AllReduceExperimentalSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
AlgAttrs
InsV2AllReduceExperimentalSoleExecutor<AlgTopoMatch, InsAlgTemplate>::BuildExperimentalAlgAttrs(const AlgAttrs& base)
{
    AlgAttrs attrs = base;
    attrs.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    attrs.engine = OpExecuteConfig::CCU_MS;
    attrs.algoTypes = {AlgoType::MESH};
    return attrs;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
std::vector<CostModelParam> InsV2AllReduceExperimentalSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
#ifndef AICPU_COMPILE
    // 直接构建修好的 AlgAttrsExperimental（规避 Experimental 段导致 algName 解析失败、algoTypes 为空）
    AlgAttrs algAttrsExperimental = BuildExperimentalAlgAttrs(AlgAttrs{});
    const AlgAttrs* attrs = &algAttrsExperimental;
#else
    // AICPU 独立核库(scatter_aicpu_kernel.so)不链接 host-only 的 AlgAttrsRegistry,
    // device 侧亦无 costmodel 调用链, 置空走 skip 分支
    const AlgAttrs* attrs = nullptr;
#endif
    // 探测路径直接调 MatchTopo（不走 CalcAlgHierarchyInfoV2 的 CHK_RET）：
    // costmodel 迭代时"不匹配"是正常事件，避免执行路径语义的 ERROR 日志刷屏
    AlgTopoMatch topoMatch;
    HcclResult matchRet
        = (attrs != nullptr) ? topoMatch.MatchTopo(topoInfo, algHierarchyInfo, *attrs) : HcclResult::HCCL_E_PARA;
    if (matchRet != HcclResult::HCCL_SUCCESS) {
        HCCL_INFO(
            "[InsV2AllReduceExperimentalSoleExecutor][CalcCostCoeff] algName=%s topo match not support, skip.",
            algName);
        return {};
    }
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = topoInfo->isPod;
    CommTopo netTypeLevel0
        = GetPhysicalLevelTopoType(topoInfo, static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]));
    std::vector<u32> portNumLevel0
        = GetPhysicalLevelPortNums(topoInfo, static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]));
    if (portNumLevel0.empty()) {
        HCCL_WARNING("[InsV2AllReduceExperimentalSoleExecutor][CalcCostCoeff] portNum is empty");
        return {};
    }
    HCCL_INFO(
        "[InsV2AllReduceExperimentalSoleExecutor][CalcCostCoeff] rankSize=%d, portNumLevel0=%d, netTypeLevel0=%d",
        rankSize, portNumLevel0, static_cast<int>(netTypeLevel0));
    return InsAlgTemplate::CalcCostCoeff(CalcCostCoeffParam{
        rankSize, 1.0f / rankSize, netTypeLevel0, BufferType::INPUT, BufferType::OUTPUT, BufferType::HCCL_BUFFER,
        portNumLevel0, isPod});
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
AlgNetMeta InsV2AllReduceExperimentalSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)param;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
#ifndef AICPU_COMPILE
    // 直接构建修好的 AlgAttrsExperimental（规避 Experimental 段导致 algName 解析失败、algoTypes 为空）
    AlgAttrs algAttrsExperimental = BuildExperimentalAlgAttrs(AlgAttrs{});
    const AlgAttrs* attrs = &algAttrsExperimental;
#else
    // AICPU 独立核库(scatter_aicpu_kernel.so)不链接 host-only 的 AlgAttrsRegistry,
    // device 侧亦无 costmodel 调用链, 置空走 skip 分支
    const AlgAttrs* attrs = nullptr;
#endif
    // 探测路径直接调 MatchTopo：无 CHK_RET 的 ERROR，且免去 V2 调用所需的多层 const_cast
    AlgTopoMatch topoMatch;
    HcclResult matchRet
        = (attrs != nullptr) ?
              topoMatch.MatchTopo(const_cast<TopoInfoWithNetLayerDetails*>(topoInfo), algHierarchyInfo, *attrs) :
              HcclResult::HCCL_E_PARA;
    if (matchRet != HcclResult::HCCL_SUCCESS) {
        HCCL_INFO(
            "[InsV2AllReduceExperimentalSoleExecutor][GetAlgNetMeta] algName=%s topo match not support, return empty.",
            algName);
        return {};
    }
    u32 rankSize = topoInfo->userRankSize;
    u32 physLevelIdx = (topoInfo->topoLevelNums > 1 && algHierarchyInfo.physicalIdxForAlgoLevels.size() > 1) ?
                           static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[1][0]) :
                           static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]);
    CommTopo netTypeLevel0 = GetPhysicalLevelTopoType(topoInfo, physLevelIdx);
    AlgNetMeta meta;
    meta.netTypes.push_back(netTypeLevel0);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1};
    meta.dataRatios = {1.0f / rankSize};
    meta.rankSizes = {rankSize};
    return meta;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceExperimentalSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    (void)comm;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, AlgAttrs{}));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceExperimentalSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel（携带算法属性，供拓扑匹配过滤）
    // 基于框架传入的 algAttrs 修正 experimental 相关字段，规避 Experimental 段导致算法名解析失败
    AlgAttrs algAttrsExperimental = BuildExperimentalAlgAttrs(algAttrs);
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrsExperimental));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceExperimentalSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    // 调用计算资源的函数
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceExperimentalSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceExperimentalSoleExecutor][Orchestrate] Orchestrate Start.");
    // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    maxTmpMemSize_ = resCtx.cclMem.size;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    if (dataTypeSize_ == 0) {
        HCCL_ERROR(
            "[InsV2AllReduceExperimentalSoleExecutor][Orchestrate] dataTypeSize_ is 0, dataType[%d] is invalid.",
            dataType_);
        return HCCL_E_INTERNAL;
    }
    if (dataCount_ > UINT64_MAX / dataTypeSize_) {
        HCCL_ERROR(
            "[InsV2AllReduceExperimentalSoleExecutor][Orchestrate] dataCount[%llu] * dataTypeSize_[%llu] is greater "
            "than UINT64_MAX",
            dataCount_, dataTypeSize_);
        return HCCL_E_INTERNAL;
    }
    dataSize_ = dataCount_ * dataTypeSize_;
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2AllReduceExperimentalSoleExecutor][Orchestrate]errNo[0x%016llx] AllReduce executor kernel run "
            "failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceExperimentalSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceExperimentalSoleExecutor][OrchestrateLoop] Start");
    // 准备资源
    TemplateResource templateAlgRes;
    if (param.engine == COMM_ENGINE_CCU) {
        templateAlgRes.ccuKernels = resCtx.ccuKernels;
    }
    if (param.engine != CommEngine::COMM_ENGINE_AIV && remoteRankToChannelInfo_.size() > 0) {
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    templateAlgRes.threads = resCtx.threads;
    templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    // 准备数据
    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    tempAlgParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
    // 不需要重复；repeat用于处理rank存在多块不连续数据块的情况（all-reduce不涉及）
    tempAlgParams.repeatNum = 1;
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;

    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
    u32 templateScratchMultiplier
        = algTemplate->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.outBuffType);

    // 计算最小传输大小
    u64 maxDataSizePerLoop = 0;
    maxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size;
    u64 transportBoundDataSize = (param.engine == CommEngine::COMM_ENGINE_AICPU_TS) ? maxTmpMemSize_ : UB_MAX_DATA_SIZE;
    HCCL_INFO("[InsV2AllReduceExperimentalSoleExecutor]maxTmpMemSize_ [%u]", maxTmpMemSize_);
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize
            = maxTmpMemSize_ / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    // 单次循环处理的数据量大小（dataTypeSize_ 非 0 已在 Orchestrate 校验）
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    HCCL_INFO(
        "[InsV2AllReduceExperimentalSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop[%llu], "
        "maxDataSizePerLoop[%llu], transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    CHK_PRT_RET(
        maxDataCountPerLoop == 0,
        HCCL_ERROR("[InsV2AllReduceExperimentalSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop is 0"),
        HCCL_E_INTERNAL);
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxDataCountPerLoop
                    + static_cast<u64>(dataCount_ % maxDataCountPerLoop != 0); // 计算迭代轮次（ceil取整）
    // count已经处理的数据
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        // dataCount_实际总数据量 和 maxDataCountPerLoop 一次搬运数据量之间不一定是整除关系，需要对尾块进行处理
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxDataCountPerLoop;
        tempAlgParams.count = currDataCount;
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;

        tempAlgParams.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParams.tailSize = tempAlgParams.sliceSize;
        tempAlgParams.inputSliceStride = 0;
        tempAlgParams.outputSliceStride = 0;
        HCCL_INFO(
            "[InsV2AllReduceExperimentalSoleExecutor] loop [%u] tempAlgParams.inputSliceStride [%u],"
            "tempAlgParams.outputSliceStride [%u] tempAlgParams.sliceSize [%u]",
            loop, tempAlgParams.inputSliceStride, tempAlgParams.outputSliceStride, tempAlgParams.sliceSize);
        HCCL_INFO(
            "[InsV2AllReduceExperimentalSoleExecutor] loop [%u] tempAlgParams.buffInfo.inBuffBaseOff [%u],"
            "tempAlgParams.buffInfo.outBuffBaseOff [%u]",
            loop, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.outBuffBaseOff);

        CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes));
        processedDataCount += currDataCount;
    }
#ifndef AICPU_COMPILE
    if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU && param.opMode != OpMode::OFFLOAD) {
        CHK_RET(FastLaunchSaveCtx(param, templateAlgRes, resCtx.notifyNumOnMainThread));
    }
#endif

    HCCL_INFO("[InsV2AllReduceExperimentalSoleExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceExperimentalSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunchSaveCtx(
    const OpParam& param, const TemplateResource& templateAlgRes, u32 notifyNumOnMainThread) const
{
    HCCL_INFO("[InsV2AllReduceExperimentalSoleExecutor] loopTimes==1, save fast launch ctx.");
    u32 threadNum = templateAlgRes.threads.size();
    u32 ccuKernelNum = templateAlgRes.submitInfos.size();
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsV2AllReduceExperimentalSoleExecutor] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "[InsV2AllReduceExperimentalSoleExecutor][HcclEngineCtxCreate] threadNum[%llu], ccuKernelNum[%llu]", threadNum,
        ccuKernelNum);

    u64 size = CcuFastLaunchCtx::GetCtxSize(threadNum, ccuKernelNum);
    // 申请ctx
    void* ctxPtr = nullptr;
    HCCL_INFO(
        "[InsV2AllReduceExperimentalSoleExecutor][HcclEngineCtxCreate] Tag[%s], size[%llu]", param.fastLaunchTag, size);
    CHK_RET(HcclEngineCtxCreate(param.hcclComm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, size, &ctxPtr));

    CcuFastLaunchCtx* ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(ctxPtr);
    // 1 算法名
    CHK_SAFETY_FUNC_RET(strcpy_s(ccuFastLaunchCtx->algName, sizeof(ccuFastLaunchCtx->algName), param.algName));
    HCCL_INFO("[InsV2AllReduceExperimentalSoleExecutor][FastLaunchSaveCtx] algName[%s]", ccuFastLaunchCtx->algName);

    // 2 thread
    ccuFastLaunchCtx->threadNum = threadNum;
    ccuFastLaunchCtx->notifyNumOnMainThread = notifyNumOnMainThread;
    ThreadHandle* threads = ccuFastLaunchCtx->GetThreadHandlePtr();
    for (u32 i = 0; i < threadNum; i++) {
        threads[i] = templateAlgRes.threads[i];
    }

    // 3 ccu kernel handle, taskArg入参
    ccuFastLaunchCtx->ccuKernelNum[0] = ccuKernelNum;
    CcuKernelSubmitInfo* kernelSubmitInfos = ccuFastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    for (u32 i = 0; i < ccuKernelNum; i++) {
        kernelSubmitInfos[i] = templateAlgRes.submitInfos[i];
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceExperimentalSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunch(
    const OpParam& param, const CcuFastLaunchCtx* fastLaunchCtx)
{
    HCCL_INFO("[InsV2AllReduceExperimentalSoleExecutor][FastLaunch] Start.");
    TemplateFastLaunchCtx tempFastLaunchCtx;
    // 1 取thread
    ThreadHandle* threads = fastLaunchCtx->GetThreadHandlePtr();
    tempFastLaunchCtx.threads.assign(threads, threads + fastLaunchCtx->threadNum);
    HCCL_INFO("[InsV2AllReduceExperimentalSoleExecutor][FastLaunch] threadNum[%llu]", fastLaunchCtx->threadNum);

    // 2 取arg
    CcuKernelSubmitInfo* ccuKernelSubmitInfos = fastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    tempFastLaunchCtx.ccuKernelSubmitInfos.assign(
        ccuKernelSubmitInfos, ccuKernelSubmitInfos + fastLaunchCtx->ccuKernelNum[0]);
    HCCL_INFO(
        "[InsV2AllReduceExperimentalSoleExecutor][FastLaunch] ccuKernelNum[%llu]", fastLaunchCtx->ccuKernelNum[0]);
    tempFastLaunchCtx.buffInfo.inputPtr = param.inputPtr;
    tempFastLaunchCtx.buffInfo.outputPtr = param.outputPtr;
    tempFastLaunchCtx.buffInfo.hcclBuff = param.hcclBuff;

    // 3 调template
    std::unique_ptr<InsAlgTemplate> algTemplate = std::make_unique<InsAlgTemplate>();
    CHK_RET(algTemplate->FastLaunch(param, tempFastLaunchCtx));
    HCCL_INFO("[InsV2AllReduceExperimentalSoleExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif

} // namespace ops_hccl_experimental

namespace ops_hccl {
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, CcuMSAllReduceExperimentalSoleMesh,
    ops_hccl_experimental::InsV2AllReduceExperimentalSoleExecutor, TopoMatchOneLevel,
    ops_hccl_experimental::CcuTempAllReduceExperimentalMesh1D);
REGISTER_ALG_ATTRS(
    CcuMSAllReduceExperimentalSoleMesh, topo.maxTopoLevelNum = 1;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS; op.isSupportProd = false;
    op.unsupportedDataTypes
    = {HcclDataType::HCCL_DATA_TYPE_INT8, HcclDataType::HCCL_DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_UINT64,
       HcclDataType::HCCL_DATA_TYPE_FP64};
    op.isSupportInplace = false;
    op.opPriorityCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->userRankSize == 2;
    });
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
} // namespace ops_hccl
