/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ins_v2_all_gather_sole_executor.h"
#include "topo_match_1d.h"
#include "ins_temp_all_gather_mesh_1D.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "ins_temp_all_gather_nhr.h"
#ifndef AICPU_COMPILE
#include "aiv_temp_all_gather_mesh_1D.h"
#if !defined(HCCL_CANN_COMPAT_850)
#include "ccu_temp_all_gather_mesh_1D_mem2mem.h"
#include "ccu_temp_all_gather_mesh_1D.h"
#include "ccu_temp_all_gather_nhr_1D_mem2mem.h"
#include "ccu_temp_all_gather_2dies_mesh_1d_mem2mem.h"
#include "ccu_temp_all_gather_2dies_mesh_1D.h"
#include "ccu_temp_all_gather_nhr_1D_multi_jetty_mem2mem.h"
#endif /* !HCCL_CANN_COMPAT_850 */
#endif
#include "topo_match_ubx.h"
namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2AllGatherSoleExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
    AlgResourceRequest &resourceRequest)
{
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    // 调用计算资源的函数
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));
    myRank_ = topoInfo->userRank;
    HCCL_DEBUG("[InsV2AllGatherSoleExecutor][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u], "
               "channels[%u]",
               myRank_, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
               resourceRequest.channels.size());
    for (auto i = 0; i < resourceRequest.notifyNumPerThread.size(); i++) {
        HCCL_DEBUG("[InsV2AllGatherSoleExecutor][CalcRes] myRank[%u], notifyNumPerThread[%u]=[%u]", myRank_, i,
                   resourceRequest.notifyNumPerThread[i]);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherSoleExecutor][Orchestrate] Orchestrate Start");
    myRank_ = resCtx.topoInfo.userRank;

    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    HCCL_DEBUG("[InsV2AllGatherSoleExecutor][Orchestrate] myRank[%u], threadsSize[%lu], "
               "dataCount[%llu], dataTypeSize[%lu]",
               myRank_, threads_.size(), dataCount_, dataTypeSize_);
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllGatherSoleExecutor][Orchestrate]errNo[0x%016llx] All Gather excutor kernel run failed",
                   HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherSoleExecutor][OrchestrateLoop] Start");

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
    // 不需要重复
    tempAlgParams.repeatNum = 1;
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;
    HCCL_INFO("[InsV2AllGatherSoleExecutor][OrchestrateLoop] myRank[%u], inputPtr[%#llx] outputPtr[%#llx], "
              "cclAddr[%#llx], cclSize[%llu], channelSize[%lu], threadSize[%lu], ",
              myRank_, param.inputPtr, param.outputPtr, resCtx.cclMem.addr, resCtx.cclMem.size,
              templateAlgRes.channels.size(), templateAlgRes.threads.size());
    // 构建template
    InsAlgTemplate algTemplate(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
    u32 templateScratchMultiplier =
        algTemplate.CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.outBuffType);
    maxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size;
    if (param.engine == COMM_ENGINE_AICPU_TS) {
        algTemplate.SetchannelsPerRank(templateAlgRes.channels);
    }
    // 中转内存单次最多能够接受的output count，注意是count不是size
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 maxDataSizePerLoop = 0;
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize =
            maxTmpMemSize_ / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    u64 maxCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;
    HCCL_INFO("[InsV2AllGatherSoleExecutor][OrchestrateLoop] myRank[%u], templateScratchMultiplier[%u] "
              "maxCountPerLoop[%llu], loopTimes[%llu]",
              myRank_, templateScratchMultiplier, maxCountPerLoop, loopTimes);
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;

        tempAlgParams.count = currDataCount;
        tempAlgParams.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParams.tailSize = tempAlgParams.sliceSize;
        tempAlgParams.inputSliceStride = 0;
        tempAlgParams.outputSliceStride = dataSize_;

        HCCL_DEBUG("[InsV2AllGatherSoleExecutor] myRank[%u], loop [%u] tempAlgParams.inputSliceStride [%u],"
                  "tempAlgParams.outputSliceStride [%u] tempAlgParams.sliceSize [%u]",
                  myRank_, loop, tempAlgParams.inputSliceStride, tempAlgParams.outputSliceStride,
                  tempAlgParams.sliceSize);
        HCCL_DEBUG("[InsV2AllGatherSoleExecutor] myRank[%u], loop [%u] tempAlgParams.buffInfo.inBuffBaseOff [%u],"
                  "tempAlgParams.buffInfo.outBuffBaseOff [%u]",
                  myRank_, loop, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.outBuffBaseOff);

        CHK_RET(algTemplate.KernelRun(param, tempAlgParams, templateAlgRes));
        processedDataCount += currDataCount;
    }

#ifndef AICPU_COMPILE
    if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU && param.opMode != OpMode::OFFLOAD) {
        CHK_RET(FastLaunchSaveCtx(param, templateAlgRes, resCtx.notifyNumOnMainThread));
    }
#endif

    HCCL_INFO("[InsV2AllGatherSoleExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunchSaveCtx(
    const OpParam &param, const TemplateResource &templateAlgRes, u32 notifyNumOnMainThread)
{
    HCCL_INFO("[InsV2AllGatherSoleExecutor] loopTimes==1, save fast launch ctx.");
    u32 threadNum = 1;
    u32 ccuKernelNum = templateAlgRes.submitInfos.size();
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsV2AllGatherSoleExecutor] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[InsV2AllGatherSoleExecutor][HcclEngineCtxCreate] threadNum[%llu], ccuKernelNum[%llu]", threadNum, ccuKernelNum);

    u64 size = CcuFastLaunchCtx::GetCtxSize(threadNum, ccuKernelNum);
    // 申请ctx
    void *ctxPtr = nullptr;
    HCCL_INFO("[InsV2AllGatherSoleExecutor][HcclEngineCtxCreate] Tag[%s], size[%llu]", param.fastLaunchTag, size);
    CHK_RET(HcclEngineCtxCreate(param.hcclComm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, size, &ctxPtr));

    CcuFastLaunchCtx *ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(ctxPtr);
    // 1 算法名
    CHK_SAFETY_FUNC_RET(strcpy_s(ccuFastLaunchCtx->algName, sizeof(ccuFastLaunchCtx->algName), param.algName));
    HCCL_INFO("[InsV2AllGatherSoleExecutor][FastLaunchSaveCtx] algName[%s]", ccuFastLaunchCtx->algName);

    // 2 thread
    ccuFastLaunchCtx->threadNum = threadNum;
    ccuFastLaunchCtx->notifyNumOnMainThread = notifyNumOnMainThread;
    ThreadHandle *threads = ccuFastLaunchCtx->GetThreadHandlePtr();
    threads[0] = templateAlgRes.threads[0];
        
    // 3 ccu kernel handle, taskArg入参
    ccuFastLaunchCtx->ccuKernelNum[0] = ccuKernelNum;
    CcuKernelSubmitInfo *kernelSubmitInfos = ccuFastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    kernelSubmitInfos[0] = templateAlgRes.submitInfos[0];
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunch(
        const OpParam &param, const CcuFastLaunchCtx *fastLaunchCtx)
{
    HCCL_INFO("[InsV2AllGatherSoleExecutor][FastLaunch] Start.");
    TemplateFastLaunchCtx tempFastLaunchCtx;
    // 1 取thread
    ThreadHandle *threads = fastLaunchCtx->GetThreadHandlePtr();
    tempFastLaunchCtx.threads.assign(threads, threads + fastLaunchCtx->threadNum);
    HCCL_INFO("[InsV2AllGatherSoleExecutor][FastLaunch] threadNum[%llu]", fastLaunchCtx->threadNum);
    
    // 2 取arg
    CcuKernelSubmitInfo *ccuKernelSubmitInfos = fastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    tempFastLaunchCtx.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + fastLaunchCtx->ccuKernelNum[0]);
    HCCL_INFO("[InsV2AllGatherSoleExecutor][FastLaunch] ccuKernelNum[%llu]", fastLaunchCtx->ccuKernelNum[0]);
    tempFastLaunchCtx.buffInfo.inputPtr = param.inputPtr;
    tempFastLaunchCtx.buffInfo.outputPtr = param.outputPtr;
    tempFastLaunchCtx.buffInfo.hcclBuff = param.hcclBuff;
    
    // 3 调template
    std::unique_ptr<InsAlgTemplate> algTemplate = std::make_unique<InsAlgTemplate>();
    CHK_RET(algTemplate->FastLaunch(param, tempFastLaunchCtx));
    HCCL_INFO("[InsV2AllGatherSoleExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherMesh1D, InsV2AllGatherSoleExecutor, TopoMatch1D,
                 InsTempAllGatherMesh1D);

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherMesh1D1DZAxisDetour, InsV2AllGatherSoleExecutor, TopoMatch1D,
                 InsTempAllGatherMesh1D1DZAxisDetour);

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherNHR, InsV2AllGatherSoleExecutor, TopoMatch1D,
                 InsTempAllGatherNHR);

#ifndef AICPU_COMPILE
#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER, CcuAllGatherMesh1DMem2Mem, InsV2AllGatherSoleExecutor, TopoMatch1D,
                 CcuTempAllGatherMesh1DMem2Mem);
#endif /* !HCCL_CANN_COMPAT_850 */

#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER, CcuAllGatherMesh1D, InsV2AllGatherSoleExecutor, TopoMatch1D,
                 CcuTempAllGatherMesh1D);
#endif /* !HCCL_CANN_COMPAT_850 */

#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER, CcuAllGatherNHR1DMem2Mem, InsV2AllGatherSoleExecutor, TopoMatch1D,
                 CcuTempAllGatherNHR1DMem2Mem);
#endif /* !HCCL_CANN_COMPAT_850 */

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER, AivAllGatherMesh1D, InsV2AllGatherSoleExecutor, TopoMatch1D,
    AivTempAllGatherMesh1D);

#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER, CcuKernelAllGather2DiesMeshMem2Mem1D, InsV2AllGatherSoleExecutor, TopoMatch1D,
    CcuTempAllGather2DiesMeshMem2Mem1D);
#endif /* !HCCL_CANN_COMPAT_850 */

#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER, CcuKernelAllGather2DiesMesh1D, InsV2AllGatherSoleExecutor, TopoMatch1D,
    CcuTempAllGather2DiesMesh1D);
#endif /* !HCCL_CANN_COMPAT_850 */

#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER, CcuAllGatherNHR1DMem2MemMultiJetty, InsV2AllGatherSoleExecutor, TopoMatch1D,
    CcuTempAllGatherNHR1DMultiJettyMem2Mem);
#endif /* !HCCL_CANN_COMPAT_850 */
                 
#endif
}  // namespace ops_hccl
