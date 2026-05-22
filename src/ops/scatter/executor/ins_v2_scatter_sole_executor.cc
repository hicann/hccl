/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_scatter_sole_executor.h"
#include "ins_temp_scatter_mesh_1D.h"
#include "ins_temp_scatter_nhr.h"
#ifndef AICPU_COMPILE
#include "aiv_temp_scatter_mesh_1D.h"
#if !defined(HCCL_CANN_COMPAT_850)
#include "ccu_temp_scatter_mesh1d.h"
#include "ccu_temp_scatter_nhr1d_mem2mem.h"
#include "ccu_kernel_scatter_nhr1d_mem2mem.h"
#endif /* !HCCL_CANN_COMPAT_850 */
#endif

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2ScatterSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    CHK_PTR_NULL(topoInfo);
    CHK_PRT_RET(algHierarchyInfo.infos.empty(), 
                HCCL_ERROR("[InsV2ScatterSoleExecutor][CalcRes] algHierarchyInfo.infos is empty"),
                HCCL_E_PARA);
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate = std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    // 调用计算资源的函数
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2ScatterSoleExecutor][Orchestrate] Orchestrate Start");
    maxTmpMemSize_ = resCtx.cclMem.size;
    myRank_ = resCtx.topoInfo.userRank;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2ScatterSoleExecutor][Orchestrate]errNo[0x%016llx] Scatter excutor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2ScatterSoleExecutor][OrchestrateLoop] Start");
    CHK_PRT_RET(resCtx.algHierarchyInfo.infos.empty(), 
                HCCL_ERROR("[InsV2ScatterSoleExecutor][OrchestrateLoop] algHierarchyInfo.infos is empty"),
                HCCL_E_PARA);
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
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate = std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
    // 初始化操作
    u32 templateScratchMultiplier = algTemplate->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType,
                                                                     tempAlgParams.buffInfo.outBuffType);
    // 中转内存单次最多能够接受的output count，注意是count不是size
    u64 maxDataSizePerLoop = 0;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    maxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size;
    HCCL_INFO("[InsV2ScatterSoleExecutor]maxTmpMemSize_ [%u]", maxTmpMemSize_);
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    u64 maxCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        tempAlgParams.count = currDataCount;
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;

        tempAlgParams.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParams.tailSize = tempAlgParams.sliceSize;
        tempAlgParams.inputSliceStride = dataSize_;  // 如果是输入，偏移是算子的output datasize
        tempAlgParams.outputSliceStride = 0;

        HCCL_INFO("[InsV2ScatterSoleExecutor] loop [%u] tempAlgParams.inputSliceStride [%u],"
                  "tempAlgParams.outputSliceStride [%u] tempAlgParams.sliceSize [%u]",
            loop,
            tempAlgParams.inputSliceStride,
            tempAlgParams.outputSliceStride,
            tempAlgParams.sliceSize);
        HCCL_INFO("[InsV2ScatterSoleExecutor] loop [%u] tempAlgParams.buffInfo.inBuffBaseOff [%u],"
                  "tempAlgParams.buffInfo.outBuffBaseOff [%u]",
            loop,
            tempAlgParams.buffInfo.inBuffBaseOff,
            tempAlgParams.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParams.repeatNum = 1;
        tempAlgParams.inputRepeatStride = 0;
        tempAlgParams.outputRepeatStride = 0;
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes));
        processedDataCount += currDataCount;
    }

#ifndef AICPU_COMPILE
    if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU && param.opMode != OpMode::OFFLOAD) {
        CHK_RET(FastLaunchSaveCtx(param, templateAlgRes, resCtx.notifyNumOnMainThread));
    }
#endif
    
    HCCL_INFO("[InsV2ScatterSoleExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
    template <typename AlgTopoMatch, typename InsAlgTemplate>
    HcclResult InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunchSaveCtx(
            const OpParam &param, const TemplateResource &templateAlgRes, u32 notifyNumOnMainThread)
    {
        HCCL_INFO("[InsV2ScatterSoleExecutor] loopTimes==1, save fast launch ctx.");
        u32 threadNum = templateAlgRes.submitInfos.size();;
        u32 ccuKernelNum = templateAlgRes.submitInfos.size();
        if (ccuKernelNum < 1) {
            HCCL_INFO("[InsV2ScatterSoleExecutor] ccu kernel num is 0, no need to save.");
            return HCCL_SUCCESS;
        }
        HCCL_INFO("[InsV2ScatterSoleExecutor][HcclEngineCtxCreate] threadNum[%llu], ccuKernelNum[%llu]", threadNum, ccuKernelNum);

        u64 size = CcuFastLaunchCtx::GetCtxSize(threadNum, ccuKernelNum);
        // 申请ctx
        void *ctxPtr = nullptr;
        HCCL_INFO("[InsV2ScatterSoleExecutor][HcclEngineCtxCreate] Tag[%s], size[%llu]", param.fastLaunchTag, size);
        CHK_RET(HcclEngineCtxCreate(param.hcclComm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, size, &ctxPtr));

        CcuFastLaunchCtx *ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(ctxPtr);
        // 1 算法名
        CHK_SAFETY_FUNC_RET(strcpy_s(ccuFastLaunchCtx->algName, sizeof(ccuFastLaunchCtx->algName), param.algName));
        HCCL_INFO("[InsV2ScatterSoleExecutor][FastLaunchSaveCtx] algName[%s]", ccuFastLaunchCtx->algName);

        // 2 thread
        ccuFastLaunchCtx->threadNum = threadNum;
        ccuFastLaunchCtx->notifyNumOnMainThread = notifyNumOnMainThread;
        ThreadHandle *threads = ccuFastLaunchCtx->GetThreadHandlePtr();

        for (int i = 0; i < threadNum; i++)
        {
            threads[i] = templateAlgRes.threads[i];
        }

        // 3 ccu kernel handle, taskArg入参
        ccuFastLaunchCtx->ccuKernelNum[0] = ccuKernelNum;
        CcuKernelSubmitInfo *kernelSubmitInfos = ccuFastLaunchCtx->GetCcuKernelSubmitInfoPtr();
        kernelSubmitInfos[0] = templateAlgRes.submitInfos[0];
        return HCCL_SUCCESS;
    }

    template <typename AlgTopoMatch, typename InsAlgTemplate>
    HcclResult InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunch(
            const OpParam &param, const CcuFastLaunchCtx *fastLaunchCtx)
    {
        HCCL_INFO("[InsV2ScatterSoleExecutor][FastLaunch] Start.");
        TemplateFastLaunchCtx tempFastLaunchCtx;
        // 1 取thread
        ThreadHandle *threads = fastLaunchCtx->GetThreadHandlePtr();
        tempFastLaunchCtx.threads.assign(threads, threads + fastLaunchCtx->threadNum);
        HCCL_INFO("[InsV2ScatterSoleExecutor][FastLaunch] threadNum[%llu]", fastLaunchCtx->threadNum);

        // 2 取arg
        CcuKernelSubmitInfo *ccuKernelSubmitInfos = fastLaunchCtx->GetCcuKernelSubmitInfoPtr();
        tempFastLaunchCtx.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + fastLaunchCtx->ccuKernelNum[0]);
        HCCL_INFO("[InsV2ScatterSoleExecutor][FastLaunch] ccuKernelNum[%llu]", fastLaunchCtx->ccuKernelNum[0]);
        tempFastLaunchCtx.buffInfo.inputPtr = param.inputPtr;
        tempFastLaunchCtx.buffInfo.outputPtr = param.outputPtr;
        tempFastLaunchCtx.buffInfo.hcclBuff = param.hcclBuff;

        // 3 调template
        std::unique_ptr<InsAlgTemplate> algTemplate = std::make_unique<InsAlgTemplate>();
        CHK_RET(algTemplate->FastLaunch(param, tempFastLaunchCtx));
        HCCL_INFO("[InsV2ScatterSoleExecutor][FastLaunch] End.");
        return HCCL_SUCCESS;
    }
#endif

// 第二个参数是Scatter的template文件
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_SCATTER, InsScatterMesh1D, InsV2ScatterSoleExecutor, TopoMatch1D, InsTempScatterMesh1D);
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_SCATTER, InsScatterNHR, InsV2ScatterSoleExecutor, TopoMatch1D, InsTempScatterNHR);
#ifndef AICPU_COMPILE
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_SCATTER, AivScatterMesh1D, InsV2ScatterSoleExecutor, TopoMatch1D,
    AivTempScatterMesh1D);
#if !defined(HCCL_CANN_COMPAT_850)
// ccu template
#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_SCATTER, CcuScatterMesh1D, InsV2ScatterSoleExecutor, TopoMatch1D, CcuTempScatterMesh1D);
#endif /* !HCCL_CANN_COMPAT_850 */
#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_SCATTER, CcuScatterNHRMem2Mem1D, InsV2ScatterSoleExecutor, TopoMatch1D, CcuTempScatterNHR1DMem2Mem);
#endif /* !HCCL_CANN_COMPAT_850 */
#endif
#endif
}  // namespace ops_hccl