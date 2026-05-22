/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_sole_executor.h"
#include "../template/aicpu/reduce_mesh_1D.h"
#include "../template/aicpu/reduce_mesh_1D_two_shot.h"
#include "../template/aicpu/reduce_nhr.h"
#include "../template/aicpu/reduce_aicpu_reduce_nhr.h"
#include "topo_match_1d.h"
#ifndef AICPU_COMPILE
#include "aiv_temp_reduce_mesh_1D.h"
#if !defined(HCCL_CANN_COMPAT_850)
#include "ccu_temp_reduce_mesh_1D_mem2mem.h"
#include "ccu_temp_reduce_mesh_1D.h"
#include "ccu_temp_reduce_nhr_1D_mem2mem.h"
#endif /* !HCCL_CANN_COMPAT_850 */
#endif

namespace ops_hccl {

template <typename AlgTopoMatch, typename AlgTemplate>
ReduceSoleExecutor<AlgTopoMatch, AlgTemplate>::ReduceSoleExecutor()
{}

template <typename AlgTopoMatch, typename AlgTemplate>
HcclResult ReduceSoleExecutor<AlgTopoMatch, AlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    CHK_PTR_NULL(topoInfo);
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate>
HcclResult ReduceSoleExecutor<AlgTopoMatch, AlgTemplate>::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    // 构建template
    const auto &topo = algHierarchyInfo.infos;
    std::shared_ptr<AlgTemplate> algTemplate =
        std::make_shared<AlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    // 调用计算资源的函数
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate>
HcclResult ReduceSoleExecutor<AlgTopoMatch, AlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[ReduceSoleExecutor][Orchestrate] Orchestrate Start channels: [%u]", resCtx.channels.size());
    // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    maxTmpMemSize_ = resCtx.cclMem.size;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
        HCCL_DEBUG("[ReduceSoleExecutor][Orchestrate] info[0].size():%u", remoteRankToChannelInfo_[0].size());
    }
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[ReduceSoleExecutor][Orchestrate]errNo[0x%016llx] Reduce excutor kernel run failed", HCCL_ERROR_CODE(ret)),
        ret);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate>
HcclResult ReduceSoleExecutor<AlgTopoMatch, AlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[ReduceSoleExecutor][OrchestrateLoop] Start");
    // 准备资源
    TemplateResource templateAlgRes;
    if (param.engine == COMM_ENGINE_CCU) {
        templateAlgRes.ccuKernels = resCtx.ccuKernels;
    }
    if (remoteRankToChannelInfo_.size() > 0) {
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    HCCL_INFO("[ReduceSoleExecutor][OrchestrateLoop] channels: %u", templateAlgRes.channels.size());
    // }
    templateAlgRes.threads = resCtx.threads;
    templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    // 准备数据
    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建template
    std::shared_ptr<AlgTemplate> algTemplate =
        std::make_shared<AlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
    u32 templateScratchMultiplier =
        algTemplate->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.outBuffType);
    // 计算最小传输大小
    u64 maxDataSizePerLoop = 0;
    maxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    HCCL_INFO("[ReduceSoleExecutor]maxTmpMemSize_ [%u]", maxTmpMemSize_);
    HCCL_INFO("[ReduceSoleExecutor]templateScratchMultiplier [%u]", templateScratchMultiplier);
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize =
            maxTmpMemSize_ / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    // 单次循环处理的数据量大小
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    HCCL_INFO("[ReduceSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
              "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        maxDataCountPerLoop,
        maxDataSizePerLoop,
        transportBoundDataSize,
        templateScratchMultiplier);
    CHK_PRT_RET(maxDataCountPerLoop == 0,
        HCCL_ERROR("[ReduceSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop is 0"),
        HCCL_E_INTERNAL);
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxDataCountPerLoop + static_cast<u64>(dataCount_ % maxDataCountPerLoop != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxDataCountPerLoop;
        tempAlgParams.count = currDataCount;
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;

        tempAlgParams.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParams.tailSize = tempAlgParams.sliceSize;
        tempAlgParams.inputSliceStride = 0;           // 如果是输入，偏移是算子的output datasize
        tempAlgParams.outputSliceStride = dataSize_;  // 如果是scratch buffer，偏移是单次循环处理的最大数据量

        HCCL_INFO("[ReduceSoleExecutor] loop [%u] tempAlgParams.inputSliceStride [%u],"
                  "tempAlgParams.outputSliceStride [%u] tempAlgParams.sliceSize [%u]",
            loop,
            tempAlgParams.inputSliceStride,
            tempAlgParams.outputSliceStride,
            tempAlgParams.sliceSize);
        HCCL_INFO("[ReduceSoleExecutor] loop [%u] tempAlgParams.buffInfo.inBuffBaseOff [%u],"
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

    HCCL_INFO("[ReduceSoleExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult ReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunchSaveCtx(
    const OpParam &param, const TemplateResource &templateAlgRes, u32 notifyNumOnMainThread)
{
    HCCL_INFO("[ReduceSoleExecutor] loopTimes==1, save fast launch ctx.");
    u32 threadNum = 1;
    u32 ccuKernelNum = templateAlgRes.submitInfos.size();
    if (ccuKernelNum < 1) {
        HCCL_INFO("[ReduceSoleExecutor] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[ReduceSoleExecutor][HcclEngineCtxCreate] threadNum[%llu], ccuKernelNum[%llu]", threadNum, ccuKernelNum);

    u64 size = CcuFastLaunchCtx::GetCtxSize(threadNum, ccuKernelNum);
    // 申请ctx
    void *ctxPtr = nullptr;
    HCCL_INFO("[ReduceSoleExecutor][HcclEngineCtxCreate] Tag[%s], size[%llu]", param.fastLaunchTag, size);
    CHK_RET(HcclEngineCtxCreate(param.hcclComm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, size, &ctxPtr));

    CcuFastLaunchCtx *ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(ctxPtr);
    // 1 算法名
    CHK_SAFETY_FUNC_RET(strcpy_s(ccuFastLaunchCtx->algName, sizeof(ccuFastLaunchCtx->algName), param.algName));
    HCCL_INFO("[ReduceSoleExecutor][FastLaunchSaveCtx] algName[%s]", ccuFastLaunchCtx->algName);

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
HcclResult ReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunch(
        const OpParam &param, const CcuFastLaunchCtx *fastLaunchCtx)
{
    HCCL_INFO("[ReduceSoleExecutor][FastLaunch] Start.");
    TemplateFastLaunchCtx tempFastLaunchCtx;
    // 1 取thread
    ThreadHandle *threads = fastLaunchCtx->GetThreadHandlePtr();
    tempFastLaunchCtx.threads.assign(threads, threads + fastLaunchCtx->threadNum);
    HCCL_INFO("[ReduceSoleExecutor][FastLaunch] threadNum[%llu]", fastLaunchCtx->threadNum);

    // 2 取arg
    CcuKernelSubmitInfo *ccuKernelSubmitInfos = fastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    tempFastLaunchCtx.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + fastLaunchCtx->ccuKernelNum[0]);
    HCCL_INFO("[ReduceSoleExecutor][FastLaunch] ccuKernelNum[%llu]", fastLaunchCtx->ccuKernelNum[0]);
    tempFastLaunchCtx.buffInfo.inputPtr = param.inputPtr;
    tempFastLaunchCtx.buffInfo.outputPtr = param.outputPtr;
    tempFastLaunchCtx.buffInfo.hcclBuff = param.hcclBuff;

    // 3 调template
    std::unique_ptr<InsAlgTemplate> algTemplate = std::make_unique<InsAlgTemplate>();
    CHK_RET(algTemplate->FastLaunch(param, tempFastLaunchCtx));
    HCCL_INFO("[ReduceSoleExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif

// 第二个参数是Reduce的template文件
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_REDUCE, ReduceMesh1D, ReduceSoleExecutor, TopoMatch1D, ReduceMesh1D);
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_REDUCE, ReduceMesh1DTwoShot, ReduceSoleExecutor, TopoMatch1D, ReduceMesh1DTwoShot);
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_REDUCE, ReduceNHR, ReduceSoleExecutor, TopoMatch1D, ReduceNHR);
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_REDUCE, ReduceAicpuReduceNHR, ReduceSoleExecutor, TopoMatch1D, ReduceAicpuReduceNHR);

#ifndef AICPU_COMPILE
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_REDUCE, AivReduceMesh1D, ReduceSoleExecutor, TopoMatch1D, AivTempReduceMesh1D);
#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE, CcuReduceMesh1DMem2Mem, ReduceSoleExecutor, TopoMatch1D, CcuTempReduceMesh1DMem2Mem);
#endif /* !HCCL_CANN_COMPAT_850 */
#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_REDUCE, CcuReduceMesh1D, ReduceSoleExecutor, TopoMatch1D, CcuTempReduceMesh1D);
#endif /* !HCCL_CANN_COMPAT_850 */
#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE, CcuReduceNHR1DMem2Mem, ReduceSoleExecutor, TopoMatch1D, CcuTempReduceNHR1DMem2Mem);
#endif /* !HCCL_CANN_COMPAT_850 */
#endif
}  // namespace ops_hccl