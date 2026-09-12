/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_broadcast_sole_executor.h"
#include <cstring>
#include "ins_temp_broadcast_mesh_1D_two_shot.h"
#include "ins_temp_broadcast_nhr.h"
#include "alg_attrs_registry.h"
#ifndef AICPU_COMPILE
#include "aiv_temp_broadcast_mesh_1D.h"
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_broadcast_mesh_1D_mem2mem.h"
#include "ccu_temp_broadcast_mesh_1D.h"
#include "ccu_temp_broadcast_nhr_1D_mem2mem.h"
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif

#include "alg_attrs_registry.h"
#include "auto_selector_base.h"
#include "hccl_aiv_utils.h"
#include <cstring>
namespace ops_hccl {
constexpr u32 BROADCAST_UBX_AIV_MAX_RANK = 8;

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2BroadcastSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void)comm;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, AlgAttrs{}));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
std::vector<CostModelParam> InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)param;
    // 探测路径直接调 MatchTopo（不走 CalcAlgHierarchyInfoV2 的 CHK_RET）：
    // costmodel 迭代时"不匹配"是正常事件，避免执行路径语义的 ERROR 日志刷屏
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
#ifndef AICPU_COMPILE
    const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(std::string(algName != nullptr ? algName : ""));
#else
    const AlgAttrs* attrs = nullptr;
#endif
    AlgTopoMatch topoMatch;
    HcclResult matchRet
        = (attrs != nullptr) ? topoMatch.MatchTopo(topoInfo, algHierarchyInfo, *attrs) : HcclResult::HCCL_E_PARA;
    if (matchRet != HcclResult::HCCL_SUCCESS) {
        HCCL_INFO("[CalcCostCoeff] algName=%s topo match not support, skip.", algName);
        lastNetType_ = CommTopo::COMM_TOPO_1DMESH;
        lastPortNum_ = {1};
        lastIsPod_ = false;
        lastRankSize_ = 0;
        return {};
    }
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = topoInfo->isPod;
    u32 physIdx = static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]);
    CommTopo netTypeLevel0 = GetPhysicalLevelTopoType(topoInfo, physIdx);
    std::vector<u32> portNumLevel0 = GetPhysicalLevelPortNums(topoInfo, physIdx);
    if (portNumLevel0.empty()) {
        HCCL_WARNING("[CalcCostCoeff] portNum is empty");
        return {};
    }

    // 缓存给 const GetAlgNetMeta 使用（当前分支 GetAlgNetMeta 无 algName 入参，无法重跑 topomatch）
    lastNetType_ = netTypeLevel0;
    lastPortNum_ = portNumLevel0;
    lastIsPod_ = isPod;
    lastRankSize_ = rankSize;
    HCCL_INFO(
        "[CalcCostCoeff] algName=%s rankSize=%d isPod=%d netType=%d portNumSize=%zu", algName, rankSize, isPod,
        static_cast<int>(netTypeLevel0), portNumLevel0.size());
    // broadcast是in-place操作，output==input，故outputBuffer=INPUT；scratch=HCCL_BUFFER
    return InsAlgTemplate::CalcCostCoeff(CalcCostCoeffParam{
        rankSize, 1.0f, netTypeLevel0, BufferType::INPUT, BufferType::INPUT, BufferType::HCCL_BUFFER, portNumLevel0,
        isPod, algName, comm, topoInfo});
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
AlgNetMeta InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)algName;
    (void)param;
    AlgNetMeta meta;
    u32 rankSize = (topoInfo != nullptr) ? topoInfo->userRankSize : lastRankSize_;
    CommTopo netType = lastNetType_;
    meta.netTypes.push_back(netType);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1};
    // SoleMesh: root 数据均分 R 份,每份 1/R; NHR/CLOS: 每步转发等量
    meta.dataRatios = {netType == CommTopo::COMM_TOPO_CLOS ? 1.0f : 1.0f / static_cast<float>(rankSize)};
    meta.rankSizes = {rankSize};
    return meta;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);

    // 调用计算资源的函数
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));
    // 在comb_exector或是parallel_exector中合并两个template的资源
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2BroadcastSoleExecutor][Orchestrate] Orchestrate Start");
    maxTmpMemSize_ = resCtx.cclMem.size; // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    supportSymmetricMemory_ = param.supportSymmetricMemory;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (supportSymmetricMemory_) {
        inputOffset_ = param.inputOffset;
        outputOffset_ = param.outputOffset;
        inputSymWindow_ = param.inputSymWindow;
        outputSymWindow_ = param.outputSymWindow;
    }
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2BroadcastSoleExecutor][Orchestrate]errNo[0x%016llx] Broadcast executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2BroadcastSoleExecutor][OrchestrateLoop] Start");
    // 准备资源
    TemplateResource templateAlgRes;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && remoteRankToChannelInfo_.size() > 0) {
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    templateAlgRes.threads = resCtx.threads;
    templateAlgRes.ccuKernels = resCtx.ccuKernels;
    templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    // 准备数据
    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.inputPtr;
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    if (std::string(param.algName) != "AicpuBroadcastSoleNHR") {
        tempAlgParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
    }
    CHK_PTR_NULL(tempAlgParams.buffInfo.inputPtr);
    CHK_PTR_NULL(tempAlgParams.buffInfo.outputPtr);
    CHK_PTR_NULL(tempAlgParams.buffInfo.hcclBuff.addr);
    tempAlgParams.buffInfo.hcclBuffBaseOff = 0;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    // 不需要重复
    tempAlgParams.repeatNum = 1;
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;

    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS
        && (resCtx.topoInfo.isPod || std::string(param.algName) == "AicpuBroadcastSoleNHRMultiLink")) {
        CHK_RET(algTemplate->SetchannelsPerRank(templateAlgRes.channels));
    }

    // 根据CCL Buffer大小和UB_MAX_DATA_SIZE，计算出一轮中最多能输出多少数据
    u32 templateScratchMultiplier
        = algTemplate->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.outBuffType);
    u64 maxDataSizePerLoop = 0;
    maxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE; // algTemplate->CalcLoopMaxCount();
    HCCL_INFO("[InsV2BroadcastSoleExecutor]maxTmpMemSize_ [%llu]", maxTmpMemSize_);
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize
            = maxTmpMemSize_ / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;

    u64 dataSize = dataCount_ * dataTypeSize_;

    u64 maxLoopOutputSize = maxDataCountPerLoop * dataTypeSize_;

    HCCL_INFO(
        "[InsV2BroadcastSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
        "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    CHK_PRT_RET(
        maxDataCountPerLoop == 0,
        HCCL_ERROR("[InsV2BroadcastSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop is 0"), HCCL_E_INTERNAL);

    u64 loopTimes = dataSize / maxLoopOutputSize + static_cast<u64>(dataSize % maxLoopOutputSize != 0);
    // 如果是对称内存，每次传输的大小不受cclbuffer和UB_MAX_DATA_SIZE的限制
    if (param.supportSymmetricMemory) {
        loopTimes = 1;
        tempAlgParams.enableRemoteMemAccess = true;
        HCCL_INFO("[InsV2BroadcastSoleExecutor][OrchestrateOpbase] %s: symmetric memory enabled", param.algName);
    }
    HCCL_INFO(
        "[InsV2BroadcastSoleExecutor][OrchestrateOpbase] myRank_[%llu], dataSize[%llu], dataTypeSize_[%llu], "
        "maxDataCountPerLoop[%llu], loopTimes[%llu]",
        myRank_, dataSize, dataTypeSize_, maxDataCountPerLoop, loopTimes);
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currloopOffset = loop * maxLoopOutputSize;
        u64 currSize = (loop == (loopTimes - 1)) ? dataSize - currloopOffset : maxLoopOutputSize;
        tempAlgParams.count = currSize / dataTypeSize_;
        // 当前搬运的数据片
        tempAlgParams.buffInfo.inBuffBaseOff = currloopOffset;
        tempAlgParams.buffInfo.outBuffBaseOff = currloopOffset;

        tempAlgParams.sliceSize = currSize;
        tempAlgParams.tailSize = tempAlgParams.sliceSize;
        HCCL_DEBUG(
            "[InsV2BroadcastSoleExecutor] Rank[%d], before generating instruction queues, currSize[%llu], "
            "currOffset[%llu].",
            myRank_, currSize, currloopOffset);
        CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes));
        HCCL_DEBUG(
            "[InsV2BroadcastSoleExecutor] Rank[%d], done generating instruction queues, currSize[%llu], "
            "currOffset[%llu].",
            myRank_, currSize, currloopOffset);
    }
#ifndef AICPU_COMPILE
    if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU && param.opMode != OpMode::OFFLOAD) {
        CHK_RET(FastLaunchSaveCtx(param, templateAlgRes, resCtx.notifyNumOnMainThread));
    }
#endif
    HCCL_INFO("[InsV2BroadcastSoleExecutor][OrchestrateLoop] End.");

    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunchSaveCtx(
    const OpParam& param, const TemplateResource& templateAlgRes, u32 notifyNumOnMainThread) const
{
    HCCL_INFO("[InsV2BroadcastSoleExecutor][FastLaunchSaveCtx] loopTimes==1, save fast launch ctx.");
    u32 threadNum = templateAlgRes.submitInfos.size();
    u32 ccuKernelNum = templateAlgRes.submitInfos.size();
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsV2BroadcastSoleExecutor][FastLaunchSaveCtx] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "[InsV2BroadcastSoleExecutor][FastLaunchSaveCtx] threadNum[%llu], ccuKernelNum[%llu]", threadNum, ccuKernelNum);

    u64 size = CcuFastLaunchCtx::GetCtxSize(threadNum, ccuKernelNum);
    // 申请ctx
    void* ctxPtr = nullptr;
    HCCL_INFO("[InsV2BroadcastSoleExecutor][FastLaunchSaveCtx] Tag[%s], size[%llu]", param.fastLaunchTag, size);
    CHK_RET(HcclEngineCtxCreate(param.hcclComm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, size, &ctxPtr));

    CcuFastLaunchCtx* ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(ctxPtr);
    // 1 算法名
    CHK_SAFETY_FUNC_RET(strcpy_s(ccuFastLaunchCtx->algName, sizeof(ccuFastLaunchCtx->algName), param.algName));
    HCCL_INFO("[InsV2BroadcastSoleExecutor][FastLaunchSaveCtx] algName[%s]", ccuFastLaunchCtx->algName);

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
    for (int i = 0; i < ccuKernelNum; i++) {
        kernelSubmitInfos[i] = templateAlgRes.submitInfos[i];
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunch(
    const OpParam& param, const CcuFastLaunchCtx* fastLaunchCtx)
{
    HCCL_INFO("[InsV2BroadcastSoleExecutor][FastLaunch] Start.");
    TemplateFastLaunchCtx tempFastLaunchCtx;
    // 1 取thread
    ThreadHandle* threads = fastLaunchCtx->GetThreadHandlePtr();
    tempFastLaunchCtx.threads.assign(threads, threads + fastLaunchCtx->threadNum);
    HCCL_INFO("[InsV2BroadcastSoleExecutor][FastLaunch] threadNum[%llu]", fastLaunchCtx->threadNum);

    // 2 取arg
    CcuKernelSubmitInfo* ccuKernelSubmitInfos = fastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    tempFastLaunchCtx.ccuKernelSubmitInfos.assign(
        ccuKernelSubmitInfos, ccuKernelSubmitInfos + fastLaunchCtx->ccuKernelNum[0]);
    HCCL_INFO("[InsV2BroadcastSoleExecutor][FastLaunch] ccuKernelNum[%llu]", fastLaunchCtx->ccuKernelNum[0]);
    tempFastLaunchCtx.buffInfo.inputPtr = param.inputPtr;
    tempFastLaunchCtx.buffInfo.outputPtr = param.inputPtr;
    tempFastLaunchCtx.buffInfo.hcclBuff = param.hcclBuff;

    // 3 调template
    std::unique_ptr<InsAlgTemplate> algTemplate = std::make_unique<InsAlgTemplate>();
    CHK_RET(algTemplate->FastLaunch(param, tempFastLaunchCtx));
    HCCL_INFO("[InsV2BroadcastSoleExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif

REGISTER_ALG_ATTRS(
    AicpuBroadcastSoleMeshTwoShot, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.isSupportLevel0PcieMix = true; topo.requireAllMeshConnected = true; topo.maxTopoLevelNum = 1;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->level0Topo != Level0Shape::MESH_1D_CLOS
               || AutoSelectorBase::IsLayerAllConnetedWithTopo(topo, 0, CommTopo::COMM_TOPO_1DMESH);
    };
    op.unsupportedDataTypes = UNSUPPORTED_64BIT;);
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_BROADCAST, AicpuBroadcastSoleMeshTwoShot, InsV2BroadcastSoleExecutor, TopoMatchOneLevel,
    InsTempBroadcastMesh1DTwoShot);
REGISTER_ALG_ATTRS(
    AicpuBroadcastSoleNHR, topo.isSupportLevel1Nhr = true;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
    op.unsupportedDataTypes = UNSUPPORTED_64BIT;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        if (topo->level0Topo != Level0Shape::MESH_1D_CLOS) {
            return true;
        }
        return !topo->level0PcieMix
               && !AutoSelectorBase::IsLayerAllConnetedWithTopo(topo, 0, CommTopo::COMM_TOPO_1DMESH);
    });
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_BROADCAST, AicpuBroadcastSoleNHR, InsV2BroadcastSoleExecutor, TopoMatchOneLevel,
    InsTempBroadcastNHR);
REGISTER_ALG_ATTRS(AicpuBroadcastSoleNHRMultiLink,
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
                   topo.isSupportLevel1Nhr = true; topo.maxTopoLevelNum = 2);
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_BROADCAST, AicpuBroadcastSoleNHRMultiLink, InsV2BroadcastSoleExecutor, TopoMatchOneLevel,
    InsTempBroadcastNHR);

#ifndef AICPU_COMPILE
REGISTER_ALG_ATTRS(
    AivBroadcastSoleMesh, topo.maxSupportRankSize = MAX_RANK_SIZE;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2; topo.isSupportLevel1Nhr = true; topo.isSupportLevel0PcieMix = true;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        if (topo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            return topo->level0PcieMix || topo->userRankSize <= BROADCAST_UBX_AIV_MAX_RANK;
        }
        return true;
    };);
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_BROADCAST, AivBroadcastSoleMesh, InsV2BroadcastSoleExecutor, TopoMatchOneLevel,
    AivTempBroadcastMesh1D);
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_ALG_ATTRS(
    CcuSchedBroadcastSoleMesh, topo.maxSupportRankSize = CCU_SCHED_MAX_RANK_SIZE;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2;
    topo.isSupportLevel0PcieMix = true; topo.requireAllMeshConnected = true;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        if (topo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            return topo->level0PcieMix;
        }
        return true;
    };
    op.unsupportedDataTypes = UNSUPPORTED_INT8_AND_64BIT;);
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_BROADCAST, CcuSchedBroadcastSoleMesh, InsV2BroadcastSoleExecutor, TopoMatchOneLevel,
    CcuTempBroadcastMesh1DMem2Mem);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_ALG_ATTRS(
    CcuMSBroadcastSoleMesh, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.maxTopoLevelNum = 1; topo.isSupportLevel0PcieMix = true; topo.requireAllMeshConnected = true;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        if (topo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            return topo->level0PcieMix;
        }
        return true;
    };
    op.unsupportedDataTypes = UNSUPPORTED_INT8_AND_64BIT;);
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_BROADCAST, CcuMSBroadcastSoleMesh, InsV2BroadcastSoleExecutor, TopoMatchOneLevel,
    CcuTempBroadcastMesh1D);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_ALG_ATTRS(CcuSchedBroadcastSoleNHR, topo.maxSupportRankSize = CCU_SCHED_MAX_RANK_SIZE;
                   topo.isSupportLevel1Nhr = true; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS;
                   op.unsupportedDataTypes = UNSUPPORTED_INT8_AND_64BIT);
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_BROADCAST, CcuSchedBroadcastSoleNHR, InsV2BroadcastSoleExecutor, TopoMatchOneLevel,
    CcuTempBroadcastNHR1DMem2Mem);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif
} // namespace ops_hccl
