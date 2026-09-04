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
#include "ins_temp_all_gather_mesh_1D.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_all_gather_nhr_dpu.h"
#ifndef AICPU_COMPILE
#include "aiv_temp_all_gather_mesh_1D.h"
#include "ccu_temp_all_gather_mesh_1D_mem2mem.h"
#include "ccu_temp_all_gather_mesh_1D.h"
#include "ccu_temp_all_gather_nhr_1D_mem2mem.h"
#include "ccu_temp_all_gather_2dies_mesh_1d_mem2mem.h"
#include "ccu_temp_all_gather_2dies_mesh_1D.h"
#include "ccu_temp_all_gather_nhr_1D_multi_jetty_mem2mem.h"
#endif
#include "ccu_temp_all_gather_concurrent_mesh_mem2mem_nhr.h"
#include "topo_match_one_level.h"
#include "topo_match_concurrent_v2.h"
#include "alg_attrs_registry.h"
#include "hccl_aiv_utils.h"
#include "auto_selector_base.h"
#include "hccl_res.h"
namespace ops_hccl {

constexpr u32 MAX_RANK_NUM_FOR_CONCURRENT_ALGO = 4;
constexpr u32 AG_UBX_AIV_BIGDATA_RANK_UPPER = 8; // 与selector保持一致：UBX大数据场景AIV算法rank上限
constexpr u32 AG_UBX_AIV_BIGDATA_RANK_LOWER = 4; // 与selector保持一致：UBX大数据场景AIV算法rank下限
constexpr u32 DEVICE_NUM_PER_MODULE_8 = 8;
constexpr u32 CCU_MAX_SIZE = 64;
constexpr u64 AG_2P_DETOUR_DATA_SIZE = 4 * 1024 * 1024;

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2AllGatherSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void)comm;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, AlgAttrs{}));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
std::vector<CostModelParam> InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)algName;
    (void)comm;
    AlgHierarchyInfoForAllLevel algHierarchyInfo; // TODO: unused for now, costmodel fallback
    (void)algHierarchyInfo;
    // TODO: CalcAlgHierarchyInfo(comm, topoInfo, algHierarchyInfo);
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = true;
    auto rs = CostModelManager::Global()->CalcRankSizeByTopo(topoInfo);
    u32 rankSizeLevel0 = rs.level0;
    // TODO: CommTopo netTypeLevel0 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[0]);
    CommTopo netTypeLevel0 = topoInfo->topoLevelNums > 1 ? CommTopo::COMM_TOPO_CLOS : CommTopo::COMM_TOPO_1DMESH;
    // TODO: std::vector<u32> portNumLevel0 = GetPortNumLevel(topoInfo, algHierarchyInfo.index[0]);
    std::vector<u32> portNumLevel0 = {1};
    HCCL_INFO(
        "[CalcCostCoeff] rankSize=%d, rankSizeLevel0=%d, portNumLevel0=%d, netTypeLevel0=%d", rankSize, rankSizeLevel0,
        portNumLevel0, static_cast<int>(netTypeLevel0));
    return InsAlgTemplate::CalcCostCoeff(CalcCostCoeffParam{
        rankSize, 1.0f, netTypeLevel0, BufferType::INPUT, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
        portNumLevel0, isPod});
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
AlgNetMeta InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const
{
    (void)param;
    auto rs = CostModelManager::Global()->CalcRankSizeByTopo(topoInfo);
    u32 rankSizeLevel0 = rs.level0;
    u32 rankSizeLevel1 = rs.level1;
    (void)rankSizeLevel1;
    // TODO: CommTopo netTypeLevel0 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[0]);
    CommTopo netTypeLevel0 = CommTopo::COMM_TOPO_1DMESH;
    AlgNetMeta meta;
    meta.netTypes.push_back(netTypeLevel0);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1};
    meta.dataRatios = {1.0f};
    meta.rankSizes = {rankSizeLevel0};
    return meta;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    // 调用计算资源的函数 AicpuAllGatherSoleNHR 在计算资源时按照channels取最大，实际使用资源由SetchannelsPerRank使能
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));
    myRank_ = topoInfo->userRank;
    HCCL_DEBUG(
        "[InsV2AllGatherSoleExecutor][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u], "
        "channels[%u]",
        myRank_, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
        resourceRequest.channels.size());
    for (auto i = 0; i < resourceRequest.notifyNumPerThread.size(); i++) {
        HCCL_DEBUG(
            "[InsV2AllGatherSoleExecutor][CalcRes] myRank[%u], notifyNumPerThread[%u]=[%u]", myRank_, i,
            resourceRequest.notifyNumPerThread[i]);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllGatherSoleExecutor][Orchestrate] Orchestrate Start");
    myRank_ = resCtx.topoInfo.userRank;
    supportSymmetricMemory_ = param.supportSymmetricMemory;

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
    HCCL_DEBUG(
        "[InsV2AllGatherSoleExecutor][Orchestrate] myRank[%u], threadsSize[%lu], "
        "dataCount[%llu], dataTypeSize[%lu]",
        myRank_, threads_.size(), dataCount_, dataTypeSize_);
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2AllGatherSoleExecutor][Orchestrate]errNo[0x%016llx] All Gather executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
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
    templateAlgRes.dieSplitRatio = resCtx.dieSplitRatio;
    // 准备数据
    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    tempAlgParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
    // 不需要重复
    tempAlgParams.repeatNum = 1;
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;
    HCCL_INFO(
        "[InsV2AllGatherSoleExecutor][OrchestrateLoop] myRank[%u], inputPtr[%#llx] outputPtr[%#llx], "
        "cclAddr[%#llx], cclSize[%llu], channelSize[%lu], threadSize[%lu], ",
        myRank_, param.inputPtr, param.outputPtr, resCtx.cclMem.addr, resCtx.cclMem.size,
        templateAlgRes.channels.size(), templateAlgRes.threads.size());
    // 构建template
    InsAlgTemplate algTemplate(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
    u32 templateScratchMultiplier
        = algTemplate.CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.outBuffType);
    maxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size;
    if (param.engine == COMM_ENGINE_AICPU_TS && std::string(param.algName) != "AicpuAllGatherSoleNHR") {
        CHK_RET(algTemplate.SetchannelsPerRank(templateAlgRes.channels));
    }
    // 中转内存单次最多能够接受的output count，注意是count不是size
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 maxDataSizePerLoop = 0;
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize
            = maxTmpMemSize_ / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    u64 maxCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;

    // 如果是对称内存，每次传输的大小不受cclbuffer和UB_MAX_DATA_SIZE的限制
    if (param.supportSymmetricMemory) {
        loopTimes = 1;
        tempAlgParams.enableRemoteMemAccess = true;
        HCCL_INFO("[InsV2AllGatherSoleExecutor][OrchestrateLoop] %s: symmetric memory enabled", param.algName);
    }
    HCCL_INFO(
        "[InsV2AllGatherSoleExecutor][OrchestrateLoop] myRank[%u], templateScratchMultiplier[%u] "
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

        HCCL_DEBUG(
            "[InsV2AllGatherSoleExecutor] myRank[%u], loop [%u] tempAlgParams.inputSliceStride [%u],"
            "tempAlgParams.outputSliceStride [%u] tempAlgParams.sliceSize [%u]",
            myRank_, loop, tempAlgParams.inputSliceStride, tempAlgParams.outputSliceStride, tempAlgParams.sliceSize);
        HCCL_DEBUG(
            "[InsV2AllGatherSoleExecutor] myRank[%u], loop [%u] tempAlgParams.buffInfo.inBuffBaseOff [%u],"
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
    const OpParam& param, const TemplateResource& templateAlgRes, u32 notifyNumOnMainThread) const
{
    HCCL_INFO("[InsV2AllGatherSoleExecutor] loopTimes==1, save fast launch ctx.");
    // 按 template 实际申请的线程数保存,兼容单线程算法与多线程算法(NHR 2 线程、concurrent 3 线程)
    u32 threadNum = static_cast<u32>(templateAlgRes.threads.size());
    u32 ccuKernelNum = templateAlgRes.submitInfos.size();
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsV2AllGatherSoleExecutor] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "[InsV2AllGatherSoleExecutor][HcclEngineCtxCreate] threadNum[%llu], ccuKernelNum[%llu]", threadNum,
        ccuKernelNum);

    u64 size = CcuFastLaunchCtx::GetCtxSize(threadNum, ccuKernelNum);
    // 申请ctx
    void* ctxPtr = nullptr;
    HCCL_INFO("[InsV2AllGatherSoleExecutor][HcclEngineCtxCreate] Tag[%s], size[%llu]", param.fastLaunchTag, size);
    CHK_RET(HcclEngineCtxCreate(param.hcclComm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, size, &ctxPtr));

    CcuFastLaunchCtx* ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(ctxPtr);
    // 1 算法名
    CHK_SAFETY_FUNC_RET(strcpy_s(ccuFastLaunchCtx->algName, sizeof(ccuFastLaunchCtx->algName), param.algName));
    HCCL_INFO("[InsV2AllGatherSoleExecutor][FastLaunchSaveCtx] algName[%s]", ccuFastLaunchCtx->algName);

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
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunch(
    const OpParam& param, const CcuFastLaunchCtx* fastLaunchCtx)
{
    HCCL_INFO("[InsV2AllGatherSoleExecutor][FastLaunch] Start.");
    TemplateFastLaunchCtx tempFastLaunchCtx;
    // 1 取thread
    ThreadHandle* threads = fastLaunchCtx->GetThreadHandlePtr();
    tempFastLaunchCtx.threads.assign(threads, threads + fastLaunchCtx->threadNum);
    HCCL_INFO("[InsV2AllGatherSoleExecutor][FastLaunch] threadNum[%llu]", fastLaunchCtx->threadNum);

    // 2 取arg
    CcuKernelSubmitInfo* ccuKernelSubmitInfos = fastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    tempFastLaunchCtx.ccuKernelSubmitInfos.assign(
        ccuKernelSubmitInfos, ccuKernelSubmitInfos + fastLaunchCtx->ccuKernelNum[0]);
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

REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLGATHER, AicpuAllGatherSoleMesh, InsV2AllGatherSoleExecutor, TopoMatchOneLevel,
    InsTempAllGatherMesh1D);
REGISTER_ALG_ATTRS(
    AicpuAllGatherSoleMesh, topo.isSupportLevel0PcieMix = true; topo.requireAllMeshConnected = true;
    topo.maxTopoLevelNum = 1; topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        bool isEqual = false;
        if (topo->level0Topo != Level0Shape::MESH_1D_CLOS) {
            return false;
        }
        AutoSelectorBase::CheckMeshNumEqualToClosNum(topo, isEqual);
        return topo->level0Topo == Level0Shape::MESH_1D_CLOS && isEqual
               && topo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO;
    });

REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLGATHER, AicpuAllGatherSoleMeshConcur, InsV2AllGatherSoleExecutor, TopoMatchOneLevel,
    InsTempAllGatherMesh1D1DZAxisDetour);
REGISTER_ALG_ATTRS(AicpuAllGatherSoleMeshConcur, topo.maxTopoLevelNum = 1;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D);

REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLGATHER, AicpuAllGatherSoleNHR, InsV2AllGatherSoleExecutor, TopoMatchOneLevel,
    InsTempAllGatherNHR);
REGISTER_ALG_ATTRS(
    AicpuAllGatherSoleNHR, topo.isSupportLevel1Nhr = true;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        // UBX矩形拓扑(8/16P等)：老selector小数据量/非对称场景会选SoleNHR，需与
        // PipeLineUBX/MultiJetty同条件命中priority，否则被ApplyTopoPriority移出候选
        if (topo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            bool isEqual = false;
            bool isMultiple = false;
            AutoSelectorBase::CheckMeshNumEqualToClosNum(topo, isEqual);
            AutoSelectorBase::CheckClosNumMultipleOfMeshNum(topo, isMultiple);
            return !(isEqual && topo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) && isMultiple;
        }
        // 该分支对应旧selector多层(topoLevelNums > 1)场景，单层时localNetInsSizeOfLayer仅1个元素，
        // 访问[1]会越界；且单层MESH_1D旧selector不选SoleNHR，直接返回false
        if (topo->topoLevelNums <= 1 || topo->netLayerDetails.localNetInsSizeOfLayer.size() < 2) {
            return false;
        }
        return topo->topLevelUboe
               && !(
                   (topo->level0Symmetric && topo->level1Symmetric)
                   && topo->deviceNumPerModule == DEVICE_NUM_PER_MODULE_8)
               && !(
                   !(topo->level0Symmetric && topo->level1Symmetric)
                   || topo->netLayerDetails.localNetInsSizeOfLayer[1] == 1)
               && topo->Level0Nhr && topo->netLayerDetails.localNetInsSizeOfLayer[0] == 1;
    });

REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLGATHER, AicpuAllGatherSoleNHRMultiLink, InsV2AllGatherSoleExecutor, TopoMatchOneLevel,
    InsTempAllGatherNHR);
REGISTER_ALG_ATTRS(
    AicpuAllGatherSoleNHRMultiLink, topo.supportLevel0Topos = LEVEL0_TOPO_CLOS;
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->level0Topo == Level0Shape::CLOS;
    });

#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLGATHER, CcuSchedAllGatherSoleMesh, InsV2AllGatherSoleExecutor, TopoMatchOneLevel,
    CcuTempAllGatherMesh1DMem2Mem);
REGISTER_ALG_ATTRS(
    CcuSchedAllGatherSoleMesh, topo.isSupportLevel0PcieMix = true; topo.requireAllMeshConnected = true;
    topo.maxTopoLevelNum = 2; topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
    op.isSupportInplace = false; topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        // UBX参与候选
        return topo->topoLevelNums == 1 && topo->level0Topo == Level0Shape::MESH_1D_CLOS;
    };

    op.opCustomCheck = [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) -> bool {
        return !(topo->topoLevelNums == 2 && topo->userRankSize > CCU_MAX_SIZE);
    });
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)

#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLGATHER, CcuMSAllGatherSoleMesh, InsV2AllGatherSoleExecutor, TopoMatchOneLevel,
    CcuTempAllGatherMesh1D);
REGISTER_ALG_ATTRS(
    CcuMSAllGatherSoleMesh, topo.maxTopoLevelNum = 1;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS; topo.isSupportLevel0PcieMix = true;
    topo.requireAllMeshConnected = true; op.isSupportInplace = false;
    // UBX定制机型场景命中topoPriority，避免被其他算法的topoPriorityCheck提前淘汰
    // （topo阶段无数据量信息，数据量条件由opCustomCheck/opPriorityCheck在op阶段保证）
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->topoLevelNums == 1 && topo->level0Topo == Level0Shape::MESH_1D_CLOS && !topo->level0PcieMix;
    });
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)

#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLGATHER, CcuSchedAllGatherSoleNHR, InsV2AllGatherSoleExecutor, TopoMatchOneLevel,
    CcuTempAllGatherNHR1DMem2Mem);
REGISTER_ALG_ATTRS(
    CcuSchedAllGatherSoleNHR, topo.isSupportLevel1Nhr = true; topo.maxTopoLevelNum = 2;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS; op.isSupportInplace = false;
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        // CLOS定制机型（单层/多层）命中，避免被其他算法的topoPriorityCheck提前淘汰
        if (topo->level0Topo == Level0Shape::CLOS) {
            return true;
        }
        return !topo->netLayerDetails.localNetInsSizeOfLayer.empty()
               && topo->netLayerDetails.localNetInsSizeOfLayer[0] == 1;
    };
    op.opCustomCheck = [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) -> bool {
        return !(topo->topoLevelNums == 2 && topo->userRankSize > CCU_MAX_SIZE);
    });
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif

#ifndef AICPU_COMPILE
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLGATHER, AivAllGatherSoleMesh, InsV2AllGatherSoleExecutor, TopoMatchOneLevel,
    AivTempAllGatherMesh1D);
REGISTER_ALG_ATTRS(
    AivAllGatherSoleMesh, topo.maxTopoLevelNum = 2;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.isSupportLevel0PcieMix = true; topo.isSupportLevel1Nhr = true;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->userRankSize <= MAX_RANK_SIZE;
    };
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->userRankSize <= MAX_RANK_SIZE;
    };
    op.opCustomCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails* topo) -> bool {
        void* bufAddr = nullptr;
        uint64_t bufSize = 0;
        if (HcclGetHcclBuffer(opParam.hcclComm, &bufAddr, &bufSize) != HCCL_SUCCESS) {
            return false;
        }
        u64 dataSize = opParam.DataDes.count * DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
        return dataSize <= bufSize * AIV_MAX_CCL_LOOP_NUM;
    });

#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLGATHER, CcuMSAllGatherSoleMesh2Die, InsV2AllGatherSoleExecutor, TopoMatchOneLevel,
    CcuTempAllGather2DiesMesh1D);
REGISTER_ALG_ATTRS(CcuMSAllGatherSoleMesh2Die, topo.maxTopoLevelNum = 1; topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
                   topo.supportLevel0MeshTypes = MESH_TYPE_TWO_DIE_REGULAR; op.isSupportInplace = false);
#endif // !HCCL_CANN_COMPAT_850
#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLGATHER, CcuSchedAllGatherSoleMesh2Die, InsV2AllGatherSoleExecutor, TopoMatchOneLevel,
    CcuTempAllGather2DiesMeshMem2Mem1D);
REGISTER_ALG_ATTRS(CcuSchedAllGatherSoleMesh2Die, topo.isSupportLevel0PcieMix = true;
                   topo.requireAllMeshConnected = true; topo.maxTopoLevelNum = 1;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
                   topo.supportLevel0MeshTypes = MESH_TYPE_TWO_DIE_REGULAR; op.isSupportInplace = false);
#endif // !HCCL_CANN_COMPAT_850
#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLGATHER, CcuSchedAllGatherSoleNHRMultiLink, InsV2AllGatherSoleExecutor, TopoMatchOneLevel,
    CcuTempAllGatherNHR1DMultiJettyMem2Mem);
REGISTER_ALG_ATTRS(
    CcuSchedAllGatherSoleNHRMultiLink, topo.maxTopoLevelNum = 1; topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS;
    op.isSupportInplace = false; topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        bool isEqual = false;
        bool isMultiple = false;
        AutoSelectorBase::CheckMeshNumEqualToClosNum(topo, isEqual);
        AutoSelectorBase::CheckClosNumMultipleOfMeshNum(topo, isMultiple);
        return !(isEqual && topo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) && !isMultiple;
    };);

#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)

#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLGATHER, CcuSchedAllGatherSoleMeshConcur, InsV2AllGatherSoleExecutor, TopoMatchConcurrentV2,
    CcuTempAllGatherConcurrentMeshMem2MemNHR);
REGISTER_ALG_ATTRS(
    CcuSchedAllGatherSoleMeshConcur, topo.isSupportLevel0PcieMix = true; topo.requireAllMeshConnected = true;
    topo.maxTopoLevelNum = 1; topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.supportDevTypes = {HcclDevType::DEV_TYPE_960}; topo.topoCustomCheck =
                                                            [](const TopoInfoWithNetLayerDetails* topo) {
                                                                return topo->netLayerDetails.netLayerNum > 1;
                                                            };
    topo.topoPriorityCheck =
        [](const TopoInfoWithNetLayerDetails* topo) {
            return topo->netLayerDetails.netLayerNum > 1;
        };
    op.isSupportInplace = false;);
#endif // !HCCL_CANN_COMPAT_850
#endif // AICPU_COMPILE

} // namespace ops_hccl
