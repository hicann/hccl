/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_scatter_sole_executor.h"
#include "ins_temp_reduce_scatter_mesh_1D.h"
#include "aiv_temp_reduce_scatter_mesh_1D.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "ins_temp_reduce_scatter_mesh_1D_meshchunk.h"
#include "ins_temp_reduce_scatter_aicpu_reduce_nhr.h"
#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "ccu_temp_reduce_scatter_concurrent_mesh_nhr.h"
#include "topo_match_concurrent.h"
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_reduce_scatter_mesh_1D_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh_1D.h"
#include "ccu_temp_reduce_scatter_nhr_1D_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh_1D_2die_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh2die.h"
#include "ccu_temp_reduce_scatter_nhr_1D_multi_jetty_mem2mem.h"
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif

#include "alg_attrs_registry.h"
#include "auto_selector_base.h"
namespace ops_hccl {
#include "hccl_aiv_utils.h"
constexpr u32 DEVICE_NUM_PER_MODULE_8 = 8;
constexpr u32 MAX_RANK_NUM_FOR_CONCURRENT_ALGO = 4;
constexpr u32 RS_CCU_2DIE_RANK_SIZE = 16;
constexpr u32 RS_CCU_2DIE_FRAME_NUM = 2;
constexpr u64 TMP_MEM_RESERVE_SIZE = 1 * 1024 * 1024;

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2ReduceScatterSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void)comm;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, AlgAttrs{}));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
std::vector<CostModelParam> InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcCostCoeff(
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
    CommTopo netTypeLevel0 = (rankSize <= 8) ? CommTopo::COMM_TOPO_1DMESH : CommTopo::COMM_TOPO_CLOS;
    // TODO: std::vector<u32> portNumLevel0 = GetPortNumLevel(topoInfo, algHierarchyInfo.index[0]);
    std::vector<u32> portNumLevel0 = {1};
    HCCL_INFO(
        "[CalcCostCoeff] rankSize=%d, rankSizeLevel0=%d, portNumLevel0=%d, netTypeLevel0=%d", rankSize, rankSizeLevel0,
        portNumLevel0, static_cast<int>(netTypeLevel0));
    return InsAlgTemplate::CalcCostCoeff(CalcCostCoeffParam{
        rankSize, 1.0f, netTypeLevel0, BufferType::INPUT, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
        portNumLevel0, isPod, algName});
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
AlgNetMeta InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const
{
    (void)param;
    auto rs = CostModelManager::Global()->CalcRankSizeByTopo(topoInfo);
    u32 rankSizeLevel0 = rs.level0;
    u32 rankSizeLevel1 = rs.level1;
    (void)rankSizeLevel0;
    (void)rankSizeLevel1;
    u32 rankSize = topoInfo->userRankSize;
    // TODO: CommTopo netTypeLevel0 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[0]);
    CommTopo netTypeLevel0 = CommTopo::COMM_TOPO_1DMESH;
    AlgNetMeta meta;
    meta.netTypes.push_back(netTypeLevel0);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1};
    meta.dataRatios = {1.0f};
    meta.rankSizes = {rankSize};
    return meta;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    // 调用计算资源的函数 AicpuReduceScatterSoleNHR 在计算资源时按照channels取最大，实际使用资源由SetchannelsPerRank使能
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2ReduceScatterSoleExecutor][Orchestrate] Orchestrate Start");
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
    // 对dataTypeSize是否为0进行校验
    CHK_PRT_RET(
        dataTypeSize_ == 0, HCCL_ERROR("[InsV2ReduceScatterSoleExecutor][Orchestrate] dataTypeSize is 0"),
        HCCL_E_INTERNAL);
    dataSize_ = dataCount_ * dataTypeSize_;

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2ReduceScatterSoleExecutor][Orchestrate]errNo[0x%016llx] Reduce scatter executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2ReduceScatterSoleExecutor][OrchestrateLoop] Start");
    // 准备资源
    TemplateResource templateAlgRes;
    if (param.engine == COMM_ENGINE_CCU) {
        templateAlgRes.ccuKernels = resCtx.ccuKernels;
    }
    if (remoteRankToChannelInfo_.size() > 0) {
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    templateAlgRes.threads = resCtx.threads;
    templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    templateAlgRes.dieSplitRatio = resCtx.dieSplitRatio;
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
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
    u32 templateScratchMultiplier
        = algTemplate->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.outBuffType);
    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS && std::string(param.algName) != "AicpuReduceScatterSoleNHR") {
        algTemplate->SetchannelsPerRank(templateAlgRes.channels);
    }

    // 计算最小传输大小
    u64 maxDataSizePerLoop = 0;
    maxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size;
    if (param.engine != COMM_ENGINE_AIV) {
        maxTmpMemSize_ = maxTmpMemSize_ - TMP_MEM_RESERVE_SIZE;
    }
    u64 transportBoundDataSize = (param.engine == CommEngine::COMM_ENGINE_AICPU_TS) ? maxTmpMemSize_ : UB_MAX_DATA_SIZE;
    HCCL_INFO("[InsV2ReduceScatterSoleExecutor]maxTmpMemSize_ [%u]", maxTmpMemSize_);
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize
            = maxTmpMemSize_ / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    // 单次循环处理的数据量大小
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    HCCL_INFO(
        "[InsV2ReduceScatterSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
        "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    CHK_PRT_RET(
        maxDataCountPerLoop == 0,
        HCCL_ERROR("[InsV2ReduceScatterSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop is 0"), HCCL_E_INTERNAL);
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxDataCountPerLoop + static_cast<u64>(dataCount_ % maxDataCountPerLoop != 0);
    tempAlgParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxDataCountPerLoop;
        tempAlgParams.count = currDataCount;
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;

        tempAlgParams.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParams.tailSize = tempAlgParams.sliceSize;
        // 这里的stride当成传统意义上的stride 间隔
        tempAlgParams.inputSliceStride = dataSize_; // 如果是输入，偏移是算子的output datasize
        tempAlgParams.outputSliceStride = 0;

        HCCL_INFO(
            "[InsV2ReduceScatterSoleExecutor] loop [%u] tempAlgParams.inputSliceStride [%u],"
            "tempAlgParams.outputSliceStride [%u] tempAlgParams.sliceSize [%u]",
            loop, tempAlgParams.inputSliceStride, tempAlgParams.outputSliceStride, tempAlgParams.sliceSize);
        HCCL_INFO(
            "[InsV2ReduceScatterSoleExecutor] loop [%u] tempAlgParams.buffInfo.inBuffBaseOff [%u],"
            "tempAlgParams.buffInfo.outBuffBaseOff [%u]",
            loop, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.outBuffBaseOff);
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

    HCCL_INFO("[InsV2ReduceScatterSoleExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunchSaveCtx(
    const OpParam& param, const TemplateResource& templateAlgRes, u32 notifyNumOnMainThread) const
{
    HCCL_INFO("[InsV2ReduceScatterSoleExecutor] loopTimes==1, save fast launch ctx.");
    u32 threadNum = templateAlgRes.threads.size();
    u32 ccuKernelNum = templateAlgRes.submitInfos.size();
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsV2ReduceScatterSoleExecutor] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "[InsV2ReduceScatterSoleExecutor][HcclEngineCtxCreate] threadNum[%llu], ccuKernelNum[%llu]", threadNum,
        ccuKernelNum);

    u64 size = CcuFastLaunchCtx::GetCtxSize(threadNum, ccuKernelNum);
    // 申请ctx
    void* ctxPtr = nullptr;
    HCCL_INFO("[InsV2ReduceScatterSoleExecutor][HcclEngineCtxCreate] Tag[%s], size[%llu]", param.fastLaunchTag, size);
    CHK_RET(HcclEngineCtxCreate(param.hcclComm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, size, &ctxPtr));

    CcuFastLaunchCtx* ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(ctxPtr);
    // 1 算法名
    CHK_SAFETY_FUNC_RET(strcpy_s(ccuFastLaunchCtx->algName, sizeof(ccuFastLaunchCtx->algName), param.algName));
    HCCL_INFO("[InsV2ReduceScatterSoleExecutor][FastLaunchSaveCtx] algName[%s]", ccuFastLaunchCtx->algName);

    // 2 thread（存全部线程，适配 concurrent 多线程场景）
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
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunch(
    const OpParam& param, const CcuFastLaunchCtx* fastLaunchCtx)
{
    HCCL_INFO("[InsV2ReduceScatterSoleExecutor][FastLaunch] Start.");
    TemplateFastLaunchCtx tempFastLaunchCtx;
    // 1 取thread
    ThreadHandle* threads = fastLaunchCtx->GetThreadHandlePtr();
    tempFastLaunchCtx.threads.assign(threads, threads + fastLaunchCtx->threadNum);
    HCCL_INFO("[InsV2ReduceScatterSoleExecutor][FastLaunch] threadNum[%llu]", fastLaunchCtx->threadNum);

    // 2 取arg
    CcuKernelSubmitInfo* ccuKernelSubmitInfos = fastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    tempFastLaunchCtx.ccuKernelSubmitInfos.assign(
        ccuKernelSubmitInfos, ccuKernelSubmitInfos + fastLaunchCtx->ccuKernelNum[0]);
    HCCL_INFO("[InsV2ReduceScatterSoleExecutor][FastLaunch] ccuKernelNum[%llu]", fastLaunchCtx->ccuKernelNum[0]);
    tempFastLaunchCtx.buffInfo.inputPtr = param.inputPtr;
    tempFastLaunchCtx.buffInfo.outputPtr = param.outputPtr;
    tempFastLaunchCtx.buffInfo.hcclBuff = param.hcclBuff;

    // 3 调template
    std::unique_ptr<InsAlgTemplate> algTemplate = std::make_unique<InsAlgTemplate>();
    CHK_RET(algTemplate->FastLaunch(param, tempFastLaunchCtx));
    HCCL_INFO("[InsV2ReduceScatterSoleExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif

// 第二个参数是Reduce Scatter的template文件
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AicpuReduceScatterSoleMesh, InsV2ReduceScatterSoleExecutor, TopoMatchOneLevel,
    InsTempReduceScatterMesh1D);
REGISTER_ALG_ATTRS(AicpuReduceScatterSoleMesh, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
                   topo.maxTopoLevelNum = 1; topo.isSupportLevel0PcieMix = true; topo.requireAllMeshConnected = true);
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AicpuReduceScatterSoleMeshChunk, InsV2ReduceScatterSoleExecutor,
    TopoMatchOneLevel, InsTempReduceScatterMesh1DMeshChunk);
REGISTER_ALG_ATTRS(AicpuReduceScatterSoleMeshChunk,
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
                   topo.maxTopoLevelNum = 1; topo.isSupportLevel0PcieMix = true; topo.requireAllMeshConnected = true;
                   op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_64BIT);
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AicpuReduceScatterSoleNHR, InsV2ReduceScatterSoleExecutor, TopoMatchOneLevel,
    InsTempReduceScatterNHR);
REGISTER_ALG_ATTRS(
    AicpuReduceScatterSoleNHR, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.isSupportLevel1Nhr = true; op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_64BIT;
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return (topo->topLevelUboe
                && !(
                    (topo->level0Symmetric && topo->level1Symmetric)
                    && topo->deviceNumPerModule == DEVICE_NUM_PER_MODULE_8)
                && !(
                    !(topo->level0Symmetric && topo->level1Symmetric)
                    || topo->netLayerDetails.localNetInsSizeOfLayer[1] == 1)
                && topo->Level0Nhr && topo->netLayerDetails.localNetInsSizeOfLayer[0] == 1)
               || (!topo->netLayerDetails.localNetInsSizeOfLayer.empty()
                   && topo->netLayerDetails.localNetInsSizeOfLayer[0] == 1);
    });
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AicpuReduceScatterSoleNHRAicpuReduce, InsV2ReduceScatterSoleExecutor,
    TopoMatchOneLevel, InsTempReduceScatterAicpuReduceNHR);
REGISTER_ALG_ATTRS(AicpuReduceScatterSoleNHRAicpuReduce,
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS | LEVEL0_TOPO_CLOS;
                   topo.isSupportLevel1Nhr = true);
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AicpuReduceScatterSoleNHRMultiLink, InsV2ReduceScatterSoleExecutor,
    TopoMatchOneLevel, InsTempReduceScatterNHR);
REGISTER_ALG_ATTRS(AicpuReduceScatterSoleNHRMultiLink, topo.supportLevel0Topos = LEVEL0_TOPO_CLOS;
                   op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_64BIT);
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AicpuReduceScatterSoleMeshConcur, InsV2ReduceScatterSoleExecutor,
    TopoMatchOneLevel, InsTempReduceScatterMesh1DZAxisDetour);
REGISTER_ALG_ATTRS(AicpuReduceScatterSoleMeshConcur,
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
                   topo.maxTopoLevelNum = 1; topo.isSupportLevel0PcieMix = true; topo.requireAllMeshConnected = true;
                   op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_64BIT);
#ifndef AICPU_COMPILE
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AivReduceScatterSoleMesh, InsV2ReduceScatterSoleExecutor, TopoMatchOneLevel,
    AivTempReduceScatterMesh1D);
REGISTER_ALG_ATTRS(
    AivReduceScatterSoleMesh,
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.maxTopoLevelNum = 2; topo.isSupportLevel0PcieMix = true; topo.isSupportLevel1Nhr = true;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->userRankSize <= ops_hccl::MAX_RANK_SIZE;
    };
    op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_UINT64_FP64;
    op.opCustomCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails* topo) -> bool {
        u64 totalSize = opParam.DataDes.count * DATATYPE_SIZE_TABLE[opParam.DataDes.dataType] * topo->userRankSize;
        void* bufAddr = nullptr;
        uint64_t bufSize = 0;
        if (HcclGetHcclBuffer(opParam.hcclComm, &bufAddr, &bufSize) != HCCL_SUCCESS) {
            return false;
        }
        return totalSize <= bufSize * ops_hccl::AIV_MAX_CCL_LOOP_NUM;
    });

#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CcuSchedReduceScatterSoleMesh, InsV2ReduceScatterSoleExecutor,
    TopoMatchOneLevel, CcuTempReduceScatterMesh1DMem2Mem);
REGISTER_ALG_ATTRS(
    CcuSchedReduceScatterSoleMesh, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.maxTopoLevelNum = 2; op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_INT8_AND_64BIT;
    op.isSupportInplace = false; topo.isSupportLevel0PcieMix = true; topo.requireAllMeshConnected = true;
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        bool isEqual = false;
        if (topo->level0Topo != Level0Shape::MESH_1D_CLOS) {
            return false;
        }
        AutoSelectorBase::CheckMeshNumEqualToClosNum(topo, isEqual);
        return topo->level0Topo == Level0Shape::MESH_1D_CLOS && isEqual
               && topo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO
               && AutoSelectorBase::CalcFrameNum(topo) <= MAX_FRAME_NUM_FOR_CCU_ALGO;
    });
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CcuMSReduceScatterSoleMesh, InsV2ReduceScatterSoleExecutor, TopoMatchOneLevel,
    CcuTempReduceScatterMesh1D);
REGISTER_ALG_ATTRS(
    CcuMSReduceScatterSoleMesh, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.maxTopoLevelNum = 1; op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_INT8_AND_64BIT;
    op.isSupportInplace = false; topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        bool isEqual = false;
        if (topo->level0Topo != Level0Shape::MESH_1D_CLOS) {
            return false;
        }
        AutoSelectorBase::CheckMeshNumEqualToClosNum(topo, isEqual);
        return topo->level0Topo == Level0Shape::MESH_1D_CLOS && isEqual
               && topo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO
               && AutoSelectorBase::CalcFrameNum(topo) <= MAX_FRAME_NUM_FOR_CCU_ALGO;
    });
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CcuSchedReduceScatterSoleNHR, InsV2ReduceScatterSoleExecutor,
    TopoMatchOneLevel, CcuTempReduceScatterNHR1DMem2Mem);
REGISTER_ALG_ATTRS(
    CcuSchedReduceScatterSoleNHR, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS;
    topo.isSupportLevel1Nhr = true; topo.maxTopoLevelNum = 2; op.isSupportProd = false;
    op.unsupportedDataTypes = UNSUPPORTED_64BIT; op.isSupportInplace = false;
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return AutoSelectorBase::CalcFrameNum(topo) > MAX_FRAME_NUM_FOR_CCU_ALGO
               || (!topo->netLayerDetails.localNetInsSizeOfLayer.empty()
                   && topo->netLayerDetails.localNetInsSizeOfLayer[0] == 1);
    });
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CcuSchedReduceScatterSoleMesh2Die, InsV2ReduceScatterSoleExecutor,
    TopoMatchOneLevel, CcuTempReduceScatterMeshMem2Mem1D2Die);
REGISTER_ALG_ATTRS(
    CcuSchedReduceScatterSoleMesh2Die, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.supportLevel0MeshTypes = MESH_TYPE_NOT_MESH | MESH_TYPE_SINGLE_DIE | MESH_TYPE_TWO_DIE_REGULAR;
    topo.maxTopoLevelNum = 2; topo.isSupportLevel0PcieMix = true; topo.requireAllMeshConnected = true;
    op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_INT8_AND_64BIT; op.isSupportInplace = false;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        // MESH_1D场景：仅支持 2框16卡组网，不满足则过滤
        if (topo->topoLevelNums > 1) {
            return topo->level0Topo == Level0Shape::MESH_1D && topo->userRankSize == RS_CCU_2DIE_RANK_SIZE
                   && AutoSelectorBase::CalcFrameNum(topo) == RS_CCU_2DIE_FRAME_NUM;
        } else if (topo->topoLevelNums == 1) {
            return topo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR;
        }
        return true;
    });
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CcuMSReduceScatterSoleMesh2Die, InsV2ReduceScatterSoleExecutor,
    TopoMatchOneLevel, CcuTempReduceScatterMesh2Die);
REGISTER_ALG_ATTRS(
    CcuMSReduceScatterSoleMesh2Die, topo.supportLevel0MeshTypes = MESH_TYPE_TWO_DIE_REGULAR; topo.maxTopoLevelNum = 1;
    op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_INT8_AND_64BIT; op.isSupportInplace = false;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return AutoSelectorBase::CalcFrameNum(topo) <= MAX_FRAME_NUM_FOR_CCU_ALGO;
    });
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CcuSchedReduceScatterSoleNHRMultiLink, InsV2ReduceScatterSoleExecutor,
    TopoMatchOneLevel, CcuTempReduceScatterNhrMultiJettyMem2Mem1D);
REGISTER_ALG_ATTRS(
    CcuSchedReduceScatterSoleNHRMultiLink, topo.maxTopoLevelNum = 1; topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS;
    op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_INT8_AND_64BIT; op.isSupportInplace = false;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return AutoSelectorBase::CalcFrameNum(topo) <= MAX_FRAME_NUM_FOR_CCU_ALGO;
    };
    op.opPriorityCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails* topo) -> bool {
        bool isEqual = false;
        bool isMultiple = false;
        if (topo->level0Topo != Level0Shape::MESH_1D_CLOS) {
            return false;
        }
        AutoSelectorBase::CheckMeshNumEqualToClosNum(topo, isEqual);
        AutoSelectorBase::CheckClosNumMultipleOfMeshNum(topo, isMultiple);
        u64 dataSize = opParam.DataDes.count * DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
        return !(isEqual && topo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO)
               && !(isMultiple && dataSize >= SMALL_COUNT_512KB);
    });
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CcuMSReduceScatterSoleMeshConcur, InsV2ReduceScatterSoleExecutor,
    TopoMatchConcurrentV2, CcuTempReduceScatterConcurrentMeshNHR);
REGISTER_ALG_ATTRS(
    CcuMSReduceScatterSoleMeshConcur, topo.maxTopoLevelNum = 1; topo.supportDevTypes = {HcclDevType::DEV_TYPE_960};
    op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_INT8_AND_64BIT; op.isSupportInplace = false;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return AutoSelectorBase::CalcFrameNum(topo) <= MAX_FRAME_NUM_FOR_CCU_ALGO;
    };
    op.opPriorityCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails* topo) -> bool {
        u64 dataSize = opParam.DataDes.count * DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
        bool isEqual = false;
        if (topo->level0Topo != Level0Shape::MESH_1D_CLOS) {
            return false;
        }
        AutoSelectorBase::CheckMeshNumEqualToClosNum(topo, isEqual);
        bool isMultiple = false;
        if (topo->level0Topo != Level0Shape::MESH_1D_CLOS) {
            return false;
        }
        AutoSelectorBase::CheckClosNumMultipleOfMeshNum(topo, isMultiple);
        return !(isEqual && topo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO)
               && !(isMultiple && dataSize >= SMALL_COUNT_512KB);
    });
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#endif

} // namespace ops_hccl
