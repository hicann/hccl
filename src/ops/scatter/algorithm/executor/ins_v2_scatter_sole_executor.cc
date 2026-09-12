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
#include "alg_attrs_registry.h"
#include "hccl_aiv_utils.h"
#include "hccl_res.h"
#ifndef AICPU_COMPILE
#include "aiv_temp_scatter_mesh_1D.h"
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_scatter_mesh1d.h"
#include "ccu_temp_scatter_nhr1d_mem2mem.h"
#include "ccu_kernel_scatter_nhr1d_mem2mem.h"
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2ScatterSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
std::vector<CostModelParam> InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)param;
    // 拓扑事实(netType/portNum)取自 V2 topo match 选中的物理层(覆盖全通信域的最低有效层),
    // 替代原按 algoType/topoLevelNums 的推断与 {1}/{6,2}/{8} 打桩
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
#ifndef AICPU_COMPILE
    const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(std::string(algName));
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
        HCCL_INFO("[CalcCostCoeff] algName=%s topo match not support, skip.", algName);
        return {};
    }
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = topoInfo->isPod;
    u32 physIdx = static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]);
    CommTopo netType = GetPhysicalLevelTopoType(topoInfo, physIdx);
    std::vector<u32> portNum = GetPhysicalLevelPortNums(topoInfo, physIdx);
    // 匹配层链路降级(portNums 为空)时算法不参与 costmodel(与其余算子统一口径)
    if (portNum.empty()) {
        HCCL_WARNING("[CalcCostCoeff] portNum is empty");
        return {};
    }
    HCCL_INFO(
        "[CalcCostCoeff] algName=%s rankSize=%d, physIdx=%u, portNum(size=%zu, first=%u), netType=%d", algName,
        rankSize, physIdx, portNum.size(), portNum[0], static_cast<int>(netType));
    std::vector<CostModelParam> params = InsAlgTemplate::CalcCostCoeff(CalcCostCoeffParam{
        rankSize, 1.0f, netType, BufferType::INPUT, BufferType::OUTPUT, BufferType::HCCL_BUFFER, portNum, isPod,
        algName, comm, topoInfo});
    return params;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
AlgNetMeta InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)algName;
    (void)param;
    // netType 与 CalcCostCoeff 同源: V2 匹配层的真实 topoType
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
#ifndef AICPU_COMPILE
    const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(std::string(algName));
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
        HCCL_INFO("[GetAlgNetMeta] algName=%s topo match not support, return empty.", algName);
        return {};
    }
    u32 rankSize = (topoInfo != nullptr) ? topoInfo->userRankSize : 1;
    CommTopo netType
        = GetPhysicalLevelTopoType(topoInfo, static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]));
    AlgNetMeta meta;
    meta.netTypes.push_back(netType);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1};
    // 与 CalcCostCoeff 的 dataRatio=1.0f 逐段相等(对齐 all_gather parallel 的 meta==CC 约定)
    meta.dataRatios = {1.0f};
    meta.rankSizes = {rankSize};
    return meta;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void)comm;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, AlgAttrs{}));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    CHK_PTR_NULL(topoInfo);
    CHK_PRT_RET(
        algHierarchyInfo.infos.empty(),
        HCCL_ERROR("[InsV2ScatterSoleExecutor][CalcRes] algHierarchyInfo.infos is empty"), HCCL_E_PARA);
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    // 调用计算资源的函数
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
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
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2ScatterSoleExecutor][Orchestrate]errNo[0x%016llx] Scatter executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2ScatterSoleExecutor][OrchestrateLoop] Start");
    CHK_PRT_RET(
        resCtx.algHierarchyInfo.infos.empty(),
        HCCL_ERROR("[InsV2ScatterSoleExecutor][OrchestrateLoop] algHierarchyInfo.infos is empty"), HCCL_E_PARA);
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
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS
        && (resCtx.topoInfo.isPod || std::string(param.algName) == "AicpuScatterSoleNHR")) {
        CHK_RET(algTemplate->SetchannelsPerRank(templateAlgRes.channels));
    }
    // 初始化操作
    u32 templateScratchMultiplier
        = algTemplate->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.outBuffType);
    // 中转内存单次最多能够接受的output count，注意是count不是size
    u64 maxDataSizePerLoop = 0;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    maxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size;
    HCCL_INFO("[InsV2ScatterSoleExecutor]maxTmpMemSize_ [%u]", maxTmpMemSize_);
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
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        tempAlgParams.count = currDataCount;
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;

        tempAlgParams.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParams.tailSize = tempAlgParams.sliceSize;
        tempAlgParams.inputSliceStride = dataSize_; // 如果是输入，偏移是算子的output datasize
        tempAlgParams.outputSliceStride = 0;

        HCCL_INFO(
            "[InsV2ScatterSoleExecutor] loop [%u] tempAlgParams.inputSliceStride [%u],"
            "tempAlgParams.outputSliceStride [%u] tempAlgParams.sliceSize [%u]",
            loop, tempAlgParams.inputSliceStride, tempAlgParams.outputSliceStride, tempAlgParams.sliceSize);
        HCCL_INFO(
            "[InsV2ScatterSoleExecutor] loop [%u] tempAlgParams.buffInfo.inBuffBaseOff [%u],"
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

    HCCL_INFO("[InsV2ScatterSoleExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunchSaveCtx(
    const OpParam& param, const TemplateResource& templateAlgRes, u32 notifyNumOnMainThread) const
{
    HCCL_INFO("[InsV2ScatterSoleExecutor] loopTimes==1, save fast launch ctx.");
    u32 threadNum = templateAlgRes.submitInfos.size();
    ;
    u32 ccuKernelNum = templateAlgRes.submitInfos.size();
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsV2ScatterSoleExecutor] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "[InsV2ScatterSoleExecutor][HcclEngineCtxCreate] threadNum[%llu], ccuKernelNum[%llu]", threadNum, ccuKernelNum);

    u64 size = CcuFastLaunchCtx::GetCtxSize(threadNum, ccuKernelNum);
    // 申请ctx
    void* ctxPtr = nullptr;
    HCCL_INFO("[InsV2ScatterSoleExecutor][HcclEngineCtxCreate] Tag[%s], size[%llu]", param.fastLaunchTag, size);
    CHK_RET(HcclEngineCtxCreate(param.hcclComm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, size, &ctxPtr));

    CcuFastLaunchCtx* ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(ctxPtr);
    // 1 算法名
    CHK_SAFETY_FUNC_RET(strcpy_s(ccuFastLaunchCtx->algName, sizeof(ccuFastLaunchCtx->algName), param.algName));
    HCCL_INFO("[InsV2ScatterSoleExecutor][FastLaunchSaveCtx] algName[%s]", ccuFastLaunchCtx->algName);

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
HcclResult InsV2ScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunch(
    const OpParam& param, const CcuFastLaunchCtx* fastLaunchCtx)
{
    HCCL_INFO("[InsV2ScatterSoleExecutor][FastLaunch] Start.");
    TemplateFastLaunchCtx tempFastLaunchCtx;
    // 1 取thread
    ThreadHandle* threads = fastLaunchCtx->GetThreadHandlePtr();
    tempFastLaunchCtx.threads.assign(threads, threads + fastLaunchCtx->threadNum);
    HCCL_INFO("[InsV2ScatterSoleExecutor][FastLaunch] threadNum[%llu]", fastLaunchCtx->threadNum);

    // 2 取arg
    CcuKernelSubmitInfo* ccuKernelSubmitInfos = fastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    tempFastLaunchCtx.ccuKernelSubmitInfos.assign(
        ccuKernelSubmitInfos, ccuKernelSubmitInfos + fastLaunchCtx->ccuKernelNum[0]);
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
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_SCATTER, AicpuScatterSoleMesh, InsV2ScatterSoleExecutor, TopoMatchOneLevel,
    InsTempScatterMesh1D);
// supportLevel0Topos 对齐旧 selector 实际选入面：SoleMesh 在 MESH_1D(:191)/MESH_1D_CLOS 全连(:195)/
// CLOS 非 PcieMix(:205) 三形态下均被选中
REGISTER_ALG_ATTRS(AicpuScatterSoleMesh, topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS | LEVEL0_TOPO_CLOS;
                   topo.isSupportLevel0PcieMix = true; topo.requireAllMeshConnected = true);
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_SCATTER, AicpuScatterSoleNHR, InsV2ScatterSoleExecutor, TopoMatchOneLevel, InsTempScatterNHR);
// SoleNHR 是各非 Mesh 分支兜底：3 级非对称/Level1Nhr/localNetIns==1/CLOS 均选它。
// MESH_1D_CLOS 形态由 SoleMesh(全连)/UBX/Pcie 算法处理，SoleNHR 不参与（对齐旧 selector）
REGISTER_ALG_ATTRS(AicpuScatterSoleNHR, topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS; topo.isSupportLevel1Nhr = true);
#ifndef AICPU_COMPILE
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_SCATTER, AivScatterSoleMesh, InsV2ScatterSoleExecutor, TopoMatchOneLevel,
    AivTempScatterMesh1D);
REGISTER_ALG_ATTRS(
    AivScatterSoleMesh, topo.maxSupportRankSize = MAX_RANK_SIZE; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS | LEVEL0_TOPO_CLOS;
    topo.isSupportLevel0PcieMix = true; topo.isSupportLevel1Nhr = true;
    // 参照 allgather AIV 注册(AivAllGatherSoleMesh)迁移旧 selector 数据量限制
    // (scatter_auto_selector.cc:271)：totalSize <= cclBufferSize * AIV_MAX_CCL_LOOP_NUM
    op.opCustomCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails* topo) -> bool {
        void* bufAddr = nullptr;
        uint64_t bufSize = 0;
        if (HcclGetHcclBuffer(opParam.hcclComm, &bufAddr, &bufSize) != HCCL_SUCCESS) {
            return false;
        }
        u64 totalSize = opParam.DataDes.count * DATATYPE_SIZE_TABLE[opParam.DataDes.dataType] * topo->userRankSize;
        return totalSize <= bufSize * AIV_MAX_CCL_LOOP_NUM;
    });
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
// ccu template
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_SCATTER, CcuSchedScatterSoleMesh, InsV2ScatterSoleExecutor, TopoMatchOneLevel,
    CcuTempScatterMesh1D);
// inplace 排除对齐旧 selector：仅单级 MESH_1D 分支查 inplace（scatter_auto_selector.cc:108），
// 同算法的 MESH_1D_CLOS 全连分支不查；其余 scatter 算法执行侧均支持 inplace（kernel isInputOutputEqual 等）
REGISTER_ALG_ATTRS(CcuSchedScatterSoleMesh, topo.maxSupportRankSize = CCU_SCHED_MAX_RANK_SIZE; topo.maxTopoLevelNum = 1;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
                   topo.isSupportLevel0PcieMix = true; topo.requireAllMeshConnected = true;
                   op.isSupportInplace = false;);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_SCATTER, CcuSchedScatterSoleNHR, InsV2ScatterSoleExecutor, TopoMatchOneLevel,
    CcuTempScatterNHR1DMem2Mem);
REGISTER_ALG_ATTRS(CcuSchedScatterSoleNHR, topo.maxSupportRankSize = CCU_SCHED_MAX_RANK_SIZE;
                   topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS; topo.isSupportLevel1Nhr = true);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif
#endif
} // namespace ops_hccl
