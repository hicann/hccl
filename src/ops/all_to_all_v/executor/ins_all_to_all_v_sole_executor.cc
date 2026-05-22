/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "template_utils.h"
#include "ins_all_to_all_v_sole_executor.h"
#ifndef AICPU_COMPILE
#if !defined(HCCL_CANN_COMPAT_850)
#include "ccu_temp_all_to_all_v_mesh_1D.h"
#include "ccu_temp_all_to_all_v_mesh2die.h"
#include "ccu_temp_all_to_all_v_mesh_1D_multi_jetty.h"
#endif /* !HCCL_CANN_COMPAT_850 */
#endif

#define CONST_ZERO 0
#define CONST_ONE 1
#define CONST_TWO 2
#define CONST_THREE 3

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsAlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsAlltoAllVSoleExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(HcclComm comm,
    TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ =  SIZE_TABLE[dataType_];
    dataCount_ = param.DataDes.count;
    dataSize_ = dataCount_ * dataTypeSize_;

    HCCL_INFO("[InsAlltoAllVSoleExecutor][InitCommInfo] myRank [%u], rankSize [%u], devType [%u], dataType_ [%u], "
        "dataCount_ [%llu]", myRank_, rankSize_, devType_, dataType_, dataCount_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(HcclComm comm, const OpParam& param,
                       const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
                       AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, topoInfo));

    std::vector<std::vector<u32>> tempAlgHierachyInfo;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        tempAlgHierachyInfo.push_back(algHierarchyInfo.infos[0][1]);    // clos拓扑，包含所有rank
    } else {
        tempAlgHierachyInfo = algHierarchyInfo.infos[0];
    }

    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate = 
        std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, tempAlgHierachyInfo);
    // 调用计算资源的函数
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetAlltoAllLocalSendRecvInfo(
    const OpParam &param, A2ASendRecvInfo &localSendRecvInfo) const
{
    HCCL_DEBUG("[GetAlltoAllLocalSendRecvInfo] rank[%u], userRankSize[%u]", myRank_, rankSize_);
    localSendRecvInfo.sendCounts.resize(rankSize_, 0);
    localSendRecvInfo.sendDispls.resize(rankSize_, 0);
    localSendRecvInfo.sendLength.resize(rankSize_, 0);
    localSendRecvInfo.sendOffset.resize(rankSize_, 0);

    localSendRecvInfo.recvCounts.resize(rankSize_, 0);
    localSendRecvInfo.recvDispls.resize(rankSize_, 0);
    localSendRecvInfo.recvLength.resize(rankSize_, 0);
    localSendRecvInfo.recvOffset.resize(rankSize_, 0);

    CHK_PRT_RET(param.varMemSize != ALL_TO_ALL_V_VECTOR_NUM * rankSize_ * sizeof(u64),
    HCCL_ERROR("[InsV2AlltoAllVSoleExecutor][OrchestrateLoop] param.varMemSize [%llu] is invalid",
        param.varMemSize), HCCL_E_PARA);

    const u64* data = reinterpret_cast<const u64*>(param.varData);
    for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * rankSize_; i++) {
        u64 val = i / rankSize_;
        u64 curRank = i % rankSize_;
        switch(val) {
            case CONST_ZERO:
                localSendRecvInfo.sendCounts[curRank] = data[i];
                localSendRecvInfo.sendLength[curRank] = data[i] * dataTypeSize_;
                HCCL_INFO("data[i]: %u, localSendRecvInfo.sendLength: %u", data[i], localSendRecvInfo.sendLength[curRank]);
                break;
            case CONST_ONE:
                localSendRecvInfo.recvCounts[curRank] = data[i];
                localSendRecvInfo.recvLength[curRank] = data[i] * dataTypeSize_;
                HCCL_INFO("data[i]: %u, localSendRecvInfo.recvLength: %u", data[i], localSendRecvInfo.recvLength[curRank]);
                break;
            case CONST_TWO:
                localSendRecvInfo.sendDispls[curRank] = data[i];
                localSendRecvInfo.sendOffset[curRank] = data[i] * dataTypeSize_;
                HCCL_INFO("data[i]: %u, localSendRecvInfo.sendOffset: %u", data[i], localSendRecvInfo.sendOffset[curRank]);
                break;
            case CONST_THREE:
                localSendRecvInfo.recvDispls[curRank] = data[i];
                localSendRecvInfo.recvOffset[curRank] = data[i] * dataTypeSize_;
                HCCL_INFO("data[i]: %u, localSendRecvInfo.recvOffset: %u", data[i], localSendRecvInfo.recvOffset[curRank]);
                break;
            default:
                break;
        }
    }
    HCCL_DEBUG("[GetAlltoAllLocalSendRecvInfo] GetAlltoAllLocalSendRecvInfo success");
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsAlltoAllVSoleExecutor][Orchestrate] Orchestrate Start");

    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, &resCtx.topoInfo));

    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    CHK_PRT_RET(GetAlltoAllLocalSendRecvInfo(param, localSendRecvInfo_),
        HCCL_ERROR("[InitCommInfo] unable to init DataInfo."),
        HcclResult::HCCL_E_PARA);

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsAlltoAllVSoleExecutor][Orchestrate]errNo[0x%016llx] All to All excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsAlltoAllVSoleExecutor][OrchestrateLoop] Start");
    
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
    algTemplate->SetA2ASendRecvInfo(localSendRecvInfo_);

    // 准备资源
    TemplateResource templateAlgRes;
    if (remoteRankToChannelInfo_.size() > 0) {
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    if (param.engine == COMM_ENGINE_CCU) {
        templateAlgRes.ccuKernels = resCtx.ccuKernels;
    }
    templateAlgRes.threads = resCtx.threads;
    templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams.buffInfo.inBuffBaseOff = 0;
    tempAlgParams.buffInfo.outBuffBaseOff = 0;
    tempAlgParams.buffInfo.hcclBuffBaseOff = 0;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.repeatNum = 1;  // 不需要重复
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;

    CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes));

#ifndef AICPU_COMPILE
    if (param.engine == CommEngine::COMM_ENGINE_CCU && param.opType != HcclCMDType::HCCL_CMD_ALLTOALLVC && param.opMode != OpMode::OFFLOAD) {
        CHK_RET(FastLaunchSaveCtx(param, templateAlgRes, resCtx.notifyNumOnMainThread));
    }
#endif

    HCCL_INFO("[InsAlltoAllVSoleExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunchSaveCtx(
    const OpParam &param, const TemplateResource &templateAlgRes, u32 notifyNumOnMainThread)
{
    HCCL_INFO("[InsAlltoAllVSoleExecutor] save fast launch ctx.");
    u32 threadNum = 1;
    u32 ccuKernelNum = templateAlgRes.submitInfos.size();
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsAlltoAllVSoleExecutor] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "[InsAlltoAllVSoleExecutor][HcclEngineCtxCreate] threadNum[%llu], ccuKernelNum[%llu]", threadNum, ccuKernelNum);

    u64 size = CcuFastLaunchCtx::GetCtxSize(threadNum, ccuKernelNum);
    // 申请ctx
    void *ctxPtr = nullptr;
    HCCL_INFO("[InsAlltoAllVSoleExecutor][HcclEngineCtxCreate] Tag[%s], size[%llu]", param.fastLaunchTag, size);
    CHK_RET(HcclEngineCtxCreate(param.hcclComm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, size, &ctxPtr));

    CcuFastLaunchCtx *ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx *>(ctxPtr);
    // 1 算法名:
    CHK_SAFETY_FUNC_RET(strcpy_s(ccuFastLaunchCtx->algName, sizeof(ccuFastLaunchCtx->algName), param.algName));
    HCCL_INFO("[InsAlltoAllVSoleExecutor][FastLaunchSaveCtx] algName[%s]", ccuFastLaunchCtx->algName);

    // 2 thread
    ccuFastLaunchCtx->threadNum = threadNum;
    ccuFastLaunchCtx->notifyNumOnMainThread = notifyNumOnMainThread;
    ThreadHandle *threads = ccuFastLaunchCtx->GetThreadHandlePtr();
    threads[0] = templateAlgRes.threads[0];

    // 3 ccu kernel handle, taskArg入参
    ccuFastLaunchCtx->ccuKernelNum[0] = ccuKernelNum;
    CcuKernelSubmitInfo *kernels = ccuFastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    kernels[0] = templateAlgRes.submitInfos[0];
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunch(
        const OpParam &param, const CcuFastLaunchCtx *fastLaunchCtx)
{
    HCCL_INFO("[InsAlltoAllVSoleExecutor][FastLaunch] Start.");
    TemplateFastLaunchCtx tempFastLaunchCtx;
    // 1 取线程
    ThreadHandle *threads = fastLaunchCtx->GetThreadHandlePtr();
    tempFastLaunchCtx.threads.assign(threads, threads + fastLaunchCtx->threadNum);
    HCCL_INFO("[InsAlltoAllVSoleExecutor][FastLaunch] threadNum[%llu]", fastLaunchCtx->threadNum);
    
    // 2 取arg
    CcuKernelSubmitInfo *ccuKernelSubmitInfos = fastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    tempFastLaunchCtx.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + fastLaunchCtx->ccuKernelNum[0]);
    HCCL_INFO("[InsAlltoAllVSoleExecutor][FastLaunch] ccuKernelNum[%llu]", fastLaunchCtx->ccuKernelNum[0]);
    tempFastLaunchCtx.buffInfo.inputPtr = param.inputPtr;
    tempFastLaunchCtx.buffInfo.outputPtr = param.outputPtr;
    
    // 3 调template
    std::unique_ptr<InsAlgTemplate> algTemplate = std::make_unique<InsAlgTemplate>();
    CHK_RET(algTemplate->FastLaunch(param, tempFastLaunchCtx));
    HCCL_INFO("[InsAlltoAllVSoleExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif

// 第二个参数是All to AllV的template文件
#ifndef AICPU_COMPILE
#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALLV, CcuAlltoAllVMesh1D, InsAlltoAllVSoleExecutor, TopoMatch1D,
    CcuTempAlltoAllVMesh1D);
#endif /* !HCCL_CANN_COMPAT_850 */

#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALLVC, CcuAlltoAllVCMesh1D, InsAlltoAllVSoleExecutor, TopoMatch1D,
    CcuTempAlltoAllVMesh1D);
#endif /* !HCCL_CANN_COMPAT_850 */

#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALLV, CcuAllToAllVMesh2Die, InsAlltoAllVSoleExecutor, TopoMatch1D,
    CcuTempAlltoAllVMesh2Die);
#endif /* !HCCL_CANN_COMPAT_850 */

#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALLV,
                CcuAllToAllVMesh1DMultiJetty,
                InsAlltoAllVSoleExecutor,
                TopoMatchUBX,
                CcuTempAllToAllVMesh1DMultiJetty);
#endif /* !HCCL_CANN_COMPAT_850 */
#endif
}