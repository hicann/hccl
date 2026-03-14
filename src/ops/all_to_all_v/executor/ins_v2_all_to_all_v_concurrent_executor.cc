/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "ins_v2_all_to_all_v_concurrent_executor.h"
#ifndef AICPU_COMPILE
#include "ccu_temp_all_to_all_v_mesh_1D_multi_jetty.h"
#include "ccu_kernel_all_to_all_v_mesh1d_multi_jetty.h"
#endif
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {
constexpr uint32_t CONST_0 = 0;
constexpr uint32_t CONST_1 = 1;
constexpr uint32_t CONST_2 = 2;
constexpr uint32_t CONST_3 = 3;
constexpr uint32_t CONST_4 = 4;

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2AllToAllVConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2AllToAllVConcurrentExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllVConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(HcclComm comm,
    TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllVConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitCommInfo(const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ =  SIZE_TABLE[dataType_];
    dataCount_ = param.DataDes.count;
    dataSize_ = dataCount_ * dataTypeSize_;

    HCCL_INFO("[InsV2AllToAllVConcurrentExecutor][InitCommInfo] myRank [%u], rankSize [%u], devType [%u], dataType_ [%u], "
        "dataCount_ [%llu]", myRank_, rankSize_, devType_, dataType_, dataCount_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllVConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::SetAlltoAllLocalSendRecvInfo(
    const OpParam &param)
{
    HCCL_DEBUG("[SetAlltoAllLocalSendRecvInfo] rank[%u], userRankSize[%u]", myRank_, rankSize_);
    localSendRecvInfo_.sendCounts.resize(rankSize_, 0);
    localSendRecvInfo_.sendDispls.resize(rankSize_, 0);
    localSendRecvInfo_.sendLength.resize(rankSize_, 0);
    localSendRecvInfo_.sendOffset.resize(rankSize_, 0);

    localSendRecvInfo_.recvCounts.resize(rankSize_, 0);
    localSendRecvInfo_.recvDispls.resize(rankSize_, 0);
    localSendRecvInfo_.recvLength.resize(rankSize_, 0);
    localSendRecvInfo_.recvOffset.resize(rankSize_, 0);

    CHK_PRT_RET(param.varMemSize != ALL_TO_ALL_V_VECTOR_NUM * rankSize_ * sizeof(u64),
    HCCL_ERROR("[InsV2AlltoAllVSoleExecutor][OrchestrateLoop] param.varMemSize [%llu] is invalid",
        param.varMemSize), HCCL_E_PARA);

    const u64* data = reinterpret_cast<const u64*>(param.varData);
    for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * rankSize_; i++) {
        u64 val = i / rankSize_;
        u64 curRank = i % rankSize_;
        switch(val) {
            case CONST_0:
                localSendRecvInfo_.sendCounts[curRank] = data[i];
                localSendRecvInfo_.sendLength[curRank] = data[i] * dataTypeSize_;
                break;
            case CONST_1:
                localSendRecvInfo_.recvCounts[curRank] = data[i];
                localSendRecvInfo_.recvLength[curRank] = data[i] * dataTypeSize_;
                break;
            case CONST_2:
                localSendRecvInfo_.sendDispls[curRank] = data[i];
                localSendRecvInfo_.sendOffset[curRank] = data[i] * dataTypeSize_;
                break;
            case CONST_3:
                localSendRecvInfo_.recvDispls[curRank] = data[i];
                localSendRecvInfo_.recvOffset[curRank] = data[i] * dataTypeSize_;
                break;
            default:
                break;
        }
    }
    HCCL_DEBUG("[SetAlltoAllLocalSendRecvInfo] SetAlltoAllLocalSendRecvInfo success");
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllVConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::SplitA2ASendRecvInfo(
    A2ASendRecvInfo &sendRecvInfoFirst, A2ASendRecvInfo &sendRecvInfoLast)
{
    HCCL_DEBUG("[SplitA2ASendRecvInfo] rank[%u], userRankSize[%u]", myRank_, rankSize_);

    uint32_t factorMesh = rankSize_ - 1;
    uint32_t factorClos = CONST_4;
    uint32_t factor = factorMesh + factorClos;
    // 初始化sendRecvInfoFirst
    sendRecvInfoFirst.sendCounts.resize(rankSize_, 0);
    sendRecvInfoFirst.sendDispls.resize(rankSize_, 0);
    sendRecvInfoFirst.sendLength.resize(rankSize_, 0);
    sendRecvInfoFirst.sendOffset.resize(rankSize_, 0);

    sendRecvInfoFirst.recvCounts.resize(rankSize_, 0);
    sendRecvInfoFirst.recvDispls.resize(rankSize_, 0);
    sendRecvInfoFirst.recvLength.resize(rankSize_, 0);
    sendRecvInfoFirst.recvOffset.resize(rankSize_, 0);

    // 初始化sendRecvInfoLast
    sendRecvInfoLast.sendCounts.resize(rankSize_, 0);
    sendRecvInfoLast.sendDispls.resize(rankSize_, 0);
    sendRecvInfoLast.sendLength.resize(rankSize_, 0);
    sendRecvInfoLast.sendOffset.resize(rankSize_, 0);

    sendRecvInfoLast.recvCounts.resize(rankSize_, 0);
    sendRecvInfoLast.recvDispls.resize(rankSize_, 0);
    sendRecvInfoLast.recvLength.resize(rankSize_, 0);
    sendRecvInfoLast.recvOffset.resize(rankSize_, 0);

    // 按照(rankSize_ - 1 ： jettyNum)切分每一个rank的数据
    for(int i = 0; i < rankSize_; i++) {
        // 设置sendRecvInfoFirst数据
        sendRecvInfoFirst.sendCounts[i] = localSendRecvInfo_.sendCounts[i] / factor * factorClos;
        sendRecvInfoFirst.sendDispls[i] = localSendRecvInfo_.sendDispls[i];
        sendRecvInfoFirst.sendLength[i] = sendRecvInfoFirst.sendCounts[i] * dataTypeSize_;
        sendRecvInfoFirst.sendOffset[i] = sendRecvInfoFirst.sendDispls[i] * dataTypeSize_;

        sendRecvInfoFirst.recvCounts[i] = localSendRecvInfo_.recvCounts[i] / factor * factorClos;
        sendRecvInfoFirst.recvDispls[i] = localSendRecvInfo_.recvDispls[i];
        sendRecvInfoFirst.recvLength[i] = sendRecvInfoFirst.recvCounts[i] * dataTypeSize_;
        sendRecvInfoFirst.recvOffset[i] = sendRecvInfoFirst.recvDispls[i] * dataTypeSize_;

        HCCL_DEBUG("[SplitA2ASendRecvInfo][sendRecvInfoFirst] rank[%u], sendCounts[%llu], sendDispls[%llu] "\
            "recvCounts[%llu], recvDispls[%llu], sendLength[%llu], recvLength[%llu]", myRank_, sendRecvInfoFirst.sendCounts[i],
            sendRecvInfoFirst.sendDispls[i], sendRecvInfoFirst.recvCounts[i], sendRecvInfoFirst.recvDispls[i],
            sendRecvInfoFirst.sendLength[i], sendRecvInfoFirst.recvLength[i]);

        // 设置sendRecvInfoLast数据
        sendRecvInfoLast.sendCounts[i] = localSendRecvInfo_.sendCounts[i] - sendRecvInfoFirst.sendCounts[i];
        sendRecvInfoLast.sendDispls[i] = localSendRecvInfo_.sendDispls[i] + sendRecvInfoFirst.sendCounts[i];
        sendRecvInfoLast.sendLength[i] = sendRecvInfoLast.sendCounts[i] * dataTypeSize_;
        sendRecvInfoLast.sendOffset[i] = sendRecvInfoLast.sendDispls[i] * dataTypeSize_;

        sendRecvInfoLast.recvCounts[i] = localSendRecvInfo_.recvCounts[i] - sendRecvInfoFirst.recvCounts[i];
        sendRecvInfoLast.recvDispls[i] = localSendRecvInfo_.recvDispls[i] + sendRecvInfoFirst.recvCounts[i];
        sendRecvInfoLast.recvLength[i] = sendRecvInfoLast.recvCounts[i] * dataTypeSize_;
        sendRecvInfoLast.recvOffset[i] = sendRecvInfoLast.recvDispls[i] * dataTypeSize_;

        HCCL_DEBUG("[SplitA2ASendRecvInfo][sendRecvInfoLast] rank[%u], sendCounts[%llu], sendDispls[%llu] "\
            "recvCounts[%llu], recvDispls[%llu], sendLength[%llu], recvLength[%llu]", myRank_, sendRecvInfoLast.sendCounts[i],
            sendRecvInfoLast.sendDispls[i], sendRecvInfoLast.recvCounts[i], sendRecvInfoLast.recvDispls[i],
            sendRecvInfoLast.sendLength[i], sendRecvInfoLast.recvLength[i]);;
    }
    HCCL_DEBUG("[SplitA2ASendRecvInfo] SplitA2ASendRecvInfo success");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllVConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    if (param.engine != CommEngine::COMM_ENGINE_CCU) {
        HCCL_ERROR("[InsV2AllToAllVConcurrentExecutor] only support ccu");
        return HCCL_E_NOT_SUPPORT;
    }

    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, topoInfo));

    std::vector<std::vector<u32>> subCommRanks0 = {algHierarchyInfo.infos[0][0]};
    std::vector<std::vector<u32>> subCommRanks1 = {algHierarchyInfo.infos[0][1]};

    // 构建template
    std::shared_ptr<InsAlgTemplate0> algTemplateClos =
        std::make_shared<InsAlgTemplate0>(param, topoInfo->userRank, subCommRanks0);
    std::shared_ptr<InsAlgTemplate1> algTemplateMesh =
        std::make_shared<InsAlgTemplate1>(param, topoInfo->userRank, subCommRanks1);

    // 调用计算资源的函数
    AlgResourceRequest resReq0;
    AlgResourceRequest resReq1;
    algTemplateClos->CalcRes(comm, param, topoInfo, resReq0);
    algTemplateMesh->CalcRes(comm, param, topoInfo, resReq1);

    resourceRequest.notifyNumOnMainThread = resReq0.slaveThreadNum + 1;  // 用于两个template间同步
    resourceRequest.slaveThreadNum = resReq0.slaveThreadNum + resReq1.slaveThreadNum + 1;
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReq0.notifyNumPerThread.begin(),
                                              resReq0.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(resReq1.notifyNumOnMainThread + 1);  // 用于两个template间同步
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReq1.notifyNumPerThread.begin(),
                                              resReq1.notifyNumPerThread.end());

    // ubx机型algHierarchyInfo的level0存在两个topo，4p及以下使用clos topo与mesh topo分别建链
    std::vector<HcclChannelDesc> channelDescs0;
    CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks0, channelDescs0, CommTopo::COMM_TOPO_CLOS));
    resReq0.ccuKernelInfos[0].channels = channelDescs0;

    std::vector<uint32_t> jettyNums;
    CHK_RET(SetJettyNums(jettyNums, true));
    resReq0.ccuKernelInfos[0].kernelArg = std::make_shared<CcuKernelArgAllToAllVMesh1DMultiJetty>(subCommRanks0[0].size(),
                                                                                    topoInfo->userRank,
                                                                                    param,
                                                                                    subCommRanks0,
                                                                                    jettyNums);

    std::vector<HcclChannelDesc> channelDescs1;
    CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks1, channelDescs1, CommTopo::COMM_TOPO_1DMESH));
    resReq1.ccuKernelInfos[0].channels = channelDescs1;

    CHK_RET(SetJettyNums(jettyNums, false));
    resReq1.ccuKernelInfos[0].kernelArg = std::make_shared<CcuKernelArgAllToAllVMesh1DMultiJetty>(subCommRanks1[0].size(),
                                                                                    topoInfo->userRank,
                                                                                    param,
                                                                                    subCommRanks1,
                                                                                    jettyNums);

    resourceRequest.ccuKernelNum.emplace_back(resReq0.ccuKernelNum[0]);
    resourceRequest.ccuKernelNum.emplace_back(resReq1.ccuKernelNum[0]);
    resourceRequest.ccuKernelInfos.emplace_back(resReq0.ccuKernelInfos[0]);
    resourceRequest.ccuKernelInfos.emplace_back(resReq1.ccuKernelInfos[0]);

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllVConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::SetJettyNums(
    std::vector<uint32_t>& jettyNums, const bool multijetty) const
{
    jettyNums.resize(rankSize_, 0);
    for (int i = 0; i < rankSize_; i++) {
        if (i == myRank_) {
            jettyNums[i] = CONST_1;
        } else if (multijetty) {
            jettyNums[i] = CONST_4;
        } else {
            jettyNums[i] = CONST_1;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllVConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    if (param.engine != CommEngine::COMM_ENGINE_CCU) {
        HCCL_ERROR("[InsV2AllToAllVConcurrentExecutor] only support ccu");
        return HCCL_E_NOT_SUPPORT;
    }
    
    HCCL_INFO("[InsV2AllToAllVConcurrentExecutor][Orchestrate] Orchestrate Start");
    HcclResult ret;

    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, &resCtx.topoInfo));

    std::vector<std::vector<u32>> subCommRanks0 = {resCtx.algHierarchyInfo.infos[0][0]};
    std::vector<std::vector<u32>> subCommRanks1 = {resCtx.algHierarchyInfo.infos[0][1]};

    // 构建template
    std::shared_ptr<InsAlgTemplate0> algTemplateClos =
        std::make_shared<InsAlgTemplate0>(param, resCtx.topoInfo.userRank, subCommRanks0);
    std::shared_ptr<InsAlgTemplate1> algTemplateMesh =
        std::make_shared<InsAlgTemplate1>(param, resCtx.topoInfo.userRank, subCommRanks1);

    CHK_PRT_RET(SetAlltoAllLocalSendRecvInfo(param),
        HCCL_ERROR("[InitCommInfo] unable to init DataInfo."),
        HcclResult::HCCL_E_PARA);

    A2ASendRecvInfo sendRecvInfoTempClos;
    A2ASendRecvInfo sendRecvInfoTempMesh;
    CHK_RET(SplitA2ASendRecvInfo(sendRecvInfoTempClos, sendRecvInfoTempMesh));
    algTemplateClos->SetA2ASendRecvInfo(sendRecvInfoTempClos);
    algTemplateMesh->SetA2ASendRecvInfo(sendRecvInfoTempMesh);

    // 准备资源
    TemplateResource templateAlgResClos;
    templateAlgResClos.threads.push_back(resCtx.threads[0]);
    templateAlgResClos.ccuKernels.push_back(resCtx.ccuKernels[0]);

    TemplateResource templateAlgResMesh;
    templateAlgResMesh.threads.push_back(resCtx.threads[1]);
    templateAlgResMesh.ccuKernels.push_back(resCtx.ccuKernels[1]);

    // 准备数据
    TemplateDataParams tempAlgParamsClos;
    tempAlgParamsClos.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsClos.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsClos.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsClos.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsClos.buffInfo.inputSize = param.inputSize;
    tempAlgParamsClos.buffInfo.outputSize = param.outputSize;
    tempAlgParamsClos.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsClos.buffInfo.outBuffBaseOff = 0;

    TemplateDataParams tempAlgParamsMesh;
    tempAlgParamsMesh.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsMesh.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsMesh.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsMesh.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsMesh.buffInfo.inputSize = param.inputSize;
    tempAlgParamsMesh.buffInfo.outputSize = param.outputSize;
    tempAlgParamsMesh.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsMesh.buffInfo.outBuffBaseOff = 0;

    ThreadHandle mainThread = resCtx.threads[0];
    std::vector<ThreadHandle> subThreads = {resCtx.threads[1]};
    std::vector<u32> notifyIdxMainToSub = {0};
    std::vector<u32> notifyIdxSubToMain = {0};

    CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub));

    ret = algTemplateClos->KernelRun(param, tempAlgParamsClos, templateAlgResClos);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllToAllVConcurrentExecutor][Orchestrate]errNo[0x%016llx] alltoallv concurrent excutor kernel 0 run failed",
            HCCL_ERROR_CODE(ret)), ret);

    ret = algTemplateMesh->KernelRun(param, tempAlgParamsMesh, templateAlgResMesh);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllToAllVConcurrentExecutor][Orchestrate]errNo[0x%016llx] alltoallv concurrent excutor kernel 1 run failed",
            HCCL_ERROR_CODE(ret)), ret);

    CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain));

    HCCL_INFO("[InsV2AllToAllVConcurrentExecutor][Orchestrate] Orchestrate End");
    return HcclResult::HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLTOALLV,
                                CcuAllToAllVMesh1DConcurrent,
                                InsV2AllToAllVConcurrentExecutor,
                                TopoMatchUBX,
                                CcuTempAllToAllVMesh1DMultiJetty,
                                CcuTempAllToAllVMesh1DMultiJetty);
#endif

}