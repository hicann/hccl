/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_broadcast_parallel_executor.h"
#include "ins_temp_all_gather_mesh_1D.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_scatter_mesh_1D.h"
#include "ins_temp_scatter_nhr.h"
#include "topo_match_multilevel.h"
#include "topo_match_pcie_mix.h"
#if !defined(HCCL_CANN_COMPAT_850)
#include "ccu_temp_all_gather_mesh_1D_mem2mem.h"
#include "ccu_temp_all_gather_nhr_1D_mem2mem.h"
#include "ccu_temp_scatter_mesh1d.h"
#include "ccu_temp_scatter_nhr1d_mem2mem.h"
#endif /* !HCCL_CANN_COMPAT_850 */
#include <cmath>

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::InsBroadcastParallelExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::CalcAlgHierarchyInfo(
    HcclComm comm,
    TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::CalcRes(
    HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    myRank_ = topoInfo->userRank;
    HCCL_INFO("[InsBroadcastParallelExecutor] CalcRes start, rank[%d]", myRank_);
    std::vector<std::vector<u32>> temp0HierarchyInfo;
    std::vector<std::vector<u32>> temp1HierarchyInfo;
    if(!topoInfo->level0PcieMix && topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS ) {
        temp0HierarchyInfo = {algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        u32 meshRankSize = algHierarchyInfo.infos[0][0].size();
        for(auto rank : algHierarchyInfo.infos[0][1]) {
            if(rank % meshRankSize == topoInfo->userRank % meshRankSize) {
                closRanks.push_back(rank);
            }
        }
        temp1HierarchyInfo = {closRanks};
    } else {
        temp0HierarchyInfo = algHierarchyInfo.infos[0];
        temp1HierarchyInfo = algHierarchyInfo.infos[1];
    }

    // 实例化算法模板类
    // 构建template
    std::shared_ptr<InsAlgTemplate0> algTemplate0 = std::make_shared<InsAlgTemplate0>(param, topoInfo->userRank, temp0HierarchyInfo);
    std::shared_ptr<InsAlgTemplate1> algTemplate1 = std::make_shared<InsAlgTemplate1>(param, topoInfo->userRank, temp1HierarchyInfo);
    std::shared_ptr<InsAlgTemplate2> algTemplate2 = std::make_shared<InsAlgTemplate2>(param, topoInfo->userRank, temp0HierarchyInfo);
    std::shared_ptr<InsAlgTemplate3> algTemplate3 = std::make_shared<InsAlgTemplate3>(param, topoInfo->userRank, temp1HierarchyInfo);

   // 计算资源
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    AlgResourceRequest intraTempRequest1;
    AlgResourceRequest interTempRequest1;
    AlgResourceRequest intraTempRequestFinal;
    AlgResourceRequest interTempRequestFinal;

    if (param.engine == COMM_ENGINE_CCU) {
        root_ = param.root;
        // 计算localRankSize和localRoot
        intraLocalRankSize_ = GetRankSize(temp0HierarchyInfo);
        interLocalRankSize_ = GetRankSize(temp1HierarchyInfo);
        rankSize_ = intraLocalRankSize_ * interLocalRankSize_;
        HCCL_INFO("[CalcRes] localRankSize: myRank[%d] intraLocalRankSize[%u] interLocalRankSize[%u] rankSize_[%u]",
                myRank_, intraLocalRankSize_, interLocalRankSize_, rankSize_);
        CHK_RET(CalcLocalRoot());
        algTemplate0->SetRoot(intraLocalRoot_);
        algTemplate1->SetRoot(interLocalRoot_);
    }
    CHK_RET(algTemplate0->CalcRes(comm, param, topoInfo, intraTempRequest));
    CHK_RET(algTemplate1->CalcRes(comm, param, topoInfo, interTempRequest));
    CHK_RET(algTemplate2->CalcRes(comm, param, topoInfo, intraTempRequest1));
    CHK_RET(algTemplate3->CalcRes(comm, param, topoInfo, interTempRequest1)); 

    for (auto &KernelInfo : intraTempRequest.ccuKernelInfos) {
        KernelInfo.resGroup = 0;
    }
    for (auto &KernelInfo : interTempRequest.ccuKernelInfos) {
        KernelInfo.resGroup = 0;
    }
    for (auto &KernelInfo : intraTempRequest1.ccuKernelInfos) {
        KernelInfo.resGroup = 1;
    }
    for (auto &KernelInfo : interTempRequest1.ccuKernelInfos) {
        KernelInfo.resGroup = 1;
    }

    u32 slaveThreadNumIntra = intraTempRequest.slaveThreadNum;
    if(intraTempRequest.slaveThreadNum >= intraTempRequest1.slaveThreadNum){
        intraTempRequestFinal.notifyNumPerThread = intraTempRequest.notifyNumPerThread;
    } else {
        slaveThreadNumIntra = intraTempRequest1.slaveThreadNum;
        intraTempRequestFinal.notifyNumPerThread = intraTempRequest1.notifyNumPerThread;
    }
    u32 slaveThreadNumInter = interTempRequest.slaveThreadNum;
    if(interTempRequest.slaveThreadNum >= interTempRequest1.slaveThreadNum){
        interTempRequestFinal.notifyNumPerThread = interTempRequest.notifyNumPerThread;
    } else {
        slaveThreadNumInter = interTempRequest1.slaveThreadNum;
        interTempRequestFinal.notifyNumPerThread = interTempRequest1.notifyNumPerThread;
    }

    resourceRequest.notifyNumOnMainThread = 2;  // 用于两个template间同步
    resourceRequest.slaveThreadNum = slaveThreadNumIntra + slaveThreadNumInter + 4;
    resourceRequest.notifyNumPerThread.emplace_back(intraTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.emplace_back(intraTempRequest1.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              intraTempRequestFinal.notifyNumPerThread.begin(),
                                              intraTempRequestFinal.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(interTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.emplace_back(interTempRequest1.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              interTempRequestFinal.notifyNumPerThread.begin(),
                                              interTempRequestFinal.notifyNumPerThread.end());
    if (param.engine != COMM_ENGINE_CCU) {
        resourceRequest.channels.emplace_back(intraTempRequest.channels[0]);
        resourceRequest.channels.emplace_back(interTempRequest.channels[0]);
    } else {
        // ccu scatter
        HCCL_INFO("[InsBroadcastParallelExecutor][CalcRes] intraTemplate has [%d] kernels.", intraTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            intraTempRequest.ccuKernelInfos.begin(),
                                            intraTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(intraTempRequest.ccuKernelNum[0]);
        HCCL_INFO("[InsBroadcastParallelExecutor][CalcRes] interTemplate has [%d] kernels.", interTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            interTempRequest.ccuKernelInfos.begin(),
                                            interTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(interTempRequest.ccuKernelNum[0]);
        // // ccu allgather
        HCCL_INFO("[InsBroadcastParallelExecutor][CalcRes] intraTemplate1 has [%d] kernels.", intraTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            intraTempRequest1.ccuKernelInfos.begin(),
                                            intraTempRequest1.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(intraTempRequest1.ccuKernelNum[0]);
        HCCL_INFO("[InsBroadcastParallelExecutor][CalcRes] interTemplate1 has [%d] kernels.", interTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            interTempRequest1.ccuKernelInfos.begin(),
                                            interTempRequest1.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(interTempRequest1.ccuKernelNum[0]);
    }

    HCCL_DEBUG("[InsBroadcastParallelExecutor][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u], "
               "channels[%u]", myRank_, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
               resourceRequest.channels.size());
    for (auto i = 0; i < resourceRequest.notifyNumPerThread.size(); i++) {
        HCCL_DEBUG("[InsBroadcastParallelExecutor][CalcRes] myRank[%u], notifyNumPerThread[%u]=[%u]", myRank_, i,
                   resourceRequest.notifyNumPerThread[i]);
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsBroadcastParallelExecutor][Orchestrate] Orchestrate Start");

    maxTmpMemSize_ = resCtx.cclMem.size;
    myRank_ = resCtx.topoInfo.userRank;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    HCCL_INFO("[InsBroadcastParallelExecutor][Orchestrate] threads_size[%d]", threads_.size());
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
        intraLinks_ = remoteRankToChannelInfo_[0];
        interLinks_ = remoteRankToChannelInfo_[1];
    }
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ =  DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    root_ = param.root; 
    // 获取算法Topo信息
    vTopo_ = resCtx.algHierarchyInfo.infos;         // 本通信域内的通信平面

    if(resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix) {
        temp0HierarchyInfo_ = {resCtx.algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        u32 meshSize = resCtx.algHierarchyInfo.infos[0][0].size();
        for(auto rank : resCtx.algHierarchyInfo.infos[0][1]) {
            if(rank % meshSize == resCtx.topoInfo.userRank % meshSize) {
                closRanks.push_back(rank);
            }
        }
        temp1HierarchyInfo_ = {closRanks};
    } else {
        temp0HierarchyInfo_ = resCtx.algHierarchyInfo.infos[0];
        temp1HierarchyInfo_ = resCtx.algHierarchyInfo.infos[1];
    }

    // 计算localRankSize和localRoot
    intraLocalRankSize_ = GetRankSize(temp0HierarchyInfo_);      
    interLocalRankSize_ = GetRankSize(temp1HierarchyInfo_);
    rankSize_ = intraLocalRankSize_ * interLocalRankSize_;
    HCCL_INFO("[Orchestrate] localRankSize: myRank[%d] intraLocalRankSize[%u] interLocalRankSize[%u] rankSize_[%u]",
              myRank_, intraLocalRankSize_, interLocalRankSize_, rankSize_);

    CHK_RET(CalcLocalRoot());

    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntra(param, resCtx.topoInfo.userRank, temp0HierarchyInfo_);
    InsAlgTemplate1 tempAlgInter(param, resCtx.topoInfo.userRank, temp1HierarchyInfo_);
    InsAlgTemplate2 tempAlgIntra1(param, resCtx.topoInfo.userRank, temp0HierarchyInfo_);
    InsAlgTemplate3 tempAlgInter1(param, resCtx.topoInfo.userRank, temp1HierarchyInfo_);

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx, tempAlgIntra, tempAlgInter, tempAlgIntra1, tempAlgInter1);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsBroadcastParallelExecutor][Orchestrate]errNo[0x%016llx] Reduce scatter excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::GetParallelDataSplit(
    std::vector<float> &splitDataSize) const
{
    // to do 先做等分，后续根据性能做调整
    double splitData = 0.5;
    splitDataSize.push_back(splitData);
    splitDataSize.push_back(splitData);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
uint64_t InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo) const
{
    uint64_t count = 1;
    for (const auto &i : vTopo) {
        count *= i.size();
    }
    return count;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::CalcLocalRoot()
{
    CHK_PRT_RET(root_ >= rankSize_,
        HCCL_ERROR("[CalcLocalRoot] root[%u] is out of rankSize[%u]", root_, rankSize_),
        HcclResult::HCCL_E_INTERNAL);
    rankIdxLevel0_ = myRank_ % intraLocalRankSize_;
    rankIdxLevel1_ = myRank_ / intraLocalRankSize_;
    interLocalRoot_ = root_ / intraLocalRankSize_ * intraLocalRankSize_ + rankIdxLevel0_;
    intraLocalRoot_ = root_ % intraLocalRankSize_ + rankIdxLevel1_ * intraLocalRankSize_;
    HCCL_DEBUG("[DEBUG] new myrank[%u], rankIdxLevel0_[%u] ,rankIdxLevel1_[%u] interlocalroot[%u] intralocalroot[%u]",
               myRank_, rankIdxLevel0_,rankIdxLevel1_, interlocalroot, intralocalroot);
    HCCL_INFO("[CalcLocalRoot] localRoot: myRank[%d] intraLocalRoot[%u] interLocalRoot[%u]",
              myRank_, intraLocalRoot_, interLocalRoot_);
    return HcclResult::HCCL_SUCCESS;
}

// Aicpu
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::PrepareResForTemplate(
    InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter, InsAlgTemplate2 &tempAlgIntra1)
{

    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    AlgResourceRequest intraTempRequest1;
    tempAlgIntra.GetRes(intraTempRequest);
    tempAlgInter.GetRes(interTempRequest);
    tempAlgIntra1.GetRes(intraTempRequest1);
    auto intraThreadsNum = intraTempRequest.slaveThreadNum + 1;
    auto intraThreadsNum1 = intraTempRequest1.slaveThreadNum + 1;
    auto intraThreadsNumFinal = std::max(intraThreadsNum, intraThreadsNum1);
    auto intraNotifyOnMainThread = intraTempRequest.notifyNumOnMainThread;
    auto interNotifyOnMainThread = interTempRequest.notifyNumOnMainThread;

    intraThreads_.clear();
    intraThreads_.emplace_back(threads_[1]);
    if (intraThreadsNum + 2 >= 3){
        for(u32 i = 3 ; i < intraThreadsNum + 2; i++) {
            intraThreads_.emplace_back(threads_[i]);
        }
    }

    interThreads_.clear();
    interThreads_.emplace_back(threads_[intraThreadsNumFinal + 2]);
    if (threads_.size() >= intraThreadsNumFinal + 4){
        for(u32 i = intraThreadsNumFinal + 4 ; i < threads_.size(); i++) {
            interThreads_.emplace_back(threads_[i]);
        }
    }

    // 用于两个算法同步
    mainThread_ = threads_.at(0);
    templateMainThreads_.clear();
    templateMainThreads_.emplace_back(intraThreads_.at(0));
    templateMainThreads_.emplace_back(interThreads_.at(0));
    syncNotifyOnTemplates_ = {intraNotifyOnMainThread, interNotifyOnMainThread};
    syncNotifyOnMain_ = {0, 1};

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::PrepareResForTemplate23(
    InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate2 &tempAlgIntra1, InsAlgTemplate3 &tempAlgInter1)
{

    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest1;
    AlgResourceRequest intraTempRequest1;
    tempAlgIntra.GetRes(intraTempRequest);
    tempAlgInter1.GetRes(interTempRequest1);
    tempAlgIntra1.GetRes(intraTempRequest1);
    auto intraThreadsNum = intraTempRequest.slaveThreadNum + 1;
    auto intraThreadsNum1 = intraTempRequest1.slaveThreadNum + 1;
    auto intraThreadsNumFinal = std::max(intraThreadsNum, intraThreadsNum1);
    auto intraNotifyOnMainThread = intraTempRequest1.notifyNumOnMainThread;
    auto interNotifyOnMainThread = interTempRequest1.notifyNumOnMainThread;

    intraThreads_.assign(threads_.begin() + 2, threads_.begin() + intraThreadsNum1 + 2);
    interThreads_.assign(threads_.begin() + intraThreadsNumFinal + 3, threads_.end());
    // 用于两个算法同步
    mainThread_ = threads_.at(0);

    templateMainThreads_.clear();
    templateMainThreads_.emplace_back(intraThreads_.at(0));
    templateMainThreads_.emplace_back(interThreads_.at(0));
    syncNotifyOnTemplates_ = {intraNotifyOnMainThread, interNotifyOnMainThread};
    syncNotifyOnMain_ = {0, 1};

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::PrepareResForTemplateResource(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, TemplateResource &intraTempAlgRes, TemplateResource &interTempAlgRes, bool isScatter)
{
    if (param.engine == COMM_ENGINE_CCU) {
        if (isScatter) {
            intraTempAlgRes.ccuKernels.insert(intraTempAlgRes.ccuKernels.end(),
                                              resCtx.ccuKernels.begin(),
                                              resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
            interTempAlgRes.ccuKernels.insert(interTempAlgRes.ccuKernels.end(),
                                                resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
                                                resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);
        } else {
            intraTempAlgRes.ccuKernels.insert(intraTempAlgRes.ccuKernels.end(),
                                              resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1],
                                              resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1] + resCtx.ccuKernelNum[2]);
            interTempAlgRes.ccuKernels.insert(interTempAlgRes.ccuKernels.end(),
                                                resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1] + resCtx.ccuKernelNum[2],
                                                resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1] + resCtx.ccuKernelNum[2] + resCtx.ccuKernelNum[3]);
        }
    } else {
        intraTempAlgRes.channels = intraLinks_;
        interTempAlgRes.channels = interLinks_;
    }
    intraTempAlgRes.threads = intraThreads_;
    intraTempAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    for(auto i: intraTempAlgRes.channels) {
        HCCL_DEBUG("[InsBroadcastParallelExecutor][PrepareResForTemplateResource],intraTempAlgRes.channels, myRank_[%u], channels[%u]= size[%u] ",
        myRank_, i.first, i.second.size());
    }
    interTempAlgRes.threads = interThreads_;
    interTempAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    for(auto i: interTempAlgRes.channels) {
        HCCL_DEBUG("[InsBroadcastParallelExecutor][PrepareResForTemplateResource],interTempAlgRes.channels, myRank_[%u], channels[%u]= size[%u] ",
        myRank_, i.first, i.second.size());
    }
    HCCL_DEBUG("[InsBroadcastParallelExecutor][PrepareResForTemplateResource] AlgTemplate intraThreads_size[%d] "
                "interThreads_size[%d] ccuKernelNumSize[%zu] ccuKernelsSize[%zu] isScatter[%u]",
                intraThreads_.size(), interThreads_.size(), resCtx.ccuKernelNum.size(), resCtx.ccuKernels.size(), isScatter);

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::GenDataParamstempAlg(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 sliceCount, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, const u32 LocalRankSize) const
{
    dataParams.buffInfo.inputPtr = dataParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER ? resCtx.cclMem.addr : param.inputPtr;
    dataParams.buffInfo.outputPtr = dataParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER ? resCtx.cclMem.addr : param.inputPtr;
    dataParams.buffInfo.inputSize = dataParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER ? resCtx.cclMem.size : param.inputSize;
    dataParams.buffInfo.outputSize = dataParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER ? resCtx.cclMem.size : param.outputSize;
    dataParams.buffInfo.hcclBuff = resCtx.cclMem;
    dataParams.buffInfo.hcclBuffBaseOff = scratchOffsetCount * dataTypeSize_;
    dataParams.buffInfo.inBuffBaseOff = dataParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER ? dataParams.buffInfo.hcclBuffBaseOff : dataOffset;
    dataParams.buffInfo.outBuffBaseOff = dataParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER ? dataParams.buffInfo.hcclBuffBaseOff : dataOffset;
    GenDataParamsAllRank(sliceCount, LocalRankSize, dataParams);
    dataParams.repeatNum = 1;
    dataParams.inputRepeatStride = 0;
    dataParams.outputRepeatStride = 0;
    HCCL_INFO("[InsBroadcastParallelExecutor][GenDataParamstempAlg]myRank_[%u] hcclBuffBaseOff[%llu]",myRank_, dataParams.buffInfo.hcclBuffBaseOff);
    
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::GenDataParamsAllRank(
    const u64 sliceCount, const u32 LocalRankSize, TemplateDataParams &dataParams) const
{
    u64 curSize = 0;
    u64 sliceSize = sliceCount * dataTypeSize_;
    u64 rankStride = RoundDown(sliceSize, (LocalRankSize * dataTypeSize_)) * dataTypeSize_;
    for (u64 i = 0; i < LocalRankSize; i++) {
        curSize = (i == (LocalRankSize - 1)) ? (sliceSize - rankStride * i) : rankStride;
    }
    dataParams.tailSize = curSize;
    dataParams.sliceSize = rankStride;
    dataParams.count = dataParams.sliceSize / dataTypeSize_;
    dataParams.inputSliceStride = rankStride;
    dataParams.outputSliceStride = rankStride;
    HCCL_INFO("[InsBroadcastParallelExecutor][GenDataParamsAllRank] tailSize[%llu] sliceSize[%llu] count[%llu] inputSliceStride[%llu] outputSliceStride[%llu]",
                dataParams.tailSize, dataParams.sliceSize, dataParams.count, dataParams.inputSliceStride, dataParams.outputSliceStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::GenDataParamsBufferType(
    const BufferType inBuffType, const BufferType outBuffType, const BufferType hcclBuffType, TemplateDataParams &dataParams) const
{
    dataParams.buffInfo.inBuffType = inBuffType;
    dataParams.buffInfo.outBuffType = outBuffType;
    dataParams.buffInfo.hcclBuffType = hcclBuffType;

    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::PrePareDataParamstempAlgInter(
    const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount)
{
    u32 myRankIdx = tempVirtRankMapInter_.at(myRank_);
    u64 sendDataSize = 0;
    u64 curSize = 0;
    allRankDisplsInter_.clear();
    allRankSliceSizeInter_.clear();
    u64 rankStride = RoundDown(currCountPart * dataTypeSize_, (intraLocalRankSize_ * dataTypeSize_)) * dataTypeSize_;
    for (u64 i = 0; i < intraLocalRankSize_; i++) {
        curSize = (i == (intraLocalRankSize_ - 1)) ? (currCountPart * dataTypeSize_ - rankStride * i) : rankStride;
        allRankSliceSizeInter_.push_back(curSize);
        allRankDisplsInter_.push_back(sendDataSize);
        sendDataSize += curSize;
    }
    dataOffset0Inter_ = dataOffset + allRankDisplsInter_[myRankIdx];
    currCountPart0_ = allRankSliceSizeInter_[myRankIdx] / dataTypeSize_;
    scratchOffsetCountInterStage1_ = scratchOffsetCount + allRankDisplsInter_[myRankIdx] / dataTypeSize_;

    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::PrePareDataParamstempAlgIntra(
    const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount)
{
    u64 myRankIdx = tempVirtRankMapIntra_.at(myRank_);
    u64 sendDataSize = 0;
    u64 curSize = 0;
    allRankDisplsInter_.clear();
    allRankSliceSizeInter_.clear();
    u64 rankStride = RoundDown(currCountPart * dataTypeSize_, (interLocalRankSize_ * dataTypeSize_)) * dataTypeSize_;
    for (u64 i = 0; i < interLocalRankSize_; i++) {
        curSize = (i == (interLocalRankSize_ - 1)) ? (currCountPart  * dataTypeSize_ - rankStride * i) : rankStride;
        allRankSliceSizeInter_.push_back(curSize);
        allRankDisplsInter_.push_back(sendDataSize);
        sendDataSize += curSize;
    }
    dataOffset0Intra_ = dataOffset + allRankDisplsInter_[myRankIdx];
    currCountPart1_ = allRankSliceSizeInter_[myRankIdx] / dataTypeSize_;
    scratchOffsetCountIntraStage1_ = scratchOffsetCount + allRankDisplsInter_[myRankIdx] / dataTypeSize_;

    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter,
    InsAlgTemplate2 &tempAlgIntra1, InsAlgTemplate3 &tempAlgInter1)
{
    HCCL_INFO("[InsBroadcastParallelExecutor] AlgTemplate intra server is [%s]", tempAlgIntra.Describe().c_str());
    HCCL_INFO("[InsBroadcastParallelExecutor] AlgTemplate inter server is [%s]", tempAlgInter.Describe().c_str());

    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PTR_NULL(resCtx.cclMem.addr);

    std::vector<float> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);

    u32 multipleIntra = tempAlgIntra.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 multipleInter = tempAlgInter.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    if(multipleIntra > 0 || multipleInter > 0){
        multipleIntra = 1;
        multipleInter = 1;
    }

    // 按照intraData0+interData1，以及intraData1+interData0两种方式分别计算，取multiple最大需求
    float multiple0 = dataSplitSize.at(0) * float(multipleIntra) + dataSplitSize.at(1) * float(multipleInter);
    float multiple1 = dataSplitSize.at(1) * float(multipleIntra) + dataSplitSize.at(0) * float(multipleInter);
    float multiple = std::max(multiple0, multiple1);

    // 数据切分
    u64 sliceCountUB = std::min(static_cast<u64>(UB_MAX_DATA_SIZE) / dataTypeSize_, dataCount_);
    float onceSliceCountPercent = std::max(dataSplitSize.at(0) * float(1.0 / intraLocalRankSize_), dataSplitSize.at(1) * float(1.0 / interLocalRankSize_));
    u64 sliceCountUB0 = onceSliceCountPercent > 0 ? std::floor(sliceCountUB / onceSliceCountPercent) : sliceCountUB;
    u64 sliceCount = sliceCountUB;
    if (multiple > 0 && maxTmpMemSize_ > 0) {
        u64 scratchCount = maxTmpMemSize_ / dataTypeSize_;  // 按照count来切分
        sliceCount = std::min(static_cast<u64>(float(scratchCount) / multiple), sliceCountUB0);
        sliceCount = std::min(sliceCount, dataCount_);
    }
    HCCL_DEBUG("[InsBroadcastParallelExecutor][OrchestrateLoop] dataCount_[%lu], myRank_[%d], sliceCountUB[%d], sliceCountUB0[%d], sliceCount[%d]",
              dataCount_, myRank_, sliceCountUB, sliceCountUB0, sliceCount);

    u64 sliceCountPart0 = static_cast<u64>(float(sliceCount) * dataSplitSize.at(0));
    u64 sliceCountPart1 = sliceCount - sliceCountPart0;

    if(sliceCount == 0){
        HCCL_WARNING("The divisor cannot be zero.");
        return HcclResult::HCCL_SUCCESS;
    }
    // 计算循环次数
    u32 loopTimes = dataCount_ / sliceCount + ((dataCount_ % sliceCount == 0) ? 0 : 1);
    // 计算尾块
    u64 finalSliceCount = dataCount_ - (loopTimes - 1) * sliceCount;
    u64 finalSliceCountPart0 = static_cast<u64>(float(finalSliceCount) * dataSplitSize.at(0));
    u64 finalSliceCountPart1 = finalSliceCount - finalSliceCountPart0;
    // 计算Scratch偏移，数据尾块必然小于常规块，不用额外计算尾块时的Scratch偏移
    u64 scratchOffsetCountIntraStage0 = 0;
    u64 scratchOffsetCountInterStage0 = sliceCountPart0 * multipleIntra;
    u64 scratchOffsetCountInterStage1 = 0;
    u64 scratchOffsetCountIntraStage1 = sliceCountPart0 * multipleInter;
    HCCL_DEBUG("[InsBroadcastParallelExecutor][OrchestrateLoop] dataCount_[%lu], myRank_[%d], sliceCountPart0[%d], multipleIntra[%d]",
              dataCount_, myRank_, sliceCountPart0, multipleIntra);
    HCCL_DEBUG("[InsBroadcastParallelExecutor][OrchestrateLoop] myRank_[%d],scratchOffsetCountInterStage0[%d], scratchOffsetCountIntraStage1[%d] loopTimes[%u]",
               myRank_, scratchOffsetCountInterStage0, scratchOffsetCountIntraStage1, loopTimes);
    TemplateDataParams tempAlgParamsIntra0;
    TemplateDataParams tempAlgParamsInter0;
    TemplateDataParams tempAlgParamsInter1;
    TemplateDataParams tempAlgParamsIntra1;
    TemplateDataParams tempAlgParamsIntra01;
    TemplateDataParams tempAlgParamsInter01;
    TemplateDataParams tempAlgParamsInter11;
    TemplateDataParams tempAlgParamsIntra11;

    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCountPart0 = (loopIndex == loopTimes - 1) ? finalSliceCountPart0 : sliceCountPart0;
        u64 currCountPart1 = (loopIndex == loopTimes - 1) ? finalSliceCountPart1 : sliceCountPart1;
        u64 dataOffset0 = loopIndex * sliceCount * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + currCountPart0 * dataTypeSize_;
        // 计算算法模板所需资源
        TemplateResource intraTempAlgRes;
        TemplateResource interTempAlgRes;
        CHK_RET(PrepareResForTemplate(tempAlgIntra, tempAlgInter, tempAlgIntra1));
        PrepareResForTemplateResource(param, resCtx, intraTempAlgRes, interTempAlgRes, true);

        // 第一步开始前同步
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
        HCCL_DEBUG("[InsBroadcastParallelExecutor][OrchestrateLoop] RunTemplateInter1 myRank_[%d], dataOffset0[%d], currCountPart0[%d], loopTimes[%d]",
                   myRank_, dataOffset0, currCountPart0, loopTimes);
        HCCL_DEBUG("[InsBroadcastParallelExecutor][OrchestrateLoop] RunTemplateInter1 myRank_[%d], dataOffset0[%d], currCountPart0[%d], scratchOffsetCountInterStage0[%d] "
                    "dataOffset1[%d], currCountPart1[%d]",
                   myRank_, dataOffset0, currCountPart0, scratchOffsetCountInterStage0, dataOffset1, currCountPart1);
        RunTemplateIntra0(param, resCtx, dataOffset0, currCountPart0, scratchOffsetCountIntraStage0, tempAlgParamsIntra0, intraTempAlgRes, tempAlgIntra);
        RunTemplateInter1(param, resCtx, dataOffset1, currCountPart1, scratchOffsetCountInterStage0, tempAlgParamsInter1, interTempAlgRes, tempAlgInter);
        // 第一步做完后回到主流做尾同步
        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
#ifndef AICPU_COMPILE
    if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU) {
        ccuKernelLaunchNumIntra0_ = intraTempAlgRes.submitInfos.size();
        ccuKernelLaunchNumInter1_ = interTempAlgRes.submitInfos.size();
    }
#endif

        // 第二步开始前同步
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
        //server 间地址偏移
        for (int i = 0; i < temp0HierarchyInfo_[0].size(); i++) {
            tempVirtRankMapInter_.insert(std::make_pair(temp0HierarchyInfo_[0][i], i));
        }
        RunTemplateInter0(param, resCtx, dataOffset0, currCountPart0, scratchOffsetCountInterStage1, tempAlgParamsInter0, interTempAlgRes, tempAlgInter);

        //server 内地址偏移
        for (int i = 0; i < temp1HierarchyInfo_[0].size(); i++) {
            tempVirtRankMapIntra_.insert(std::make_pair(temp1HierarchyInfo_[0][i], i));
        }
        RunTemplateIntra1(param, resCtx, dataOffset1, currCountPart1, scratchOffsetCountIntraStage1, tempAlgParamsIntra1, intraTempAlgRes, tempAlgIntra);
        // 尾同步
        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
#ifndef AICPU_COMPILE
    if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU) {
        ccuKernelLaunchNumInter0_ = interTempAlgRes.submitInfos.size() - ccuKernelLaunchNumInter1_;
        ccuKernelLaunchNumIntra1_ = intraTempAlgRes.submitInfos.size() - ccuKernelLaunchNumIntra0_;
    }
#endif
        // 计算算法模板所需资源 InsAlgTemplate2 &tempAlgIntra1, InsAlgTemplate3 &tempAlgInter1
        TemplateResource intraTempAlgRes1;
        TemplateResource interTempAlgRes1;
        CHK_RET(PrepareResForTemplate23(tempAlgIntra, tempAlgIntra1, tempAlgInter1));
        PrepareResForTemplateResource(param, resCtx, intraTempAlgRes1, interTempAlgRes1, false);

        // 第三步开始前同步
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
        RunTemplateInter01(param, resCtx, dataOffset0, currCountPart0, scratchOffsetCountInterStage1, tempAlgParamsInter01, interTempAlgRes1, tempAlgInter1);
        RunTemplateIntra11(param, resCtx, dataOffset1, currCountPart1, scratchOffsetCountIntraStage1, tempAlgParamsIntra11, intraTempAlgRes1, tempAlgIntra1);
        // 尾同步
        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
#ifndef AICPU_COMPILE
    if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU) {
        ccuKernelLaunchNumInter01_ = interTempAlgRes1.submitInfos.size();
        ccuKernelLaunchNumIntra11_ = intraTempAlgRes1.submitInfos.size();
    }
#endif
        // 第四步开始前同步
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
        RunTemplateIntra01(param, resCtx, dataOffset0, currCountPart0, scratchOffsetCountIntraStage0, tempAlgParamsIntra01, intraTempAlgRes1, tempAlgIntra1);
        RunTemplateInter11(param, resCtx, dataOffset1, currCountPart1, scratchOffsetCountInterStage0, tempAlgParamsInter11, interTempAlgRes1, tempAlgInter1);
        // 第四步做完后回到主流做尾同步
        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
#ifndef AICPU_COMPILE
    if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU && param.opMode != OpMode::OFFLOAD) {
        CHK_RET(FastLaunchSaveCtx(param, intraTempAlgRes, interTempAlgRes, intraTempAlgRes1, interTempAlgRes1, resCtx.notifyNumOnMainThread));
    }
#endif
    }

    HCCL_INFO("[InsBroadcastParallelExecutor][OrchestrateLoop] End.myRank[%u]", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::FastLaunchSaveCtx(
    const OpParam &param, const TemplateResource &intraTempAlgRes, const TemplateResource &interTempAlgRes,
    const TemplateResource &intraTempAlgRes1, const TemplateResource &interTempAlgRes1, u32 notifyNumOnMainThread)
{
    HCCL_INFO("[InsBroadcastParallelExecutor][FastLaunchSaveCtx] loopTimes==1, save fast launch ctx.");
    ccuKernelLaunchNumIntra01_ = intraTempAlgRes1.submitInfos.size() - ccuKernelLaunchNumIntra11_;
    ccuKernelLaunchNumInter11_ = interTempAlgRes1.submitInfos.size() - ccuKernelLaunchNumInter01_;
    u32 threadNum = threads_.size();
    u32 ccuKernelNum = ccuKernelLaunchNumIntra1_ + ccuKernelLaunchNumInter0_ + ccuKernelLaunchNumIntra0_ + ccuKernelLaunchNumInter1_ +
                        ccuKernelLaunchNumIntra11_ + ccuKernelLaunchNumInter01_ + ccuKernelLaunchNumIntra01_ + ccuKernelLaunchNumInter11_;
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsBroadcastParallelExecutor][FastLaunchSaveCtx] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[InsBroadcastParallelExecutor][FastLaunchSaveCtx] threadNum[%llu], ccuKernelNum[%llu]", threadNum, ccuKernelNum);

    u64 size = CcuFastLaunchCtx::GetCtxSize(threadNum, ccuKernelNum);
    // 申请ctx
    void *ctxPtr = nullptr;
    HCCL_INFO("[InsBroadcastParallelExecutor][FastLaunchSaveCtx] Tag[%s], size[%llu]", param.fastLaunchTag, size);
    CHK_RET(HcclEngineCtxCreate(param.hcclComm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, size, &ctxPtr));

    CcuFastLaunchCtx *ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(ctxPtr);
    // 1 算法名
    CHK_SAFETY_FUNC_RET(strcpy_s(ccuFastLaunchCtx->algName, sizeof(ccuFastLaunchCtx->algName), param.algName));
    HCCL_INFO("[InsBroadcastParallelExecutor][FastLaunchSaveCtx] algName[%s]", ccuFastLaunchCtx->algName);

    // 2 thread
    ccuFastLaunchCtx->threadNum = threadNum;
    ccuFastLaunchCtx->notifyNumOnMainThread = notifyNumOnMainThread;
    ThreadHandle *threads = ccuFastLaunchCtx->GetThreadHandlePtr();
    for (u32 i = 0; i < threadNum; i++) {
        threads[i] = threads_[i];
    }

    // 3 ccu kernel handle, taskArg入参
    u32 templateIdx = 0;
    //scatter
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumIntra0_;
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumInter1_;
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumInter0_;
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumIntra1_;
    // allgather
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumInter01_;
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumIntra11_;
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumIntra01_;
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumInter11_;
    CcuKernelSubmitInfo *kernelSubmitInfos = ccuFastLaunchCtx->GetCcuKernelSubmitInfoPtr();

    u32 kernelIdx = 0;
    // scatter
    for (u32 i = 0; i < ccuKernelLaunchNumIntra0_; i++) {
        kernelSubmitInfos[kernelIdx++] = intraTempAlgRes.submitInfos[i];
    }
    for (u32 i = 0; i < ccuKernelLaunchNumInter1_; i++) {
        kernelSubmitInfos[kernelIdx++] = interTempAlgRes.submitInfos[i];
    }
    for (u32 i = ccuKernelLaunchNumInter1_; i < ccuKernelLaunchNumInter0_ + ccuKernelLaunchNumInter1_; i++) {
        kernelSubmitInfos[kernelIdx++] = interTempAlgRes.submitInfos[i];
    }
    for (u32 i = ccuKernelLaunchNumIntra0_; i < ccuKernelLaunchNumIntra1_ + ccuKernelLaunchNumIntra0_; i++) {
        kernelSubmitInfos[kernelIdx++] = intraTempAlgRes.submitInfos[i];
    }
    // allgather
    for (u32 i = 0; i < ccuKernelLaunchNumInter01_; i++) {
        kernelSubmitInfos[kernelIdx++] = interTempAlgRes1.submitInfos[i];
    }
    for (u32 i = 0; i < ccuKernelLaunchNumIntra11_; i++) {
        kernelSubmitInfos[kernelIdx++] = intraTempAlgRes1.submitInfos[i];
    }
    for (u32 i = ccuKernelLaunchNumIntra11_; i < ccuKernelLaunchNumIntra01_ + ccuKernelLaunchNumIntra11_; i++) {
        kernelSubmitInfos[kernelIdx++] = intraTempAlgRes1.submitInfos[i];
    }
    for (u32 i = ccuKernelLaunchNumInter01_; i < ccuKernelLaunchNumInter01_ + ccuKernelLaunchNumInter11_; i++) {
        kernelSubmitInfos[kernelIdx++] = interTempAlgRes1.submitInfos[i];
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::FastLaunch(
        const OpParam &param, const CcuFastLaunchCtx *ctx)
{
    InsAlgTemplate0 intraTempAlg{};
    InsAlgTemplate1 interTempAlg{};
    InsAlgTemplate2 intraTempAlg1{};
    InsAlgTemplate3 interTempAlg1{};

    // 保存scatter信息
    TemplateFastLaunchCtx tempFastLaunchCtxIntra0, tempFastLaunchCtxInter0;
    TemplateFastLaunchCtx tempFastLaunchCtxInter1, tempFastLaunchCtxIntra1;
    //保存allgather信息
    TemplateFastLaunchCtx tempFastLaunchCtxIntra01, tempFastLaunchCtxInter01;
    TemplateFastLaunchCtx tempFastLaunchCtxInter11, tempFastLaunchCtxIntra11;

    TemplateResource templateAlgResIntra, templateAlgResInter;
    ThreadHandle *threads = ctx->GetThreadHandlePtr();
    threads_.assign(threads, threads + ctx->threadNum);
    PrepareResForTemplate(intraTempAlg, interTempAlg, intraTempAlg1);

    CcuKernelSubmitInfo *ccuKernelSubmitInfos = ctx->GetCcuKernelSubmitInfoPtr();

    std::vector<float> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);
    dataCount_ = param.DataDes.count;
    u64 SliceCountPart0 = static_cast<u64>(float(dataCount_) * dataSplitSize.at(0));
    u64 SliceCountPart1 = dataCount_ - SliceCountPart0;

    //第一步开始前同步
    HCCL_INFO("[InsBroadcastParallelExecutor][FastLaunch] Intra0 ccuKernelNum[%llu]", ctx->ccuKernelNum[0]);
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
    //数据0的server内的mesh算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra0, param.inputPtr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxIntra0.threads = intraThreads_;
    tempFastLaunchCtxIntra0.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[0]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[0];
    //数据1的server间的nhr算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter1, param.inputPtr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxInter1.threads = interThreads_;
    tempFastLaunchCtxInter1.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[1]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[1];
    //把每个template需要的queue传进去，比如stars的mesh要传多条queue
    CHK_RET(FastLaunchTemplateIntra0(param, ctx->ccuKernelNum[0], tempFastLaunchCtxIntra0, intraTempAlg));
    CHK_RET(FastLaunchTemplateInter1(param, ctx->ccuKernelNum[1], tempFastLaunchCtxInter1, interTempAlg));
    //第一步做完后回到主流做尾同步
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

    //第二步开始前同步
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
    //数据0的server间的nhr算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter0, param.hcclBuff.addr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxInter0.threads = interThreads_;
    tempFastLaunchCtxInter0.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[2]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[2];
    //数据1的server内的mesh算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra1, param.hcclBuff.addr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxIntra1.threads = intraThreads_;
    tempFastLaunchCtxIntra1.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[3]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[3];
    // step2 scatter 数据0的server间的nhr算法
    CHK_RET(FastLaunchTemplateInter0(param, ctx->ccuKernelNum[2], tempFastLaunchCtxInter0, interTempAlg));
    CHK_RET(FastLaunchTemplateIntra1(param, ctx->ccuKernelNum[3], tempFastLaunchCtxIntra1, intraTempAlg));

    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
    CHK_RET(PrepareResForTemplate23(intraTempAlg, intraTempAlg1, interTempAlg1));
    // step 3
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
    // 数据0 allgather nhr
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter01, param.hcclBuff.addr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxInter01.threads = interThreads_;
    tempFastLaunchCtxInter01.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[4]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[4];
    // 数据1 allgather mesh
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra11, param.hcclBuff.addr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxIntra11.threads = intraThreads_;
    tempFastLaunchCtxIntra11.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[5]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[5];
    CHK_RET(FastLaunchTemplateInter01(param, ctx->ccuKernelNum[4], tempFastLaunchCtxInter01, interTempAlg1));
    CHK_RET(FastLaunchTemplateIntra11(param, ctx->ccuKernelNum[5], tempFastLaunchCtxIntra11, intraTempAlg1));
    //尾同步
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
    //step 4
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
    //数据0 allgather mesh
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra01, param.hcclBuff.addr, param.inputPtr, param.hcclBuff));
    tempFastLaunchCtxIntra01.threads = intraThreads_;
    tempFastLaunchCtxIntra01.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[6]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[6];
    //数据1的 allgather nhr
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter11, param.hcclBuff.addr, param.inputPtr, param.hcclBuff));
    tempFastLaunchCtxInter11.threads = interThreads_;
    tempFastLaunchCtxInter11.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[7]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[7];
    CHK_RET(FastLaunchTemplateIntra01(param, ctx->ccuKernelNum[6], tempFastLaunchCtxIntra01, intraTempAlg1));
    CHK_RET(FastLaunchTemplateInter11(param, ctx->ccuKernelNum[7], tempFastLaunchCtxInter11, interTempAlg1));
    //尾同步
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

    HCCL_INFO("[InsBroadcastParallelExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateIntra0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, TemplateResource& templateResource, InsAlgTemplate0 &tempAlgIntra)
{
    // server内topo包含root_的rank进行展开，其它rank不展开
    if (intraLocalRoot_ == root_ && currCountPart > 0) {
        //数据0的server内的mesh算法
        HCCL_INFO("[InsBroadcastParallelExecutor][RunTemplateIntra0] myRank_[%u] intraLocalRoot[%u] currCountPart[%u]",
                    myRank_, intraLocalRoot_, currCountPart);
        GenDataParamsBufferType(BufferType::INPUT, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, dataParams);
        GenDataParamstempAlg(param, resCtx, dataOffset, currCountPart, scratchOffsetCount, dataParams, intraLocalRankSize_);
        CHK_RET(tempAlgIntra.KernelRun(param, dataParams, templateResource));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::FastLaunchTemplateIntra0(
    const OpParam &param, const u32 kernelNum, TemplateFastLaunchCtx &tempFastLaunchCtxIntra, InsAlgTemplate0 &tempAlgIntra)
{
    if (kernelNum > 0) {
        //数据0的server内的mesh算法
        HCCL_INFO("[InsBroadcastParallelExecutor][FastLaunchTemplateIntra0] kernelNum[%u]", kernelNum);
        CHK_RET(tempAlgIntra.FastLaunch(param, tempFastLaunchCtxIntra));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateInter1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, TemplateResource& templateResource, InsAlgTemplate1 &tempAlgInter)
{
    // server间topo包含root_的rank进行展开，其它rank不展开
    if (interLocalRoot_ == root_ && currCountPart > 0) {
        //数据1的server间的nhr算法
        HCCL_INFO("[InsBroadcastParallelExecutor][RunTemplateInter1] myRank_[%u] interLocalRoot[%u] currCountPart[%u]",
                    myRank_, interLocalRoot_, currCountPart);
        GenDataParamsBufferType(BufferType::INPUT, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, dataParams);
        GenDataParamstempAlg(param, resCtx, dataOffset, currCountPart, scratchOffsetCount, dataParams, interLocalRankSize_);
        CHK_RET(tempAlgInter.KernelRun(param, dataParams, templateResource));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::FastLaunchTemplateInter1(
    const OpParam &param, const u32 kernelNum, TemplateFastLaunchCtx &tempFastLaunchCtxIntra, InsAlgTemplate1 &tempAlgInter)
{
    if (kernelNum > 0) {
        //数据1的server间的nhr算法
        HCCL_INFO("[InsBroadcastParallelExecutor][FastLaunchTemplateInter1] kernelNum[%u]", kernelNum);
        CHK_RET(tempAlgInter.FastLaunch(param, tempFastLaunchCtxIntra));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateInter0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, TemplateResource& templateResource, InsAlgTemplate1 &tempAlgInter)
{
    PrePareDataParamstempAlgInter(dataOffset, currCountPart, scratchOffsetCount);
    if (currCountPart0_ > 0) {
        // 数据0的server间的nhr算法
        HCCL_DEBUG("[InsBroadcastParallelExecutor][RunTemplateInter0] myRank_[%d], dataOffset[%d], currCountPart[%d], scratchOffsetCountInterStage[%d]",
                   myRank_, dataOffset0Inter_, currCountPart0_, scratchOffsetCountInterStage1_);
        GenDataParamsBufferType(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, dataParams);
        GenDataParamstempAlg(param, resCtx, dataOffset0Inter_, currCountPart0_, scratchOffsetCountInterStage1_, dataParams, interLocalRankSize_);
        tempAlgInter.SetRoot(interLocalRoot_);
        CHK_RET(tempAlgInter.KernelRun(param, dataParams, templateResource));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::FastLaunchTemplateInter0(
    const OpParam &param, const u32 kernelNum, TemplateFastLaunchCtx &tempFastLaunchCtxInter, InsAlgTemplate1 &tempAlgInter)
{
    if (kernelNum > 0) {
        HCCL_INFO("[InsBroadcastParallelExecutor][FastLaunchTemplateInter0] kernelNum[%u]", kernelNum);
        CHK_RET(tempAlgInter.FastLaunch(param, tempFastLaunchCtxInter));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateIntra1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, TemplateResource& templateResource, InsAlgTemplate0 &tempAlgIntra)
{
    PrePareDataParamstempAlgIntra(dataOffset, currCountPart, scratchOffsetCount);
    if (currCountPart1_ > 0) {
        // 数据1的server内的mesh算法
        HCCL_DEBUG("[InsBroadcastParallelExecutor][RunTemplateIntra1] myRank_[%d], dataOffset[%d], currCountPart[%d], scratchOffsetCountInterStage[%d]",
                   myRank_, dataOffset0Intra_, currCountPart1_, scratchOffsetCountIntraStage1_);
        GenDataParamsBufferType(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, dataParams);
        GenDataParamstempAlg(param, resCtx, dataOffset0Intra_, currCountPart1_, scratchOffsetCountIntraStage1_, dataParams, intraLocalRankSize_);
        tempAlgIntra.SetRoot(intraLocalRoot_);
        CHK_RET(tempAlgIntra.KernelRun(param, dataParams, templateResource));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::FastLaunchTemplateIntra1(
    const OpParam &param, const u32 kernelNum, TemplateFastLaunchCtx &tempFastLaunchCtxIntra, InsAlgTemplate0 &tempAlgIntra)
{
    if (kernelNum > 0) {
        HCCL_INFO("[InsBroadcastParallelExecutor][FastLaunchTemplateIntra1] kernelNum[%u]", kernelNum);
        CHK_RET(tempAlgIntra.FastLaunch(param, tempFastLaunchCtxIntra));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateInter01(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, TemplateResource& templateResource, InsAlgTemplate3 &tempAlgInter1)
{
    PrePareDataParamstempAlgInter(dataOffset, currCountPart, scratchOffsetCount);
    if (currCountPart0_ > 0) {
        // 数据0的server间的nhr算法
        HCCL_DEBUG("[InsBroadcastParallelExecutor][RunTemplateInter01] myRank_[%d], dataOffset[%d], currCountPart[%d], scratchOffsetCountInterStage[%d]",
                   myRank_, dataOffset0Inter_, currCountPart0_, scratchOffsetCountInterStage1_);
        GenDataParamsBufferType(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, dataParams);
        GenDataParamstempAlg(param, resCtx, dataOffset0Inter_, currCountPart0_, scratchOffsetCountInterStage1_, dataParams, interLocalRankSize_);
        CHK_RET(tempAlgInter1.KernelRun(param, dataParams, templateResource));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::FastLaunchTemplateInter01(
    const OpParam &param, const u32 kernelNum, TemplateFastLaunchCtx &tempFastLaunchCtxInter, InsAlgTemplate3 &tempAlgInter1)
{
    if (kernelNum > 0) {
        HCCL_INFO("[InsBroadcastParallelExecutor][FastLaunchTemplateInter01] kernelNum[%u]", kernelNum);
        CHK_RET(tempAlgInter1.FastLaunch(param, tempFastLaunchCtxInter));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateIntra11(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, TemplateResource& templateResource, InsAlgTemplate2 &tempAlgIntra1)
{
    PrePareDataParamstempAlgIntra(dataOffset, currCountPart, scratchOffsetCount);
    if (currCountPart1_ > 0) {
        // 数据1的server内的mesh算法
        HCCL_DEBUG("[InsBroadcastParallelExecutor][RunTemplateIntra11] myRank_[%d], dataOffset[%d], currCountPart[%d], scratchOffsetCountInterStage[%d]",
                   myRank_, dataOffset0Intra_, currCountPart1_, scratchOffsetCountIntraStage1_);
        GenDataParamsBufferType(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, dataParams);
        GenDataParamstempAlg(param, resCtx, dataOffset0Intra_, currCountPart1_, scratchOffsetCountIntraStage1_, dataParams, intraLocalRankSize_);
        CHK_RET(tempAlgIntra1.KernelRun(param, dataParams, templateResource));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::FastLaunchTemplateIntra11(
    const OpParam &param, const u32 kernelNum, TemplateFastLaunchCtx &tempFastLaunchCtxIntra, InsAlgTemplate2 &tempAlgIntra1)
{
    if (kernelNum > 0) {
        HCCL_INFO("[InsBroadcastParallelExecutor][FastLaunchTemplateIntra11] kernelNum[%u]", kernelNum);
        CHK_RET(tempAlgIntra1.FastLaunch(param, tempFastLaunchCtxIntra));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateIntra01(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, TemplateResource& templateResource, InsAlgTemplate2 &tempAlgIntra1)
{
    if (currCountPart > 0) {
        //数据0的server内的mesh算法
        GenDataParamsBufferType(BufferType::HCCL_BUFFER, BufferType::INPUT, BufferType::HCCL_BUFFER, dataParams);
        GenDataParamstempAlg(param, resCtx, dataOffset, currCountPart, scratchOffsetCount, dataParams, intraLocalRankSize_);
        HCCL_DEBUG("[InsBroadcastParallelExecutor][RunTemplateIntra01] dataOffset[%d], myRank_[%d], inBuffBaseOff[%d], outBuffBaseOff[%d], hcclBuffBaseOff[%d]",
            dataOffset, myRank_, dataParams.buffInfo.inBuffBaseOff, dataParams.buffInfo.outBuffBaseOff, dataParams.buffInfo.hcclBuffBaseOff);
        CHK_RET(tempAlgIntra1.KernelRun(param, dataParams, templateResource));
        }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::FastLaunchTemplateIntra01(
    const OpParam &param, const u32 kernelNum, TemplateFastLaunchCtx &tempFastLaunchCtxIntra, InsAlgTemplate2 &tempAlgIntra1)
{
    if (kernelNum > 0) {
        HCCL_INFO("[InsBroadcastParallelExecutor][FastLaunchTemplateIntra01] kernelNum[%u]", kernelNum);
        CHK_RET(tempAlgIntra1.FastLaunch(param, tempFastLaunchCtxIntra));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateInter11(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, TemplateResource& templateResource, InsAlgTemplate3 &tempAlgInter1)
{
    if (currCountPart > 0) {
        //数据1的server间的nhr算法
        HCCL_DEBUG("[InsBroadcastParallelExecutor][RunTemplateInter11] myRank_[%d], dataOffset[%d], currCountPart[%d], scratchOffsetCountInterStage[%d]",
                   myRank_, dataOffset, currCountPart, scratchOffsetCount);
        GenDataParamsBufferType(BufferType::HCCL_BUFFER, BufferType::INPUT, BufferType::HCCL_BUFFER, dataParams);
        GenDataParamstempAlg(param, resCtx, dataOffset, currCountPart, scratchOffsetCount, dataParams, interLocalRankSize_);
        CHK_RET(tempAlgInter1.KernelRun(param, dataParams, templateResource));
        }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::FastLaunchTemplateInter11(
    const OpParam &param, const u32 kernelNum, TemplateFastLaunchCtx &tempFastLaunchCtxInter, InsAlgTemplate3 &tempAlgInter1)
{
    if (kernelNum > 0) {
        HCCL_INFO("[InsBroadcastParallelExecutor][FastLaunchTemplateInter11] kernelNum[%u]", kernelNum);
        CHK_RET(tempAlgInter1.FastLaunch(param, tempFastLaunchCtxInter));
    }
    return HcclResult::HCCL_SUCCESS;
}

// 算法注册
#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_BROADCAST, InsBroadcastParallelMesh1DNHR, InsBroadcastParallelExecutor,
    TopoMatchMultilevel, InsTempScatterMesh1D, InsTempScatterNHR, InsTempAllGatherMesh1D, InsTempAllGatherNHR);
REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_BROADCAST, InsBroadcastParallelMesh1DNHRUBX, InsBroadcastParallelExecutor,
    TopoMatchUBX, InsTempScatterMesh1D, InsTempScatterNHR, InsTempAllGatherMesh1D, InsTempAllGatherNHR);
REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_BROADCAST, InsBroadcastParallelMesh1DNHRPcie,
    InsBroadcastParallelExecutor, TopoMatchPcieMix, InsTempScatterMesh1D, InsTempScatterNHR,
    InsTempAllGatherMesh1D, InsTempAllGatherNHR);
#endif /* !HCCL_CANN_COMPAT_850 */
#ifndef AICPU_COMPILE
#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_BROADCAST, CcuBroadcastParallelMesh1DNHR, InsBroadcastParallelExecutor,
    TopoMatchMultilevel, CcuTempScatterMesh1D, CcuTempScatterNHR1DMem2Mem, CcuTempAllGatherMesh1DMem2Mem, CcuTempAllGatherNHR1DMem2Mem);
REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_BROADCAST, CcuBroadcastParallelMesh1DNHRUBX, InsBroadcastParallelExecutor,
    TopoMatchUBX, CcuTempScatterMesh1D, CcuTempScatterNHR1DMem2Mem, CcuTempAllGatherMesh1DMem2Mem, CcuTempAllGatherNHR1DMem2Mem);
#endif /* !HCCL_CANN_COMPAT_850 */
#endif
}  // namespace Hccl