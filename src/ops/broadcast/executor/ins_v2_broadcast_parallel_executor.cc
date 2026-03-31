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
#include "ccu_temp_all_gather_mesh_1D_mem2mem.h"
#include "ccu_temp_all_gather_nhr_1D_mem2mem.h"
#include "ccu_temp_scatter_mesh1d.h"
#include "ccu_temp_scatter_nhr1d_mem2mem.h"
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

    // 实例化算法模板类
    // 构建template
    std::shared_ptr<InsAlgTemplate0> algTemplate0 = std::make_shared<InsAlgTemplate0>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> algTemplate1 = std::make_shared<InsAlgTemplate1>(param, topoInfo->userRank, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate2> algTemplate2 = std::make_shared<InsAlgTemplate2>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate3> algTemplate3 = std::make_shared<InsAlgTemplate3>(param, topoInfo->userRank, algHierarchyInfo.infos[1]);

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
        intraLocalRankSize_ = GetRankSize(algHierarchyInfo.infos[0]);
        interLocalRankSize_ = GetRankSize(algHierarchyInfo.infos[1]);
        rankSize_ = intraLocalRankSize_ * interLocalRankSize_;
        HCCL_INFO("[CalcRes] localRankSize: myRank[%d] intraLocalRankSize[%u] interLocalRankSize[%u] rankSize_[%u]",
                myRank_, intraLocalRankSize_, interLocalRankSize_, rankSize_);
        CHK_RET(CalcLocalRoot());
        algTemplate0->SetRoot(intraLocalRoot_);
        algTemplate1->SetRoot(interLocalRoot_);
    }
    algTemplate0->CalcRes(comm, param, topoInfo, intraTempRequest);
    algTemplate1->CalcRes(comm, param, topoInfo, interTempRequest);
    algTemplate2->CalcRes(comm, param, topoInfo, intraTempRequest1);
    algTemplate3->CalcRes(comm, param, topoInfo, interTempRequest1); 

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

    // 计算localRankSize和localRoot
    intraLocalRankSize_ = GetRankSize(resCtx.algHierarchyInfo.infos[0]);
    interLocalRankSize_ = GetRankSize(resCtx.algHierarchyInfo.infos[1]);
    rankSize_ = intraLocalRankSize_ * interLocalRankSize_;
    HCCL_INFO("[Orchestrate] localRankSize: myRank[%d] intraLocalRankSize[%u] interLocalRankSize[%u] rankSize_[%u]",
              myRank_, intraLocalRankSize_, interLocalRankSize_, rankSize_);

    CHK_RET(CalcLocalRoot());

    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntra(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
    InsAlgTemplate1 tempAlgInter(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[1]);
    InsAlgTemplate2 tempAlgIntra1(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
    InsAlgTemplate3 tempAlgInter1(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[1]);

    // 算法展开
    HcclResult ret = GenInsQues(param, resCtx, tempAlgIntra, tempAlgInter, tempAlgIntra1, tempAlgInter1);
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
    const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter, InsAlgTemplate2 &tempAlgIntra1)
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
    const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate2 &tempAlgIntra1, InsAlgTemplate3 &tempAlgInter1)
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
    dataParams.buffInfo.inputSize = param.inputSize;
    dataParams.buffInfo.outputSize = param.outputSize;
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
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::GenInsQues(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter,
    InsAlgTemplate2 &tempAlgIntra1, InsAlgTemplate3 &tempAlgInter1)
{
    HCCL_INFO("[InsBroadcastParallelExecutor] AlgTemplate intra server is [%s]", tempAlgIntra.Describe().c_str());
    HCCL_INFO("[InsBroadcastParallelExecutor] AlgTemplate inter server is [%s]", tempAlgInter.Describe().c_str());

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
    HCCL_DEBUG("[InsBroadcastParallelExecutor][GenInsQues] dataCount_[%lu], myRank_[%d], sliceCountUB[%d], sliceCountUB0[%d], sliceCount[%d]",
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
    HCCL_DEBUG("[InsBroadcastParallelExecutor][GenInsQues] dataCount_[%lu], myRank_[%d], sliceCountPart0[%d], multipleIntra[%d]",
              dataCount_, myRank_, sliceCountPart0, multipleIntra);
    HCCL_DEBUG("[InsBroadcastParallelExecutor][GenInsQues] myRank_[%d],scratchOffsetCountInterStage0[%d], scratchOffsetCountIntraStage1[%d] loopTimes[%u]",
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
        CHK_RET(PrepareResForTemplate(resCtx, tempAlgIntra, tempAlgInter, tempAlgIntra1));
        TemplateResource intraTempAlgRes;
        TemplateResource interTempAlgRes;
        PrepareResForTemplateResource(param, resCtx, intraTempAlgRes, interTempAlgRes, true);

        // 第一步开始前同步
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
        HCCL_DEBUG("[InsBroadcastParallelExecutor][GenInsQues] RunTemplateInter1 myRank_[%d], dataOffset0[%d], currCountPart0[%d], loopTimes[%d]",
                   myRank_, dataOffset0, currCountPart0, loopTimes);
        HCCL_DEBUG("[InsBroadcastParallelExecutor][GenInsQues] RunTemplateInter1 myRank_[%d], dataOffset0[%d], currCountPart0[%d], scratchOffsetCountInterStage0[%d] "
                    "dataOffset1[%d], currCountPart1[%d]",
                   myRank_, dataOffset0, currCountPart0, scratchOffsetCountInterStage0, dataOffset1, currCountPart1);
        RunTemplateIntra0(param, resCtx, dataOffset0, currCountPart0, scratchOffsetCountIntraStage0, tempAlgParamsIntra0, intraTempAlgRes, tempAlgIntra);
        RunTemplateInter1(param, resCtx, dataOffset1, currCountPart1, scratchOffsetCountInterStage0, tempAlgParamsInter1, interTempAlgRes, tempAlgInter);
        // 第一步做完后回到主流做尾同步
        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

        // 第二步开始前同步
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
        //server 间地址偏移
        for (int i = 0; i < resCtx.algHierarchyInfo.infos[0][0].size(); i++) {
            tempVirtRankMapInter_.insert(std::make_pair(resCtx.algHierarchyInfo.infos[0][0][i], i));
        }
        RunTemplateInter0(param, resCtx, dataOffset0, currCountPart0, scratchOffsetCountInterStage1, tempAlgParamsInter0, interTempAlgRes, tempAlgInter);

        //server 内地址偏移
        for (int i = 0; i < resCtx.algHierarchyInfo.infos[1][0].size(); i++) {
            tempVirtRankMapIntra_.insert(std::make_pair(resCtx.algHierarchyInfo.infos[1][0][i], i));
        }
        RunTemplateIntra1(param, resCtx, dataOffset1, currCountPart1, scratchOffsetCountIntraStage1, tempAlgParamsIntra1, intraTempAlgRes, tempAlgIntra);
        // 尾同步
        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

        // 计算算法模板所需资源 InsAlgTemplate2 &tempAlgIntra1, InsAlgTemplate3 &tempAlgInter1
        CHK_RET(PrepareResForTemplate23(resCtx, tempAlgIntra, tempAlgIntra1, tempAlgInter1));
        TemplateResource intraTempAlgRes1;
        TemplateResource interTempAlgRes1;
        PrepareResForTemplateResource(param, resCtx, intraTempAlgRes1, interTempAlgRes1, false);

        // 第三步开始前同步
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
        RunTemplateInter01(param, resCtx, dataOffset0, currCountPart0, scratchOffsetCountInterStage1, tempAlgParamsInter01, interTempAlgRes1, tempAlgInter1);
        RunTemplateIntra11(param, resCtx, dataOffset1, currCountPart1, scratchOffsetCountIntraStage1, tempAlgParamsIntra11, intraTempAlgRes1, tempAlgIntra1);
        // 尾同步
        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

        // 第四步开始前同步
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
        RunTemplateIntra01(param, resCtx, dataOffset0, currCountPart0, scratchOffsetCountIntraStage0, tempAlgParamsIntra01, intraTempAlgRes1, tempAlgIntra1);
        RunTemplateInter11(param, resCtx, dataOffset1, currCountPart1, scratchOffsetCountInterStage0, tempAlgParamsInter11, interTempAlgRes1, tempAlgInter1);
        // 第四步做完后回到主流做尾同步
        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
    }

    HCCL_INFO("[InsBroadcastParallelExecutor][GenInsQues] End.myRank[%u]", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateIntra0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, const TemplateResource& templateResource, InsAlgTemplate0 &tempAlgIntra)
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
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateInter1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, const TemplateResource& templateResource, InsAlgTemplate1 &tempAlgInter)
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
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateInter0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, const TemplateResource& templateResource, InsAlgTemplate1 &tempAlgInter)
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
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateIntra1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, const TemplateResource& templateResource, InsAlgTemplate0 &tempAlgIntra)
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
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateInter01(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, const TemplateResource& templateResource, InsAlgTemplate3 &tempAlgInter1)
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
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateIntra11(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, const TemplateResource& templateResource, InsAlgTemplate2 &tempAlgIntra1)
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
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateIntra01(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, const TemplateResource& templateResource, InsAlgTemplate2 &tempAlgIntra1)
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
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::RunTemplateInter11(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams, const TemplateResource& templateResource, InsAlgTemplate3 &tempAlgInter1)
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

// 算法注册
REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_BROADCAST, InsBroadcastParallelMesh1DNHR, InsBroadcastParallelExecutor,
    TopoMatchMultilevel, InsTempScatterMesh1D, InsTempScatterNHR, InsTempAllGatherMesh1D, InsTempAllGatherNHR);
#ifndef AICPU_COMPILE
REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_BROADCAST, CcuBroadcastParallelMesh1DNHR, InsBroadcastParallelExecutor,
    TopoMatchMultilevel, CcuTempScatterMesh1D, CcuTempScatterNHR1DMem2Mem, CcuTempAllGatherMesh1DMem2Mem, CcuTempAllGatherNHR1DMem2Mem);
#endif
}  // namespace Hccl