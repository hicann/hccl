/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_scatter_parallel_executor.h"
#include "ins_temp_scatter_mesh_1D.h"
#include "ins_temp_scatter_nhr.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2ScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2ScatterParallelExecutor()
{
    
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(HcclComm comm,
    TopoInfo* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}


template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 构建template
    std::shared_ptr<InsAlgTemplate0> intraAlgTemplate = std::make_shared<InsAlgTemplate0>(param, topoInfo->userRank, algHierarchyInfo.infos[COMM_LEVEL0]);
    std::shared_ptr<InsAlgTemplate1> interAlgTemplate = std::make_shared<InsAlgTemplate1>(param, topoInfo->userRank, algHierarchyInfo.infos[COMM_LEVEL1]);
    
    // 调用计算资源的函数,暂不考虑Detour
    AlgResourceRequest intraResourceRequest;
    AlgResourceRequest interResourceRequest;

    intraAlgTemplate->CalcRes(comm, param, topoInfo, intraResourceRequest);
    interAlgTemplate->CalcRes(comm, param, topoInfo, interResourceRequest);

    // 合并两个template的资源
    resourceRequest.notifyNumOnMainThread = 2;  // 用于两个template间同步
    resourceRequest.slaveThreadNum = intraResourceRequest.slaveThreadNum + interResourceRequest.slaveThreadNum + 2;
    resourceRequest.notifyNumPerThread.emplace_back(intraResourceRequest.slaveThreadNum + 1); // intra模板控制流
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              intraResourceRequest.notifyNumPerThread.begin(),
                                              intraResourceRequest.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(interResourceRequest.slaveThreadNum + 1); // inter模板控制流
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              interResourceRequest.notifyNumPerThread.begin(),
                                              interResourceRequest.notifyNumPerThread.end());
    HCCL_DEBUG("[InsV2ScatterParallelExecutor][CalcRes] intraResourceRequest.channels[0].size[%u]",intraResourceRequest.channels[0].size());                                          
    resourceRequest.channels.emplace_back(intraResourceRequest.channels[0]);
    resourceRequest.channels.emplace_back(interResourceRequest.channels[0]);
    HCCL_DEBUG("[InsV2ScatterParallelExecutor][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u], "
               "channels[%u]",
               myRank_, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
               resourceRequest.channels.size());
    for (auto i = 0; i < resourceRequest.notifyNumPerThread.size(); i++) {
    HCCL_DEBUG("[InsV2ScatterParallelExecutor][CalcRes] myRank[%u], notifyNumPerThread[%u]=[%u]", myRank_, i,
                   resourceRequest.notifyNumPerThread[i]);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
uint64_t InsV2ScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo)
{
    uint64_t count = 1;
    for (const auto &i : vTopo) {
        count *= i.size();
    }
    return count;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2ScatterParallelExecutor][Orchestrate] Orchestrate Start");
    maxTmpMemSize_ = resCtx.cclMem.size;
    myRank_ = resCtx.topoInfo.userRank;
    root_ = param.root;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    HCCL_INFO("[InsV2ScatterParallelExecutor][Orchestrate] resCtx.threads.size()[%u], resCtx.cclMem.size[%u]", resCtx.threads.size(), resCtx.cclMem.size);
    if (param.engine != CommEngine::COMM_ENGINE_AIV) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ =  DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    HCCL_INFO("[CalcLocalRankSize] rankSizeLevel0_: resCtx.algHierarchyInfo.infos[0][0].size()[%d] resCtx.algHierarchyInfo.infos[0][1].size()[%u]",
              resCtx.algHierarchyInfo.infos[0][0].size(), resCtx.algHierarchyInfo.infos[0][1].size());
    rankSizeLevel0_ = GetRankSize(resCtx.algHierarchyInfo.infos[0]);
    rankSizeLevel1_ = GetRankSize(resCtx.algHierarchyInfo.infos[1]);
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    
    HCCL_INFO("[CalcLocalRankSize] localRankSize: myRank[%d] rankSizeLevel0_[%u] rankSizeLevel1_[%u]",
              myRank_, rankSizeLevel0_, rankSizeLevel1_);

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2ScatterParallelExecutor][Orchestrate]errNo[0x%016llx] Scatter excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2ScatterParallelExecutor][OrchestrateLoop] Start");

    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 实例化算法模板类
    HCCL_INFO("[InsV2ScatterParallelExecutor][OrchestrateLoop] param.root[%u]", param.root);
    InsAlgTemplate0 tempAlgIntra(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]); //server内算法，比如mesh  
    InsAlgTemplate1 tempAlgInter(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[1]); //server间算法，比如nhr     
    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(param, resCtx, tempAlgIntra, tempAlgInter));
    
    CHK_RET(GenInsQuesHost(param, resCtx, tempAlgIntra, tempAlgInter));
    HCCL_INFO("[InsV2ScatterParallelExecutor][OrchestrateLoop] Orchestrate success.");

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2ScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(std::vector<double> &splitDataSize) const
{
    // to do 先做等分，后续根据性能做调整
    double splitData = 0.5;
    splitDataSize.push_back(splitData);
    splitDataSize.push_back(splitData);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PreSyncInterTemplates()
{
    std::vector<ThreadHandle> subThreads{intraThreads_.at(0), interThreads_.at(0)};
    std::vector<u32> notifyIdxMainToSub{static_cast<u32>(intraThreads_.size() - 1), static_cast<u32>(interThreads_.size() - 1)}; //使用末尾的notifiy进行同步
    CHK_RET(PreSyncInterThreads(threads_.at(0), subThreads, notifyIdxMainToSub));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PostSyncInterTemplates()
{
    std::vector<ThreadHandle> subThreads{intraThreads_.at(0), interThreads_.at(0)};
    std::vector<u32> notifyIdxSubToMain_{0, 1};
    CHK_RET(PostSyncInterThreads(threads_.at(0), subThreads, notifyIdxSubToMain_));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenInsQuesHost(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter)
{
    HCCL_INFO("[InsV2ScatterParallelExecutor] AlgTemplate intra server is [%s]", tempAlgIntra.Describe().c_str());
    HCCL_INFO("[InsV2ScatterParallelExecutor] AlgTemplate inter server is [%s]", tempAlgInter.Describe().c_str());
    std::vector<double> dataSplitSize;
    GetParallelDataSplit(dataSplitSize); // <0.5, 0.5>
    double hcclBuffMultipleIntra = std::max(dataSplitSize.at(0) * rankSizeLevel1_, dataSplitSize.at(1) * rankSizeLevel0_); // intra都是mesh // x/y方向最大rank数 * y/x方向的dataSplitSize
    double hcclBuffMultipleInter = std::max(dataSplitSize.at(1) * rankSizeLevel0_ * rankSizeLevel1_, dataSplitSize.at(0) * rankSizeLevel0_ * rankSizeLevel1_);
    
    double totalScratchMultiple = hcclBuffMultipleIntra + hcclBuffMultipleInter;
    u64 hcclMemBlockSize = maxTmpMemSize_;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    if (totalScratchMultiple > 0) {
        // data0和data1的count需要和申请的scratch mem大小对应
        u64 tmpMemBlockCount = u64(maxTmpMemSize_  / totalScratchMultiple) / dataTypeSize_;
        hcclMemBlockSize = (u64(dataSplitSize.at(0) * tmpMemBlockCount) + u64(dataSplitSize.at(1) * tmpMemBlockCount)) * dataTypeSize_;
    }
    u64 intraScratchOffset = 0;
    u64 interScratchOffset = hcclBuffMultipleIntra * hcclMemBlockSize;
    u64 maxCountPerLoop = std::min(static_cast<u64>(hcclMemBlockSize), static_cast<u64>(UB_MAX_DATA_SIZE)) / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / dataTypeSize_; 

    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    TemplateDataParams tempAlgParamsIntra0;
    TemplateDataParams tempAlgParamsInter0;
    TemplateDataParams tempAlgParamsInter1;
    TemplateDataParams tempAlgParamsIntra1;

    // 准备资源
    TemplateResource intraTemplateAlgRes;
    intraTemplateAlgRes.channels = remoteRankToChannelInfo_[0];
    intraTemplateAlgRes.threads = intraThreads_;
    intraTemplateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    TemplateResource interTemplateAlgRes;
    interTemplateAlgRes.channels = remoteRankToChannelInfo_[1];
    interTemplateAlgRes.threads = interThreads_;
    interTemplateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCount = (loopIndex == loopTimes - 1) ? (dataCount_ - loopIndex * maxCountPerLoop) : maxCountPerLoop;
        u64 dataCountPerLoopAixs0 = dataSplitSize.at(0) * currCount;
        u64 dataCountPerLoopAixs1 = currCount - dataCountPerLoopAixs0;

        u64 dataOffset0 = loopIndex * maxCountPerLoop * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + dataCountPerLoopAixs0 * dataTypeSize_;
        //第一步开始前同步
        PreSyncInterTemplates();
        if (rankIdxLevel1_ == root_ / rankSizeLevel0_) { //数据0的server内的mesh算法
            GenTemplateAlgParamsIntra0(param, resCtx, dataOffset0, dataCountPerLoopAixs0, intraScratchOffset, tempAlgParamsIntra0);
            CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra0, intraTemplateAlgRes));
        }
        if (rankIdxLevel0_ == root_ % rankSizeLevel0_) { //数据1的server间的nhr算法
            GenTemplateAlgParamsInter1(param, resCtx, dataOffset1, dataCountPerLoopAixs1, interScratchOffset, tempAlgParamsInter1);
            CHK_RET(tempAlgInter.KernelRun(param, tempAlgParamsInter1, interTemplateAlgRes));
        }
        //第一步做完后回到主流做尾同步
        PostSyncInterTemplates();
        HCCL_INFO("[InsV2ScatterParallelExecutor][GenInsQuesHost] Stage1 Finished!! myRank_[%u], rankIdxLevel0_[%u], rankIdxLevel1_[%u], rankSizeLevel0_[%u], rankSizeLevel1_[%u], inter_new_root[%u], intra_new_root[%u]", myRank_, rankIdxLevel0_, rankIdxLevel1_, rankSizeLevel0_, rankSizeLevel1_, root_ / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_, root_ % rankSizeLevel0_ + rankIdxLevel1_ * rankSizeLevel0_);
        //第二步开始前同步
        PreSyncInterTemplates();
        //数据0的server间的nhr算法
        GenTemplateAlgParamsInter0(param, resCtx, dataOffset0, dataCountPerLoopAixs0, intraScratchOffset, tempAlgParamsInter0);
        tempAlgInter.SetRoot(root_ / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_); // 与root同框的同列rank作为新server间模板的root
        CHK_RET(tempAlgInter.KernelRun(param, tempAlgParamsInter0, interTemplateAlgRes));
        //数据1的server内的mesh算法
        GenTemplateAlgParamsIntra1(param, resCtx, dataOffset1,  dataCountPerLoopAixs1, interScratchOffset,  tempAlgParamsIntra1);
        tempAlgIntra.SetRoot(root_ % rankSizeLevel0_ + rankIdxLevel1_ * rankSizeLevel0_); // 各框内与root相连的rank作为新server内模板的新root
        CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra1, intraTemplateAlgRes));
        //尾同步
        PostSyncInterTemplates();
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2ScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra0(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
                                                                    const u64 dataCountPerLoopAixs0,
                                                                    const u64 scratchOffset,
                                                                    TemplateDataParams &tempAlgParamsIntra0) const
{
    tempAlgParamsIntra0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsIntra0.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsIntra0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsIntra0.buffInfo.inBuffType =  BufferType::INPUT;
    tempAlgParamsIntra0.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra0.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsIntra0.buffInfo.outBuffBaseOff = scratchOffset;
    tempAlgParamsIntra0.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsIntra0.sliceSize = dataCountPerLoopAixs0 * dataTypeSize_;

    tempAlgParamsIntra0.inputSliceStride = dataSize_;
    tempAlgParamsIntra0.outputSliceStride = 0;
    tempAlgParamsIntra0.repeatNum = rankSizeLevel1_;
    tempAlgParamsIntra0.inputRepeatStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsIntra0.outputRepeatStride = tempAlgParamsIntra0.sliceSize;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2ScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter0(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
                                                                    const u64 dataCountPerLoopAixs0,
                                                                    const u64 scratchOffset,
                                                                    TemplateDataParams &tempAlgParamsInter0) const
{
    tempAlgParamsInter0.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsInter0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsInter0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsInter0.buffInfo.inBuffType =  BufferType::HCCL_BUFFER;
    tempAlgParamsInter0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsInter0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter0.buffInfo.inBuffBaseOff = scratchOffset;
    tempAlgParamsInter0.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsInter0.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsInter0.sliceSize = dataCountPerLoopAixs0 * dataTypeSize_;

    tempAlgParamsInter0.inputSliceStride = tempAlgParamsInter0.sliceSize;
    tempAlgParamsInter0.outputSliceStride = 0;
    tempAlgParamsInter0.repeatNum = 1;
    tempAlgParamsInter0.inputRepeatStride = 0;
    tempAlgParamsInter0.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2ScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter1(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
                                                                    const u64 dataCountPerLoopAixs1,
                                                                    const u64 scratchOffset,
                                                                    TemplateDataParams &tempAlgParamsInter1) const
{
    tempAlgParamsInter1.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsInter1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsInter1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsInter1.buffInfo.inBuffType =  BufferType::INPUT;
    tempAlgParamsInter1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter1.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsInter1.buffInfo.outBuffBaseOff = scratchOffset; 
    tempAlgParamsInter1.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsInter1.sliceSize = dataCountPerLoopAixs1 * dataTypeSize_;

    tempAlgParamsInter1.inputSliceStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsInter1.outputSliceStride = 0;
    tempAlgParamsInter1.repeatNum = rankSizeLevel0_;
    tempAlgParamsInter1.inputRepeatStride = dataSize_;
    tempAlgParamsInter1.outputRepeatStride = tempAlgParamsInter1.sliceSize;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2ScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra1(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
                                                                    const u64 dataCountPerLoopAixs1,
                                                                    const u64 scratchOffset,
                                                                    TemplateDataParams &tempAlgParamsIntra1) const
{
    tempAlgParamsIntra1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsIntra1.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsIntra1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsIntra1.buffInfo.inBuffType =  BufferType::HCCL_BUFFER;
    tempAlgParamsIntra1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra1.buffInfo.inBuffBaseOff = scratchOffset;
    tempAlgParamsIntra1.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsIntra1.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsIntra1.sliceSize = dataCountPerLoopAixs1 * dataTypeSize_;

    tempAlgParamsIntra1.inputSliceStride = tempAlgParamsIntra1.sliceSize;
    tempAlgParamsIntra1.outputSliceStride = 0;
    tempAlgParamsIntra1.repeatNum = 1;
    tempAlgParamsIntra1.inputRepeatStride = 0;
    tempAlgParamsIntra1.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter)
{
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    u64 intraThreadsNum = tempAlgIntra.GetThreadNum();
    intraThreads_.assign(threads_.begin() + 1, threads_.begin() + 1 + intraThreadsNum);
    interThreads_.assign(threads_.begin() + intraThreadsNum + 1, threads_.end());
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_SCATTER, InsScatterParallelMesh1DNHR, InsV2ScatterParallelExecutor, TopoMatchMultilevel,
    InsTempScatterMesh1D, InsTempScatterNHR);
}