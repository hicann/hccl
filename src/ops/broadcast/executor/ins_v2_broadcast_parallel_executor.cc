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
#include "ins_temp_broadcast_mesh_1D_two_shot.h"
#include "ins_temp_broadcast_nhr.h"
#include "topo_match_multilevel.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsBroadcastParallelExecutor()
{
}


template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(HcclComm comm,
    TopoInfo* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam& param,
    const TopoInfo* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    myRank_ = topoInfo->userRank;
    HCCL_INFO("[InsBroadcastParallelExecutor] CalcRes start, rank[%d]", myRank_);

    // 实例化算法模板类
    // 构建template
    std::shared_ptr<InsAlgTemplate0> algTemplate0 = std::make_shared<InsAlgTemplate0>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> algTemplate1 = std::make_shared<InsAlgTemplate1>(param, topoInfo->userRank, algHierarchyInfo.infos[1]);

   // 计算资源
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;

    algTemplate0->CalcRes(comm, param, topoInfo, intraTempRequest);
    algTemplate1->CalcRes(comm, param, topoInfo, interTempRequest);

    resourceRequest.notifyNumOnMainThread = 2;  // 用于两个template间同步
    resourceRequest.slaveThreadNum = intraTempRequest.slaveThreadNum + interTempRequest.slaveThreadNum + 2;
    resourceRequest.notifyNumPerThread.emplace_back(intraTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              intraTempRequest.notifyNumPerThread.begin(),
                                              intraTempRequest.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(interTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              interTempRequest.notifyNumPerThread.begin(),
                                              interTempRequest.notifyNumPerThread.end());
    resourceRequest.channels.emplace_back(intraTempRequest.channels[0]);
    resourceRequest.channels.emplace_back(interTempRequest.channels[0]);

    HCCL_DEBUG("[InsBroadcastParallelExecutor][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u], "
               "channels[%u]", myRank_, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
               resourceRequest.channels.size());
    for (auto i = 0; i < resourceRequest.notifyNumPerThread.size(); i++) {
        HCCL_DEBUG("[InsBroadcastParallelExecutor][CalcRes] myRank[%u], notifyNumPerThread[%u]=[%u]", myRank_, i,
                   resourceRequest.notifyNumPerThread[i]);
    }

    return HcclResult::HCCL_SUCCESS;
}


template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsBroadcastParallelExecutor][Orchestrate] Orchestrate Start");

    maxTmpMemSize_ = resCtx.cclMem.size;
    myRank_ = resCtx.topoInfo.userRank;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    HCCL_INFO("[InsBroadcastParallelExecutor][Orchestrate] threads_size[%d]", threads_.size());
    if (param.engine != CommEngine::COMM_ENGINE_AIV) {
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

    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(resCtx, tempAlgIntra, tempAlgInter));

    // 算法展开
    HcclResult ret = GenInsQues(param, resCtx, tempAlgIntra, tempAlgInter);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsBroadcastParallelExecutor][Orchestrate]errNo[0x%016llx] Reduce scatter excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);
    return HcclResult::HCCL_SUCCESS;
}


template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(
    std::vector<float> &splitDataSize) const
{
    // to do 先做等分，后续根据性能做调整
    double splitData = 0.5;
    splitDataSize.push_back(splitData);
    splitDataSize.push_back(splitData);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
uint64_t InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo)
{
    uint64_t count = 1;
    for (const auto &i : vTopo) {
        count *= i.size();
    }
    return count;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcLocalRoot()
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
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter)
{

    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    tempAlgIntra.GetRes(intraTempRequest);
    tempAlgInter.GetRes(interTempRequest);
    auto intraThreadsNum = intraTempRequest.slaveThreadNum + 1;
    auto interThreadsNum = intraTempRequest.slaveThreadNum + 1;
    auto intraNotifyOnMainThread = intraTempRequest.notifyNumOnMainThread;
    auto interNotifyOnMainThread = interTempRequest.notifyNumOnMainThread;
 
    intraThreads_.assign(threads_.begin() + 1, threads_.begin() + intraThreadsNum + 1);
    interThreads_.assign(threads_.begin() + intraThreadsNum + 1, threads_.end());
    // 用于两个算法同步
    mainThread_ = threads_.at(0);
    templateMainThreads_.emplace_back(intraThreads_.at(0));
    templateMainThreads_.emplace_back(interThreads_.at(0));
    syncNotifyOnTemplates_ = {intraNotifyOnMainThread, interNotifyOnMainThread};
    syncNotifyOnMain_ = {0, 1};

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenDataParams(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 sliceCount, const u64 scratchOffsetCount,
    TemplateDataParams &dataParams) const
{
    dataParams.buffInfo.inputPtr = param.inputPtr;
    dataParams.buffInfo.outputPtr = param.inputPtr;
    dataParams.buffInfo.inputSize = param.inputSize;
    dataParams.buffInfo.outputSize = param.outputSize;
    dataParams.buffInfo.hcclBuff = resCtx.cclMem;
    dataParams.buffInfo.inBuffType = BufferType::INPUT;
    dataParams.buffInfo.outBuffType = BufferType::INPUT;
    dataParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    dataParams.buffInfo.inBuffBaseOff = dataOffset;
    dataParams.buffInfo.outBuffBaseOff = dataOffset;
    dataParams.buffInfo.hcclBuffBaseOff = scratchOffsetCount * dataTypeSize_;
    dataParams.sliceSize = sliceCount * dataTypeSize_;
    dataParams.count = sliceCount;

    dataParams.inputSliceStride = 0;
    dataParams.outputSliceStride = 0;
    dataParams.repeatNum = 1;
    dataParams.inputRepeatStride = 0;
    dataParams.outputRepeatStride = 0;
    return;
}


template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenInsQues(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter)
{
    HCCL_INFO("[InsBroadcastParallelExecutor] AlgTemplate intra server is [%s]", tempAlgIntra.Describe().c_str());
    HCCL_INFO("[InsBroadcastParallelExecutor] AlgTemplate inter server is [%s]", tempAlgInter.Describe().c_str());

    TemplateResource intraTempAlgRes;
    intraTempAlgRes.channels = intraLinks_;
    intraTempAlgRes.threads = intraThreads_;
    intraTempAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    for(auto i: intraTempAlgRes.channels) {
        HCCL_DEBUG("[InsBroadcastParallelExecutor][GenInsQues],intraTempAlgRes.channels, myRank_[%u], channels[%u]= size[%u] ",
        myRank_, i.first, i.second.size());
    }
    TemplateResource interTempAlgRes;
    interTempAlgRes.channels = interLinks_;
    interTempAlgRes.threads = interThreads_;
    interTempAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    for(auto i: interTempAlgRes.channels) {
        HCCL_DEBUG("[InsBroadcastParallelExecutor][GenInsQues],interTempAlgRes.channels, myRank_[%u], channels[%u]= size[%u] ",
        myRank_, i.first, i.second.size());
    }
    HCCL_DEBUG("[InsBroadcastParallelExecutor][GenInsQues] AlgTemplate intraThreads_size[%d] interThreads_size[%d]",
              intraThreads_.size(), interThreads_.size());

    std::vector<float> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);

    // 实际上，两个输入参数无效
    u32 multipleIntra = tempAlgIntra.CalcScratchMultiple(BufferType::INPUT, BufferType::INPUT);
    u32 multipleInter = tempAlgInter.CalcScratchMultiple(BufferType::INPUT, BufferType::INPUT);


    // 按照intraData0+interData1，以及intraData1+interData0两种方式分别计算，取multiple最大需求
    float multiple0 = dataSplitSize.at(0) * float(multipleIntra) + dataSplitSize.at(1) * float(multipleInter);
    float multiple1 = dataSplitSize.at(1) * float(multipleIntra) + dataSplitSize.at(0) * float(multipleInter);
    float multiple = std::max(multiple0, multiple1);

    // 数据切分
    u64 sliceCount = std::min(static_cast<u64>(UB_MAX_DATA_SIZE) / dataTypeSize_, dataCount_);
    if (multiple > 0 && maxTmpMemSize_ > 0) {
        u64 scratchCount = maxTmpMemSize_ / dataTypeSize_;  // 按照count来切分
        sliceCount = static_cast<u64>(float(scratchCount) / multiple);  // 向下取整，防止Scratch溢出
    }

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
    HCCL_DEBUG("[InsBroadcastParallelExecutor][GenInsQues] dataCount_[%d], myRank_[%d], sliceCountPart0[%d], multipleIntra[%d], scratchOffsetCountInterStage0[%d], scratchOffsetCountIntraStage1[%d]",
              dataCount_, myRank_, sliceCountPart0, multipleIntra, scratchOffsetCountInterStage0, scratchOffsetCountIntraStage1);

    TemplateDataParams tempAlgParamsIntra0;

    TemplateDataParams tempAlgParamsInter0;
    TemplateDataParams tempAlgParamsInter1;
    TemplateDataParams tempAlgParamsIntra1;


    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCountPart0 = (loopIndex == loopTimes - 1) ? finalSliceCountPart0 : sliceCountPart0;
        u64 currCountPart1 = (loopIndex == loopTimes - 1) ? finalSliceCountPart1 : sliceCountPart1;
        u64 dataOffset0 = loopIndex * sliceCount * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + currCountPart0 * dataTypeSize_;

        // 第一步开始前同步
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
        // server内topo包含root_的rank进行展开，其它rank不展开
        if (intraLocalRoot_ == root_ && currCountPart0 > 0) {
            //数据0的server内的mesh算法
            GenDataParams(param, resCtx, dataOffset0, currCountPart0, scratchOffsetCountIntraStage0, tempAlgParamsIntra0);
            HCCL_DEBUG("[GenInsQues] step 1 server broadcastmesh: myRank[%d] intraLocalRoot[%u] dataOffset0[%u] currCountPart0[%u] scratchOffset[%u]",
              myRank_, intraLocalRoot_, dataOffset0, currCountPart0, scratchOffsetCountIntraStage0);
            tempAlgInter.SetRoot(param.root);
            CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra0, intraTempAlgRes));
        }
        // server间topo包含root_的rank进行展开，其它rank不展开
        if (interLocalRoot_ == root_ && currCountPart1 > 0) {
            //数据1的server间的nhr算法
            GenDataParams(param, resCtx, dataOffset1, currCountPart1, scratchOffsetCountInterStage0, tempAlgParamsInter1);
            HCCL_DEBUG("[GenInsQues] step 1 server broadcastnhr: myRank[%d] interLocalRoot_[%u] dataOffset1[%u], currCountPart1[%u], scratchOffset[%u]",
              myRank_, interLocalRoot_, dataOffset1, currCountPart1, scratchOffsetCountInterStage0);
            tempAlgInter.SetRoot(param.root);
            CHK_RET(tempAlgInter.KernelRun(param, tempAlgParamsInter1, interTempAlgRes));
        }
        // 第一步做完后回到主流做尾同步
        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

        HCCL_INFO("[GenInsQues] after step 1 PostSyncInterThreads: myRank[%d]", myRank_);
        // 第二步开始前同步
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
        if (currCountPart0 > 0) {
            // 数据0的server间的nhr算法
            GenDataParams(param, resCtx, dataOffset0, currCountPart0, scratchOffsetCountInterStage1, tempAlgParamsInter0);
            HCCL_DEBUG("[GenInsQues] step 2 server broadcastnhr: myRank[%d] dataOffset0[%d], currCountPart0[%d], scratchOffset[%d]",
              myRank_, dataOffset0, currCountPart0, scratchOffsetCountInterStage1);
            tempAlgInter.SetRoot(interLocalRoot_);
            CHK_RET(tempAlgInter.KernelRun(param, tempAlgParamsInter0, interTempAlgRes));
        }
        if (currCountPart1 > 0) {
            // 数据1的server内的mesh算法
            GenDataParams(param, resCtx, dataOffset1, currCountPart1, scratchOffsetCountIntraStage1, tempAlgParamsIntra1);
            HCCL_DEBUG("[GenInsQues] step 2 server broadcastmesh: myRank[%d] dataOffset1[%u] currCountPart1[%d], scratchOffset[%d]",
              myRank_, dataOffset1, currCountPart1, scratchOffsetCountIntraStage1);
            tempAlgIntra.SetRoot(intraLocalRoot_);
            CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra1, intraTempAlgRes));
        }
        // 尾同步
        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));


    }
    HCCL_INFO("[InsBroadcastParallelExecutor][GenInsQues] End.myRank[%d]", myRank_);
    return HcclResult::HCCL_SUCCESS;
}


// 算法注册
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_BROADCAST, InsBroadcastParallelMesh1DNHR, InsBroadcastParallelExecutor,
    TopoMatchMultilevel, InsTempBroadcastMesh1DTwoShot, InsTempBroadcastNHR);

}  // namespace Hccl