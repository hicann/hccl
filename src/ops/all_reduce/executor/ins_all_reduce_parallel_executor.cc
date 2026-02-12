/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <math.h>

#include "ins_all_reduce_parallel_executor.h"
#include "ins_temp_all_reduce_mesh_1D_two_shot.h"
#include "ins_temp_all_reduce_nhr.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsAllReduceParallelExecutor(){}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::~InsAllReduceParallelExecutor(){}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfo* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(HcclComm comm,
    const OpParam& param, const TopoInfo* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    InsAlgTemplate0 tempAlgIntra(param, topoInfo->userRank, algHierarchyInfo.infos.at(0));
    InsAlgTemplate1 tempAlgInter(param, topoInfo->userRank, algHierarchyInfo.infos.at(1));

    // 计算子算法所需资源
    AlgResourceRequest resReqIntra;
    AlgResourceRequest resReqInter;
    CHK_RET(tempAlgIntra.CalcRes(comm, param, topoInfo, resReqIntra));
    CHK_RET(tempAlgInter.CalcRes(comm, param, topoInfo, resReqInter));

    // 用第intra算法的主流作为Executor的主流
    resourceRequest.slaveThreadNum = resReqIntra.slaveThreadNum + resReqInter.slaveThreadNum + 1;

    // 每个算法的主流需要1个额外Notify用于算法之间同步
    resourceRequest.notifyNumOnMainThread = resReqIntra.notifyNumOnMainThread + 1;
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
        resReqIntra.notifyNumPerThread.begin(), resReqIntra.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(resReqInter.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
        resReqInter.notifyNumPerThread.begin(), resReqInter.notifyNumPerThread.end());

    // 每个算法的channels分别作为Executor中channels的Level0层和level1层
    resourceRequest.channels.emplace_back(resReqIntra.channels.at(0));
    resourceRequest.channels.emplace_back(resReqInter.channels.at(0));

    HCCL_INFO("[InsAllReduceParallelExecutor] Calculate resource finished."
        "intra algo resource request: threadNum[%u], main thread notifyNum[%u], channelNum[%u]."
        "inter algo resource request: threadNum[%u], main thread notifyNum[%u], channelNum[%u].",
        resReqIntra.slaveThreadNum + 1, resReqIntra.notifyNumOnMainThread + 1, resReqIntra.channels.at(0).size(),
        resReqInter.slaveThreadNum + 1, resReqInter.notifyNumOnMainThread + 1, resReqInter.channels.at(0).size());

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsAllReduceParallelExecutor] Orchestrate start");

    // 初始化基本信息
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ =  DATATYPE_SIZE_TABLE[dataType_];
    dataCount_ = param.DataDes.count;
    dataSize_ = dataCount_ * dataTypeSize_;
    reduceOp_ = param.reduceType;

    // 初始化资源信息
    maxTmpMemSize_ = resCtx.cclMem.size;
    threads_ = resCtx.threads;

    // 分配资源
    InsAlgTemplate0 tempAlgIntra(param, myRank_, resCtx.algHierarchyInfo.infos.at(0));
    InsAlgTemplate1 tempAlgInter(param, myRank_, resCtx.algHierarchyInfo.infos.at(1));
    CHK_RET(PrepareResForTemplate(param, resCtx, tempAlgIntra, tempAlgInter));

    // 算法展开
    CHK_RET(OrchestrateLoop(param, resCtx, tempAlgIntra, tempAlgInter));

    HCCL_INFO("[InsAllReduceParallelExecutor] Orchestrate finished");

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(
    std::vector<float> &splitDataSize) const
{
    // to do 先做等分，后续根据性能做调整
    double splitData = 0.5;
    splitDataSize.push_back(splitData);
    splitDataSize.push_back(splitData);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter)
{
    // 获取Template资源信息
    AlgResourceRequest algResIntra;
    AlgResourceRequest algResInter;
    CHK_RET(tempAlgIntra.GetRes(algResIntra));
    CHK_RET(tempAlgInter.GetRes(algResInter));

    // 分配threads
    u64 intraThreadsNum = algResIntra.slaveThreadNum + 1;
    u64 interThreadsNum = algResInter.slaveThreadNum + 1;
    CHK_PRT_RET(intraThreadsNum + interThreadsNum > threads_.size(),
        HCCL_ERROR("[InsAllReduceParallelExecutor] threadsNum[%u] is insufficient, need[%u], intra[%u], inter[%u]",
            threads_.size(), intraThreadsNum + interThreadsNum, intraThreadsNum, interThreadsNum), 
            HcclResult::HCCL_E_INTERNAL);

    intraThreads_.assign(threads_.begin(), threads_.begin() + intraThreadsNum);
    interThreads_.assign(threads_.begin() + intraThreadsNum, threads_.end());

    // 两个算法的主线程用于算法间同步
    syncThreads_.emplace_back(intraThreads_.at(0));
    syncThreads_.emplace_back(interThreads_.at(0));

    // 两个算法的主线程中的最后一个Notify用于算法间同步
    preSyncNotifyList_.emplace_back(algResInter.notifyNumOnMainThread);
    postSyncNotifyList_.emplace_back(algResIntra.notifyNumOnMainThread);

    // 分配channels
    if (param.engine != CommEngine::COMM_ENGINE_AIV) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
        intraChannelInfo_ = remoteRankToChannelInfo_.at(0);
        interChannelInfo_ = remoteRankToChannelInfo_.at(1);
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter)
{
    HCCL_INFO("[InsAllReduceParallelExecutor][OrchestrateLoop] Start");
    
    // 计算数据切分
    u64 alignedSize = 128;  // 假设需要128字节对齐，太大会导致后续maxCountPerLoop计算有问题
    u64 memBlockSize = UB_MAX_DATA_SIZE;
    u32 multipleIntra = tempAlgIntra.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 multipleInter = tempAlgInter.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    std::vector<float> dataSplitRateList;
    GetParallelDataSplit(dataSplitRateList);
    float dataSplitRate = dataSplitRateList.at(0);
    CalcSendDataSize(memBlockSize, dataSplitRate, multipleIntra, multipleInter);

    // 算法资源参数
    TemplateResource algResIntra;
    algResIntra.channels = intraChannelInfo_;
    algResIntra.threads = intraThreads_;
    algResIntra.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    TemplateResource algResInter;
    algResInter.channels = interChannelInfo_;
    algResInter.threads = interThreads_;
    algResInter.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    
    // dataSplitSize为分数，这里maxCountPerLoop对10取整，ScratchBufferSize为1M时可能会导致maxCountPerLoop为0；
    u64 maxCountPerLoop = (memBlockSize / dataTypeSize_ / 10 / alignedSize) * 10 * alignedSize;
    CHK_PRT_RET(maxCountPerLoop == 0,
        HCCL_ERROR("[InsAllReduceParallelExecutor] memBlockSize:%llu, maxCountPerLoop==0!.", memBlockSize),
        HcclResult::HCCL_E_INTERNAL);
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    // 循环展开
    TemplateDataParams algParamsIntraStage0;
    TemplateDataParams algParamsInterStage0;
    TemplateDataParams algParamsIntraStage1;
    TemplateDataParams algParamsInterStage1;
    for (u64 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCount = (loopIndex == loopTimes - 1) ? (dataCount_ - loopIndex * maxCountPerLoop) : maxCountPerLoop;
        u64 dataCountPart0 = static_cast<u64>(dataSplitRate * currCount);
        u64 dataCountPart1 = currCount - dataCountPart0;
        u64 dataOffsetPart0 = loopIndex * maxCountPerLoop * dataTypeSize_;
        u64 dataOffsetPart1 = dataOffsetPart0 + dataCountPart0 * dataTypeSize_;
        
        // 第一步，双轴同步AllReduce
        CHK_RET(PreSync(syncThreads_));
        if (dataCountPart0 > 0) {
            GenAlgParamsStage0(param, resCtx, dataOffsetPart0, dataCountPart0, 0, algParamsIntraStage0);
            CHK_RET(tempAlgIntra.KernelRun(param, algParamsIntraStage0, algResIntra));
        }
        if (dataCountPart1 > 0) {
            GenAlgParamsStage0(param, resCtx, dataOffsetPart1, dataCountPart1, parallelHcclBuffOffsetStage0_,
                algParamsInterStage0);
            CHK_RET(tempAlgInter.KernelRun(param, algParamsInterStage0, algResInter));
        }
        CHK_RET(PostSync(syncThreads_));

        // 第二步，数据换轴后，双轴同步AllReduce
        CHK_RET(PreSync(syncThreads_));
        if (dataCountPart0 > 0) {
            // 数据0的server间的nhr算法
            GenAlgParamsStage1(param, resCtx, dataOffsetPart0, dataCountPart0, parallelHcclBuffOffsetStage1_,
                algParamsInterStage1);
            CHK_RET(tempAlgInter.KernelRun(param, algParamsInterStage1, algResInter));
        }
        if (dataCountPart1 > 0) {
            // 数据1的server内的mesh算法
            GenAlgParamsStage1(param, resCtx, dataOffsetPart1, dataCountPart1, 0, algParamsIntraStage1);
            CHK_RET(tempAlgIntra.KernelRun(param, algParamsIntraStage1, algResIntra));
        }
        // 尾同步
        CHK_RET(PostSync(syncThreads_));
    }
    HCCL_INFO("[InsAllReduceParallelExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcSendDataSize(
    u64 &memBlockSize, float &SplitRate, u32 &multipleIntra, u32 &multipleInter)
{
    std::vector<float> dataSplitSizeList;
    GetParallelDataSplit(dataSplitSizeList);
    uint64_t templateNum = 2;
    if (multipleIntra == 0 && multipleInter == 0) {
        memBlockSize = UB_MAX_DATA_SIZE + UB_MAX_DATA_SIZE;
    } else if ((multipleIntra == 0 && multipleInter > 0) || (multipleInter == 0 && multipleIntra > 0)) {
        // 因为数据要交替在两个template中执行，因此最终要以数据处理量小的template为准
        if (multipleIntra > 0) {
            memBlockSize = std::min(static_cast<u64>(UB_MAX_DATA_SIZE), maxTmpMemSize_ / multipleIntra) * templateNum;
            intraHcclBuffSizeStage0_ = maxTmpMemSize_;
            intraHcclBuffSizeStage1_ = maxTmpMemSize_;
        } else {
            memBlockSize = std::min(static_cast<u64>(UB_MAX_DATA_SIZE), maxTmpMemSize_ / multipleInter) * templateNum;
            interHcclBuffSizeStage0_ = maxTmpMemSize_;
            interHcclBuffSizeStage0_ = maxTmpMemSize_;
        }
    } else {  // multipleIntra >0 && multipleInter >0, 理论上dataSplitSize[0]=0.5时，scratch buffer利用率最大
        SplitRate = dataSplitSizeList.at(0);
        u32 subMultiple0 = static_cast<u32>(std::ceil(SplitRate * multipleIntra + (1 - SplitRate) * multipleInter));
        u32 subMultiple1 = static_cast<u32>(std::ceil((1 - SplitRate) * multipleIntra + SplitRate * multipleInter));
        u64 totalScratchMultiple = std::max(subMultiple0, subMultiple1);
        memBlockSize = std::min(static_cast<u64>(UB_MAX_DATA_SIZE), maxTmpMemSize_ / totalScratchMultiple);

        parallelHcclBuffOffsetStage0_ = static_cast<u64>(memBlockSize * SplitRate * multipleIntra);
        parallelHcclBuffOffsetStage1_ = static_cast<u64>(memBlockSize * (1 - SplitRate) * multipleIntra);
        intraHcclBuffSizeStage0_ = parallelHcclBuffOffsetStage0_;
        interHcclBuffSizeStage0_ = parallelHcclBuffOffsetStage1_;
        intraHcclBuffSizeStage1_ = parallelHcclBuffOffsetStage1_;
        interHcclBuffSizeStage0_ = parallelHcclBuffOffsetStage0_;
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenAlgParamsStage0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 dataCount, 
    const u64 hcclBuffBaseOff, TemplateDataParams &tempAlgParams) const
{
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    tempAlgParams.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParams.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParams.buffInfo.hcclBuffBaseOff = hcclBuffBaseOff;
    
    tempAlgParams.count = dataCount;
    tempAlgParams.sliceSize = dataCount * dataTypeSize_;
    tempAlgParams.tailSize = tempAlgParams.sliceSize;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenAlgParamsStage1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 dataCount,
    const u64 hcclBuffBaseOff, TemplateDataParams &tempAlgParams) const
{
    tempAlgParams.buffInfo.inputPtr = param.outputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParams.buffInfo.inBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    tempAlgParams.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParams.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParams.buffInfo.hcclBuffBaseOff = hcclBuffBaseOff;

    tempAlgParams.count = dataCount;
    tempAlgParams.sliceSize = dataCount * dataTypeSize_;
    tempAlgParams.tailSize = tempAlgParams.sliceSize;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PreSync(
    const std::vector<ThreadHandle> &threads)
{
    if (threads.size() > 1) {
        std::vector<ThreadHandle> slaveThreads(threads.begin() + 1, threads.end());
        CHK_RET(PreSyncInterThreads(threads.at(0), slaveThreads, preSyncNotifyList_));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PostSync(
    const std::vector<ThreadHandle> &threads)
{
    if (threads.size() > 1) {
        std::vector<ThreadHandle> slaveThreads(threads.begin() + 1, threads.end());
        CHK_RET(PostSyncInterThreads(threads.at(0), slaveThreads, postSyncNotifyList_));
    }
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE, InsAllReduceParallelMesh1DNHR,
    InsAllReduceParallelExecutor, TopoMatchMultilevel, InsTempAllReduceMesh1DTwoShot, InsTempAllReduceNHR);

}