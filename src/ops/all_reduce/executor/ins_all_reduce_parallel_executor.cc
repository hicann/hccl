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
#ifndef AICPU_COMPILE
#include "ccu_temp_all_reduce_mesh_1D_mem2mem.h"
#include "ccu_temp_all_reduce_nhr_1D_mem2mem.h"
#include "ccu_temp_all_reduce_nhr_mem2mem_1D_multi_jetty.h"
#endif

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsAllReduceParallelExecutor(){}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::~InsAllReduceParallelExecutor(){}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(HcclComm comm,
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    // 构建template
    std::vector<std::vector<u32>> temp0HierarchyInfo;
    std::vector<std::vector<u32>> temp1HierarchyInfo;
    if(topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        temp0HierarchyInfo = {algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        u32 meshSize = algHierarchyInfo.infos[0][0].size();
        for(auto rank : algHierarchyInfo.infos[0][1]) {
            if(rank % meshSize == topoInfo->userRank % meshSize) {
                closRanks.push_back(rank);
            }
        }
        temp1HierarchyInfo = {closRanks};
    } else {
        temp0HierarchyInfo = algHierarchyInfo.infos[0];
        temp1HierarchyInfo = algHierarchyInfo.infos[1];
    }
    InsAlgTemplate0 tempAlgIntra(param, topoInfo->userRank, temp0HierarchyInfo);
    InsAlgTemplate1 tempAlgInter(param, topoInfo->userRank, temp1HierarchyInfo);

    // 计算子算法所需资源
    AlgResourceRequest resReqIntra;
    AlgResourceRequest resReqInter;
    CHK_RET(tempAlgIntra.CalcRes(comm, param, topoInfo, resReqIntra));
    CHK_RET(tempAlgInter.CalcRes(comm, param, topoInfo, resReqInter));

    constexpr u32 SUB_MAIN_THREAD_NUM = 2;
    // 用第intra算法的主流作为Executor的主流
    resourceRequest.slaveThreadNum = resReqIntra.slaveThreadNum + resReqInter.slaveThreadNum + SUB_MAIN_THREAD_NUM;

    // 每个算法的主流需要1个额外Notify用于算法之间同步
    resourceRequest.notifyNumOnMainThread = SUB_MAIN_THREAD_NUM;
    resourceRequest.notifyNumPerThread.emplace_back(resReqIntra.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
        resReqIntra.notifyNumPerThread.begin(), resReqIntra.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(resReqInter.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
        resReqInter.notifyNumPerThread.begin(), resReqInter.notifyNumPerThread.end());

    // 每个算法的channels分别作为Executor中channels的Level0层和level1层
    if (param.engine != COMM_ENGINE_CCU) {
        resourceRequest.channels.emplace_back(resReqIntra.channels[0]);
        resourceRequest.channels.emplace_back(resReqInter.channels[0]);
    } else {
        // ccu
        HCCL_INFO("[InsAllReduceParallelExecutor][CalcRes] intraTemplate has [%d] kernels.", resReqIntra.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            resReqIntra.ccuKernelInfos.begin(),
                                            resReqIntra.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(resReqIntra.ccuKernelNum[0]);
        HCCL_INFO("[InsAllReduceParallelExecutor][CalcRes] interTemplate has [%d] kernels.", resReqInter.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            resReqInter.ccuKernelInfos.begin(),
                                            resReqInter.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(resReqInter.ccuKernelNum[0]);
    }

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

    std::vector<std::vector<u32>> temp0HierarchyInfo;
    std::vector<std::vector<u32>> temp1HierarchyInfo;
    if(resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS) {
        temp0HierarchyInfo = {resCtx.algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        u32 meshSize = resCtx.algHierarchyInfo.infos[0][0].size();
        for(auto rank : resCtx.algHierarchyInfo.infos[0][1]) {
            if(rank % meshSize == resCtx.topoInfo.userRank % meshSize) {
                closRanks.push_back(rank);
            }
        }
        temp1HierarchyInfo = {closRanks};
    } else {
        temp0HierarchyInfo = resCtx.algHierarchyInfo.infos[0];
        temp1HierarchyInfo = resCtx.algHierarchyInfo.infos[1];
    }
    InsAlgTemplate0 tempAlgIntra(param, myRank_, temp0HierarchyInfo);
    InsAlgTemplate1 tempAlgInter(param, myRank_, temp1HierarchyInfo);

    // 分配资源
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

    u64 intraThreadsNum = algResIntra.slaveThreadNum + 1;
    u64 interThreadsNum = algResInter.slaveThreadNum + 1;
    CHK_PRT_RET(intraThreadsNum + interThreadsNum > threads_.size(),
        HCCL_ERROR("[InsAllReduceParallelExecutor] threadsNum[%u] is insufficient, need[%u], intra[%u], inter[%u]",
            threads_.size(), intraThreadsNum + interThreadsNum, intraThreadsNum, interThreadsNum), 
            HcclResult::HCCL_E_INTERNAL);

    intraThreads_.assign(threads_.begin() + 1, threads_.begin() + intraThreadsNum + 1);
    interThreads_.assign(threads_.begin() + intraThreadsNum + 1, threads_.end());
    // 用于两个算法同步
    mainThread_ = threads_.at(0);
    templateMainThreads_.emplace_back(intraThreads_.at(0));
    templateMainThreads_.emplace_back(interThreads_.at(0));
    syncNotifyOnTemplates_ = {algResIntra.notifyNumOnMainThread, algResInter.notifyNumOnMainThread};
    syncNotifyOnMain_ = {0, 1};

    // 分配channels
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
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
    TemplateResource algResInter;

    if (param.engine == COMM_ENGINE_CCU) {
        algResIntra.ccuKernels.insert(algResIntra.ccuKernels.end(),
                                              resCtx.ccuKernels.begin(),
                                              resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
        algResInter.ccuKernels.insert(algResInter.ccuKernels.end(),
                                              resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
                                              resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);
    } else {
        algResIntra.channels = intraChannelInfo_;
        algResInter.channels = interChannelInfo_;
        algResIntra.aivCommInfoPtr = resCtx.aivCommInfoPtr;
        algResInter.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    }
    algResIntra.threads = intraThreads_;
    algResInter.threads = interThreads_;
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
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
        if (dataCountPart0 > 0) {
            GenAlgParamsStage0(param, resCtx, dataOffsetPart0, dataCountPart0, 0, algParamsIntraStage0);
            CHK_RET(tempAlgIntra.KernelRun(param, algParamsIntraStage0, algResIntra));
        }
        if (dataCountPart1 > 0) {
            GenAlgParamsStage0(param, resCtx, dataOffsetPart1, dataCountPart1, parallelHcclBuffOffsetStage0_,
                algParamsInterStage0);
            CHK_RET(tempAlgInter.KernelRun(param, algParamsInterStage0, algResInter));
        }
        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

        // 第二步，数据换轴后，双轴同步AllReduce
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
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
        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
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
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCount, const u64 hcclBuffBaseOff, TemplateDataParams &tempAlgParams) const
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

    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    
    tempAlgParams.count = dataCount;
    tempAlgParams.sliceSize = dataCount * dataTypeSize_;
    tempAlgParams.tailSize = tempAlgParams.sliceSize;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenAlgParamsStage1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCount, const u64 hcclBuffBaseOff, TemplateDataParams &tempAlgParams) const
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

    tempAlgParams.buffInfo.inputSize = param.outputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;

    tempAlgParams.count = dataCount;
    tempAlgParams.sliceSize = dataCount * dataTypeSize_;
    tempAlgParams.tailSize = tempAlgParams.sliceSize;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    return;
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE, InsAllReduceParallelMesh1DNHR,
    InsAllReduceParallelExecutor, TopoMatchMultilevel, InsTempAllReduceMesh1DTwoShot, InsTempAllReduceNHR);

#ifndef AICPU_COMPILE
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE, CcuAllReduceParallelMesh1DNHR, 
    InsAllReduceParallelExecutor, TopoMatchMultilevel, CcuTempAllReduceMeshMem2Mem1D, CcuTempAllReduceNHRMem2Mem1D);
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE, CcuAllReduceParallelNHR1DMutiJetty, 
    InsAllReduceParallelExecutor, TopoMatchUBX, CcuTempAllReduceMeshMem2Mem1D, CcuTempAllReduceNhrMem2Mem1DMultiJetty);
#endif
}