/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_reduce_scatter_parallel_executor.h"
#include <cmath>
#include "ins_temp_reduce_scatter_mesh_1D.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_temp_reduce_scatter_nhr_1D_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh_1D_mem2mem.h"


namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsReduceScatterParallelExecutor()
    : InsCollAlgBase()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::~InsReduceScatterParallelExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam& param,
    const TopoInfo* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    // 构建template
    InsAlgTemplate0 intraTempAlg(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    InsAlgTemplate1 interTempAlg(param, topoInfo->userRank, algHierarchyInfo.infos[1]);
 
    // 调用计算资源的函数
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    intraTempAlg.CalcRes(comm, param, topoInfo, intraTempRequest);
    interTempAlg.CalcRes(comm, param, topoInfo, interTempRequest);
    // 申请一条控制thread作为主thread，该thread仅用于两个template之间同步
    resourceRequest.notifyNumOnMainThread = 2;
    // 由于主thread被单独作为控制thread，因此总的slaveThread需要额外加上两个template的主thread
    resourceRequest.slaveThreadNum = intraTempRequest.slaveThreadNum + interTempRequest.slaveThreadNum + 2;
    // 第一个template的zhuthread需要的notify数量，+1是因为需要和控制thread做同步
    resourceRequest.notifyNumPerThread.emplace_back(intraTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              intraTempRequest.notifyNumPerThread.begin(),
                                              intraTempRequest.notifyNumPerThread.end());
    // 这一条是interTemplate的主thread，需要+1是为了和控制thread进行同步
    resourceRequest.notifyNumPerThread.emplace_back(interTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              interTempRequest.notifyNumPerThread.begin(),
                                              interTempRequest.notifyNumPerThread.end());
    if (param.engine != COMM_ENGINE_CCU) {
        resourceRequest.channels.emplace_back(intraTempRequest.channels[0]);
        resourceRequest.channels.emplace_back(interTempRequest.channels[0]);
    } else {
        // ccu
        HCCL_INFO("[InsReduceScatterParallelExecutor][CalcRes] intraTemplate has [%d] kernels.", intraTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            intraTempRequest.ccuKernelInfos.begin(),
                                            intraTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(intraTempRequest.ccuKernelNum[0]);
        HCCL_INFO("[InsReduceScatterParallelExecutor][CalcRes] interTemplate has [%d] kernels.", interTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            interTempRequest.ccuKernelInfos.begin(),
                                            interTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(interTempRequest.ccuKernelNum[0]);
    }
 
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfo *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

// HOST 侧算法入口，将对应的 instruction 添加到指令队列中
// 传入的insQue为一条主流
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, 
    const u64 dataOffset, const u64 dataCountPerLoopAixs0,
    std::vector<u64> &scratchOffVec, TemplateDataParams &tempAlgParamsIntra0) const
{
    tempAlgParamsIntra0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsIntra0.buffInfo.inputSize = param.inputSize;
    tempAlgParamsIntra0.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsIntra0.buffInfo.outputSize = resCtx.cclMem.size;
    tempAlgParamsIntra0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsIntra0.buffInfo.inBuffType =  BufferType::INPUT;
    tempAlgParamsIntra0.buffInfo.outBuffType = BufferType::HCCL_BUFFER; // 第一步最后的数据存储在scratch buffer上
    tempAlgParamsIntra0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra0.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsIntra0.buffInfo.outBuffBaseOff = scratchOffVec[0] + rankIdxLevel0_ * dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsIntra0.buffInfo.hcclBuffBaseOff = scratchOffVec[0];
    tempAlgParamsIntra0.sliceSize = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsIntra0.count = dataCountPerLoopAixs0;

    tempAlgParamsIntra0.inputSliceStride = dataSize_;
    tempAlgParamsIntra0.outputSliceStride = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsIntra0.repeatNum = rankSizeLevel1_;
    tempAlgParamsIntra0.inputRepeatStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsIntra0.outputRepeatStride = dataCountPerLoopAixs0 * dataTypeSize_ * rankSizeLevel0_;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, 
    const u64 dataOffset, const u64 dataCountPerLoopAixs0,
    std::vector<u64> &scratchOffVec, TemplateDataParams &tempAlgParamsInter0) const
{
    tempAlgParamsInter0.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsInter0.buffInfo.inputSize = resCtx.cclMem.size;
    tempAlgParamsInter0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsInter0.buffInfo.outputSize = param.outputSize;
    tempAlgParamsInter0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsInter0.buffInfo.inBuffType =  BufferType::HCCL_BUFFER;
    tempAlgParamsInter0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsInter0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter0.buffInfo.inBuffBaseOff = scratchOffVec[0] + rankIdxLevel0_ * dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsInter0.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsInter0.buffInfo.hcclBuffBaseOff = scratchOffVec[2]; 
    tempAlgParamsInter0.sliceSize = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsInter0.count = dataCountPerLoopAixs0;

    tempAlgParamsInter0.inputSliceStride = dataCountPerLoopAixs0 * dataTypeSize_ * rankSizeLevel0_;
    tempAlgParamsInter0.outputSliceStride = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsInter0.repeatNum = 1;
    tempAlgParamsInter0.inputRepeatStride = 0;
    tempAlgParamsInter0.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    const u64 dataOffset, const u64 dataCountPerLoopAixs1,
    std::vector<u64> &scratchOffVec, TemplateDataParams &tempAlgParamsInter1) const
{
    tempAlgParamsInter1.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsInter1.buffInfo.inputSize = param.inputSize;
    tempAlgParamsInter1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsInter1.buffInfo.outputSize = resCtx.cclMem.size;
    tempAlgParamsInter1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsInter1.buffInfo.inBuffType =  BufferType::INPUT;
    tempAlgParamsInter1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter1.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsInter1.buffInfo.outBuffBaseOff = scratchOffVec[3];
    tempAlgParamsInter1.buffInfo.hcclBuffBaseOff = scratchOffVec[3];
    tempAlgParamsInter1.sliceSize = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsInter1.count = dataCountPerLoopAixs1;

    tempAlgParamsInter1.inputSliceStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsInter1.outputSliceStride = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsInter1.repeatNum = rankSizeLevel0_;
    tempAlgParamsInter1.inputRepeatStride = dataSize_;
    tempAlgParamsInter1.outputRepeatStride = dataCountPerLoopAixs1 * dataTypeSize_ * rankSizeLevel1_;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, 
    const u64 dataOffset, const u64 dataCountPerLoopAixs1, 
    std::vector<u64> &scratchOffVec, TemplateDataParams &tempAlgParamsIntra1) const
{
    tempAlgParamsIntra1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsIntra1.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsIntra1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsIntra1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra1.buffInfo.inBuffBaseOff = scratchOffVec[3]; 
    tempAlgParamsIntra1.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsIntra1.buffInfo.hcclBuffBaseOff = scratchOffVec[1];
    tempAlgParamsIntra1.sliceSize = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsIntra1.count = dataCountPerLoopAixs1;

    tempAlgParamsIntra1.inputSliceStride = dataCountPerLoopAixs1 * dataTypeSize_ * rankSizeLevel1_;
    tempAlgParamsIntra1.outputSliceStride = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsIntra1.repeatNum = 1;
    tempAlgParamsIntra1.inputRepeatStride = 0;
    tempAlgParamsIntra1.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(std::vector<float> &splitDataSize) const
{
    // to do 先做等分，后续根据性能做调整
    double splitData = 0.5;
    splitDataSize.push_back(splitData);
    splitDataSize.push_back(splitData);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra,
    InsAlgTemplate1 &tempAlgInter)
{
    u64 intraThreadsNum = tempAlgIntra.GetThreadNum();
    u64 interThreadsNum = tempAlgInter.GetThreadNum();
    intraThreads_.assign(threads_.begin() + 1, threads_.begin() + 1 + intraThreadsNum);
    interThreads_.assign(threads_.begin() + 1 + intraThreadsNum, threads_.end());
    // 用于两个算法同步
    controlThread_ = threads_.at(0);
    templateMainThreads_.push_back(intraThreads_.at(0));
    templateMainThreads_.push_back(interThreads_.at(0));
    // 获取两个template各自的主thread上有多少notify
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    CHK_RET(tempAlgIntra.GetRes(intraTempRequest));
    CHK_RET(tempAlgInter.GetRes(interTempRequest));
    notifyIdxControlToTemplates_.push_back(intraTempRequest.notifyNumOnMainThread);
    notifyIdxControlToTemplates_.push_back(interTempRequest.notifyNumOnMainThread);
    notifyIdxTemplatesToControl_.push_back(0);
    notifyIdxTemplatesToControl_.push_back(1);
    return HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsReduceScatterParallelExecutor][Orchestrate] Orchestrate Start");

    // cclBuffer的大小
    maxTmpMemSize_ = resCtx.cclMem.size;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
        intraChannelMap_ = remoteRankToChannelInfo_[0];
        interChannelMap_ = remoteRankToChannelInfo_[1];
    }
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    rankSizeLevel0_ = GetRankSize(resCtx.algHierarchyInfo.infos[0]);
    rankSizeLevel1_ = GetRankSize(resCtx.algHierarchyInfo.infos[1]);
    myRank_ = resCtx.topoInfo.userRank;
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;

    // 实例化算法模板类
    // 构建template
    InsAlgTemplate0 intraTempAlg(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
    InsAlgTemplate1 interTempAlg(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[1]);
    // 将计算资源分配个每个算法
    PrepareResForTemplate(param, resCtx, intraTempAlg, interTempAlg);
    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx, intraTempAlg, interTempAlg);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllGatherParallelExecutor][Orchestrate]errNo[0x%016llx] All Gather excutor kernel run failed",
                   HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra,
    InsAlgTemplate1 &tempAlgInter)
{
    HCCL_INFO("[InsReduceScatterParallelExecutor][OrchestrateLoop] Start");
    HCCL_INFO("[InsReduceScatterParallelExecutor] AlgTemplate inter server is [%s]", tempAlgIntra.Describe().c_str());
    HCCL_INFO("[InsReduceScatterParallelExecutor] AlgTemplate intra server is [%s]", tempAlgInter.Describe().c_str());
    std::vector<float> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);
    u64 alignedSize = 16 * 1024; //假设需要16K对齐
    BufferType inBuffType = BufferType::INPUT;
    BufferType outBuffType = BufferType::OUTPUT;
    u32 intraScatchteMultipleStage0 = tempAlgIntra.CalcScratchMultiple(inBuffType, outBuffType);
    u32 interScatchteMultipleStage0 = tempAlgInter.CalcScratchMultiple(inBuffType, outBuffType);
    u32 intraScatchteMultipleStage1 = tempAlgIntra.CalcScratchMultiple(outBuffType, outBuffType);
    u32 interScatchteMultipleStage1 = tempAlgInter.CalcScratchMultiple(outBuffType, outBuffType);
    if (interScatchteMultipleStage0 == 0 || interScatchteMultipleStage1 == 0) {
        interScatchteMultipleStage0 = rankSizeLevel1_;
        interScatchteMultipleStage1 = rankSizeLevel1_;
    }
    u32 scratchMultipleIntra0 = static_cast<u32>(std::ceil(dataSplitSize[0] * intraScatchteMultipleStage0 * rankSizeLevel1_));
    u32 scratchMultipleIntra1 = static_cast<u32>(std::ceil(dataSplitSize[1] * intraScatchteMultipleStage1));
    u32 scratchMultipleInter1 = static_cast<u32>(std::ceil(dataSplitSize[1] * interScatchteMultipleStage0 * rankSizeLevel0_));
    u32 scratchMultipleInter0 = static_cast<u32>(std::ceil(dataSplitSize[0] * interScatchteMultipleStage1));
    u32 totalScratchMultiple = scratchMultipleIntra0 + scratchMultipleIntra1 + scratchMultipleInter0 + scratchMultipleInter1;
    u64 scratchMemBlockSize = maxTmpMemSize_;
    if (totalScratchMultiple > 0) {
        scratchMemBlockSize = (maxTmpMemSize_ / alignedSize / totalScratchMultiple) * alignedSize;
    }
    u64 intra0ScratchOffset = 0;
    u64 intra1ScratchOffset = intra0ScratchOffset + scratchMultipleIntra0 * scratchMemBlockSize;
    u64 inter0ScratchOffset = intra1ScratchOffset + scratchMultipleIntra1 * scratchMemBlockSize;
    u64 inter1ScratchOffset = inter0ScratchOffset + scratchMultipleInter0 * scratchMemBlockSize;
    std::vector<u64> scratchOffVec = {intra0ScratchOffset, intra1ScratchOffset, inter0ScratchOffset, inter1ScratchOffset};

    // dataSplitSize为分数，这里maxCountPerLoop对10取整
    u64 maxCountPerLoop = (std::min(static_cast<u64>(scratchMemBlockSize), static_cast<u64>(UB_MAX_DATA_SIZE)) / dataTypeSize_ / 10) * 10; 
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    TemplateDataParams tempAlgParamsIntra0;
    TemplateDataParams tempAlgParamsInter0;
    TemplateDataParams tempAlgParamsInter1;
    TemplateDataParams tempAlgParamsIntra1;

    TemplateResource templateAlgResIntra;
    TemplateResource templateAlgResInter;
    if (param.engine == COMM_ENGINE_CCU) {
        templateAlgResIntra.ccuKernels.insert(templateAlgResIntra.ccuKernels.end(),
                                              resCtx.ccuKernels.begin(),
                                              resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
        templateAlgResInter.ccuKernels.insert(templateAlgResInter.ccuKernels.end(),
                                              resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
                                              resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);
    } else {
        templateAlgResIntra.channels = intraChannelMap_;
        templateAlgResInter.channels = interChannelMap_;
    }
    templateAlgResIntra.threads = intraThreads_;
    templateAlgResInter.threads = interThreads_;
    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCount = (loopIndex == loopTimes - 1) ? (dataCount_ - loopIndex * maxCountPerLoop) : maxCountPerLoop;
        u64 dataCountPerLoopAixs0 = static_cast<u64>(dataSplitSize[0] * currCount);
        u64 dataCountPerLoopAixs1 = currCount - dataCountPerLoopAixs0;
        
        u64 dataOffset0 = loopIndex * maxCountPerLoop * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + dataCountPerLoopAixs0 * dataTypeSize_;
        
        //第一步开始前同步
        CHK_RET(PreSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxControlToTemplates_));
        //数据0的server内的mesh算法
        GenTemplateAlgParamsIntra0(param, resCtx, dataOffset0, dataCountPerLoopAixs0, scratchOffVec, tempAlgParamsIntra0);
        //把每个template需要的queue传进去，比如stars的mesh要传多条queue
        CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra0, templateAlgResIntra));
        //数据1的server间的nhr算法
        GenTemplateAlgParamsInter1(param, resCtx, dataOffset1, dataCountPerLoopAixs1, scratchOffVec, tempAlgParamsInter1);
        CHK_RET(tempAlgInter.KernelRun(param, tempAlgParamsInter1, templateAlgResInter));
        //第一步做完后回到主流做尾同步
        CHK_RET(PostSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxTemplatesToControl_));

        //第二步开始前同步
        CHK_RET(PreSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxControlToTemplates_));
        //数据0的server间的nhr算法
        GenTemplateAlgParamsInter0(param, resCtx, dataOffset0, dataCountPerLoopAixs0, scratchOffVec, tempAlgParamsInter0);
        CHK_RET(tempAlgInter.KernelRun(param, tempAlgParamsInter0, templateAlgResInter));
        //数据1的server内的mesh算法
        GenTemplateAlgParamsIntra1(param, resCtx, dataOffset1,  dataCountPerLoopAixs1, scratchOffVec, tempAlgParamsIntra1);
        CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra1, templateAlgResIntra));
        //尾同步
        CHK_RET(PostSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxTemplatesToControl_));
    }
    HCCL_INFO("[InsReduceScatterParallelExecutor][OrchestrateLoop] End.");
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
uint64_t InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetRankSize(
    const std::vector<std::vector<u32>> &subCommRanks)
{
    uint64_t count = 1;
    for (const auto &i : subCommRanks) {
        count *= i.size();
    }
    return count;
}

// 算法注册
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, InsReduceScatterParallelMesh1DNHR,
    InsReduceScatterParallelExecutor, TopoMatchMultilevel, InsTempReduceScatterMesh1D, InsTempReduceScatterNHR);
#ifndef AICPU_COMPILE
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CcuReduceScatterParallelMesh1DNHR,
    InsReduceScatterParallelExecutor, TopoMatchMultilevel, CcuTempReduceScatterMesh1DMem2Mem, CcuTempReduceScatterNHR1DMem2Mem);
#endif
}