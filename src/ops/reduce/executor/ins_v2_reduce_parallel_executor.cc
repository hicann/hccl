/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>

#include "ins_v2_reduce_parallel_executor.h"

#include "../template/aicpu/ins_temp_reduce_mesh_1D.h"
#include "../template/aicpu/ins_temp_reduce_nhr.h"
#include "topo_match_multilevel.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2ReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2ReduceParallelExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfo *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(HcclComm comm,
    const OpParam &param, const TopoInfo *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
    AlgResourceRequest &resourceRequest)
{
    // 构建template
    InsAlgTemplate0 intraTempAlg(param, topoInfo->userRank, algHierarchyInfo.infos.at(0));
    InsAlgTemplate1 interTempAlg(param, topoInfo->userRank, algHierarchyInfo.infos.at(1));

    // 调用计算资源的函数
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    
    intraTempAlg.CalcRes(comm, param, topoInfo, intraTempRequest);
    interTempAlg.CalcRes(comm, param, topoInfo, interTempRequest);
    // 申请一条控制thread作为主thread，该thread仅用于两个template之间同步
    resourceRequest.notifyNumOnMainThread = 2;
    // 由于主thread被单独作为控制thread，因此总的slaveThread需要额外加上两个template的主thread
    resourceRequest.slaveThreadNum = intraTempRequest.slaveThreadNum + interTempRequest.slaveThreadNum + 2;
    // 第一个template的主thread需要的notify数量，+1是因为需要和控制thread做同步
    resourceRequest.notifyNumPerThread.emplace_back(intraTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              intraTempRequest.notifyNumPerThread.begin(),
                                              intraTempRequest.notifyNumPerThread.end());
    // 这一条是interTemplate的主thread，需要+1是为了和控制thread进行同步
    resourceRequest.notifyNumPerThread.emplace_back(interTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              interTempRequest.notifyNumPerThread.begin(),
                                              interTempRequest.notifyNumPerThread.end());
    resourceRequest.channels.emplace_back(intraTempRequest.channels.at(0));
    resourceRequest.channels.emplace_back(interTempRequest.channels.at(0));

    myRank_ = topoInfo->userRank;
    HCCL_DEBUG("Need Resource notifyNumOnMainThread[%d], channels[%d], slaveThreadNum[%d] | intraTemplate "
               "notifyNumOnMainThread[%d], channels[%d] | interTemplate notifyNumOnMainThread[%d], channels[%d]",
        resourceRequest.notifyNumOnMainThread,
        resourceRequest.channels.size(),
        resourceRequest.slaveThreadNum,
        intraTempRequest.notifyNumOnMainThread,
        intraTempRequest.channels.size(),
        interTempRequest.notifyNumOnMainThread,
        interTempRequest.channels.size());

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2ReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParams0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, u64 dataOffset, u64 dataCountPerLoopAxis0,
    u64 scratchOffset, TemplateDataParams &tempAlgParams0) const
{
    tempAlgParams0.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams0.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams0.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParams0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams0.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParams0.buffInfo.outBuffBaseOff = scratchOffset;
    tempAlgParams0.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParams0.sliceSize = dataCountPerLoopAxis0 * dataTypeSize_;
    tempAlgParams0.tailSize = tempAlgParams0.sliceSize;
    tempAlgParams0.count = dataCountPerLoopAxis0;
    tempAlgParams0.inputSliceStride = 0;  // 输入数据仅有1个slice，不需要 stride
    tempAlgParams0.outputSliceStride = 0;
    tempAlgParams0.repeatNum = 1;
    tempAlgParams0.inputRepeatStride = 0;
    tempAlgParams0.outputRepeatStride = 0;

    HCCL_DEBUG(
        "[InsV2ReduceParallelExecutor][GenTemplateAlgParamsIntra0] rank[%d] inBuffBaseOff[%llu] "
        "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu] intraLocalRankSize_[%u] "
        "interLocalRankSize_[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]",
        myRank_,
        tempAlgParams0.buffInfo.inBuffBaseOff,
        tempAlgParams0.buffInfo.outBuffBaseOff,
        tempAlgParams0.buffInfo.hcclBuffBaseOff,
        tempAlgParams0.sliceSize,
        tempAlgParams0.outputSliceStride,
        intraLocalRankSize_,
        interLocalRankSize_,
        rankIdxLevel0_,
        rankIdxLevel1_);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2ReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, u64 dataOffset, u64 dataCountPerLoopAxis1,
    u64 scratchOffset, u64 othScratchOffset, TemplateDataParams &tempAlgParams1) const
{
    tempAlgParams1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParams1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams1.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams1.buffInfo.inBuffBaseOff = othScratchOffset;
    tempAlgParams1.buffInfo.outBuffBaseOff = dataOffset;  // for example 0 2 4 | 1 3 5
    tempAlgParams1.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParams1.sliceSize = dataCountPerLoopAxis1 * dataTypeSize_;
    tempAlgParams1.tailSize = tempAlgParams1.sliceSize;
    tempAlgParams1.count = dataCountPerLoopAxis1;

    tempAlgParams1.inputSliceStride = 0;  // 输入数据仅有1个slice，不需要 stride
    tempAlgParams1.outputSliceStride = 0;
    tempAlgParams1.repeatNum = 1;
    tempAlgParams1.inputRepeatStride = 0;
    tempAlgParams1.outputRepeatStride = 0;
    HCCL_DEBUG("[InsV2ReduceParallelExecutor][GenTemplateAlgParams1] rank[%u] inBuffBaseOff[%llu] "
               "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu]",
        myRank_,
        tempAlgParams1.buffInfo.inBuffBaseOff,
        tempAlgParams1.buffInfo.outBuffBaseOff,
        tempAlgParams1.buffInfo.hcclBuffBaseOff,
        tempAlgParams1.sliceSize,
        tempAlgParams1.outputSliceStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2ReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, u64 dataOffset, u64 dataCountPerLoopAxis1,
    u64 scratchOffset, u64 othScratchOffset, TemplateDataParams &tempAlgParams1) const
{
    (void)scratchOffset;
    tempAlgParams1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParams1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams1.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams1.buffInfo.inBuffBaseOff = othScratchOffset;
    tempAlgParams1.buffInfo.outBuffBaseOff = dataOffset;
    // 这里与Intra1不同，因为NHR算法比较特殊，不需要使用除输入的CCL BUFFER外其他额外的CCL BUFFER，
    // 因此直接使用inBuffBaseOff
    tempAlgParams1.buffInfo.hcclBuffBaseOff = tempAlgParams1.buffInfo.inBuffBaseOff;
    tempAlgParams1.sliceSize = dataCountPerLoopAxis1 * dataTypeSize_;
    tempAlgParams1.tailSize = tempAlgParams1.sliceSize;
    tempAlgParams1.count = dataCountPerLoopAxis1;

    tempAlgParams1.inputSliceStride = 0;  // 输入数据仅有1个slice，不需要 stride
    tempAlgParams1.outputSliceStride = 0;
    tempAlgParams1.repeatNum = 1;
    tempAlgParams1.inputRepeatStride = 0;
    tempAlgParams1.outputRepeatStride = 0;
    HCCL_DEBUG("[InsV2ReduceParallelExecutor][GenTemplateAlgParams1] rank[%u] inBuffBaseOff[%llu] "
               "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu]",
        myRank_,
        tempAlgParams1.buffInfo.inBuffBaseOff,
        tempAlgParams1.buffInfo.outBuffBaseOff,
        tempAlgParams1.buffInfo.hcclBuffBaseOff,
        tempAlgParams1.sliceSize,
        tempAlgParams1.outputSliceStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
uint64_t InsV2ReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo)
{
    uint64_t count = 1;
    for (const auto &i : vTopo) {
        count *= i.size();
    }
    return count;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2ReduceParallelExecutor][Orchestrate] Orchestrate Start");
    maxTmpMemSize_ = resCtx.cclMem.size;  // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    myRank_ = resCtx.topoInfo.userRank;
    root_ = param.root;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
        intraLinkMap_ = remoteRankToChannelInfo_.at(0);
        interLinkMap_ = remoteRankToChannelInfo_.at(1);
    }
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    intraLocalRankSize_ = GetRankSize(resCtx.algHierarchyInfo.infos[0]);
    interLocalRankSize_ = GetRankSize(resCtx.algHierarchyInfo.infos[1]);
    rankIdxLevel0_ = myRank_ % intraLocalRankSize_;
    rankIdxLevel1_ = myRank_ / intraLocalRankSize_;
    CHK_RET(CalcLocalRoot());
    HCCL_DEBUG(
        "[InsV2ReduceParallelExecutor][Orchestrate]myRank[%u], intraLocalRankSize_[%u], interLocalRankSize_[%u], "
        "rankIdxLevel0_[%u], rankIdxLevel1_[%u]",
        myRank_,
        intraLocalRankSize_,
        interLocalRankSize_,
        rankIdxLevel0_,
        rankIdxLevel1_);
    // 实例化算法模板类
    // 构建template

    OpParam paramIntra = param;
    paramIntra.root = intraLocalRoot_;
    OpParam paramInter = param;
    paramInter.root = interLocalRoot_;

    InsAlgTemplate0 intraTempAlg(paramIntra, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos.at(0));
    InsAlgTemplate1 interTempAlg(paramInter, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos.at(1));
    // 将计算资源分配个每个算法
    PrepareResForTemplate(param, resCtx, intraTempAlg, interTempAlg);
    // 算法展开

    HcclResult ret = OrchestrateLoop(param, resCtx, intraTempAlg, interTempAlg);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2ReduceParallelExecutor][Orchestrate]errNo[0x%016llx] All Gather excutor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra,
    InsAlgTemplate1 &tempAlgInter)
{
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    u64 intraThreadsNum = tempAlgIntra.GetThreadNum();
    u64 interThreadsNum = tempAlgInter.GetThreadNum();
    HCCL_DEBUG("[InsV2ReduceParallelExecutor][PrepareResForTemplate]myRank_[%d], threads_[%d], intraThreadsNum[%d], "
               "interThreadsNum[%d]",
        myRank_,
        threads_.size(),
        intraThreadsNum,
        interThreadsNum);
    intraThreads_.assign(threads_.begin() + 1, threads_.begin() + intraThreadsNum + 1);
    interThreads_.assign(threads_.begin() + intraThreadsNum + 1, threads_.end());
    HCCL_DEBUG("[InsV2ReduceParallelExecutor][PrepareResForTemplate]myRank_[%d],  threads_[%d], intraThreads_[%d], "
               "interThreads_[%d]",
        myRank_,
        threads_.size(),
        intraThreads_.size(),
        interThreads_.size());

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2ReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(
    std::vector<float> &splitDataSize) const
{
    splitDataSize = {0.5, 0.5};
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra,
    InsAlgTemplate1 &tempAlgInter)
{
    HCCL_DEBUG("[InsV2ReduceParallelExecutor][OrchestrateLoop] Rank[%u], inputPtr[%#llx] outputPtr[%#llx], "
               "cclAddr[%#llx], cclSize[%u]",
        myRank_,
        param.inputPtr,
        param.outputPtr,
        resCtx.cclMem.addr,
        resCtx.cclMem.size);
    HCCL_INFO("[InsV2ReduceParallelExecutor] AlgTemplate inter server is [%s]", tempAlgIntra.Describe().c_str());
    HCCL_INFO("[InsV2ReduceParallelExecutor] AlgTemplate intra server is [%s]", tempAlgInter.Describe().c_str());
    u64 alignedSize = 16 * 1024;  // 假设需要16K对齐
    std::vector<float> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);
    u32 intraScratchMultipleStage0 = tempAlgIntra.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 interScratchMultipleStage0 = tempAlgInter.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 intraScratchMultipleStage1 = tempAlgIntra.CalcScratchMultiple(BufferType::OUTPUT, BufferType::OUTPUT);
    u32 interScratchMultipleStage1 = tempAlgInter.CalcScratchMultiple(BufferType::OUTPUT, BufferType::OUTPUT);
    u32 scratchMultipleIntra = static_cast<u32>(std::max(std::ceil(dataSplitSize[0] * intraScratchMultipleStage0),
        std::ceil(dataSplitSize[1] * intraScratchMultipleStage1)));
    u32 scratchMultipleInter = static_cast<u32>(std::max(std::ceil(dataSplitSize[1] * interScratchMultipleStage0),
        std::ceil(dataSplitSize[0] * interScratchMultipleStage1)));
    u32 totalScratchMultiple = scratchMultipleIntra + scratchMultipleInter;
    HCCL_DEBUG("[InsV2ReduceParallelExecutor][OrchestrateLoop] total:[%u], scratchMultipleIntra[%u]",
        totalScratchMultiple,
        scratchMultipleIntra);
    u64 scratchMemBlockSize = maxTmpMemSize_;
    if (totalScratchMultiple > 0) {
        scratchMemBlockSize = (maxTmpMemSize_ / alignedSize / totalScratchMultiple) * alignedSize;
    }
    u64 intraScratchOffset = 0;
    u64 interScratchOffset = scratchMultipleIntra * scratchMemBlockSize;

    u64 maxCountPerLoop = std::min<u64>(scratchMemBlockSize, UB_MAX_DATA_SIZE) / dataTypeSize_;

    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    TemplateResource interTempAlgRes;
    interTempAlgRes.channels = interLinkMap_;
    interTempAlgRes.threads = interThreads_;
    interTempAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    TemplateResource intraTempAlgRes;
    intraTempAlgRes.channels = intraLinkMap_;
    intraTempAlgRes.threads = intraThreads_;
    intraTempAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    TemplateDataParams tempAlgParamsIntra0;
    TemplateDataParams tempAlgParamsInter0;
    TemplateDataParams tempAlgParamsInter1;
    TemplateDataParams tempAlgParamsIntra1;

    OpParam paramIntra = param;
    paramIntra.root = intraLocalRoot_;
    OpParam paramInter = param;
    paramInter.root = interLocalRoot_;

    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCount = (loopIndex + 1 == loopTimes) ? (dataCount_ - loopIndex * maxCountPerLoop) : maxCountPerLoop;
        u64 dataCountPerLoopAxis0 = currCount / 2;
        u64 dataCountPerLoopAxis1 = currCount - dataCountPerLoopAxis0;
        // 第一步开始前同步
        CHK_RET(PreSyncInterThreads(threads_.at(0), {intraThreads_.at(0), interThreads_.at(0)}, {0, 0}));
        u64 dataOffset0 = loopIndex * maxCountPerLoop * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + dataCountPerLoopAxis0 * dataTypeSize_;
        // 数据0的server内的mesh算法
        GenTemplateAlgParams0(
            paramIntra, resCtx, dataOffset0, dataCountPerLoopAxis0, intraScratchOffset, tempAlgParamsIntra0);
        // 把每个template需要的queue传进去，比如stars的mesh要传多条queue
        CHK_RET(tempAlgIntra.KernelRun(paramIntra, tempAlgParamsIntra0, intraTempAlgRes));
        // 数据1的server间的nhr算法
        GenTemplateAlgParams0(
            paramInter, resCtx, dataOffset1, dataCountPerLoopAxis1, interScratchOffset, tempAlgParamsInter1);
        CHK_RET(tempAlgInter.KernelRun(paramInter, tempAlgParamsInter1, interTempAlgRes));
        // 第一步做完后回到主流做尾同步
        CHK_RET(PostSyncInterThreads(threads_.at(0), {intraThreads_.at(0), interThreads_.at(0)}, {0, 1}));

        HCCL_DEBUG("[InsV2ReduceParallelExecutor][OrchestrateLoop] rank[%u] finish step 1", myRank_);
        if ((myRank_ != intraLocalRoot_) && (myRank_ != interLocalRoot_)) {
            continue;
        }

        // 第二步开始前同步
        CHK_RET(PreSyncInterThreads(threads_.at(0), {intraThreads_.at(0), interThreads_.at(0)}, {0, 0}));
        if (myRank_ == intraLocalRoot_) {
            // 数据0的server间的nhr算法
            GenTemplateAlgParamsInter1(paramInter,
                resCtx,
                dataOffset0,
                dataCountPerLoopAxis0,
                interScratchOffset,
                intraScratchOffset,
                tempAlgParamsInter0);
            CHK_RET(tempAlgInter.KernelRun(paramInter, tempAlgParamsInter0, interTempAlgRes));
        }
        // 数据1的server内的mesh算法
        if (myRank_ == interLocalRoot_) {
            GenTemplateAlgParamsIntra1(paramIntra,
                resCtx,
                dataOffset1,
                dataCountPerLoopAxis1,
                intraScratchOffset,
                interScratchOffset,
                tempAlgParamsIntra1);
            CHK_RET(tempAlgIntra.KernelRun(paramIntra, tempAlgParamsIntra1, intraTempAlgRes));
        }
        // 尾同步
        CHK_RET(PostSyncInterThreads(threads_.at(0), {intraThreads_.at(0), interThreads_.at(0)}, {0, 1}));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcLocalRoot()
{
    CHK_PRT_RET(
        root_ == UINT32_MAX, HCCL_ERROR("[CalcLocalRoot] root[%u] is invalid", root_), HcclResult::HCCL_E_INTERNAL);

    intraLocalRoot_ = intraLocalRankSize_ * (myRank_ / intraLocalRankSize_) + (root_ % intraLocalRankSize_);
    interLocalRoot_ = intraLocalRankSize_ * (root_ / intraLocalRankSize_) + (myRank_ % intraLocalRankSize_);
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_REDUCE, InsReduceParallelMesh1DNHR, InsV2ReduceParallelExecutor,
    TopoMatchMultilevel, InsTempReduceMesh1D, InsTempReduceNHR);

// 算法注册