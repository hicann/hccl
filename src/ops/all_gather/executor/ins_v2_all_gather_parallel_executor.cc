/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_parallel_executor.h"
#include <cmath>
#include "alg_data_trans_wrapper.h"
#include "ins_temp_all_gather_mesh_1D.h"
#include "ins_temp_all_gather_nhr.h"
#ifndef AICPU_COMPILE
#if !defined(HCCL_CANN_COMPAT_850)
#include "ccu_temp_all_gather_nhr_1D_mem2mem.h"
#include "ccu_temp_all_gather_mesh_1D_mem2mem.h"
#include "ccu_temp_all_gather_nhr_1D_multi_jetty_mem2mem.h"
#endif /* !HCCL_CANN_COMPAT_850 */
#endif
#include "alg_data_trans_wrapper.h"

#include "topo_match_multilevel.h"
#include "topo_match_ubx.h"
#include "topo_match_pcie_mix.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2AllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2AllGatherParallelExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
    AlgResourceRequest &resourceRequest)
{
    myRank_ = topoInfo->userRank;
    // 构建template
    std::vector<std::vector<u32>> intraHierarchyInfo;
    std::vector<std::vector<u32>> interHierarchyInfo;
    if(topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        intraHierarchyInfo = {algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        u32 meshSize = algHierarchyInfo.infos[0][0].size();
        for(auto rank : algHierarchyInfo.infos[0][1]) {
            if(rank % meshSize == topoInfo->userRank % meshSize) {
                closRanks.push_back(rank);
            }
        }
        interHierarchyInfo = {closRanks};
    } else {
        constexpr u32 TOPO_NUM = 2;
        CHK_PRT_RET(algHierarchyInfo.infos.size() < TOPO_NUM || algHierarchyInfo.infos[0].empty() || algHierarchyInfo.infos[1].empty(),
                     HCCL_ERROR("[InsAllGatherParallelExecutor][CalcRes] Invalid topoInfo"),
                     HcclResult::HCCL_E_INTERNAL);
        intraHierarchyInfo = algHierarchyInfo.infos[0];
        interHierarchyInfo = algHierarchyInfo.infos[1];
    }

    InsAlgTemplate0 intraTempAlg(param, topoInfo->userRank, intraHierarchyInfo);
    InsAlgTemplate1 interTempAlg(param, topoInfo->userRank, interHierarchyInfo);
    
    // 调用计算资源的函数
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    CHK_RET(intraTempAlg.CalcRes(comm, param, topoInfo, intraTempRequest));
    CHK_RET(interTempAlg.CalcRes(comm, param, topoInfo, interTempRequest));
    constexpr u32 SUB_MAIN_THREAD_NUM = 2;
    resourceRequest.notifyNumOnMainThread = SUB_MAIN_THREAD_NUM;  // 用于两个template间同步
    resourceRequest.slaveThreadNum = intraTempRequest.slaveThreadNum + interTempRequest.slaveThreadNum + SUB_MAIN_THREAD_NUM;
    resourceRequest.notifyNumPerThread.emplace_back(intraTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              intraTempRequest.notifyNumPerThread.begin(),
                                              intraTempRequest.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(interTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              interTempRequest.notifyNumPerThread.begin(),
                                              interTempRequest.notifyNumPerThread.end());
    if (param.engine != COMM_ENGINE_CCU) {
        CHK_PRT_RET(intraTempRequest.channels.empty() || interTempRequest.channels.empty(),
                     HCCL_ERROR("[InsAllGatherParallelExecutor][CalcRes] intraTemplate or interTemplate has empty channels."),
                     HcclResult::HCCL_E_INTERNAL);
        resourceRequest.channels.emplace_back(intraTempRequest.channels[0]);
        resourceRequest.channels.emplace_back(interTempRequest.channels[0]);
    } else {
        // ccu
        HCCL_INFO("[InsAllGatherParallelExecutor][CalcRes] intraTemplate has [%d] kernels.", intraTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            intraTempRequest.ccuKernelInfos.begin(),
                                            intraTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(intraTempRequest.ccuKernelNum[0]);
        HCCL_INFO("[InsAllGatherParallelExecutor][CalcRes] interTemplate has [%d] kernels.", interTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            interTempRequest.ccuKernelInfos.begin(),
                                            interTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(interTempRequest.ccuKernelNum[0]);
    }
    HCCL_DEBUG("[InsV2AllGatherParallelExecutor][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u], "
               "channels[%u]",
               myRank_, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
               resourceRequest.channels.size());
    for (auto i = 0; i < resourceRequest.notifyNumPerThread.size(); i++) {
        HCCL_DEBUG("[InsV2AllGatherParallelExecutor][CalcRes] myRank[%u], notifyNumPerThread[%u]=[%u]", myRank_, i,
                   resourceRequest.notifyNumPerThread[i]);
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAixs0, const u64 scratchOffset, TemplateDataParams &tempAlgParamsIntra0) const
{
    tempAlgParamsIntra0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsIntra0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsIntra0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsIntra0.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsIntra0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra0.buffInfo.inputSize = param.inputSize;
    tempAlgParamsIntra0.buffInfo.outputSize = param.outputSize;

    tempAlgParamsIntra0.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsIntra0.buffInfo.outBuffBaseOff = rankIdxLevel1_ * rankSizeLevel0_ * dataSize_ + dataOffset;
    tempAlgParamsIntra0.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsIntra0.sliceSize = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsIntra0.count = dataCountPerLoopAixs0;
    tempAlgParamsIntra0.tailSize = tempAlgParamsIntra0.sliceSize;

    tempAlgParamsIntra0.inputSliceStride = 0;
    tempAlgParamsIntra0.outputSliceStride = dataSize_;
    tempAlgParamsIntra0.repeatNum = 1;
    tempAlgParamsIntra0.inputRepeatStride = 0;
    tempAlgParamsIntra0.outputRepeatStride = 0;
    tempAlgParamsIntra0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    HCCL_DEBUG(
        "[InsV2AllGatherParallelExecutor][GenTemplateAlgParamsIntra0] rank[%d] inBuffBaseOff[%llu] "
        "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu] rankSizeLevel0[%u] "
        "rankSizeLevel1[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]",
        myRank_, tempAlgParamsIntra0.buffInfo.inBuffBaseOff, tempAlgParamsIntra0.buffInfo.outBuffBaseOff,
        tempAlgParamsIntra0.buffInfo.hcclBuffBaseOff, tempAlgParamsIntra0.sliceSize,
        tempAlgParamsIntra0.outputSliceStride, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAixs0, const u64 scratchOffset, TemplateDataParams &tempAlgParamsInter0) const
{
    tempAlgParamsInter0.buffInfo.inputPtr = param.outputPtr;
    tempAlgParamsInter0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsInter0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsInter0.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsInter0.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsInter0.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsInter0.buffInfo.inBuffType = BufferType::OUTPUT;
    tempAlgParamsInter0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsInter0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter0.buffInfo.inputSize = param.inputSize;
    tempAlgParamsInter0.buffInfo.outputSize = param.outputSize;
    tempAlgParamsInter0.sliceSize = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsInter0.count = dataCountPerLoopAixs0;
    tempAlgParamsInter0.tailSize = tempAlgParamsInter0.sliceSize;

    tempAlgParamsInter0.inputSliceStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsInter0.outputSliceStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsInter0.repeatNum = rankSizeLevel0_;
    tempAlgParamsInter0.inputRepeatStride = dataSize_;
    tempAlgParamsInter0.outputRepeatStride = dataSize_;
    tempAlgParamsInter0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
    HCCL_DEBUG("[InsV2AllGatherParallelExecutor][GenTemplateAlgParamsInter0] rank[%u] inBuffBaseOff[%llu] "
               "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu] "
               "outputRepeatStride[%llu]",
               myRank_, tempAlgParamsInter0.buffInfo.inBuffBaseOff, tempAlgParamsInter0.buffInfo.outBuffBaseOff,
               tempAlgParamsInter0.buffInfo.hcclBuffBaseOff, tempAlgParamsInter0.sliceSize,
               tempAlgParamsInter0.outputSliceStride, tempAlgParamsInter0.outputRepeatStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAixs1, const u64 scratchOffset, TemplateDataParams &tempAlgParamsInter1) const
{
    tempAlgParamsInter1.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsInter1.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsInter1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsInter1.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsInter1.buffInfo.outBuffBaseOff = rankIdxLevel0_ * dataSize_ + dataOffset;
    tempAlgParamsInter1.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsInter1.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsInter1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsInter1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter1.buffInfo.inputSize = param.inputSize;
    tempAlgParamsInter1.buffInfo.outputSize = param.outputSize;
    tempAlgParamsInter1.sliceSize = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsInter1.count = dataCountPerLoopAixs1;
    tempAlgParamsInter1.tailSize = tempAlgParamsInter1.sliceSize;

    tempAlgParamsInter1.inputSliceStride = 0;
    tempAlgParamsInter1.outputSliceStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsInter1.repeatNum = 1;
    tempAlgParamsInter1.inputRepeatStride = 0;
    tempAlgParamsInter1.outputRepeatStride = 0;
    tempAlgParamsInter1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
    HCCL_DEBUG("[InsV2AllGatherParallelExecutor][GenTemplateAlgParamsInter1] rank[%u] inBuffBaseOff[%llu] "
               "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu]",
               myRank_, tempAlgParamsInter1.buffInfo.inBuffBaseOff, tempAlgParamsInter1.buffInfo.outBuffBaseOff,
               tempAlgParamsInter1.buffInfo.hcclBuffBaseOff, tempAlgParamsInter1.sliceSize,
               tempAlgParamsInter1.outputSliceStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAixs1, const u64 scratchOffset, TemplateDataParams &tempAlgParamsIntra1) const
{
    tempAlgParamsIntra1.buffInfo.inputPtr = param.outputPtr;
    tempAlgParamsIntra1.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsIntra1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsIntra1.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsIntra1.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsIntra1.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsIntra1.buffInfo.inBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra1.buffInfo.inputSize = param.inputSize;
    tempAlgParamsIntra1.buffInfo.outputSize = param.outputSize;
    tempAlgParamsIntra1.sliceSize = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsIntra1.count = dataCountPerLoopAixs1;
    tempAlgParamsIntra1.tailSize = tempAlgParamsIntra1.sliceSize;

    tempAlgParamsIntra1.inputSliceStride = dataSize_;
    tempAlgParamsIntra1.outputSliceStride = dataSize_;
    tempAlgParamsIntra1.repeatNum = rankSizeLevel1_;
    tempAlgParamsIntra1.inputRepeatStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsIntra1.outputRepeatStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsIntra1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
    HCCL_DEBUG("[InsV2AllGatherParallelExecutor][GenTemplateAlgParamsIntra1] rank[%u] inBuffBaseOff[%llu] "
               "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu] "
               "outputRepeatStride[%llu]",
               myRank_, tempAlgParamsIntra1.buffInfo.inBuffBaseOff, tempAlgParamsIntra1.buffInfo.outBuffBaseOff,
               tempAlgParamsIntra1.buffInfo.hcclBuffBaseOff, tempAlgParamsIntra1.sliceSize,
               tempAlgParamsIntra1.outputSliceStride, tempAlgParamsIntra1.outputRepeatStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
uint64_t InsV2AllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo) const
{
    uint64_t count = 1;
    for (const auto &i : vTopo) {
        count *= i.size();
    }
    return count;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherParallelExecutor][Orchestrate] Orchestrate Start");
    maxTmpMemSize_ = resCtx.cclMem.size;  // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    myRank_ = resCtx.topoInfo.userRank;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
        intraLinkMap_ = remoteRankToChannelInfo_[0];
        interLinkMap_ = remoteRankToChannelInfo_[1];
    }
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    if(resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix) {
        intraHierarchyInfo_ = {resCtx.algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        u32 meshSize = resCtx.algHierarchyInfo.infos[0][0].size();
        for(auto rank : resCtx.algHierarchyInfo.infos[0][1]) {
            if(rank % meshSize == resCtx.topoInfo.userRank % meshSize) {
                closRanks.push_back(rank);
            }
        }
        interHierarchyInfo_ = {closRanks};
    } else {
        intraHierarchyInfo_ = resCtx.algHierarchyInfo.infos[0];
        interHierarchyInfo_ = resCtx.algHierarchyInfo.infos[1];
    }
    rankSizeLevel0_ = GetRankSize(intraHierarchyInfo_);
    rankSizeLevel1_ = GetRankSize(interHierarchyInfo_);
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    // 实例化算法模板类
    // 构建template
    InsAlgTemplate0 intraTempAlg(param, resCtx.topoInfo.userRank, intraHierarchyInfo_);
    InsAlgTemplate1 interTempAlg(param, resCtx.topoInfo.userRank, interHierarchyInfo_);
    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS) {
        interTempAlg.SetchannelsPerRank(interLinkMap_);
    }
    // 将计算资源分配个每个算法
    PrepareResForTemplate(intraTempAlg, interTempAlg);
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
HcclResult InsV2AllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter)
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
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(
    std::vector<float> &splitDataSize) const
{
    double splitData = multipleDimensionSplitRatio_;
    splitDataSize.push_back(1 - splitData);
    splitDataSize.push_back(splitData);
    HCCL_INFO("[InsV2AllGatherParallelExecutor] splitDataSize is %f, %f", splitDataSize[0], splitDataSize[1]);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra,
    InsAlgTemplate1 &tempAlgInter)
{
    HCCL_INFO("[InsV2AllGatherParallelExecutor] AlgTemplate intra server is [%s]", tempAlgIntra.Describe().c_str());
    HCCL_INFO("[InsV2AllGatherParallelExecutor] AlgTemplate inter server is [%s]", tempAlgInter.Describe().c_str());
    multipleDimensionSplitRatio_ = param.opConfig.multipleDimensionSplitRatio;
    std::vector<float> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);
    u32 intraScatchteMultipleStage0 = tempAlgIntra.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 interScatchteMultipleStage0 = tempAlgInter.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 intraScatchteMultipleStage1 = tempAlgIntra.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 interScatchteMultipleStage1 = tempAlgInter.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 scratchMultipleIntra =
        static_cast<u32>(std::max(std::ceil(dataSplitSize[0] * intraScatchteMultipleStage0),
                                  std::ceil(dataSplitSize[1] * intraScatchteMultipleStage1 * rankSizeLevel1_)));
    u32 scratchMultipleInter =
        static_cast<u32>(std::max(std::ceil(dataSplitSize[1] * interScatchteMultipleStage0),
                                  std::ceil(dataSplitSize[0] * interScatchteMultipleStage1 * rankSizeLevel0_)));
    u32 totalScratchMultiple = scratchMultipleIntra + scratchMultipleInter;
    u64 scratchMemBlockSize = maxTmpMemSize_;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    if (totalScratchMultiple > 0) {
        scratchMemBlockSize = (maxTmpMemSize_ / HCCL_MIN_SLICE_ALIGN / totalScratchMultiple) * HCCL_MIN_SLICE_ALIGN;
        scratchMemBlockSize = std::min(scratchMemBlockSize, transportBoundDataSize);
    }
    u64 intraScratchOffset = 0;
    u64 interScratchOffset = scratchMultipleIntra * scratchMemBlockSize;

    u64 maxCountPerLoop =
        (std::min(static_cast<u64>(scratchMemBlockSize), static_cast<u64>(UB_MAX_DATA_SIZE)) / dataTypeSize_ / 10) * 10;

    // ============ 循环前的数据对齐操作 ============
    u64 alignSize = AICPU_ALIGN_SIZE; // 用于4k对齐
    
    // 根据 dataSplitSize 计算两块数据的大小
    u64 dataCountPerLoopAixs0 = static_cast<u64>(dataSplitSize[0] * maxCountPerLoop);
    u64 dataCountPerLoopAixs1 = maxCountPerLoop - dataCountPerLoopAixs0;
    
    // 对两块数据分别进行4K对齐（向下取整）
    if (dataCountPerLoopAixs0 * dataTypeSize_ >= alignSize) {
        dataCountPerLoopAixs0 = dataCountPerLoopAixs0 * dataTypeSize_ / alignSize * alignSize / dataTypeSize_;
    }
    if (dataCountPerLoopAixs1 * dataTypeSize_ >= alignSize) {
        dataCountPerLoopAixs1 = dataCountPerLoopAixs1 * dataTypeSize_ / alignSize * alignSize / dataTypeSize_;
    }
    
    // 用对齐后的数据量之和重新刷新 maxCountPerLoop
    maxCountPerLoop = dataCountPerLoopAixs0 + dataCountPerLoopAixs1;
    // ====================================================

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

    if (param.engine == COMM_ENGINE_CCU) {
        intraTempAlgRes.ccuKernels.insert(intraTempAlgRes.ccuKernels.end(),
                                              resCtx.ccuKernels.begin(),
                                              resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
        interTempAlgRes.ccuKernels.insert(interTempAlgRes.ccuKernels.end(),
                                              resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
                                              resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);
    } else {
        intraTempAlgRes.channels = intraLinkMap_;
        interTempAlgRes.channels = interLinkMap_;
    }
    
    // 新增：声明尾块变量
    u64 finalDataCountPerLoopAixs0 = dataCountPerLoopAixs0;
    u64 finalDataCountPerLoopAixs1 = dataCountPerLoopAixs1;
    
    // 如果存在尾块，单独计算尾块的对齐
    if (loopTimes > 1) {
        u64 finalCount = dataCount_ - (loopTimes - 1) * maxCountPerLoop;
        finalDataCountPerLoopAixs0 = static_cast<u64>(dataSplitSize[0] * finalCount);
        finalDataCountPerLoopAixs1 = finalCount - finalDataCountPerLoopAixs0;
    } else {
        finalDataCountPerLoopAixs0 = static_cast<u64>(dataSplitSize[0] * dataCount_);
        finalDataCountPerLoopAixs1 = dataCount_ - finalDataCountPerLoopAixs0;
    }

    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        // 使用预计算的对齐后数据量
        u64 currCountPart0 = (loopIndex == loopTimes - 1) ? finalDataCountPerLoopAixs0 : dataCountPerLoopAixs0;
        u64 currCountPart1 = (loopIndex == loopTimes - 1) ? finalDataCountPerLoopAixs1 : dataCountPerLoopAixs1;
        
        // 第一步开始前同步
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
        u64 dataOffset0 = loopIndex * maxCountPerLoop * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + currCountPart0 * dataTypeSize_;
        // 数据0的server内的mesh算法
        GenTemplateAlgParamsIntra0(param, resCtx, dataOffset0, currCountPart0, intraScratchOffset,
                                   tempAlgParamsIntra0);
        // 把每个template需要的queue传进去，比如stars的mesh要传多条queue
        CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra0, intraTempAlgRes));
        // 数据1的server间的nhr算法
        GenTemplateAlgParamsInter1(param, resCtx, dataOffset1, currCountPart1, interScratchOffset,
                                   tempAlgParamsInter1);
        CHK_RET(tempAlgInter.KernelRun(param, tempAlgParamsInter1, interTempAlgRes));
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
        // 数据0的server间的nhr算法
        GenTemplateAlgParamsInter0(param, resCtx, dataOffset0, currCountPart0, interScratchOffset,
                                   tempAlgParamsInter0);
        CHK_RET(tempAlgInter.KernelRun(param, tempAlgParamsInter0, interTempAlgRes));
        // 数据1的server内的mesh算法
        GenTemplateAlgParamsIntra1(param, resCtx, dataOffset1, currCountPart1, intraScratchOffset,
                                   tempAlgParamsIntra1);
        CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra1, intraTempAlgRes));
        // 尾同步
        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
    }
#ifndef AICPU_COMPILE
    if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU && param.opMode != OpMode::OFFLOAD) {
        CHK_RET(FastLaunchSaveCtx(param, intraTempAlgRes, interTempAlgRes, resCtx.notifyNumOnMainThread));
    }
#endif
    HCCL_INFO("[InsV2AllGatherParallelExecutor][OrchestrateLoop] End.");
    return HcclResult::HCCL_SUCCESS;
};

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::FastLaunchSaveCtx(
    const OpParam &param, const TemplateResource &templateAlgResIntra, const TemplateResource &templateAlgResInter, u32 notifyNumOnMainThread)
{
    HCCL_INFO("[InsV2AllGatherParallelExecutor] loopTimes==1, save fast launch ctx.");
    ccuKernelLaunchNumIntra1_ = templateAlgResIntra.submitInfos.size() - ccuKernelLaunchNumIntra0_;
    ccuKernelLaunchNumInter0_ = templateAlgResInter.submitInfos.size() - ccuKernelLaunchNumInter1_;
    u32 threadNum = threads_.size();
    u32 ccuKernelNum = ccuKernelLaunchNumIntra1_ + ccuKernelLaunchNumInter0_ + ccuKernelLaunchNumIntra0_ + ccuKernelLaunchNumInter1_;
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsV2AllGatherParallelExecutor] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[InsV2AllGatherParallelExecutor][HcclEngineCtxCreate] threadNum[%llu], ccuKernelNum[%llu]", threadNum, ccuKernelNum);

    std::vector<u32> ccuKernelNumList = {ccuKernelLaunchNumIntra0_, ccuKernelLaunchNumInter1_, ccuKernelLaunchNumInter0_, ccuKernelLaunchNumIntra1_};
    std::vector<std::vector<CcuKernelSubmitInfo>> submitInfosList = {templateAlgResIntra.submitInfos, templateAlgResInter.submitInfos};
    return FastLaunchSaveCtxTwoTemplate(param, threadNum, ccuKernelNum, threads_, ccuKernelNumList, submitInfosList, notifyNumOnMainThread);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::FastLaunch(
        const OpParam &param, const CcuFastLaunchCtx *ctx)
{
    InsAlgTemplate0 intraTempAlg{};
    InsAlgTemplate1 interTempAlg{};
    
    TemplateFastLaunchCtx tempFastLaunchCtxIntra0, tempFastLaunchCtxInter0;
    TemplateFastLaunchCtx tempFastLaunchCtxInter1, tempFastLaunchCtxIntra1;

    TemplateResource templateAlgResIntra, templateAlgResInter;
    ThreadHandle *threads = ctx->GetThreadHandlePtr();
    threads_.assign(threads, threads + ctx->threadNum);
    PrepareResForTemplate(intraTempAlg, interTempAlg);
    
    CcuKernelSubmitInfo *ccuKernelSubmitInfos = ctx->GetCcuKernelSubmitInfoPtr();
    
    //第一步开始前同步
    HCCL_INFO("[InsV2AllGatherParallelExecutor][FastLaunch] Intra0 ccuKernelNum[%llu]", ctx->ccuKernelNum[0]);
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
    //数据0的server内的mesh算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra0, param.inputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxIntra0.threads = intraThreads_;
    tempFastLaunchCtxIntra0.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[0]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[0];
    //把每个template需要的queue传进去，比如stars的mesh要传多条queue
    if (ctx->ccuKernelNum[0] > 0) {
        CHK_RET(intraTempAlg.FastLaunch(param, tempFastLaunchCtxIntra0));
    }
    //数据1的server间的nhr算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter1, param.inputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxInter1.threads = interThreads_;
    tempFastLaunchCtxInter1.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[1]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[1];
    if (ctx->ccuKernelNum[1] > 0) {
        CHK_RET(interTempAlg.FastLaunch(param, tempFastLaunchCtxInter1));
    }
    
    //第一步做完后回到主流做尾同步
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

    //第二步开始前同步
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
    //数据0的server间的nhr算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter0, param.outputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxInter0.threads = interThreads_;
    tempFastLaunchCtxInter0.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[2]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[2];
    if (ctx->ccuKernelNum[2] > 0) {
        CHK_RET(interTempAlg.FastLaunch(param, tempFastLaunchCtxInter0));
    }
    //数据1的server内的mesh算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra1, param.outputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxIntra1.threads = intraThreads_;
    tempFastLaunchCtxIntra1.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[3]);
    if (ctx->ccuKernelNum[3] > 0) {
        CHK_RET(intraTempAlg.FastLaunch(param, tempFastLaunchCtxIntra1));
    }
    
    //尾同步
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

    HCCL_INFO("[InsV2AllGatherParallelExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif

#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherParallelMesh1DNHR,
                               InsV2AllGatherParallelExecutor, TopoMatchMultilevel, InsTempAllGatherMesh1D,
                               InsTempAllGatherNHR);
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherParallelMesh1DNHRMultiJetty,
                               InsV2AllGatherParallelExecutor, TopoMatchUBX, InsTempAllGatherMesh1D,
                               InsTempAllGatherNHR);
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherParallelMesh1DNHRPcie,
                               InsV2AllGatherParallelExecutor, TopoMatchPcieMix, InsTempAllGatherMesh1D,
                               InsTempAllGatherNHR);
#endif /* !HCCL_CANN_COMPAT_850 */
#ifndef AICPU_COMPILE
#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, CcuAllGatherParallelMesh1DNHR,
    InsV2AllGatherParallelExecutor, TopoMatchMultilevel, CcuTempAllGatherMesh1DMem2Mem, CcuTempAllGatherNHR1DMem2Mem);
#endif /* !HCCL_CANN_COMPAT_850 */

#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, CcuAllGatherParallelMesh1DNHRMemMultiJetty,
    InsV2AllGatherParallelExecutor, TopoMatchUBX, CcuTempAllGatherMesh1DMem2Mem, CcuTempAllGatherNHR1DMultiJettyMem2Mem);
#endif /* !HCCL_CANN_COMPAT_850 */

#endif
}
// 算法注册