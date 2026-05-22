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
#include "reduce_parallel_executor.h"
#include "coll_alg_v2_exec_registry.h"
#include "ins_temp_all_gather_mesh_1D.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_reduce_scatter_mesh_1D.h"
#include "ins_temp_reduce_scatter_nhr.h"
#if !defined(HCCL_CANN_COMPAT_850)
#include "ccu_temp_all_gather_mesh_1D_mem2mem.h"
#include "ccu_temp_all_gather_nhr_1D_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh_1D_mem2mem.h"
#include "ccu_temp_reduce_scatter_nhr_1D_mem2mem.h"
#endif /* !HCCL_CANN_COMPAT_850 */
#include "topo_match_multilevel.h"
#include "topo_match_ubx.h"
#include "topo_match_pcie_mix.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::ReduceParallelExecutor()
{}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult
    ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    CHK_PTR_NULL(topoInfo);
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    CHK_PTR_NULL(topoInfo);
    myRank_ = topoInfo->userRank;
    HCCL_INFO("[ReduceParallelExecutor] CalcRes start, rank[%d]", myRank_);

    // 实例化算法模板类
    // 构建template
    std::vector<std::vector<u32>> temp0HierarchyInfo;
    std::vector<std::vector<u32>> temp1HierarchyInfo;
    if(topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
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

    // reduceScatter intra
    algTemplatePtrArr_.at(0).at(0) =
        std::make_shared<AlgTemplate0>(param, topoInfo->userRank, temp0HierarchyInfo);
    // reduceScatter inter
    algTemplatePtrArr_.at(0).at(1) =
        std::make_shared<AlgTemplate1>(param, topoInfo->userRank, temp1HierarchyInfo);
    // allGather intra
    algTemplatePtrArr_.at(1).at(0) =
        std::make_shared<AlgTemplate2>(param, topoInfo->userRank, temp0HierarchyInfo);
    // allGather inter
    algTemplatePtrArr_.at(1).at(1) =
        std::make_shared<AlgTemplate3>(param, topoInfo->userRank, temp1HierarchyInfo);

    // 计算资源
    AlgResourceRequest reduceScatterIntraTempRequest;
    AlgResourceRequest reduceScatterInterTempRequest;
    AlgResourceRequest allGatherIntraTempRequest;
    AlgResourceRequest allGatherInterTempRequest;
    AlgResourceRequest intraTempRequestFinal;
    AlgResourceRequest interTempRequestFinal;

    CHK_RET(algTemplatePtrArr_.at(0).at(0)->CalcRes(comm, param, topoInfo, reduceScatterIntraTempRequest));
    CHK_RET(algTemplatePtrArr_.at(0).at(1)->CalcRes(comm, param, topoInfo, reduceScatterInterTempRequest));
    CHK_RET(algTemplatePtrArr_.at(1).at(0)->CalcRes(comm, param, topoInfo, allGatherIntraTempRequest));
    CHK_RET(algTemplatePtrArr_.at(1).at(1)->CalcRes(comm, param, topoInfo, allGatherInterTempRequest));

    for (auto &KernelInfo : reduceScatterIntraTempRequest.ccuKernelInfos) {
        KernelInfo.resGroup = 0;
    }
    for (auto &KernelInfo : reduceScatterInterTempRequest.ccuKernelInfos) {
        KernelInfo.resGroup = 0;
    }
    for (auto &KernelInfo : allGatherIntraTempRequest.ccuKernelInfos) {
        KernelInfo.resGroup = 1;
    }
    for (auto &KernelInfo : allGatherInterTempRequest.ccuKernelInfos) {
        KernelInfo.resGroup = 1;
    }

    u32 slaveThreadNumIntraMax = 0;
    if (reduceScatterIntraTempRequest.slaveThreadNum >= allGatherIntraTempRequest.slaveThreadNum) {
        slaveThreadNumIntraMax = reduceScatterIntraTempRequest.slaveThreadNum;
        intraTempRequestFinal.notifyNumPerThread = reduceScatterIntraTempRequest.notifyNumPerThread;
    } else {
        slaveThreadNumIntraMax = allGatherIntraTempRequest.slaveThreadNum;
        intraTempRequestFinal.notifyNumPerThread = allGatherIntraTempRequest.notifyNumPerThread;
    }
    u32 slaveThreadNumInterMax = 0;
    if (reduceScatterInterTempRequest.slaveThreadNum >= allGatherInterTempRequest.slaveThreadNum) {
        slaveThreadNumInterMax = reduceScatterInterTempRequest.slaveThreadNum;
        interTempRequestFinal.notifyNumPerThread = reduceScatterInterTempRequest.notifyNumPerThread;
    } else {
        slaveThreadNumInterMax = allGatherInterTempRequest.slaveThreadNum;
        interTempRequestFinal.notifyNumPerThread = allGatherInterTempRequest.notifyNumPerThread;
    }

    resourceRequest.notifyNumOnMainThread = stageSize_;  // 用于intra和inter两个template间同步
    // intra主流 + intra从流 + inter主流 + inter从流
    resourceRequest.slaveThreadNum = stageSize_ + slaveThreadNumIntraMax + stageSize_ + slaveThreadNumInterMax;
    resourceRequest.notifyNumPerThread.emplace_back(reduceScatterIntraTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.emplace_back(allGatherIntraTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
        intraTempRequestFinal.notifyNumPerThread.begin(),
        intraTempRequestFinal.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(reduceScatterInterTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.emplace_back(allGatherInterTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
        interTempRequestFinal.notifyNumPerThread.begin(),
        interTempRequestFinal.notifyNumPerThread.end());

    if (param.engine != COMM_ENGINE_CCU) {
        resourceRequest.channels.emplace_back(reduceScatterIntraTempRequest.channels.at(0));
        resourceRequest.channels.emplace_back(reduceScatterInterTempRequest.channels.at(0));
    } else {
        HCCL_INFO("[ReduceParallelExecutor][CalcRes] reduceScatterIntraTemp has [%d] kernels.", reduceScatterIntraTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            reduceScatterIntraTempRequest.ccuKernelInfos.begin(),
                                            reduceScatterIntraTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(reduceScatterIntraTempRequest.ccuKernelNum[0]);
        HCCL_INFO("[ReduceParallelExecutor][CalcRes] reduceScatterInterTemp has [%d] kernels.", reduceScatterInterTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            reduceScatterInterTempRequest.ccuKernelInfos.begin(),
                                            reduceScatterInterTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(reduceScatterInterTempRequest.ccuKernelNum[0]);
        HCCL_INFO("[ReduceParallelExecutor][CalcRes] allGatherIntraTemp has [%d] kernels.", allGatherIntraTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            allGatherIntraTempRequest.ccuKernelInfos.begin(),
                                            allGatherIntraTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(allGatherIntraTempRequest.ccuKernelNum[0]);
        HCCL_INFO("[ReduceParallelExecutor][CalcRes] allGatherInterTemp has [%d] kernels.", allGatherInterTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            allGatherInterTempRequest.ccuKernelInfos.begin(),
                                            allGatherInterTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(allGatherInterTempRequest.ccuKernelNum[0]);
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::CalcLocalRoot()
{
    CHK_PRT_RET(root_ >= rankSize_,
        HCCL_ERROR("[ReduceParallelExecutor][CalcLocalRoot] root[%u] is out of rankSize[%u]", root_, rankSize_),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(intraLocalRankSize_ == 0,
        HCCL_ERROR("[ReduceParallelExecutor][CalcLocalRoot] intraLocalRankSize_ is 0"),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(interLocalRankSize_ == 0,
        HCCL_ERROR("[ReduceParallelExecutor][CalcLocalRoot] interLocalRankSize_ is 0"),
        HcclResult::HCCL_E_INTERNAL);
    rankIdxLevel0_ = myRank_ % intraLocalRankSize_;
    rankIdxLevel1_ = myRank_ / intraLocalRankSize_;
    intraLocalRoot_ = root_ / intraLocalRankSize_ * intraLocalRankSize_ + rankIdxLevel0_;
    interLocalRoot_ = root_ % intraLocalRankSize_ + rankIdxLevel1_ * intraLocalRankSize_;
    HCCL_INFO("[ReduceParallelExecutor][CalcLocalRoot] myRank[%d] intraLocalRoot[%u] interLocalRoot[%u]",
        myRank_,
        intraLocalRoot_,
        interLocalRoot_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
uint64_t ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo) const
{
    uint64_t count = 1;
    for (const auto &i : vTopo) {
        count *= i.size();
    }
    return count;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[ReduceParallelExecutor][Orchestrate] Orchestrate Start");

    maxTmpMemSize_ = resCtx.cclMem.size;  // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    myRank_ = resCtx.topoInfo.userRank;
    threads_ = resCtx.threads;
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    param_ = param;
    CHK_PTR_NULL(resCtx.cclMem.addr);
    resCtx_ = resCtx;
    HCCL_INFO("[ReduceParallelExecutor][Orchestrate] threads_ size[%d]", threads_.size());
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo;
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo));
        intraLinks_ = remoteRankToChannelInfo.at(0);
        interLinks_ = remoteRankToChannelInfo.at(1);
    }
    dataCount_ = param_.DataDes.count;
    dataType_ = param_.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param_.DataDes.dataType];

    root_ = param.root;
    vTopo_ = resCtx.algHierarchyInfo.infos;     // 本通信域内的通信平面

    if(resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix) {
        temp0HierarchyInfo_ = {resCtx.algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRankList;
        u32 meshSize = resCtx.algHierarchyInfo.infos[0][0].size();
        for(auto rank : resCtx.algHierarchyInfo.infos[0][1]) {
            if(rank % meshSize == resCtx.topoInfo.userRank % meshSize) {
                closRankList.push_back(rank);
            }
        }
        temp1HierarchyInfo_ = {closRankList};
    } else {
        temp0HierarchyInfo_ = resCtx.algHierarchyInfo.infos[0];
        temp1HierarchyInfo_ = resCtx.algHierarchyInfo.infos[1];
    }
    vTopo_ = {temp0HierarchyInfo_, temp1HierarchyInfo_};
    intraLocalRankSize_ = GetRankSize(temp0HierarchyInfo_);
    interLocalRankSize_ = GetRankSize(temp1HierarchyInfo_);
    rankSize_ = intraLocalRankSize_ * interLocalRankSize_;
    HCCL_DEBUG("[ReduceParallelExecutor][Orchestrate] myRank[%u], intraLocalRankSize_[%u], interLocalRankSize_[%u]",
        myRank_,
        intraLocalRankSize_,
        interLocalRankSize_);

    CHK_RET(CalcLocalRoot());

    // 实例化算法模板类
    algTemplatePtrArr_.at(0).at(0) = std::make_shared<AlgTemplate0>(param, myRank_, temp0HierarchyInfo_);
    algTemplatePtrArr_.at(0).at(1) = std::make_shared<AlgTemplate1>(param, myRank_, temp1HierarchyInfo_);
    algTemplatePtrArr_.at(1).at(0) = std::make_shared<AlgTemplate2>(param, myRank_, temp0HierarchyInfo_);
    algTemplatePtrArr_.at(1).at(1) = std::make_shared<AlgTemplate3>(param, myRank_, temp1HierarchyInfo_);

    // 算法展开
    CHK_RET(OrchestrateImpl());

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2,
    AlgTemplate3>::PrepareResForStage(u32 stage)
{
    std::array<std::array<AlgResourceRequest, stepSize_>, stageSize_> tempRequestArr;
    std::array<u32, stageSize_> intraThreadsNum;
    for (u32 stageIdx = 0; stageIdx < stageSize_; stageIdx++) {
        for (u32 stepIdx = 0; stepIdx < stepSize_; stepIdx++) {
            algTemplatePtrArr_.at(stageIdx).at(stepIdx)->GetRes(tempRequestArr.at(stageIdx).at(stepIdx));
        }
        intraThreadsNum.at(stageIdx) = tempRequestArr.at(stageIdx).at(0).slaveThreadNum + 1;
    }

    u32 intraThreadsNumMax = std::max(intraThreadsNum.at(0), intraThreadsNum.at(1));
    u32 interThreadsNumMax = std::max(tempRequestArr.at(0).at(1).slaveThreadNum + 1,
                                      tempRequestArr.at(1).at(1).slaveThreadNum + 1);
    // 预期threads数量= 全局主流 + intra主流  +        intra从流         + inter主流   +          inter从流 
    u32 expectedThreadsNum = 1 + stageSize_ + (intraThreadsNumMax - 1) + stageSize_ + (interThreadsNumMax - 1);
    CHK_PRT_RET(threads_.size() < expectedThreadsNum,
        HCCL_ERROR("[ReduceParallelExecutor][PrepareRes] act:[%u] exp:[%u]", threads_.size(), expectedThreadsNum),
        HcclResult::HCCL_E_INTERNAL);

    // 第0条流是全局主流
    intraThreads_ = {threads_.at(1 + stage)};
    intraThreads_.insert(intraThreads_.end(), threads_.begin() + stageSize_ + 1,
        threads_.begin() + stageSize_ + intraThreadsNum.at(stage));
    interThreads_ = {threads_.at(intraThreadsNumMax + stageSize_ + stage)};
    interThreads_.insert(interThreads_.end(), threads_.begin() + intraThreadsNumMax + stageSize_ + stageSize_,
        threads_.end());

    mainThread_ = threads_.at(0);
    templateMainThreads_ = {intraThreads_.at(0), interThreads_.at(0)};

    syncNotifyOnTemplates_ = {tempRequestArr.at(stage).at(0).notifyNumOnMainThread,
                              tempRequestArr.at(stage).at(1).notifyNumOnMainThread};
    syncNotifyOnMain_ = {0, 1};

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2,
    AlgTemplate3>::PrepareResForStage2(u32 stage)
{
    if (param_.engine == COMM_ENGINE_CCU) {
        tempAlgResArr_.at(stage * 2).ccuKernels.clear();
        tempAlgResArr_.at(stage * 2 + 1).ccuKernels.clear();
        if (stage == 0) {
            tempAlgResArr_.at(stage * 2).ccuKernels.insert(tempAlgResArr_.at(stage * 2).ccuKernels.end(),
                                               resCtx_.ccuKernels.begin(),
                                               resCtx_.ccuKernels.begin() + resCtx_.ccuKernelNum[0]);
            tempAlgResArr_.at(stage * 2 + 1).ccuKernels.insert(tempAlgResArr_.at(stage * 2 + 1).ccuKernels.end(),
                                               resCtx_.ccuKernels.begin() + resCtx_.ccuKernelNum[0],
                                               resCtx_.ccuKernels.begin() + resCtx_.ccuKernelNum[0] + resCtx_.ccuKernelNum[1]);
        } else {
            tempAlgResArr_.at(stage * 2).ccuKernels.insert(tempAlgResArr_.at(stage * 2).ccuKernels.end(),
                                               resCtx_.ccuKernels.begin() + resCtx_.ccuKernelNum[0] + resCtx_.ccuKernelNum[1],
                                               resCtx_.ccuKernels.begin() + resCtx_.ccuKernelNum[0] + resCtx_.ccuKernelNum[1] + resCtx_.ccuKernelNum[2]);
            tempAlgResArr_.at(stage * 2 + 1).ccuKernels.insert(tempAlgResArr_.at(stage * 2 + 1).ccuKernels.end(),
                                               resCtx_.ccuKernels.begin() + resCtx_.ccuKernelNum[0] + resCtx_.ccuKernelNum[1] + resCtx_.ccuKernelNum[2],
                                               resCtx_.ccuKernels.begin() + resCtx_.ccuKernelNum[0] + resCtx_.ccuKernelNum[1] + resCtx_.ccuKernelNum[2] + resCtx_.ccuKernelNum[3]);
        }
    } else {
        tempAlgResArr_.at(stage * 2).channels = intraLinks_;
        tempAlgResArr_.at(stage * 2 + 1).channels = interLinks_;
    }

    tempAlgResArr_.at(stage * 2).threads = intraThreads_;
    tempAlgResArr_.at(stage * 2).aivCommInfoPtr = resCtx_.aivCommInfoPtr;

    tempAlgResArr_.at(stage * 2 + 1).threads = interThreads_;
    tempAlgResArr_.at(stage * 2 + 1).aivCommInfoPtr = resCtx_.aivCommInfoPtr;

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
TemplateDataParams ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2,
    AlgTemplate3>:: GenDataParamsTempAlg(u32 dataSliceIdx, u32 stageIdx, u32 stepIdx, bool isInter)
{
    TemplateDataParams dataParams;
    bool isFirstStep = (stageIdx == 0 && stepIdx == 0);

    dataParams.buffInfo.inBuffType = isFirstStep ? BufferType::INPUT : BufferType::HCCL_BUFFER;
    dataParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    dataParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 数据在inputBuffer上的起始偏移量，仅用于本次loop第一步firstStep时，要从inputBuffer上取数据
    const u64 inputBufferOffset = dataOffsetPerLoop_.at(dataSliceIdx);
    // totalCount是本次template处理，所有卡需要处理的总数据个数
    u64 totalCount = dataCountPerLoop_.at(dataSliceIdx);
    // dataSliceBaseOffset根据当前要处理的是哪片数据（dataSliceIdx），决定要输出到的ccl缓存起始位置
    const u64 dataSliceBaseOffset = dataSliceIdx == 0 ? 0 : dataCountPerLoop_.at(0) * dataTypeSize_;
    // dataOffset是在非firstStep时，数据输入和输出的起始地址
    u64 dataOffset = dataSliceBaseOffset;

    // 中间step需要额外处理，即stage0-step1（第二次reduceScatter）和stage1-step0（第一次allGather）需要
    // 重新计算dataOffset和totalCount
    if ((stageIdx ^ stepIdx) == 1) {
        const u32 othLocalRankSize = isInter ? intraLocalRankSize_ : interLocalRankSize_;
        const std::map<u32, u32> &othTempVirtRankMap = virtRankMap_.at(!isInter);
        const u32 othLocalRankIdx = othTempVirtRankMap.at(myRank_);
        const u64 trivialSize = dataCountPerLoop_.at(dataSliceIdx) / othLocalRankSize * dataTypeSize_;
        const u64 tailSize = dataCountPerLoop_.at(dataSliceIdx) * dataTypeSize_ - (othLocalRankSize - 1) * trivialSize;
        dataOffset = dataSliceBaseOffset + othLocalRankIdx * trivialSize;
        totalCount = ((othLocalRankIdx + 1 == othLocalRankSize) ? tailSize : trivialSize) / dataTypeSize_;
    }
    const u64 totalSize = totalCount * dataTypeSize_;   // totalSize是本次处理，所有卡需要处理的总数据量

    dataParams.buffInfo.inputPtr = isFirstStep ? param_.inputPtr : resCtx_.cclMem.addr;
    dataParams.buffInfo.inputSize = isFirstStep ? param_.inputSize : resCtx_.cclMem.size;
    dataParams.buffInfo.outputPtr = resCtx_.cclMem.addr;
    dataParams.buffInfo.outputSize = resCtx_.cclMem.size;
    dataParams.buffInfo.hcclBuff = resCtx_.cclMem;
    dataParams.buffInfo.hcclBuffSize = resCtx_.cclMem.size;

    // 前localRankSize - 1个rank的数据片为trivialSize，最后一个rank的数据片大小为tailSize
    const u32 localRankSize = isInter ? interLocalRankSize_ : intraLocalRankSize_;
    u64 trivialSize = totalCount / localRankSize * dataTypeSize_;
    dataParams.tailSize = totalSize - (localRankSize - 1) * trivialSize;
    dataParams.sliceSize = trivialSize;
    dataParams.count = dataParams.sliceSize / dataTypeSize_;

    if (stageIdx == 0 && !isInter) {
        // 框内reduceScatter，需要使用额外的scratchBuffer（不与输入输出相同位置）
        dataParams.buffInfo.hcclBuffBaseOff = (dataCountPerLoop_.at(0) + dataCountPerLoop_.at(1)) * dataTypeSize_;
    } else {
        // AllGather阶段，使用与输入输出相同位置的scratchBuffer，避免localCopy开销
        dataParams.buffInfo.hcclBuffBaseOff = dataOffset;
    }

    if (isFirstStep) {
        dataParams.buffInfo.inBuffBaseOff = inputBufferOffset;
        dataParams.buffInfo.outBuffBaseOff = dataSliceBaseOffset;
    } else {
        dataParams.buffInfo.inBuffBaseOff = dataOffset;
        dataParams.buffInfo.outBuffBaseOff = dataOffset;
    }
    dataParams.inputSliceStride = dataParams.sliceSize;
    dataParams.outputSliceStride = dataParams.sliceSize;

    dataParams.repeatNum = 1;
    dataParams.inputRepeatStride = 0;
    dataParams.outputRepeatStride = 0;

    return dataParams;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult
    ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::OrchestrateImpl()
{
    for (u32 stage = 0; stage < stageSize_; stage++) {
        for (u32 isInter = 0; isInter < dataSplitPart_; isInter++) {
            HCCL_INFO("[ReduceParallelExecutor][OrchestrateImpl] stage[%u] isInter[%u] [%s]",
                stage,
                isInter,
                algTemplatePtrArr_.at(stage).at(isInter)->Describe().c_str());
        }
    }

    std::array<long double, dataSplitPart_> dataSplitSize{dataSplitSize0_, 1.0 - dataSplitSize0_};

    // inter模板不再需要额外的scratch，因为当input/output都在CCL BUFFER上是，NHR算法可以直接在原地进行
    const long double scratchMultipleIntra = std::max(dataSplitSize.at(0), dataSplitSize.at(1) / interLocalRankSize_);
    // + 1.0是因为要留一份scratch来临时存储中间数据
    const long double totalScratchMultiple = scratchMultipleIntra + 1.0;

    const u64 scratchMemBlockSize = maxTmpMemSize_ / totalScratchMultiple;
    CHK_PRT_RET(dataTypeSize_ == 0, "[ReduceParallelExecutor][OrchestrateImpl] dataTypeSize_ is 0", HCCL_E_INTERNAL);
    const u64 maxCountPerLoop = std::min<u64>(scratchMemBlockSize, UB_MAX_DATA_SIZE) / dataTypeSize_;
    CHK_PRT_RET(maxCountPerLoop == 0, "[ReduceParallelExecutor][OrchestrateImpl] maxCountPerLoop is 0", HCCL_E_INTERNAL);
    const u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    for (u32 isInter = 0; isInter < dataSplitPart_; isInter++) {
        for (u32 localRank = 0; localRank < vTopo_.at(isInter).at(0).size(); localRank++) {
            const u32 globalRank = vTopo_.at(isInter).at(0).at(localRank);
            virtRankMap_.at(isInter)[globalRank] = localRank;
        }
    }

    CHK_RET(OrchestrateLoop(loopTimes, maxCountPerLoop));

    HCCL_INFO("[ReduceParallelExecutor][OrchestrateImpl] myRank[%d] End.", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult
    ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::OrchestrateLoop(
        u32 loopTimes, u64 maxCountPerLoop)
{
    u64 processedCount = 0;
    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCount = (loopIndex + 1 == loopTimes) ? (dataCount_ - loopIndex * maxCountPerLoop) : maxCountPerLoop;
        dataCountPerLoop_.at(0) = static_cast<u64>(currCount * dataSplitSize0_);
        dataCountPerLoop_.at(1) = currCount - dataCountPerLoop_.at(0);
        dataOffsetPerLoop_.at(0) = loopIndex * maxCountPerLoop * dataTypeSize_;
        dataOffsetPerLoop_.at(1) = dataOffsetPerLoop_.at(0) + dataCountPerLoop_.at(0) * dataTypeSize_;

        for (u32 stageIdx = 0; stageIdx < 2; stageIdx++) {
            // 计算算法模板所需资源
            CHK_RET(PrepareResForStage(stageIdx));
            CHK_RET(PrepareResForStage2(stageIdx));
            // 每个阶段分2步执行任务编排
            for (u32 stepIdx = 0; stepIdx < 2; stepIdx++) {
                CHK_RET(OrchestrateStep(stageIdx, stepIdx));
#ifndef AICPU_COMPILE
                if (loopTimes == 1 && param_.engine == CommEngine::COMM_ENGINE_CCU) {
                    if (stageIdx == 0 && stepIdx == 0) {
                        ccuKernelLaunchNumRSIntra0_ = tempAlgResArr_.at(0).submitInfos.size();
                        ccuKernelLaunchNumRSInter1_ = tempAlgResArr_.at(1).submitInfos.size();
                    } else if (stageIdx == 0 && stepIdx == 1) {
						ccuKernelLaunchNumRSIntra1_ = tempAlgResArr_.at(0).submitInfos.size() - ccuKernelLaunchNumRSIntra0_;
                        ccuKernelLaunchNumRSInter0_ = tempAlgResArr_.at(1).submitInfos.size() - ccuKernelLaunchNumRSInter1_;
                    } else if (stageIdx == 1 && stepIdx == 0) {
						ccuKernelLaunchNumAGIntra1_ = tempAlgResArr_.at(2).submitInfos.size();
                        ccuKernelLaunchNumAGInter0_ = tempAlgResArr_.at(3).submitInfos.size();
                    } else if (stageIdx == 1 && stepIdx == 1 && param_.opMode != OpMode::OFFLOAD) {
						ccuKernelLaunchNumAGIntra0_ = tempAlgResArr_.at(2).submitInfos.size() - ccuKernelLaunchNumAGIntra1_;
    					ccuKernelLaunchNumAGInter1_ = tempAlgResArr_.at(3).submitInfos.size() - ccuKernelLaunchNumAGInter0_;
                        CHK_RET(FastLaunchSaveCtx());
                    }
                }
#endif
            }
        }
        if (myRank_ != root_) {
            continue;
        }
        const DataSlice srcSlice(resCtx_.cclMem.addr, 0, currCount * dataTypeSize_);
        const DataSlice dstSlice(param_.outputPtr, processedCount * dataTypeSize_, currCount * dataTypeSize_);
        CHK_RET(LocalCopy(threads_.at(0), srcSlice, dstSlice));

        processedCount += currCount;
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult
    ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::OrchestrateStep(
        u32 stageIdx, u32 stepIdx)
{
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
    for (u32 dataSliceIdx = 0; dataSliceIdx < dataSplitPart_; dataSliceIdx++) {
        // 第一个stage第一个step时，第一片数据跑intra，第二片数据跑inter；
        // 第一个stage第二个step时，第一片数据跑inter，第二片数据跑intra；
        // 第二个stage第一个step时，第一片数据跑inter，第二片数据跑intra；
        // 第二个stage第二个step时，第一片数据跑intra，第二片数据跑inter；
        bool isInter = (stageIdx == 1) ^ (stepIdx == 1) ^ (dataSliceIdx == 1);
        CHK_RET(RunTemplate(dataSliceIdx, stageIdx, stepIdx, isInter));
    }
    // 尾同步
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::RunTemplate(
    u32 dataSliceIdx, u32 stageIdx, u32 stepIdx, bool isInter)
{
    if (dataCountPerLoop_.at(dataSliceIdx) == 0) {
        return HCCL_SUCCESS;
    }
    const TemplateDataParams dataParams = GenDataParamsTempAlg(dataSliceIdx, stageIdx, stepIdx, isInter);
    CHK_RET(algTemplatePtrArr_.at(stageIdx).at(isInter)->KernelRun(param_, dataParams, tempAlgResArr_.at(stageIdx == 0 ? (stepIdx == dataSliceIdx ? 0 : 1) : (stepIdx == dataSliceIdx ? 3 : 2))));
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,typename AlgTemplate3>
HcclResult ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::FastLaunchSaveCtx()
{
    HCCL_INFO("[ReduceParallelExecutor][FastLaunchSaveCtx] loopTimes==1, save fast launch ctx.");
    u32 threadNum = threads_.size();
    u32 ccuKernelNum = ccuKernelLaunchNumRSIntra1_ + ccuKernelLaunchNumRSInter0_ + ccuKernelLaunchNumRSIntra0_ + ccuKernelLaunchNumRSInter1_ +
                        ccuKernelLaunchNumAGIntra1_ + ccuKernelLaunchNumAGInter0_ + ccuKernelLaunchNumAGIntra0_ + ccuKernelLaunchNumAGInter1_;
    if (ccuKernelNum < 1) {
        HCCL_INFO("[ReduceParallelExecutor][FastLaunchSaveCtx] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[ReduceParallelExecutor][FastLaunchSaveCtx] threadNum[%llu], ccuKernelNum[%llu]", threadNum, ccuKernelNum);

    u64 size = CcuFastLaunchCtx::GetCtxSize(threadNum, ccuKernelNum);
    // 申请ctx
    void *ctxPtr = nullptr;
    HCCL_INFO("[ReduceParallelExecutor][FastLaunchSaveCtx] Tag[%s], size[%llu]", param_.fastLaunchTag, size);
    CHK_RET(HcclEngineCtxCreate(param_.hcclComm, param_.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, size, &ctxPtr));

    CcuFastLaunchCtx *ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(ctxPtr);
    // 1 算法名
    CHK_SAFETY_FUNC_RET(strcpy_s(ccuFastLaunchCtx->algName, sizeof(ccuFastLaunchCtx->algName), param_.algName));
    HCCL_INFO("[ReduceParallelExecutor][FastLaunchSaveCtx] algName[%s]", ccuFastLaunchCtx->algName);

    // 2 thread
    ccuFastLaunchCtx->threadNum = threadNum;
    ccuFastLaunchCtx->notifyNumOnMainThread = resCtx_.notifyNumOnMainThread;
    ThreadHandle *threads = ccuFastLaunchCtx->GetThreadHandlePtr();
    for (u32 i = 0; i < threadNum; i++) {
        threads[i] = threads_[i];
    }

    // 3 ccu kernel handle, taskArg入参
    u32 templateIdx = 0;
    // reduce_scatter
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumRSIntra0_;
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumRSInter1_;
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumRSInter0_;
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumRSIntra1_;
    // allgather
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumAGInter0_;
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumAGIntra1_;
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumAGIntra0_;
    ccuFastLaunchCtx->ccuKernelNum[templateIdx++] = ccuKernelLaunchNumAGInter1_;
    CcuKernelSubmitInfo *kernelSubmitInfos = ccuFastLaunchCtx->GetCcuKernelSubmitInfoPtr();

    u32 kernelIdx = 0;
    // reduce_scatter
    for (u32 i = 0; i < ccuKernelLaunchNumRSIntra0_; i++) {
        kernelSubmitInfos[kernelIdx++] = tempAlgResArr_.at(0).submitInfos[i];
    }
    for (u32 i = 0; i < ccuKernelLaunchNumRSInter1_; i++) {
        kernelSubmitInfos[kernelIdx++] = tempAlgResArr_.at(1).submitInfos[i];
    }
    for (u32 i = ccuKernelLaunchNumRSInter1_; i < ccuKernelLaunchNumRSInter0_ + ccuKernelLaunchNumRSInter1_; i++) {
        kernelSubmitInfos[kernelIdx++] = tempAlgResArr_.at(1).submitInfos[i];
    }
    for (u32 i = ccuKernelLaunchNumRSIntra0_; i < ccuKernelLaunchNumRSIntra1_ + ccuKernelLaunchNumRSIntra0_; i++) {
        kernelSubmitInfos[kernelIdx++] = tempAlgResArr_.at(0).submitInfos[i];
    }
    // allgather
    for (u32 i = 0; i < ccuKernelLaunchNumAGInter0_; i++) {
        kernelSubmitInfos[kernelIdx++] = tempAlgResArr_.at(3).submitInfos[i];
    }
    for (u32 i = 0; i < ccuKernelLaunchNumAGIntra1_; i++) {
        kernelSubmitInfos[kernelIdx++] = tempAlgResArr_.at(2).submitInfos[i];
    }
    for (u32 i = ccuKernelLaunchNumAGIntra1_; i < ccuKernelLaunchNumAGIntra0_ + ccuKernelLaunchNumAGIntra1_; i++) {
        kernelSubmitInfos[kernelIdx++] = tempAlgResArr_.at(2).submitInfos[i];
    }
    for (u32 i = ccuKernelLaunchNumAGInter0_; i < ccuKernelLaunchNumAGInter0_ + ccuKernelLaunchNumAGInter1_; i++) {
        kernelSubmitInfos[kernelIdx++] = tempAlgResArr_.at(3).submitInfos[i];
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,typename AlgTemplate3>
HcclResult ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::FastLaunch(
        const OpParam &param, const CcuFastLaunchCtx *ctx)
{
    algTemplatePtrArr_.at(0).at(0) = std::make_shared<AlgTemplate0>();
    algTemplatePtrArr_.at(0).at(1) = std::make_shared<AlgTemplate1>();
    algTemplatePtrArr_.at(1).at(0) = std::make_shared<AlgTemplate2>();
    algTemplatePtrArr_.at(1).at(1) = std::make_shared<AlgTemplate3>();

    // 保存reducescatter信息
    TemplateFastLaunchCtx tempFastLaunchCtxIntra0, tempFastLaunchCtxInter0;
    TemplateFastLaunchCtx tempFastLaunchCtxInter1, tempFastLaunchCtxIntra1;
    //保存allgather信息
    TemplateFastLaunchCtx tempFastLaunchCtxIntra01, tempFastLaunchCtxInter01;
    TemplateFastLaunchCtx tempFastLaunchCtxInter11, tempFastLaunchCtxIntra11;

    TemplateResource templateAlgResIntra, templateAlgResInter;
    ThreadHandle *threads = ctx->GetThreadHandlePtr();
    threads_.assign(threads, threads + ctx->threadNum);
    PrepareResForStage(0);

    CcuKernelSubmitInfo *ccuKernelSubmitInfos = ctx->GetCcuKernelSubmitInfoPtr();

    //第一步开始前同步
    HCCL_INFO("[ReduceParallelExecutor][FastLaunch] Intra0 ccuKernelNum[%llu]", ctx->ccuKernelNum[0]);
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
    //数据0的server内的mesh算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra0, param.inputPtr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxIntra0.threads = intraThreads_;
    tempFastLaunchCtxIntra0.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[0]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[0];
	//数据0的server内的mesh算法
    CHK_RET(algTemplatePtrArr_.at(0).at(0)->FastLaunch(param, tempFastLaunchCtxIntra0));
    //数据1的server间的nhr算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter1, param.inputPtr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxInter1.threads = interThreads_;
    tempFastLaunchCtxInter1.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[1]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[1];
    //数据1的server间的nhr算法
    CHK_RET(algTemplatePtrArr_.at(0).at(1)->FastLaunch(param, tempFastLaunchCtxInter1));
    //第一步做完后回到主流做尾同步
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

    //第二步开始前同步
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
    //数据0的server间的nhr算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter0, param.hcclBuff.addr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxInter0.threads = interThreads_;
    tempFastLaunchCtxInter0.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[2]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[2];
	CHK_RET(algTemplatePtrArr_.at(0).at(1)->FastLaunch(param, tempFastLaunchCtxInter0));
    //数据1的server内的mesh算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra1, param.hcclBuff.addr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxIntra1.threads = intraThreads_;
    tempFastLaunchCtxIntra1.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[3]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[3];
    // step2 scatter 数据0的server间的nhr算法
    CHK_RET(algTemplatePtrArr_.at(0).at(0)->FastLaunch(param, tempFastLaunchCtxIntra1));
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

    // allgather
    PrepareResForStage(1);
    // step 3
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
    // 数据0 allgather nhr
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter01, param.hcclBuff.addr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxInter01.threads = interThreads_;
    tempFastLaunchCtxInter01.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[4]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[4];
	CHK_RET(algTemplatePtrArr_.at(1).at(1)->FastLaunch(param, tempFastLaunchCtxInter01));
    // 数据1 allgather mesh
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra11, param.hcclBuff.addr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxIntra11.threads = intraThreads_;
    tempFastLaunchCtxIntra11.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[5]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[5];
	CHK_RET(algTemplatePtrArr_.at(1).at(0)->FastLaunch(param, tempFastLaunchCtxIntra11));
    //尾同步
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
    //step 4
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
    //数据0 allgather mesh
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra01, param.hcclBuff.addr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxIntra01.threads = intraThreads_;
    tempFastLaunchCtxIntra01.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[6]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[6];
	CHK_RET(algTemplatePtrArr_.at(1).at(0)->FastLaunch(param, tempFastLaunchCtxIntra01));
    //数据1的 allgather nhr
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter11, param.hcclBuff.addr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxInter11.threads = interThreads_;
    tempFastLaunchCtxInter11.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[7]);
    CHK_RET(algTemplatePtrArr_.at(1).at(1)->FastLaunch(param, tempFastLaunchCtxInter11));
    //尾同步
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
    HCCL_INFO("[ReduceParallelExecutor][FastLaunch] param.userRank[%u]", param.userRank);
    HCCL_INFO("[ReduceParallelExecutor][FastLaunch] param.root[%u]", param.root);
	if (param.userRank == param.root) {
        HCCL_INFO("[ReduceParallelExecutor][FastLaunch] LocalCopy");
        const DataSlice srcSlice(param.hcclBuff.addr, 0, param.DataDes.count * DATATYPE_SIZE_TABLE[param.DataDes.dataType]);
        const DataSlice dstSlice(param.outputPtr, 0, param.DataDes.count * DATATYPE_SIZE_TABLE[param.DataDes.dataType]);
        CHK_RET(LocalCopy(threads_.at(0), srcSlice, dstSlice));
    }
    HCCL_INFO("[ReduceParallelExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif

// 算法注册
#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_REDUCE, ReduceParallelMesh1DNHR, ReduceParallelExecutor,
    TopoMatchMultilevel, InsTempReduceScatterMesh1D, InsTempReduceScatterNHR, InsTempAllGatherMesh1D,
    InsTempAllGatherNHR);
REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_REDUCE, ReduceParallelMesh1DNHRUBX, ReduceParallelExecutor,
    TopoMatchUBX, InsTempReduceScatterMesh1D, InsTempReduceScatterNHR, InsTempAllGatherMesh1D,
    InsTempAllGatherNHR);
REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_REDUCE, ReduceParallelMesh1DNHRPcie, ReduceParallelExecutor,
    TopoMatchPcieMix, InsTempReduceScatterMesh1D, InsTempReduceScatterNHR, InsTempAllGatherMesh1D, InsTempAllGatherNHR);
#endif /* !HCCL_CANN_COMPAT_850 */

#ifndef AICPU_COMPILE
#if !defined(HCCL_CANN_COMPAT_850)
    REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_REDUCE, CcuReduceParallelMesh1DNHR, ReduceParallelExecutor,
        TopoMatchMultilevel, CcuTempReduceScatterMesh1DMem2Mem, CcuTempReduceScatterNHR1DMem2Mem, CcuTempAllGatherMesh1DMem2Mem, CcuTempAllGatherNHR1DMem2Mem);
    REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_REDUCE, CcuReduceParallelMesh1DNHRUBX, ReduceParallelExecutor,
        TopoMatchUBX, CcuTempReduceScatterMesh1DMem2Mem, CcuTempReduceScatterNHR1DMem2Mem, CcuTempAllGatherMesh1DMem2Mem, CcuTempAllGatherNHR1DMem2Mem);
#endif /* !HCCL_CANN_COMPAT_850 */
#endif
}  // namespace ops_hccl