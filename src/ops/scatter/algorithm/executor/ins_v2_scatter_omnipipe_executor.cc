/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_scatter_omnipipe_executor.h"
#include "dtype_common.h"
#include "topo_match_3_level.h"
#include "ins_temp_scatter_omnipipe_mesh1d.h"
#include "ins_temp_scatter_omnipipe_nhr_dpu.h"
#include "ins_temp_scatter_omnipipe_nhr.h"
#include "topo_match_pcie_mix.h"
#include "omnipipe_template_utils.h"
namespace ops_hccl {
constexpr uint32_t HIERARCHY_SIZE_3 = 3;
constexpr uint64_t RANK_SIZE_LEVEL_2 = 2;
constexpr uint64_t RANK_SIZE_LEVEL_4 = 4;
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2ScatterOmniPipeExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InsV2ScatterOmniPipeExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InitCommInfo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO(
        "[%s]myRank[%u] userRankSize[%u] devType[%u] redOp[%u] dataType[%u] dataTypeSize[%u]", __func__, myRank_,
        rankSize_, devType_, reduceOp_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::BuildSubCommAndTempMap(
    const OpParam& param, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    subCommRanks0_.clear();
    subCommRanks1_.clear();
    subCommRanks2_.clear();
    tempLevel0_.reset();
    tempLevel1_.reset();
    tempLevel2_.reset();

    HCCL_INFO("[BuildSubCommAndTempMap]infos,%s", ThreeDVecToStrOmni(algHierarchyInfo_.infos).c_str());
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        std::vector<u32> closRanks;
        if (!algHierarchyInfo_.infos[0].empty() && !algHierarchyInfo_.infos[0][0].empty()) {
            subCommRanks0_.push_back(algHierarchyInfo_.infos[0][0]);
            u32 meshSize = algHierarchyInfo_.infos[0][0].size();
            if (!algHierarchyInfo_.infos[0][1].empty()) {
                for (auto rank : algHierarchyInfo_.infos[0][1]) {
                    if (rank % meshSize == topoInfo->userRank % meshSize) {
                        closRanks.push_back(rank);
                    }
                }
            }
        }
        subCommRanks1_.push_back(closRanks);
        omniNeedSetStepNum_ = (subCommRanks1_[0].size() == RANK_SIZE_LEVEL_4) ? OmniNeedSetStepNum::OMNIPIPE_UBX_16P :
                                                                                OmniNeedSetStepNum::OMNIPIPE_DEFAULT;
        if (!algHierarchyInfo_.infos[1].empty()) {
            subCommRanks2_ = algHierarchyInfo_.infos[1];
            omniNeedSetStepNum_
                = (subCommRanks2_[0].size() > 1) ? OmniNeedSetStepNum::OMNIPIPE_UBX_32P : omniNeedSetStepNum_;
        } else {
            subCommRanks2_.emplace_back(std::vector<u32>{myRank_});
        }
    } else if (topoType_ == TopoType::THREE_LEVEL) {
        if (!algHierarchyInfo.infos[0].empty() && !algHierarchyInfo.infos[0][0].empty()) {
            subCommRanks0_.push_back(algHierarchyInfo.infos[0][0]);
        } else {
            subCommRanks0_.emplace_back(std::vector<u32>{myRank_});
        }
        if (!algHierarchyInfo.infos[1].empty() && !algHierarchyInfo.infos[1][0].empty()) {
            subCommRanks1_.push_back(algHierarchyInfo.infos[1][0]);
        } else {
            subCommRanks1_.emplace_back(std::vector<u32>{myRank_});
        }
        if (!algHierarchyInfo.infos[2].empty() && !algHierarchyInfo.infos[2][0].empty()) {
            subCommRanks2_.push_back(algHierarchyInfo.infos[2][0]);
        } else {
            subCommRanks2_.emplace_back(std::vector<u32>{myRank_});
        }
    } else {
        if (!algHierarchyInfo_.infos[0].empty()) {
            subCommRanks0_ = algHierarchyInfo_.infos[0];
        }
        if (!algHierarchyInfo_.infos[1].empty()) {
            subCommRanks1_ = algHierarchyInfo_.infos[1];
        }
        subCommRanks2_.emplace_back(std::vector<u32>{myRank_});
    }

    // 打印子通信组信息
    for (size_t i = 0; i < subCommRanks0_.size(); ++i) {
        std::stringstream ss;
        for (size_t j = 0; j < subCommRanks0_[i].size(); ++j) {
            ss << subCommRanks0_[i][j] << " ";
        }
        HCCL_DEBUG("[%s] subCommRanks0_[%zu] content: %s", __func__, i, ss.str().c_str());
    }

    for (size_t i = 0; i < subCommRanks1_.size(); ++i) {
        std::stringstream ss;
        for (size_t j = 0; j < subCommRanks1_[i].size(); ++j) {
            ss << subCommRanks1_[i][j] << " ";
        }
        HCCL_DEBUG("[%s] subCommRanks1_[%zu] content: %s", __func__, i, ss.str().c_str());
    }

    for (size_t i = 0; i < subCommRanks2_.size(); ++i) {
        std::stringstream ss;
        for (size_t j = 0; j < subCommRanks2_[i].size(); ++j) {
            ss << subCommRanks2_[i][j] << " ";
        }
        HCCL_DEBUG("[%s] subCommRanks2_[%zu] content: %s", __func__, i, ss.str().c_str());
    }
    // 打印子通信组信息

    rankSizeLevel0_ = subCommRanks0_[0].size();
    rankSizeLevel1_ = subCommRanks1_[0].size();
    rankSizeLevel2_ = subCommRanks2_[0].size();

    // 当前rank的三轴坐标
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ % (rankSizeLevel0_ * rankSizeLevel1_) / rankSizeLevel0_;
    rankIdxLevel2_ = myRank_ / (rankSizeLevel0_ * rankSizeLevel1_);

    // root rank的三轴坐标
    u64 rootx = param.root % rankSizeLevel0_;
    u64 rooty = param.root % (rankSizeLevel0_ * rankSizeLevel1_) / rankSizeLevel0_;
    u64 rootz = param.root / (rankSizeLevel0_ * rankSizeLevel1_);

    bool isRoot = (myRank_ == param.root);
    // 表示和root同机的和root同横轴的非root rank
    isSameXAxisAsRoot = (rankIdxLevel1_ == rooty && rankIdxLevel2_ == rootz) && !isRoot;
    // 表示和root同机的和root同纵轴的非root rank
    isSameYAxisAsRoot = (rankIdxLevel0_ == rootx && rankIdxLevel2_ == rootz) && !isRoot;
    // 表示和root同Z轴的非root rank
    isSameZAxisAsRoot = false;
    if (rankSizeLevel2_ > 1) {
        isSameZAxisAsRoot = (rankIdxLevel1_ == rooty && rankIdxLevel0_ == rootx && rankIdxLevel2_ != rootz) && !isRoot;
    }
    // 表示和root同机的非root rank
    isSameSerAsRoot = (rankIdxLevel2_ == rootz) && !isRoot;

    if (rankSizeLevel0_ > 1) {
        tempLevel0_ = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0_);
        if (rankIdxLevel2_ != rootz) {
            tempLevel0_->SetRoot(
                rankIdxLevel2_ * (rankSizeLevel0_ * rankSizeLevel1_) + rankIdxLevel1_ * rankSizeLevel0_ + rootx);
        } else {
            tempLevel0_->SetRoot((myRank_ / rankSizeLevel0_) * rankSizeLevel0_ + (param.root % rankSizeLevel0_));
        }
    }
    if (rankSizeLevel1_ > 1) {
        tempLevel1_ = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1_);
        if (rankIdxLevel2_ == rootz) {
            tempLevel1_->SetRoot(param.root / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_);
        } else {
            tempLevel1_->SetRoot(
                rankIdxLevel2_ * (rankSizeLevel0_ * rankSizeLevel1_) + rooty * rankSizeLevel0_ + rankIdxLevel0_);
        }
    }
    if (rankSizeLevel2_ > 1) {
        tempLevel2_ = std::make_shared<InsAlgTemplate2>(param, myRank_, subCommRanks2_);
        tempLevel2_->SetRoot(
            param.root / (rankSizeLevel0_ * rankSizeLevel1_) * (rankSizeLevel0_ * rankSizeLevel1_)
            + rankIdxLevel1_ * rankSizeLevel0_ + rankIdxLevel0_);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    HCCL_DEBUG("[InsV2ScatterOmniPipeExecutor] CalcRes");
    // 初始化一些基本成员变量
    InitCommInfo(param, topoInfo, algHierarchyInfo);

    if (algHierarchyInfo_.infos.size() == HIERARCHY_SIZE_3 && !algHierarchyInfo_.infos[2].empty()
        && !algHierarchyInfo_.infos[2][0].empty()) {
        topoType_ = TopoType::THREE_LEVEL;
    } else {
        topoType_ = TopoType::UBX_2LEVEL;
    }

    CHK_RET(BuildSubCommAndTempMap(param, algHierarchyInfo, topoInfo));

    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    if (tempLevel0_) {
        AlgResourceRequest resReqlevel;
        CHK_RET(tempLevel0_->CalcRes(comm, param, topoInfo, resReqlevel));
        resourceRequest.slaveThreadNum += 1 + resReqlevel.slaveThreadNum;
        resourceRequest.notifyNumPerThread.emplace_back(resReqlevel.notifyNumOnMainThread + 1);
        resourceRequest.notifyNumPerThread.insert(
            resourceRequest.notifyNumPerThread.end(), resReqlevel.notifyNumPerThread.begin(),
            resReqlevel.notifyNumPerThread.end());
        resourceRequest.notifyNumOnMainThread++;
        resourceRequest.channels.push_back(resReqlevel.channels[0]);
        HCCL_DEBUG(
            "[InsV2ScatterOmniPipeExecutor] level0-CalcRes, level0 slaveThreadNum: %u, notifyNumOnMainThread: %u",
            resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    }
    if (tempLevel1_) {
        AlgResourceRequest resReqlevel;
        CHK_RET(tempLevel1_->CalcRes(comm, param, topoInfo, resReqlevel));
        resourceRequest.slaveThreadNum += 1 + resReqlevel.slaveThreadNum;
        resourceRequest.notifyNumPerThread.emplace_back(resReqlevel.notifyNumOnMainThread + 1);
        resourceRequest.notifyNumPerThread.insert(
            resourceRequest.notifyNumPerThread.end(), resReqlevel.notifyNumPerThread.begin(),
            resReqlevel.notifyNumPerThread.end());
        resourceRequest.notifyNumOnMainThread++;
        resourceRequest.channels.push_back(resReqlevel.channels[0]);
        HCCL_DEBUG(
            "[InsV2ScatterOmniPipeExecutor] level1-CalcRes, level1 slaveThreadNum: %u, notifyNumOnMainThread: %u",
            resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    }
    if (tempLevel2_) {
        AlgResourceRequest resReqlevel;
        CHK_RET(tempLevel2_->CalcRes(comm, param, topoInfo, resReqlevel));
        resourceRequest.slaveThreadNum += 1 + resReqlevel.slaveThreadNum;
        resourceRequest.notifyNumPerThread.emplace_back(resReqlevel.notifyNumOnMainThread + 1);
        resourceRequest.notifyNumPerThread.insert(
            resourceRequest.notifyNumPerThread.end(), resReqlevel.notifyNumPerThread.begin(),
            resReqlevel.notifyNumPerThread.end());
        resourceRequest.notifyNumOnMainThread++;
        resourceRequest.channels.push_back(resReqlevel.channels[0]);
        HCCL_DEBUG(
            "[InsV2ScatterOmniPipeExecutor] level2-CalcRes, level2 slaveThreadNum: %u, notifyNumOnMainThread: %u",
            resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    }

    return HCCL_SUCCESS;
}

// 该函数必须按照level0、level1、level2的顺序调用
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    PrepareResForTemplateLevel(u32 level, const std::shared_ptr<InsAlgTemplateBase>& tempBase)
{
    u32 levelThreadNum = tempBase->GetThreadNum();
    if (level == OMNIPIPE_LEVEL0) {
        levelThreads_[OMNIPIPE_LEVEL0].assign(threads_.begin() + 1, threads_.begin() + 1 + levelThreadNum);
        tempMainThreadsLevel01_.push_back(levelThreads_[0].at(0));
    } else if (level == OMNIPIPE_LEVEL1) {
        levelThreads_[OMNIPIPE_LEVEL1].assign(
            threads_.begin() + 1 + levelThreads_[0].size(),
            threads_.begin() + 1 + levelThreads_[0].size() + levelThreadNum);
        tempMainThreadsLevel01_.push_back(levelThreads_[1].at(0));
    } else if (level == OMNIPIPE_LEVEL2) {
        levelThreads_[OMNIPIPE_LEVEL2].assign(
            threads_.begin() + 1 + levelThreads_[0].size() + levelThreads_[1].size(), threads_.end());
        tempMainThreadsLevel2_.push_back(levelThreads_[OMNIPIPE_LEVEL2].at(0));
    }

    // 获取当前template各自的主thread上有多少notify
    AlgResourceRequest levelTempRequest;
    CHK_RET(tempBase->GetRes(levelTempRequest));
    if (level < OMNIPIPE_LEVEL2) {
        notifyIdxCtrlToTempLevel01_.push_back(levelTempRequest.notifyNumOnMainThread);
        notifyIdxTempToCtrlLevel01_.push_back(tempMainThreadsLevel01_.size() + tempMainThreadsLevel2_.size() - 1);
    } else {
        notifyIdxCtrlToTempLevel2_.push_back(levelTempRequest.notifyNumOnMainThread);
        notifyIdxTempToCtrlLevel2_.push_back(tempMainThreadsLevel01_.size() + tempMainThreadsLevel2_.size() - 1);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::RestoreChannelMap(
    const AlgResourceCtxSerializable& resCtx,
    std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo) const
{
    rankIdToChannelInfo.resize(OMNIPIPE_LEVEL_NUM);
    u32 level = 0;
    if (rankSizeLevel0_ > 1) {
        for (auto& channel : resCtx.channels[level]) {
            u32 remoteRank = channel.remoteRank;
            rankIdToChannelInfo[OMNIPIPE_LEVEL0][remoteRank].push_back(channel);
        }
        level++;
    }
    if (rankSizeLevel1_ > 1) {
        for (auto& channel : resCtx.channels[level]) {
            u32 remoteRank = channel.remoteRank;
            rankIdToChannelInfo[OMNIPIPE_LEVEL1][remoteRank].push_back(channel);
        }
        level++;
    }
    if (rankSizeLevel2_ > 1) {
        for (auto& channel : resCtx.channels[level]) {
            u32 remoteRank = channel.remoteRank;
            rankIdToChannelInfo[OMNIPIPE_LEVEL2][remoteRank].push_back(channel);
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2ScatterOmniPipeExecutor][Orchestrate] Orchestrate Start");
    // 参数填充
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    threads_ = resCtx.threads;

    if (algHierarchyInfo_.infos.size() == HIERARCHY_SIZE_3 && !algHierarchyInfo_.infos[2].empty()
        && !algHierarchyInfo_.infos[2][0].empty()) {
        topoType_ = TopoType::THREE_LEVEL;
    } else {
        topoType_ = TopoType::UBX_2LEVEL;
    }

    // 计算subCommRanks和template
    CHK_RET(BuildSubCommAndTempMap(param, algHierarchyInfo_, &resCtx.topoInfo));

    // 为temp分配thread
    controlThread_ = threads_.at(0);
    levelThreads_.resize(OMNIPIPE_LEVEL_NUM);
    // 先初始化remoteRankToChannelInfo_，然后为nhr赋值多channel，最后再计算资源，这样计算线程资源的时候就能获取到多channel需要的线程数
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix) {
        if (rankSizeLevel1_ > 1) {
            CHK_RET(tempLevel1_->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
        }
    }

    if (tempLevel0_) {
        CHK_RET(PrepareResForTemplateLevel(OMNIPIPE_LEVEL0, tempLevel0_));
    }
    if (tempLevel1_) {
        CHK_RET(PrepareResForTemplateLevel(OMNIPIPE_LEVEL1, tempLevel1_));
    }
    if (tempLevel2_) {
        CHK_RET(PrepareResForTemplateLevel(OMNIPIPE_LEVEL2, tempLevel2_));
    }

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2ScatterOmniPipeExecutor][Orchestrate]errNo[0x%016llx] Scatter executor "
            "kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

// 将计算出的单步slice信息初始化到templateParam中
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenTempAlgParamsIn2HCCLBuff(
        TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo, u64 processedDataCount,
        const AlgResourceCtxSerializable& resCtx, const OpParam& param)
{
    tempAlgParams.count = processedDataCount;
    tempAlgParams.dataType = dataType_;
    stepSliceInfo.buffInfo.hcclBuff = resCtx.cclMem;
    stepSliceInfo.buffInfo.inputPtr = param.inputPtr;
    stepSliceInfo.buffInfo.inputSize = param.inputSize;
    stepSliceInfo.buffInfo.outputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.outputSize = resCtx.cclMem.size;
    stepSliceInfo.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo = stepSliceInfo.buffInfo;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff
        = processedDataCount * dataTypeSize_ + stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.localCopyFlag = 0;
    tempAlgParams.repeatNum = stepSliceInfo.stepCount.size();

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenTempAlgParamsHCCLBuff2HCCLBuff(
        TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo, u64 processedDataCount,
        const AlgResourceCtxSerializable& resCtx, const OpParam& param)
{
    tempAlgParams.count = processedDataCount;
    tempAlgParams.dataType = dataType_;
    stepSliceInfo.buffInfo.hcclBuff = resCtx.cclMem;
    stepSliceInfo.buffInfo.inputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.inputSize = resCtx.cclMem.size;
    stepSliceInfo.buffInfo.outputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.outputSize = resCtx.cclMem.size;
    stepSliceInfo.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo = stepSliceInfo.buffInfo;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.localCopyFlag = 0;
    tempAlgParams.repeatNum = stepSliceInfo.stepCount.size();

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2ScatterOmniPipeExecutor][OrchestrateLoop] Start");
    // 1.计算带宽
    double bw_rs_l0 = BW_OMNI_DEFAULT;  // 50
    double bw_rs_l1 = BW_OMNI_DEFAULT;  // 50
    double bw_rs_l2 = BW_OMNI_UBX_ROCE; // 25

    if (resCtx.topoInfo.level0PcieMix) {
        if (rankSizeLevel1_ == RANK_SIZE_LEVEL_2) {
            bw_rs_l1 = BW_OMNI_PCIE_EIGHT_RS_CLOS; // 29
        } else if (rankSizeLevel1_ == RANK_SIZE_LEVEL_4) {
            bw_rs_l1 = BW_OMNI_PCIE_SIXTEEN_RS_CLOS; // 35
        }
    } else if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS) {
        bw_rs_l1 = BW_OMNI_UBX_AICPU_SC_CLOS;
    }

    // 计算等价带宽
    double eqBw0 = bw_rs_l0; // L0 mesh
    double eqBw1 = bw_rs_l1; // L1 NHR
    double eqBw2 = bw_rs_l2; // L2 NHR

    eqBw1 = rankSizeLevel1_ > 1 ? eqBw1 / (rankSizeLevel1_ - 1) : eqBw1;
    eqBw2 = rankSizeLevel2_ > 1 ? eqBw2 / (rankSizeLevel2_ - 1) : eqBw2;
    HCCL_DEBUG("[InsV2ScatterOmniPipeExecutor][OrchestrateLoop] eqBw0[%f] eqBw1[%f] eqBw2[%f]", eqBw0, eqBw1, eqBw2);

    std::vector<double> endpointAttrBwNew{eqBw0, eqBw1, eqBw2};

    // 2、计算maxCountPerloop, loopTimes
    u64 maxTmpMemSize = resCtx.cclMem.size;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 scatterDataSize = maxTmpMemSize / rankSize_;
    HCCL_DEBUG(
        "[%s] myRank[%u] maxTmpMemSize[%u] transportBoundDataSize[%u]", __func__, myRank_, maxTmpMemSize,
        transportBoundDataSize);
    u64 maxCountPerLoop = std::min(scatterDataSize, transportBoundDataSize) / HCCL_MIN_SLICE_ALIGN
                          * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    CHK_PRT_RET(maxCountPerLoop == 0, HCCL_ERROR("[%s] maxCountPerLoop is 0", __func__), HCCL_E_INTERNAL);
    HCCL_DEBUG("[%s] myRank[%u] maxCountPerLoop[%u]", __func__, myRank_, maxCountPerLoop);
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);
    HCCL_DEBUG("[%s] myRank[%u] loopTimes[%u]", __func__, myRank_, loopTimes);
    u64 perLoopSize = maxCountPerLoop * dataTypeSize_;
    perLoopSize = dataSize_ > perLoopSize ? perLoopSize : dataSize_;
    HCCL_DEBUG("[%s] perLoopSize[%u]", __func__, perLoopSize);

    // 3、计算n-1次loop的slice信息
    OmniPipeSliceParam sliceParam;
    std::vector<u64> dataSizePerLoop;
    std::vector<u64> dataWholeSize;

    for (u32 i = 0; i < rankSize_; i++) {
        dataSizePerLoop.push_back(perLoopSize);
        dataWholeSize.push_back(dataSize_);
    }
    sliceParam.dataSizePerLoop = dataSizePerLoop;
    sliceParam.dataWholeSize = dataWholeSize;
    sliceParam.endpointAttrBw = endpointAttrBwNew;
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, rankIdxLevel2_};
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_};
    sliceParam.levelAlgType = {1, 0, 1};
    sliceParam.dataTypeSize = dataTypeSize_;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = param.engine;
    sliceParam.needSetStepNum = omniNeedSetStepNum_;
    OmniPipeSliceInfo alignSliceInfo = CalcScatterOmniPipeSliceInfo(sliceParam, param.root);
    CHK_PRT_RET(alignSliceInfo.isEmpty(), HCCL_ERROR("[%s] alignSliceInfo is empty", __func__), HCCL_E_INTERNAL);

    // 4、计算第n次的loop的slice信息
    OmniPipeSliceInfo tailSliceInfo;
    u64 tailLoopSize = 0;
    if (dataCount_ > maxCountPerLoop && dataCount_ % maxCountPerLoop != 0) {
        u64 tailCount = dataCount_ % maxCountPerLoop;
        tailLoopSize = tailCount * dataTypeSize_;
        HCCL_DEBUG("[%s] myRank[%u] tailLoopSize[%u]", __func__, myRank_, tailLoopSize);
        std::vector<u64> tailPerLoop(rankSize_, tailLoopSize);
        sliceParam.dataSizePerLoop = tailPerLoop;
        tailSliceInfo = CalcScatterOmniPipeSliceInfo(sliceParam, param.root);
        CHK_PRT_RET(tailSliceInfo.isEmpty(), HCCL_ERROR("[%s] tailSliceInfo is empty", __func__), HCCL_E_INTERNAL);
    }

    u64 processedDataCount = 0;
    OmniPipeSliceInfo currentSliceInfo;
    HCCL_INFO("[InsV2ScatterOmniPipeExecutor][OrchestrateLoop]loopTimes = [%u]", loopTimes);
    std::map<u32, TemplateResource> tempResMap;
    std::map<u32, TemplateDataParams> tempAlgParamMap;
    auto initTempRes = [&](u32 level) {
        tempResMap[level].channels = remoteRankToChannelInfo_[level];
        tempResMap[level].threads = levelThreads_[level];
        tempResMap[level].npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
        tempResMap[level].dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
        tempAlgParamMap[level].buffInfo.hcclBuff = resCtx.cclMem;
    };
    if (tempLevel0_) {
        initTempRes(OMNIPIPE_LEVEL0);
    }
    if (tempLevel1_) {
        initTempRes(OMNIPIPE_LEVEL1);
    }
    if (tempLevel2_) {
        initTempRes(OMNIPIPE_LEVEL2);
    }

    // 5、进行一次loop的数据处理
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        HCCL_DEBUG("[%s] myRank[%u] currDataCount[%llu]", __func__, myRank_, currDataCount);
        // 5.1 确定当前是前n-1次loop的slice结果，还是存在尾块时最后一次loop的slice结果
        if (loop == loopTimes - 1 && !tailSliceInfo.isEmpty()) {
            perLoopSize = tailLoopSize;
            currentSliceInfo = tailSliceInfo;
        } else {
            currentSliceInfo = alignSliceInfo;
        }
        u64 level2StepCount = currentSliceInfo.dataSliceLevel2.size();
        u64 level0StepCount = currentSliceInfo.dataSliceLevel0.size() / currentSliceInfo.dataSliceLevel2.size();
        HCCL_INFO(
            "[InsV2ScatterOmniPipeExecutor][OrchestrateLoop]level2 step count = [%u], level0 step count = [%u]",
            level2StepCount, level0StepCount);

        // 5.2 for外层2d
        for (u32 i = 0; i < level2StepCount; i++) {
            HCCL_INFO("[InsV2ScatterOmniPipeExecutor][OrchestrateLoop]Step [%u] in level2", i);
            if (rankSizeLevel2_ > 1) {
                HCCL_DEBUG("rankSizeLevel2_ > 1");
                if (myRank_ == param.root) {
                    // 统一设置从userIn发到cclbuff
                    CHK_RET(GenTempAlgParamsIn2HCCLBuff(
                        tempAlgParamMap[OMNIPIPE_LEVEL2], currentSliceInfo.dataSliceLevel2[i], processedDataCount,
                        resCtx, param));
                } else {
                    // 其他rank从cclbuff接收数据
                    CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
                        tempAlgParamMap[OMNIPIPE_LEVEL2], currentSliceInfo.dataSliceLevel2[i], processedDataCount,
                        resCtx, param));
                }
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel2_, notifyIdxCtrlToTempLevel2_));
                if (myRank_ == param.root || isSameZAxisAsRoot) {
                    tempLevel2_->SetDoTask(true);
                }

                // z轴带宽比较小，所以只有第一步是root发给对应的节点，其他剩余步, 所有rank都需要执行
                if (i > 0) {
                    tempLevel2_->SetDoTask(true);
                }
            }
            // 5.3 for内层2d
            for (u32 j = 0; j < level0StepCount; j++) {
                // level0、1前同步
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxCtrlToTempLevel01_));
                // 只有root的卡统一设置从userIn发到cclbuff
                if (myRank_ == param.root) {
                    CHK_RET(GenTempAlgParamsIn2HCCLBuff(
                        tempAlgParamMap[OMNIPIPE_LEVEL0], currentSliceInfo.dataSliceLevel0[i * level0StepCount + j],
                        processedDataCount, resCtx, param));
                    CHK_RET(GenTempAlgParamsIn2HCCLBuff(
                        tempAlgParamMap[OMNIPIPE_LEVEL1], currentSliceInfo.dataSliceLevel1[i * level0StepCount + j],
                        processedDataCount, resCtx, param));
                } else {
                    // 另外的都是从cclbuff到cclbuff
                    CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
                        tempAlgParamMap[OMNIPIPE_LEVEL1], currentSliceInfo.dataSliceLevel1[i * level0StepCount + j],
                        processedDataCount, resCtx, param));
                    CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
                        tempAlgParamMap[OMNIPIPE_LEVEL0], currentSliceInfo.dataSliceLevel0[i * level0StepCount + j],
                        processedDataCount, resCtx, param));
                }
                // 和root同x轴（也就是同列的）的卡参与，其他列的卡不参与
                if (rankIdxLevel0_ == param.root % rankSizeLevel0_) {
                    tempLevel1_->SetDoTask(true);
                }
                // 和root同y轴（也就是同行的）的卡参与，其他行的卡不参与
                if (rankIdxLevel1_ == param.root % (rankSizeLevel0_ * rankSizeLevel1_) / rankSizeLevel0_) {
                    tempLevel0_->SetDoTask(true);
                }

                // 设置机内template任务的参数
                if (j == level0StepCount - 1) {
                    HCCL_DEBUG("myRank[%u] j == level0StepCount - 1", myRank_);
                    tempLevel1_->SetDoTask(true);
                    tempLevel0_->SetDoTask(true);
                } else if (j != 0) {
                    HCCL_DEBUG("myRank[%u] j != 0", myRank_);
                    if (eqBw0 > eqBw1) {
                        tempLevel1_->SetDoTask(true);
                    } else {
                        tempLevel0_->SetDoTask(true);
                    }
                }
                // 执行机内template任务
                if (rankSizeLevel0_ > 1) {
                    HCCL_DEBUG("rankSizeLevel0_ > 1");
                    tempLevel0_->xyTotalRankSize_ = rankSizeLevel0_ * rankSizeLevel1_ * rankIdxLevel2_;
                    CHK_RET(
                        tempLevel0_->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL0], tempResMap[OMNIPIPE_LEVEL0]));
                }
                if (rankSizeLevel1_ > 1) {
                    HCCL_DEBUG("rankSizeLevel1_ > 1");
                    CHK_RET(
                        tempLevel1_->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL1], tempResMap[OMNIPIPE_LEVEL1]));
                }
                // level0、1的尾同步
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxTempToCtrlLevel01_));
            }
            if (rankSizeLevel1_ > 1) {
                tempLevel1_->SetDoTask(false);
            }
            if (rankSizeLevel0_ > 1) {
                tempLevel0_->SetDoTask(false);
            }
            if (rankSizeLevel2_ > 1) {
                CHK_RET(tempLevel2_->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL2], tempResMap[OMNIPIPE_LEVEL2]));
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel2_, notifyIdxTempToCtrlLevel2_));
                HCCL_DEBUG("PostSyncInterThreads z success.");
            }
        }
        // localcopy前同步
        CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxCtrlToTempLevel01_));
        // 5.4 将当前这个loop在ccl中的数据一次性拷贝到userout中
        TemplateDataParams tempAlgParamsLocalCopy;
        tempAlgParamsLocalCopy.localCopyFlag = 1;
        tempAlgParamsLocalCopy.dataType = dataType_;
        tempAlgParamsLocalCopy.buffInfo.inputPtr = resCtx.cclMem.addr;
        tempAlgParamsLocalCopy.buffInfo.inputSize = resCtx.cclMem.size;
        tempAlgParamsLocalCopy.buffInfo.outputPtr = param.outputPtr;
        tempAlgParamsLocalCopy.buffInfo.outputSize = param.outputSize;
        tempAlgParamsLocalCopy.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamsLocalCopy.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsLocalCopy.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParamsLocalCopy.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsLocalCopy.count = currDataCount;
        tempAlgParamsLocalCopy.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParamsLocalCopy.buffInfo.inBuffBaseOff = myRank_ * perLoopSize;

        if (myRank_ == param.root) {
            tempAlgParamsLocalCopy.buffInfo.inputPtr = param.inputPtr;
            tempAlgParamsLocalCopy.buffInfo.inputSize = param.inputSize;
            tempAlgParamsLocalCopy.buffInfo.inBuffType = BufferType::INPUT;
            tempAlgParamsLocalCopy.buffInfo.inBuffBaseOff = myRank_ * dataSize_ + processedDataCount * dataTypeSize_;
        }

        HCCL_DEBUG(
            "[%s] myRank[%u] localCopy inBuffBaseOff[%lu] outBuffBaseOff[%lu] sliceSize[%lu]", __func__, myRank_,
            tempAlgParamsLocalCopy.buffInfo.inBuffBaseOff, tempAlgParamsLocalCopy.buffInfo.outBuffBaseOff,
            tempAlgParamsLocalCopy.sliceSize);
        if (rankSizeLevel0_ > 1) {
            CHK_RET(tempLevel0_->DoLocalCopy(tempAlgParamsLocalCopy, tempResMap[OMNIPIPE_LEVEL0].threads));
        } else if (rankSizeLevel1_ > 1) {
            CHK_RET(tempLevel1_->DoLocalCopy(tempAlgParamsLocalCopy, tempResMap[OMNIPIPE_LEVEL1].threads));
        }
        // localcopy后同步
        CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxTempToCtrlLevel01_));
        processedDataCount += currDataCount;
        if (rankSizeLevel2_ > 1) {
            tempLevel2_->SetDoTask(false);
        }
    }
    HCCL_INFO("[InsV2ScatterOmniPipeExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_SCATTER, DpuScatterOmniPipeMeshNHR, InsV2ScatterOmniPipeExecutor, TopoMatchUBX,
    InsTempScatterOmniPipeMesh1D, InsTempScatterOmniPipeNHR, InsTempScatterOmniPipeNHRDpu);

} // namespace ops_hccl
