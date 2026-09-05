/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_broadcast_omnipipe_executor.h"
#include "topo_match_3_level.h"
#include "ins_temp_scatter_omnipipe_mesh1d.h"
#include "ins_temp_scatter_omnipipe_nhr_dpu.h"
#include "ins_temp_scatter_omnipipe_nhr.h"
#include "ins_temp_all_gather_omnipipe_mesh_1D.h"
#include "ins_temp_all_gather_omnipipe_nhr_dpu.h"
#include "ins_temp_all_gather_omnipipe_nhr.h"
#include "omnipipe_template_utils.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {
constexpr u32 ALG_HIERARCHY_NUM3 = 3;
constexpr uint64_t RANK_SIZE_LEVEL1_2 = 2;
constexpr uint64_t RANK_SIZE_LEVEL1_4 = 4;

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::InsV2BroadcastOmniPipeExecutor()
{}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
void InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    InitTemplateBufferInfo(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateDataParams& tempAlgParam)
{
    tempAlgParam.buffInfo.inputPtr = param.inputPtr;
    tempAlgParam.buffInfo.outputPtr = param.outputPtr;
    tempAlgParam.buffInfo.hcclBuff = resCtx.cclMem;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
void InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    InitTemplateParamByLevel(
        u32 templateLevel, u32 hierarchyLevel, const std::vector<std::vector<ThreadHandle>>& levelThreads,
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, std::map<u32, TemplateResource>& tempResMap,
        std::map<u32, TemplateDataParams>& tempAlgParamMap)
{
    tempResMap[templateLevel].threads = levelThreads[hierarchyLevel];
    tempResMap[templateLevel].channels = remoteRankToChannelInfo_[hierarchyLevel];
    tempResMap[templateLevel].npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    tempResMap[templateLevel].dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    InitTemplateBufferInfo(param, resCtx, tempAlgParamMap[templateLevel]);
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
void InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::InitRankIndex()
{
    uint32_t intraSuperpodDeviceNum = rankSizeLevel0_ * rankSizeLevel1_;
    rankIdxLevel0_ = (myRank_ % intraSuperpodDeviceNum) % rankSizeLevel0_;
    rankIdxLevel1_ = (myRank_ % intraSuperpodDeviceNum) / rankSizeLevel0_;
    rankIdxLevel2_ = myRank_ / intraSuperpodDeviceNum;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    InitCommInfo(
        const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    algHierarchyInfo_ = algHierarchyInfo;

    HCCL_INFO(
        "[%s]myRank[%u] userRankSize[%u] devType[%u] dataType[%u] dataTypeSize[%u]", __func__, myRank_, rankSize_,
        devType_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    BuildUbxSubCommRanks(
        std::vector<std::vector<u32>>& subCommRanks0, std::vector<std::vector<u32>>& subCommRanks1,
        std::vector<std::vector<u32>>& subCommRanks2, const TopoInfoWithNetLayerDetails* topoInfo)
{
    if (algHierarchyInfo_.infos[0].size() < 2 || algHierarchyInfo_.infos[0][0].empty()) {
        HCCL_ERROR(
            "[%s] algHierarchyInfo_.infos[0] size[%zu] is less than 2 or infos[0][0] empty.", __func__,
            algHierarchyInfo_.infos[0].size());
        return HCCL_E_PARA;
    }
    std::vector<u32> closRanks;
    subCommRanks0 = {algHierarchyInfo_.infos[0][0]};
    u32 meshSize = algHierarchyInfo_.infos[0][0].size();
    if (!algHierarchyInfo_.infos[0][1].empty()) {
        for (auto rank : algHierarchyInfo_.infos[0][1]) {
            if (rank % meshSize == topoInfo->userRank % meshSize) {
                closRanks.push_back(rank);
            }
        }
    }
    subCommRanks1 = {closRanks};
    omniNeedSetStepNum_ = (subCommRanks1[0].size() == RANK_SIZE_LEVEL1_4) ? OmniNeedSetStepNum::OMNIPIPE_UBX_16P :
                                                                            OmniNeedSetStepNum::OMNIPIPE_DEFAULT;
    if (!algHierarchyInfo_.infos[1].empty()) {
        subCommRanks2 = algHierarchyInfo_.infos[1];
        omniNeedSetStepNum_
            = (subCommRanks2[0].size() > 1) ? OmniNeedSetStepNum::OMNIPIPE_UBX_32P : omniNeedSetStepNum_;
    } else {
        subCommRanks2.emplace_back(std::vector<u32>{myRank_});
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    BuildSubCommRanks(
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, std::vector<std::vector<u32>>& subCommRanks0,
        std::vector<std::vector<u32>>& subCommRanks1, std::vector<std::vector<u32>>& subCommRanks2,
        const TopoInfoWithNetLayerDetails* topoInfo)
{
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        return BuildUbxSubCommRanks(subCommRanks0, subCommRanks1, subCommRanks2, topoInfo);
    }
    if (topoType_ == TopoType::THREE_LEVEL) {
        if (!algHierarchyInfo.infos[0].empty() && !algHierarchyInfo.infos[0][0].empty()) {
            subCommRanks0.push_back(algHierarchyInfo.infos[0][0]);
        } else {
            subCommRanks0.emplace_back(std::vector<u32>{myRank_});
        }
        if (!algHierarchyInfo.infos[1].empty() && !algHierarchyInfo.infos[1][0].empty()) {
            subCommRanks1.push_back(algHierarchyInfo.infos[1][0]);
        } else {
            subCommRanks1.emplace_back(std::vector<u32>{myRank_});
        }
        if (!algHierarchyInfo.infos[2].empty() && !algHierarchyInfo.infos[2][0].empty()) {
            subCommRanks2.push_back(algHierarchyInfo.infos[2][0]);
        } else {
            subCommRanks2.emplace_back(std::vector<u32>{myRank_});
        }
        return HCCL_SUCCESS;
    }
    if (!algHierarchyInfo_.infos[0].empty()) {
        subCommRanks0 = algHierarchyInfo_.infos[0];
    } else {
        subCommRanks0.emplace_back(std::vector<u32>{myRank_});
    }
    if (!algHierarchyInfo_.infos[1].empty()) {
        subCommRanks1 = algHierarchyInfo_.infos[1];
    } else {
        subCommRanks1.emplace_back(std::vector<u32>{myRank_});
    }
    subCommRanks2.emplace_back(std::vector<u32>{myRank_});
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    InitRankInfoAndTemp(
        const OpParam& param, std::vector<std::vector<u32>>& subCommRanks0,
        std::vector<std::vector<u32>>& subCommRanks1, std::vector<std::vector<u32>>& subCommRanks2)
{
    rankSizeLevel0_ = subCommRanks0[0].size();
    rankSizeLevel1_ = subCommRanks1[0].size();
    rankSizeLevel2_ = subCommRanks2[0].size();
    if (rankSizeLevel0_ == 0 || rankSizeLevel1_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel0_[%u] or rankSizeLevel1_[%u] is 0.", __func__, rankSizeLevel0_, rankSizeLevel1_);
        return HCCL_E_PARA;
    }

    InitRankIndex();

    bool isRoot = (myRank_ == param.root);
    u64 rootz = param.root / (rankSizeLevel0_ * rankSizeLevel1_);
    u64 rooty = param.root % (rankSizeLevel0_ * rankSizeLevel1_) / rankSizeLevel0_;
    u64 rootx = param.root % rankSizeLevel0_;
    isSameSerAsRoot = (rankIdxLevel2_ == rootz) && !isRoot;
    isSameYAxisAsRoot = (rankIdxLevel0_ == rootx && rankIdxLevel2_ == rootz) && !isRoot;
    isSameXAxisAsRoot = (rankIdxLevel1_ == rooty && rankIdxLevel2_ == rootz) && !isRoot;
    isSameZAxisAsRoot = false;
    if (rankSizeLevel2_ > 1) {
        isSameZAxisAsRoot = (rankIdxLevel1_ == rooty && rankIdxLevel0_ == rootx && rankIdxLevel2_ != rootz) && !isRoot;
    }

    if (rankSizeLevel0_ > 1) {
        tempScatterLevel0_ = std::make_shared<InsScatterAlgTemplateX>(param, myRank_, subCommRanks0);
        if (rankIdxLevel2_ != rootz) {
            tempScatterLevel0_->SetRoot(
                rankIdxLevel2_ * (rankSizeLevel0_ * rankSizeLevel1_) + rankIdxLevel1_ * rankSizeLevel0_ + rootx);
        } else {
            tempScatterLevel0_->SetRoot((myRank_ / rankSizeLevel0_) * rankSizeLevel0_ + (param.root % rankSizeLevel0_));
        }
        tempAgLevel0_ = std::make_shared<InsAgAlgTemplateX>(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel1_ > 1) {
        tempScatterLevel1_ = std::make_shared<InsScatterAlgTemplateY>(param, myRank_, subCommRanks1);
        if (rankIdxLevel2_ == rootz) {
            tempScatterLevel1_->SetRoot(param.root / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_);
        } else {
            tempScatterLevel1_->SetRoot(
                rankIdxLevel2_ * (rankSizeLevel0_ * rankSizeLevel1_) + rooty * rankSizeLevel0_ + rankIdxLevel0_);
        }
        tempAgLevel1_ = std::make_shared<InsAgAlgTemplateY>(param, myRank_, subCommRanks1);
    }
    if (rankSizeLevel2_ > 1) {
        tempScatterLevel2_ = std::make_shared<InsScatterAlgTemplateZ>(param, myRank_, subCommRanks2);
        tempScatterLevel2_->SetRoot(
            param.root / (rankSizeLevel0_ * rankSizeLevel1_) * (rankSizeLevel0_ * rankSizeLevel1_)
            + rankIdxLevel1_ * rankSizeLevel0_ + rankIdxLevel0_);
        tempAgLevel2_ = std::make_shared<InsAgAlgTemplateZ>(param, myRank_, subCommRanks2);
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    BuildSubCommAndTempMap(
        const OpParam& param, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        std::vector<std::vector<u32>>& subCommRanks0, std::vector<std::vector<u32>>& subCommRanks1,
        std::vector<std::vector<u32>>& subCommRanks2, const TopoInfoWithNetLayerDetails* topoInfo)
{
    if (algHierarchyInfo_.infos.empty()) {
        HCCL_ERROR("[%s] algHierarchyInfo_.infos is empty.", __func__);
        return HCCL_E_PARA;
    }
    subCommRanks0.clear();
    subCommRanks1.clear();
    subCommRanks2.clear();
    tempScatterLevel0_.reset();
    tempScatterLevel1_.reset();
    tempScatterLevel2_.reset();
    tempAgLevel0_.reset();
    tempAgLevel1_.reset();
    tempAgLevel2_.reset();

    HCCL_INFO("[BuildSubCommAndTempMap]infos,%s", ThreeDVecToStrOmni(algHierarchyInfo_.infos).c_str());
    CHK_RET(BuildSubCommRanks(algHierarchyInfo, subCommRanks0, subCommRanks1, subCommRanks2, topoInfo));
    CHK_RET(InitRankInfoAndTemp(param, subCommRanks0, subCommRanks1, subCommRanks2));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    CalcResLevel(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const std::shared_ptr<InsAlgTemplateBase> tempAlg, AlgResourceRequest& resourceRequest, bool addChannel) const
{
    AlgResourceRequest resReqlevel;
    CHK_RET(tempAlg->CalcRes(comm, param, topoInfo, resReqlevel));
    resourceRequest.slaveThreadNum += resReqlevel.slaveThreadNum + 1;
    resourceRequest.notifyNumOnMainThread += 1;
    resourceRequest.notifyNumPerThread.emplace_back(
        resReqlevel.notifyNumOnMainThread + 1); // temp控制流：从流数量+主控制流
    resourceRequest.notifyNumPerThread.insert(
        resourceRequest.notifyNumPerThread.end(), resReqlevel.notifyNumPerThread.begin(),
        resReqlevel.notifyNumPerThread.end());
    if (addChannel) {
        resourceRequest.channels.emplace_back(resReqlevel.channels[0]);
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    HCCL_DEBUG("[InsV2BroadcastOmniPipeExecutor] CalcRes");
    CHK_RET(InitCommInfo(param, topoInfo, algHierarchyInfo));

    if (algHierarchyInfo_.infos.size() == ALG_HIERARCHY_NUM3 && !algHierarchyInfo_.infos[2].empty()
        && !algHierarchyInfo_.infos[2][0].empty()) {
        topoType_ = TopoType::THREE_LEVEL;
    } else {
        topoType_ = TopoType::UBX_2LEVEL;
    }

    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;
    CHK_RET(BuildSubCommAndTempMap(param, algHierarchyInfo, subCommRanks0, subCommRanks1, subCommRanks2, topoInfo));

    HCCL_DEBUG(
        "[InsV2BroadcastOmniPipeExecutor] L0[%u], L1[%u], L2[%u]", rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_);

    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    // 各级 Scatter 计算资源 (addChannel=true)
    if (tempScatterLevel0_) {
        CHK_RET(CalcResLevel(comm, param, topoInfo, tempScatterLevel0_, resourceRequest, true));
    }
    if (tempScatterLevel1_) {
        CHK_RET(CalcResLevel(comm, param, topoInfo, tempScatterLevel1_, resourceRequest, true));
    }
    if (tempScatterLevel2_) {
        CHK_RET(CalcResLevel(comm, param, topoInfo, tempScatterLevel2_, resourceRequest, true));
    }
    // 各级 AllGather 计算资源 (addChannel=false, 复用 Scatter 的 channel)
    if (tempAgLevel0_) {
        CHK_RET(CalcResLevel(comm, param, topoInfo, tempAgLevel0_, resourceRequest, false));
    }
    if (tempAgLevel1_) {
        CHK_RET(CalcResLevel(comm, param, topoInfo, tempAgLevel1_, resourceRequest, false));
    }
    if (tempAgLevel2_) {
        CHK_RET(CalcResLevel(comm, param, topoInfo, tempAgLevel2_, resourceRequest, false));
    }
    return HCCL_SUCCESS;
}

// 该函数必须按照 level0、level1、level2 的顺序调用（Scatter）
template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    PrepareResForTemplateLevelScatter(u32 level, const std::shared_ptr<InsAlgTemplateBase>& tempBase)
{
    u64 levelThreadNum = tempBase->GetThreadNum();
    if (level == OMNIPIPE_LEVEL0) {
        levelThreadsSC_[OMNIPIPE_LEVEL0].assign(threads_.begin() + 1, threads_.begin() + 1 + levelThreadNum);
        tempMainThreadsLevel01SC_.push_back(levelThreadsSC_[0].at(0));
    } else if (level == OMNIPIPE_LEVEL1) {
        levelThreadsSC_[OMNIPIPE_LEVEL1].assign(
            threads_.begin() + 1 + levelThreadsSC_[0].size(),
            threads_.begin() + 1 + levelThreadsSC_[0].size() + levelThreadNum);
        tempMainThreadsLevel01SC_.push_back(levelThreadsSC_[1].at(0));
    } else if (level == OMNIPIPE_LEVEL2) {
        levelThreadsSC_[OMNIPIPE_LEVEL2].assign(
            threads_.begin() + 1 + levelThreadsSC_[0].size() + levelThreadsSC_[1].size(),
            threads_.begin() + 1 + levelThreadsSC_[0].size() + levelThreadsSC_[1].size() + levelThreadNum);
        tempMainThreadsLevel2SC_.push_back(levelThreadsSC_[OMNIPIPE_LEVEL2].at(0));
    }

    AlgResourceRequest levelTempRequest;
    CHK_RET(tempBase->GetRes(levelTempRequest));
    if (level < OMNIPIPE_LEVEL2) {
        ntfIdxCtrlToTempLevel01SC_.push_back(levelTempRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlLevel01SC_.push_back(tempMainThreadsLevel01SC_.size() + tempMainThreadsLevel2SC_.size() - 1);
    } else {
        ntfIdxCtrlToTempLevel2SC_.push_back(levelTempRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlLevel2SC_.push_back(tempMainThreadsLevel01SC_.size() + tempMainThreadsLevel2SC_.size() - 1);
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    PrepareResForTemplateLevelAllGather(u32 level, const std::shared_ptr<InsAlgTemplateBase>& tempBase)
{
    u64 levelThreadNum = tempBase->GetThreadNum();
    // AG 线程接在所有 Scatter 线程之后
    u64 threadsNumStart = levelThreadsSC_[OMNIPIPE_LEVEL0].size() + levelThreadsSC_[OMNIPIPE_LEVEL1].size()
                          + levelThreadsSC_[OMNIPIPE_LEVEL2].size();
    if (level == OMNIPIPE_LEVEL0) {
        levelThreadsAG_[OMNIPIPE_LEVEL0].assign(
            threads_.begin() + threadsNumStart + 1, threads_.begin() + threadsNumStart + 1 + levelThreadNum);
        tempMainThreadsLevel01AG_.push_back(levelThreadsAG_[0].at(0));
    } else if (level == OMNIPIPE_LEVEL1) {
        levelThreadsAG_[OMNIPIPE_LEVEL1].assign(
            threads_.begin() + threadsNumStart + 1 + levelThreadsAG_[0].size(),
            threads_.begin() + threadsNumStart + 1 + levelThreadsAG_[0].size() + levelThreadNum);
        tempMainThreadsLevel01AG_.push_back(levelThreadsAG_[1].at(0));
    } else if (level == OMNIPIPE_LEVEL2) {
        levelThreadsAG_[OMNIPIPE_LEVEL2].assign(
            threads_.begin() + threadsNumStart + 1 + levelThreadsAG_[0].size() + levelThreadsAG_[1].size(),
            threads_.end());
        tempMainThreadsLevel2AG_.push_back(levelThreadsAG_[OMNIPIPE_LEVEL2].at(0));
    }

    AlgResourceRequest levelTempRequest;
    CHK_RET(tempBase->GetRes(levelTempRequest));
    if (level < OMNIPIPE_LEVEL2) {
        ntfIdxCtrlToTempLevel01AG_.push_back(levelTempRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlLevel01AG_.push_back(tempMainThreadsLevel01AG_.size() + tempMainThreadsLevel2AG_.size() - 1);
    } else {
        ntfIdxCtrlToTempLevel2AG_.push_back(levelTempRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlLevel2AG_.push_back(tempMainThreadsLevel01AG_.size() + tempMainThreadsLevel2AG_.size() - 1);
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    RestoreChannelMap(
        const AlgResourceCtxSerializable& resCtx,
        std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo) const
{
    rankIdToChannelInfo.resize(OMNIPIPE_LEVEL_NUM);
    const u32 channelIndexLevel0 = 0;
    const u32 channelIndexLevel1 = (rankSizeLevel0_ > 1) ? 1 : 0;
    const u32 channelIndexLevel2 = channelIndexLevel1 + ((rankSizeLevel1_ > 1) ? 1 : 0);
    if (rankSizeLevel2_ > 1) {
        for (auto& channel : resCtx.channels[channelIndexLevel2]) {
            u32 remoteRank = channel.remoteRank;
            rankIdToChannelInfo[OMNIPIPE_LEVEL2][remoteRank].push_back(channel);
        }
    }
    if (rankSizeLevel1_ > 1) {
        for (auto& channel : resCtx.channels[channelIndexLevel1]) {
            u32 remoteRank = channel.remoteRank;
            rankIdToChannelInfo[OMNIPIPE_LEVEL1][remoteRank].push_back(channel);
        }
    }
    if (rankSizeLevel0_ > 1) {
        for (auto& channel : resCtx.channels[channelIndexLevel0]) {
            u32 remoteRank = channel.remoteRank;
            rankIdToChannelInfo[OMNIPIPE_LEVEL0][remoteRank].push_back(channel);
        }
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::InitExecutorInfo(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    threads_ = resCtx.threads;
    maxTmpMemSize_ = resCtx.cclMem.size;

    if (algHierarchyInfo_.infos.size() == ALG_HIERARCHY_NUM3 && !algHierarchyInfo_.infos[2].empty()
        && !algHierarchyInfo_.infos[2][0].empty()) {
        topoType_ = TopoType::THREE_LEVEL;
    } else {
        topoType_ = TopoType::UBX_2LEVEL;
    }

    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;
    CHK_RET(BuildSubCommAndTempMap(
        param, algHierarchyInfo_, subCommRanks0, subCommRanks1, subCommRanks2, &(resCtx.topoInfo)));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2BroadcastOmniPipeExecutor][Orchestrate] Orchestrate Start");
    CHK_RET(InitExecutorInfo(param, resCtx));

    controlThread_ = threads_.at(0);
    levelThreadsSC_.resize(OMNIPIPE_LEVEL_NUM);
    levelThreadsAG_.resize(OMNIPIPE_LEVEL_NUM);

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    // UBX 下为 NHR level1 设置多channel
    if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix) {
        if (rankSizeLevel1_ > 1) {
            CHK_RET(tempScatterLevel1_->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
            CHK_RET(tempAgLevel1_->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
        }
    }

    // 为各级 Scatter/AG template 分配 thread
    if (tempScatterLevel0_) {
        CHK_RET(PrepareResForTemplateLevelScatter(OMNIPIPE_LEVEL0, tempScatterLevel0_));
    }
    if (tempScatterLevel1_) {
        CHK_RET(PrepareResForTemplateLevelScatter(OMNIPIPE_LEVEL1, tempScatterLevel1_));
    }
    if (tempScatterLevel2_) {
        CHK_RET(PrepareResForTemplateLevelScatter(OMNIPIPE_LEVEL2, tempScatterLevel2_));
    }
    if (tempAgLevel0_) {
        CHK_RET(PrepareResForTemplateLevelAllGather(OMNIPIPE_LEVEL0, tempAgLevel0_));
    }
    if (tempAgLevel1_) {
        CHK_RET(PrepareResForTemplateLevelAllGather(OMNIPIPE_LEVEL1, tempAgLevel1_));
    }
    if (tempAgLevel2_) {
        CHK_RET(PrepareResForTemplateLevelAllGather(OMNIPIPE_LEVEL2, tempAgLevel2_));
    }

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2BroadcastOmniPipeExecutor][Orchestrate]errNo[0x%016llx] Broadcast executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    GenTempAlgParamsIn2HCCLBuff(
        TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo, u64 processedDataCount,
        const AlgResourceCtxSerializable& resCtx, const OpParam& param)
{
    stepSliceInfo.buffInfo.outputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.outputSize = resCtx.cclMem.size;
    stepSliceInfo.buffInfo.inputPtr = param.inputPtr;
    stepSliceInfo.buffInfo.inputSize = param.inputSize;
    stepSliceInfo.buffInfo.hcclBuff = resCtx.cclMem;
    stepSliceInfo.buffInfo.inBuffType = BufferType::INPUT;
    stepSliceInfo.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.dataType = dataType_;
    tempAlgParams.count = processedDataCount;
    tempAlgParams.buffInfo = stepSliceInfo.buffInfo;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.repeatNum = stepSliceInfo.stepCount.size();
    tempAlgParams.localCopyFlag = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff
        = processedDataCount * dataTypeSize_ + stepSliceInfo.buffInfo.inBuffBaseOff;
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    GenTempAlgParamsHCCLBuff2HCCLBuff(
        TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo, u64 processedDataCount,
        const AlgResourceCtxSerializable& resCtx, const OpParam& param)
{
    stepSliceInfo.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.hcclBuff = resCtx.cclMem;
    stepSliceInfo.buffInfo.outputSize = resCtx.cclMem.size;
    stepSliceInfo.buffInfo.outputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.inputSize = resCtx.cclMem.size;
    stepSliceInfo.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParams.dataType = dataType_;
    tempAlgParams.count = processedDataCount;
    tempAlgParams.buffInfo = stepSliceInfo.buffInfo;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.repeatNum = stepSliceInfo.stepCount.size();
    tempAlgParams.localCopyFlag = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::GenTemplateAlgParamsByDimData(TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo)
    const
{
    // 参考allReduce 3-level: 过程中的所有step都在ccl中进行数据搬运，在template中只使用ccl的起始地址
    tempAlgParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;

    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.buffInfo.hcclBuffBaseOff = stepSliceInfo.buffInfo.hcclBuffBaseOff;

    tempAlgParams.stepSliceInfo = stepSliceInfo;
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    DoLocalCopy(
        const TemplateDataParams& tempAlgParams, const ThreadHandle& thread, const std::vector<u64>& allRankSplitData,
        const std::vector<u64>& curLoopAllRankSplitData) const
{
    std::vector<DataSlice> srcDataSlice;
    std::vector<DataSlice> dstDataSlice;

    CHK_RET(CalLocalCopySlice(
        tempAlgParams, allRankSplitData, curLoopAllRankSplitData, srcDataSlice, dstDataSlice, dataTypeSize_));

    CHK_PRT_RET(
        srcDataSlice.size() != dstDataSlice.size(),
        HCCL_ERROR("[InsV2BroadcastOmniPipeExecutor][DoLocalCopy] srcDataSlice.size != dstDataSlice.size"),
        HCCL_E_PARA);

    for (auto i = 0; i < srcDataSlice.size(); ++i) {
        CHK_RET(LocalCopy(thread, srcDataSlice[i], dstDataSlice[i]));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    CalcEndpointBandwidth(
        const AlgResourceCtxSerializable& resCtx, std::vector<double>& endpointAttrBwAvgSC,
        std::vector<double>& endpointAttrBwAvgAG)
{
    // Scatter 带宽（参考 scatter 3-level: UBX下 l1=185, l2=BW_OMNI_UBX_ROCE）
    double bw_sc_l0 = BW_OMNI_DEFAULT;  // 50
    double bw_sc_l1 = BW_OMNI_DEFAULT;  // 50
    double bw_sc_l2 = BW_OMNI_UBX_ROCE; // 25
    // AllGather 带宽（参考 all_reduce 3-level AG: UBX下 l1=BW_OMNI_UBX_AG_CLOS, l2=BW_OMNI_UBX_ROCE）
    double bw_ag_l0 = BW_OMNI_DEFAULT;
    double bw_ag_l1 = BW_OMNI_DEFAULT;
    double bw_ag_l2 = BW_OMNI_UBX_ROCE;

    if (resCtx.topoInfo.level0PcieMix) {
        if (rankSizeLevel1_ == RANK_SIZE_LEVEL1_2) {
            bw_sc_l1 = BW_OMNI_PCIE_EIGHT_RS_CLOS;
            bw_ag_l1 = BW_OMNI_PCIE_EIGHT_AG_CLOS;
        } else if (rankSizeLevel1_ == RANK_SIZE_LEVEL1_4) {
            bw_sc_l1 = BW_OMNI_PCIE_SIXTEEN_RS_CLOS;
            bw_ag_l1 = BW_OMNI_PCIE_SIXTEEN_AG_CLOS;
        }
    } else if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS) {
        bw_sc_l1 = BW_OMNI_UBX_AICPU_SC_CLOS;
        bw_ag_l1 = BW_OMNI_UBX_AG_CLOS;
    }

    // 等效带宽: Level0 mesh为本身; Level1/Level2 NHR 按 rankSize-1 均摊
    double eqBw0SC = bw_sc_l0;
    double eqBw1SC = rankSizeLevel1_ > 1 ? bw_sc_l1 / (rankSizeLevel1_ - 1) : bw_sc_l1;
    double eqBw2SC = rankSizeLevel2_ > 1 ? bw_sc_l2 / (rankSizeLevel2_ - 1) : bw_sc_l2;
    endpointAttrBwAvgSC = {eqBw0SC, eqBw1SC, eqBw2SC};

    double eqBw0AG = bw_ag_l0;
    double eqBw1AG = rankSizeLevel1_ > 1 ? bw_ag_l1 / (rankSizeLevel1_ - 1) : bw_ag_l1;
    double eqBw2AG = rankSizeLevel2_ > 1 ? bw_ag_l2 / (rankSizeLevel2_ - 1) : bw_ag_l2;
    endpointAttrBwAvgAG = {eqBw0AG, eqBw1AG, eqBw2AG};

    HCCL_INFO(
        "[%s] eqBwSC[0:%f, 1:%f, 2:%f], eqBwAG[0:%f, 1:%f, 2:%f]", __func__, eqBw0SC, eqBw1SC, eqBw2SC, eqBw0AG,
        eqBw1AG, eqBw2AG);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::CalcLoopSplitData(u64 maxTmpMemSize, u64 root, LoopSplitData& loopSplitData)
{
    // 1. 每个rank切分的总count（broadcast语义：按rank均分，root所占份额为其余rank分到的那一份）
    loopSplitData.allRankSplitData = OmniPipeSplitScatterData(rankSize_, dataCount_, dataTypeSize_, root);
    CHK_PRT_RET(
        loopSplitData.allRankSplitData.empty(), HCCL_ERROR("[%s] allRankSplitData is empty", __func__),
        HCCL_E_INTERNAL);

    // 2. loop次数受 UB 单次传输上限和scratch显存上限双约束
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 scatterDataSize = maxTmpMemSize / rankSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    HCCL_DEBUG(
        "[%s] myRank[%u] transportBoundDataSize[%u] scatterDataSize[%u]", __func__, myRank_, transportBoundDataSize,
        scatterDataSize);
    loopSplitData.maxCountPerLoop = std::min(transportBoundDataSize, scatterDataSize) / dataTypeSize_;
    CHK_PRT_RET(loopSplitData.maxCountPerLoop == 0, HCCL_ERROR("[%s] maxCountPerLoop is 0", __func__), HCCL_E_INTERNAL);

    const u64 maxRankCount
        = *std::max_element(loopSplitData.allRankSplitData.begin(), loopSplitData.allRankSplitData.end());
    loopSplitData.loopTimes
        = maxRankCount / loopSplitData.maxCountPerLoop + ((maxRankCount % loopSplitData.maxCountPerLoop == 0) ? 0 : 1);
    HCCL_DEBUG(
        "[%s] myRank[%u] maxCountPerLoop[%u] loopTimes[%u]", __func__, myRank_, loopSplitData.maxCountPerLoop,
        loopSplitData.loopTimes);

    // 3. 每个rank每个loop切分的count
    loopSplitData.multiLoopAllRankSplitData = OmniPipeSplitRankDataLoop(
        loopSplitData.allRankSplitData, loopSplitData.maxCountPerLoop, loopSplitData.loopTimes, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    InitSliceParam(
        const OpParam& param, const std::vector<u64>& allRankSplitData,
        const std::vector<std::vector<u64>>& multiLoopAllRankSplitData, OmniPipeSliceParam& sliceParam)
{
    sliceParam.dataSizePerLoop = CalcCountToDataSize(multiLoopAllRankSplitData[0], dataTypeSize_);
    sliceParam.dataWholeSize = CalcCountToDataSize(allRankSplitData, dataTypeSize_);
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, rankIdxLevel2_};
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_};
    // levelAlgType: mesh=1, nhr=0 (参考 scatter 3-level)
    sliceParam.levelAlgType = std::vector<u64>{1, 0, 1};
    sliceParam.dataTypeSize = dataTypeSize_;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = param.engine;
    sliceParam.needSetStepNum = omniNeedSetStepNum_;
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    PrepareSliceInfoForLoop(
        u64 loop, u64 root, const std::vector<u64>& allRankSplitData,
        const std::vector<std::vector<u64>>& multiLoopAllRankSplitData, const std::vector<double>& endpointAttrBwAvgSC,
        const std::vector<double>& endpointAttrBwAvgAG, OmniPipeSliceParam& sliceParam,
        OmniPipeSliceInfo& omniPipeSliceInfoSC, OmniPipeSliceInfo& omniPipeSliceInfoAG)
{
    // 首轮loop或与上轮切分不同时重新计算 sliceInfo
    if (loop != 0 && isSameLoop(multiLoopAllRankSplitData[loop - 1], multiLoopAllRankSplitData[loop])) {
        return HCCL_SUCCESS;
    }

    const std::vector<u64> curLoopDataSize = CalcCountToDataSize(multiLoopAllRankSplitData[loop], dataTypeSize_);
    const std::vector<u64> wholeDataSize = CalcCountToDataSize(allRankSplitData, dataTypeSize_);

    // Scatter：完整 user input -> 当前 loop 的紧凑 CCL buffer。
    sliceParam.dataSizePerLoop = curLoopDataSize;
    sliceParam.dataWholeSize = wholeDataSize;
    sliceParam.endpointAttrBw = endpointAttrBwAvgSC;
    omniPipeSliceInfoSC = CalcScatterOmniPipeSliceInfo(sliceParam, root);

    // AllGather：当前 loop 的紧凑 CCL buffer -> CCL buffer。
    sliceParam.dataWholeSize = curLoopDataSize;
    sliceParam.endpointAttrBw = endpointAttrBwAvgAG;
    omniPipeSliceInfoAG = CalcAGOmniPipeSliceInfo(sliceParam);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    InitTemplateParams(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, std::map<u32, TemplateResource>& tempResMap,
        std::map<u32, TemplateDataParams>& tempAlgParamMap)
{
    // Scatter 各级
    if (tempScatterLevel0_) {
        InitTemplateParamByLevel(
            OMNIPIPE_SC_LEVEL0, OMNIPIPE_LEVEL0, levelThreadsSC_, param, resCtx, tempResMap, tempAlgParamMap);
    }
    if (tempScatterLevel1_) {
        InitTemplateParamByLevel(
            OMNIPIPE_SC_LEVEL1, OMNIPIPE_LEVEL1, levelThreadsSC_, param, resCtx, tempResMap, tempAlgParamMap);
    }
    if (tempScatterLevel2_) {
        InitTemplateParamByLevel(
            OMNIPIPE_SC_LEVEL2, OMNIPIPE_LEVEL2, levelThreadsSC_, param, resCtx, tempResMap, tempAlgParamMap);
    }
    // AllGather 各级
    if (tempAgLevel0_) {
        InitTemplateParamByLevel(
            OMNIPIPE_AG_LEVEL0, OMNIPIPE_LEVEL0, levelThreadsAG_, param, resCtx, tempResMap, tempAlgParamMap);
    }
    if (tempAgLevel1_) {
        InitTemplateParamByLevel(
            OMNIPIPE_AG_LEVEL1, OMNIPIPE_LEVEL1, levelThreadsAG_, param, resCtx, tempResMap, tempAlgParamMap);
    }
    if (tempAgLevel2_) {
        InitTemplateParamByLevel(
            OMNIPIPE_AG_LEVEL2, OMNIPIPE_LEVEL2, levelThreadsAG_, param, resCtx, tempResMap, tempAlgParamMap);
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    RunScatterLevel2Step(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, u64 processedDataCount,
        OmniPipeSliceInfo& omniPipeSliceInfoSC, u32 stepZ, std::map<u32, TemplateDataParams>& tempAlgParamMap)
{
    if (rankSizeLevel2_ <= 1) {
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("rankSizeLevel2_ > 1");
    if (myRank_ == param.root) {
        CHK_RET(GenTempAlgParamsIn2HCCLBuff(
            tempAlgParamMap[OMNIPIPE_LEVEL2], omniPipeSliceInfoSC.dataSliceLevel2[stepZ], processedDataCount, resCtx,
            param));
    } else {
        CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
            tempAlgParamMap[OMNIPIPE_LEVEL2], omniPipeSliceInfoSC.dataSliceLevel2[stepZ], processedDataCount, resCtx,
            param));
    }

    CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel2SC_, ntfIdxCtrlToTempLevel2SC_));
    if (myRank_ == param.root || isSameZAxisAsRoot) {
        tempScatterLevel2_->SetDoTask(true);
    }
    if (stepZ > 0) {
        tempScatterLevel2_->SetDoTask(true);
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    RunScatterLevel01Steps(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, u64 processedDataCount,
        OmniPipeSliceInfo& omniPipeSliceInfoSC, const std::vector<double>& endpointAttrBwAvgSC, u32 stepZ,
        u32 level0StepCountSC, std::map<u32, TemplateResource>& tempResMap,
        std::map<u32, TemplateDataParams>& tempAlgParamMap)
{
    for (u32 stepXY = 0; stepXY < level0StepCountSC; stepXY++) {
        CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01SC_, ntfIdxCtrlToTempLevel01SC_));
        if (myRank_ == param.root) {
            CHK_RET(GenTempAlgParamsIn2HCCLBuff(
                tempAlgParamMap[OMNIPIPE_LEVEL0],
                omniPipeSliceInfoSC.dataSliceLevel0[stepZ * level0StepCountSC + stepXY], processedDataCount, resCtx,
                param));
            CHK_RET(GenTempAlgParamsIn2HCCLBuff(
                tempAlgParamMap[OMNIPIPE_LEVEL1],
                omniPipeSliceInfoSC.dataSliceLevel1[stepZ * level0StepCountSC + stepXY], processedDataCount, resCtx,
                param));
        } else {
            CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
                tempAlgParamMap[OMNIPIPE_LEVEL1],
                omniPipeSliceInfoSC.dataSliceLevel1[stepZ * level0StepCountSC + stepXY], processedDataCount, resCtx,
                param));
            CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
                tempAlgParamMap[OMNIPIPE_LEVEL0],
                omniPipeSliceInfoSC.dataSliceLevel0[stepZ * level0StepCountSC + stepXY], processedDataCount, resCtx,
                param));
        }
        const ScatterLevel01TaskParam taskParam{
            param.root, stepXY, level0StepCountSC, endpointAttrBwAvgSC[0] > endpointAttrBwAvgSC[1]};
        SetScatterLevel01Task(taskParam);

        if (rankSizeLevel0_ > 1) {
            HCCL_DEBUG("rankSizeLevel0_ > 1");
            tempScatterLevel0_->xyTotalRankSize_ = rankSizeLevel0_ * rankSizeLevel1_ * rankIdxLevel2_;
            CHK_RET(tempScatterLevel0_->KernelRun(
                param, tempAlgParamMap[OMNIPIPE_SC_LEVEL0], tempResMap[OMNIPIPE_SC_LEVEL0]));
        }
        if (rankSizeLevel1_ > 1) {
            HCCL_DEBUG("rankSizeLevel1_ > 1");
            CHK_RET(tempScatterLevel1_->KernelRun(
                param, tempAlgParamMap[OMNIPIPE_SC_LEVEL1], tempResMap[OMNIPIPE_SC_LEVEL1]));
        }
        CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01SC_, ntfIdxTempToCtrlLevel01SC_));
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
void InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::SetScatterLevel01Task(const ScatterLevel01TaskParam& taskParam)
{
    if (rankSizeLevel1_ > 1 && rankIdxLevel0_ == taskParam.root % rankSizeLevel0_) {
        tempScatterLevel1_->SetDoTask(true);
    }
    if (rankSizeLevel0_ > 1
        && rankIdxLevel1_ == taskParam.root % (rankSizeLevel0_ * rankSizeLevel1_) / rankSizeLevel0_) {
        tempScatterLevel0_->SetDoTask(true);
    }
    if (taskParam.step == taskParam.stepCount - 1) {
        HCCL_DEBUG("step[%u] is the last level0 step", taskParam.step);
        if (rankSizeLevel1_ > 1) {
            tempScatterLevel1_->SetDoTask(true);
        }
        if (rankSizeLevel0_ > 1) {
            tempScatterLevel0_->SetDoTask(true);
        }
    } else if (taskParam.step != 0) {
        HCCL_DEBUG("step[%u] is not the first level0 step", taskParam.step);
        if (taskParam.preferLevel1 && rankSizeLevel1_ > 1) {
            tempScatterLevel1_->SetDoTask(true);
        } else if (!taskParam.preferLevel1 && rankSizeLevel0_ > 1) {
            tempScatterLevel0_->SetDoTask(true);
        }
    }
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
void InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::ResetScatterLevel1State()
{
    if (rankSizeLevel1_ > 1) {
        tempScatterLevel1_->SetDoTask(false);
    }
    if (rankSizeLevel0_ > 1) {
        tempScatterLevel0_->SetDoTask(false);
    }
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    RunScatterStage(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, u64 processedDataCount,
        OmniPipeSliceInfo& omniPipeSliceInfoSC, const std::vector<double>& endpointAttrBwAvgSC,
        std::map<u32, TemplateResource>& tempResMap, std::map<u32, TemplateDataParams>& tempAlgParamMap)
{
    u32 level2StepCountSC = omniPipeSliceInfoSC.dataSliceLevel2.size();
    u32 level0StepCountSC = level2StepCountSC > 0 ? omniPipeSliceInfoSC.dataSliceLevel0.size() / level2StepCountSC : 0;
    HCCL_INFO(
        "[InsV2BroadcastOmniPipeExecutor][SC] level2StepCountSC[%u], level0StepCountSC[%u]", level2StepCountSC,
        level0StepCountSC);

    for (u32 stepZ = 0; stepZ < level2StepCountSC; stepZ++) {
        CHK_RET(RunScatterLevel2Step(param, resCtx, processedDataCount, omniPipeSliceInfoSC, stepZ, tempAlgParamMap));
        CHK_RET(RunScatterLevel01Steps(
            param, resCtx, processedDataCount, omniPipeSliceInfoSC, endpointAttrBwAvgSC, stepZ, level0StepCountSC,
            tempResMap, tempAlgParamMap));
        ResetScatterLevel1State();
        if (rankSizeLevel2_ > 1) {
            CHK_RET(tempScatterLevel2_->KernelRun(
                param, tempAlgParamMap[OMNIPIPE_SC_LEVEL2], tempResMap[OMNIPIPE_SC_LEVEL2]));
            CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel2SC_, ntfIdxTempToCtrlLevel2SC_));
            HCCL_DEBUG("PostSyncInterThreads z success.");
        }
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    AdaptRootDataForAllGather(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, u64 processedDataCount,
        const std::vector<u64>& allRankSplitData, const std::vector<u64>& curLoopAllRankSplitData)
{
    CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01SC_, ntfIdxCtrlToTempLevel01SC_));
    if (myRank_ == param.root) {
        u64 rootRankOffset = 0;
        for (u32 rank = 0; rank < param.root; ++rank) {
            rootRankOffset += allRankSplitData[rank];
        }
        u64 srcOff = (rootRankOffset + processedDataCount) * dataTypeSize_;
        u64 rootCclOffset = 0;
        for (u32 rank = 0; rank < param.root; ++rank) {
            rootCclOffset += curLoopAllRankSplitData[rank];
        }
        u64 dstOff = rootCclOffset * dataTypeSize_;
        u64 copyCount = curLoopAllRankSplitData[param.root];
        u64 copySize = copyCount * dataTypeSize_;

        DataSlice srcSlice(param.inputPtr, srcOff, copySize, copyCount);
        DataSlice dstSlice(resCtx.cclMem.addr, dstOff, copySize, copyCount);
        HCCL_DEBUG(
            "[%s] Root adapt localcopy, myRank[%u], srcOff %llu, dstOff %llu, count %llu", __func__, myRank_, srcOff,
            dstOff, copyCount);
        CHK_RET(LocalCopy(controlThread_, srcSlice, dstSlice));
    }
    CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01SC_, ntfIdxTempToCtrlLevel01SC_));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    RunAllGatherStage(
        const OpParam& param, OmniPipeSliceInfo& omniPipeSliceInfoAG, std::map<u32, TemplateResource>& tempResMap,
        std::map<u32, TemplateDataParams>& tempAlgParamMap)
{
    u32 level2StepCountAG = omniPipeSliceInfoAG.dataSliceLevel2.size();
    u32 level0StepCountAG = level2StepCountAG > 0 ? omniPipeSliceInfoAG.dataSliceLevel0.size() / level2StepCountAG : 0;
    HCCL_DEBUG(
        "[InsV2BroadcastOmniPipeExecutor][AG] level2StepCountAG[%u], level0StepCountAG[%u]", level2StepCountAG,
        level0StepCountAG);
    for (u32 stepZ = 0; stepZ < level2StepCountAG; stepZ++) {
        if (rankSizeLevel2_ > 1) {
            CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel2AG_, ntfIdxCtrlToTempLevel2AG_));
            CHK_RET(GenTemplateAlgParamsByDimData(
                tempAlgParamMap[OMNIPIPE_AG_LEVEL2], omniPipeSliceInfoAG.dataSliceLevel2[stepZ]));
        }

        for (u32 stepXY = 0; stepXY < level0StepCountAG; stepXY++) {
            CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01AG_, ntfIdxCtrlToTempLevel01AG_));
            u32 idx = stepZ * level0StepCountAG + stepXY;
            if (rankSizeLevel0_ > 1) {
                CHK_RET(GenTemplateAlgParamsByDimData(
                    tempAlgParamMap[OMNIPIPE_AG_LEVEL0], omniPipeSliceInfoAG.dataSliceLevel0[idx]));
                CHK_RET(tempAgLevel0_->KernelRun(
                    param, tempAlgParamMap[OMNIPIPE_AG_LEVEL0], tempResMap[OMNIPIPE_AG_LEVEL0]));
            }
            if (rankSizeLevel1_ > 1) {
                CHK_RET(GenTemplateAlgParamsByDimData(
                    tempAlgParamMap[OMNIPIPE_AG_LEVEL1], omniPipeSliceInfoAG.dataSliceLevel1[idx]));
                CHK_RET(tempAgLevel1_->KernelRun(
                    param, tempAlgParamMap[OMNIPIPE_AG_LEVEL1], tempResMap[OMNIPIPE_AG_LEVEL1]));
            }
            CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01AG_, ntfIdxTempToCtrlLevel01AG_));
        }
        if (rankSizeLevel2_ > 1) {
            CHK_RET(
                tempAgLevel2_->KernelRun(param, tempAlgParamMap[OMNIPIPE_AG_LEVEL2], tempResMap[OMNIPIPE_AG_LEVEL2]));
            CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel2AG_, ntfIdxTempToCtrlLevel2AG_));
        }
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
void InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::ResetScatterState()
{
    if (rankSizeLevel2_ > 1) {
        tempScatterLevel2_->SetDoTask(false);
    }
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    CopyAllGatherResult(
        const OpParam& param, u64 processedDataCount, u64 currDataCount, const std::vector<u64>& allRankSplitData,
        const std::vector<u64>& curLoopAllRankSplitData, TemplateDataParams& tempParamLocalcopy)
{
    tempParamLocalcopy.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempParamLocalcopy.buffInfo.inBuffBaseOff = 0;
    tempParamLocalcopy.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    tempParamLocalcopy.buffInfo.outBuffType = BufferType::OUTPUT;
    tempParamLocalcopy.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempParamLocalcopy.repeatNum = rankSize_;
    tempParamLocalcopy.localCopyFlag = 1;

    HCCL_DEBUG(
        "[%s] Post-AG localcopy start, myRank[%u], currDataCount %llu, processedDataCount %llu, repeatNum %u", __func__,
        myRank_, currDataCount, processedDataCount, rankSize_);
    CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01AG_, ntfIdxCtrlToTempLevel01AG_));
    if (myRank_ != param.root) {
        CHK_RET(DoLocalCopy(tempParamLocalcopy, controlThread_, allRankSplitData, curLoopAllRankSplitData));
    }
    CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01AG_, ntfIdxTempToCtrlLevel01AG_));
    HCCL_DEBUG("[%s] Post-AG localcopy end", __func__);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
void InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    InitCommonTemplateParam(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateDataParams& tempAlgParamsCommon)
{
    InitTemplateBufferInfo(param, resCtx, tempAlgParamsCommon);
    tempAlgParamsCommon.buffInfo.hcclBuffSize = resCtx.cclMem.size;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
void InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY, InsAgAlgTemplateZ>::
    InitLocalCopyTemplateParam(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateDataParams& tempParamLocalcopy)
{
    InitTemplateBufferInfo(param, resCtx, tempParamLocalcopy);
    tempParamLocalcopy.dataType = dataType_;
}

template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2BroadcastOmniPipeExecutor<
    AlgTopoMatch, InsScatterAlgTemplateX, InsScatterAlgTemplateY, InsScatterAlgTemplateZ, InsAgAlgTemplateX,
    InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2BroadcastOmniPipeExecutor][OrchestrateLoop] Start");

    // 公共参数
    TemplateDataParams tempAlgParamsCommon;
    InitCommonTemplateParam(param, resCtx, tempAlgParamsCommon);

    std::map<u32, TemplateResource> tempResMap;
    std::map<u32, TemplateDataParams> tempAlgParamMap;
    CHK_RET(InitTemplateParams(param, resCtx, tempResMap, tempAlgParamMap));

    // 1. 带宽
    std::vector<double> endpointAttrBwAvgSC;
    std::vector<double> endpointAttrBwAvgAG;
    CHK_RET(CalcEndpointBandwidth(resCtx, endpointAttrBwAvgSC, endpointAttrBwAvgAG));

    // 2. 数据切分与loop次数
    LoopSplitData loopSplitData;
    CHK_RET(CalcLoopSplitData(maxTmpMemSize_, param.root, loopSplitData));
    const auto& allRankSplitData = loopSplitData.allRankSplitData;
    const auto& multiLoopAllRankSplitData = loopSplitData.multiLoopAllRankSplitData;
    u32 loopTimes = loopSplitData.loopTimes;

    // 3. slice 参数
    OmniPipeSliceParam sliceParam;
    CHK_RET(InitSliceParam(param, allRankSplitData, multiLoopAllRankSplitData, sliceParam));

    u64 processedDataCount = 0;
    OmniPipeSliceInfo omniPipeSliceInfoSC;
    OmniPipeSliceInfo omniPipeSliceInfoAG;
    HCCL_INFO("[InsV2BroadcastOmniPipeExecutor][OrchestrateLoop] loopTimes = [%u]", loopTimes);

    // 预置 localcopy 参数模板 (参考 allReduce 3-level tempParamLocalcopy)
    TemplateDataParams tempParamLocalcopy;
    InitLocalCopyTemplateParam(param, resCtx, tempParamLocalcopy);

    for (u64 loop = 0; loop < loopTimes; loop++) {
        CHK_PRT_RET(
            multiLoopAllRankSplitData.size() <= loop,
            HCCL_ERROR("[InsV2BroadcastOmniPipeExecutor][Orchestrate] multiLoopAllRankSplitData.size() <= loop"),
            HCCL_E_PARA);

        // 4.1 按需重算 SC/AG sliceInfo
        CHK_RET(PrepareSliceInfoForLoop(
            loop, param.root, allRankSplitData, multiLoopAllRankSplitData, endpointAttrBwAvgSC, endpointAttrBwAvgAG,
            sliceParam, omniPipeSliceInfoSC, omniPipeSliceInfoAG));

        const auto& curLoopAllRankSplitData = multiLoopAllRankSplitData[loop];
        u64 currDataCount = curLoopAllRankSplitData[myRank_];
        HCCL_DEBUG(
            "[%s] dataCount_ %llu, processedDataCount %llu, maxCountPerLoop %llu, currDataCount %llu", __func__,
            dataCount_, processedDataCount, loopSplitData.maxCountPerLoop, currDataCount);

        CHK_RET(RunScatterStage(
            param, resCtx, processedDataCount, omniPipeSliceInfoSC, endpointAttrBwAvgSC, tempResMap, tempAlgParamMap));
        CHK_RET(
            AdaptRootDataForAllGather(param, resCtx, processedDataCount, allRankSplitData, curLoopAllRankSplitData));
        // 复位 Scatter 状态
        ResetScatterState();

        CHK_RET(RunAllGatherStage(param, omniPipeSliceInfoAG, tempResMap, tempAlgParamMap));
        CHK_RET(CopyAllGatherResult(
            param, processedDataCount, currDataCount, allRankSplitData, curLoopAllRankSplitData, tempParamLocalcopy));
        processedDataCount += loopSplitData.maxCountPerLoop;
    }

    HCCL_INFO("[InsV2BroadcastOmniPipeExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_BROADCAST, DpuBroadcastOmniPipeMeshNHR, InsV2BroadcastOmniPipeExecutor, TopoMatchUBX,
    InsTempScatterOmniPipeMesh1D, InsTempScatterOmniPipeNHR, InsTempScatterOmniPipeNHRDpu,
    InsTempAllGatherOmniPipeMesh1D, InsTempAllGatherOmniPipeNHR, InsTempAllGatherOmniPipeNHRDPU);
} // namespace ops_hccl
