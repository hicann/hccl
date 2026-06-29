/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_reduce_omnipipe_executor.h"
#include "topo_match_3_level.h"
#include "ins_temp_reduce_scatter_omnipipe_mesh_1D.h"
#include "ins_temp_reduce_scatter_omnipipe_mesh_1d_dpu.h"
#include "ins_temp_reduce_scatter_omnipipe_nhr.h"
#include "ins_temp_all_gather_omnipipe_mesh_1D.h"
#include "ins_temp_all_gather_omnipipe_nhr_dpu.h"
#include "ins_temp_all_gather_omnipipe_nhr.h"
#include "omnipipe_data_slice_calc.h"
#include <cmath>

namespace ops_hccl {
constexpr u32 ALG_HIERARCHY_NUM3 = 3;
constexpr uint64_t RANK_SIZE_LEVEL1_2 = 2;
constexpr uint64_t RANK_SIZE_LEVEL1_4 = 4;
template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
InsV2AllReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ, InsAgAlgTemplateX,
                               InsAgAlgTemplateY, InsAgAlgTemplateZ>::InsV2AllReduceOmniPipeExecutor()
{
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<
    AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ, InsAgAlgTemplateX, InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::InitCommInfo(HcclComm comm, const OpParam& param, TopoInfoWithNetLayerDetails* topoInfo,
                                     AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void) comm;
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    algHierarchyInfo_ = algHierarchyInfo;
    return HCCL_SUCCESS;
}

// 实例化实际执行以来AutoMatchMeshNhr这个类的实现
template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<
    AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ, InsAgAlgTemplateX, InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                             AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;

    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ,
                                          InsAgAlgTemplateX, InsAgAlgTemplateY,
                                          InsAgAlgTemplateZ>::CalcResLevel(HcclComm comm, const OpParam& param,
                                                                           const TopoInfoWithNetLayerDetails* topoInfo,
                                                                           const std::shared_ptr<InsAlgTemplateBase> tempAlg,
                                                                           AlgResourceRequest& resourceRequest,
                                                                           bool addChannel) const
{
    AlgResourceRequest resReqlevel;
    CHK_RET(tempAlg->CalcRes(comm, param, topoInfo, resReqlevel));
    resourceRequest.slaveThreadNum += resReqlevel.slaveThreadNum + 1;
    resourceRequest.notifyNumOnMainThread += 1;
    resourceRequest.notifyNumPerThread.emplace_back(resReqlevel.notifyNumOnMainThread +
                                                    1);  // temp2控制流：从流数量+主控制流
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReqlevel.notifyNumPerThread.begin(),
                                              resReqlevel.notifyNumPerThread.end());

    if (addChannel)
        resourceRequest.channels.emplace_back(resReqlevel.channels[0]);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<
    AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ, InsAgAlgTemplateX, InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
                                AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    algHierarchyInfo_ = algHierarchyInfo;

    if (algHierarchyInfo_.infos.size() == ALG_HIERARCHY_NUM3 &&
        !algHierarchyInfo_.infos[2].empty() && !algHierarchyInfo_.infos[2][0].empty()) {
        topoType_ = TopoType::THREE_LEVEL;
    } else {
        topoType_ = TopoType::UBX_2LEVEL;
    }

    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;
    CHK_RET(BuildSubCommAndTempMap(param, algHierarchyInfo,
            subCommRanks0, subCommRanks1, subCommRanks2, tempMap, topoInfo));

    HCCL_DEBUG("[InsV2AllReduceOmniPipeExecutor] L0[%u], L1[%u], L2[%u]", rankSizeLevel0_, rankSizeLevel1_,
               rankSizeLevel2_);

    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    for (int level = 0; level < OMNIPIPE_AR_LEVEL_NUM; level++) {
        if (tempMap.count(level) > 0) {
            CHK_RET(CalcResLevel(comm, param, topoInfo, tempMap[level], resourceRequest,
                                 level < OMNIPIPE_AG_LEVEL0 ? true : false));
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ,
                                          InsAgAlgTemplateX, InsAgAlgTemplateY,
                                          InsAgAlgTemplateZ>::Orchestrate(const OpParam& param,
                                                                          const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceOmniPipeExecutor][Orchestrate] Orchestrate Start");
    // 参数填充

    CHK_RET(InitExectorInfo(param, resCtx));

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2AllReduceOmniPipeExecutor][Orchestrate]errNo[0x%016llx] Reduce scatter excutor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    HCCL_INFO("[InsV2AllReduceOmniPipeExecutor][Orchestrate] Orchestrate END");
    return HCCL_SUCCESS;
}

// ! 已完成编码
template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ,
                                          InsAgAlgTemplateX, InsAgAlgTemplateY,
                                          InsAgAlgTemplateZ>::InitExectorInfo(const OpParam& param,
                                                                              const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    threads_ = resCtx.threads;

    if (algHierarchyInfo_.infos.size() == ALG_HIERARCHY_NUM3 &&
        !algHierarchyInfo_.infos[2].empty() && !algHierarchyInfo_.infos[2][0].empty()) {
        topoType_ = TopoType::THREE_LEVEL;
    } else {
        topoType_ = TopoType::UBX_2LEVEL;
    }

    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;
    CHK_RET(BuildSubCommAndTempMap(param, algHierarchyInfo_,
            subCommRanks0, subCommRanks1, subCommRanks2, tempMap, &(resCtx.topoInfo)));
    return HCCL_SUCCESS;
}

// 将计算出的单步slice信息初始化到templateParam中
template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<
    AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ, InsAgAlgTemplateX, InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::GenTemplateAlgParamsByDimData(TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo) const
{
    // rs特殊处理，过程中的所有step都在ccl中进行数据搬运，在template中只使用ccl的起始地址就可以了，in和out不用赋值
    tempAlgParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;

    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.buffInfo.hcclBuffBaseOff = stepSliceInfo.buffInfo.hcclBuffBaseOff;

    tempAlgParams.stepSliceInfo = stepSliceInfo;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<
    AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ, InsAgAlgTemplateX, InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::PrepareResForTemplateLevelRS(u32 level, std::shared_ptr<InsAlgTemplateBase>& tempBase)
{
    u64 levelThreadNum = tempBase->GetThreadNum();
    if (level == OMNIPIPE_LEVEL0) {
        levelThreadsRS_[OMNIPIPE_LEVEL0].assign(threads_.begin() + 1, threads_.begin() + 1 + levelThreadNum);
        tempMainThreadsLevel01RS_.push_back(levelThreadsRS_[0].at(0));
    } else if (level == OMNIPIPE_LEVEL1) {
        levelThreadsRS_[OMNIPIPE_LEVEL1].assign(threads_.begin() + 1 + levelThreadsRS_[0].size(),
                                                threads_.begin() + 1 + levelThreadsRS_[0].size() + levelThreadNum);
        tempMainThreadsLevel01RS_.push_back(levelThreadsRS_[1].at(0));
    } else if (level == OMNIPIPE_LEVEL2) {
        levelThreadsRS_[OMNIPIPE_LEVEL2].assign(
            threads_.begin() + 1 + levelThreadsRS_[OMNIPIPE_LEVEL0].size() + levelThreadsRS_[OMNIPIPE_LEVEL1].size(),
            threads_.begin() + 1 + levelThreadsRS_[OMNIPIPE_LEVEL0].size() + levelThreadsRS_[OMNIPIPE_LEVEL1].size() +
                levelThreadNum);
        tempMainThreadsLevel2RS_.push_back(levelThreadsRS_[OMNIPIPE_LEVEL2].at(0));
    }

    AlgResourceRequest levelTempRequest;
    CHK_RET(tempBase->GetRes(levelTempRequest));
    if (level < OMNIPIPE_LEVEL2) {
        ntfIdxCtrlToTempLevel01RS_.push_back(levelTempRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlLevel01RS_.push_back(tempMainThreadsLevel01RS_.size() + tempMainThreadsLevel2RS_.size() - 1);
    } else {
        ntfIdxCtrlToTempLevel2RS_.push_back(levelTempRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlLevel2RS_.push_back(tempMainThreadsLevel01RS_.size() + tempMainThreadsLevel2RS_.size() - 1);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<
    AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ, InsAgAlgTemplateX, InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::PrepareResForTemplateLevelAG(u32 level, std::shared_ptr<InsAlgTemplateBase>& tempBase)
{
    u64 levelThreadNum = tempBase->GetThreadNum();
    u64 ThreadsNumStart = levelThreadsRS_[OMNIPIPE_LEVEL0].size() + levelThreadsRS_[OMNIPIPE_LEVEL1].size() +
                          levelThreadsRS_[OMNIPIPE_LEVEL2].size();
    if (level == OMNIPIPE_LEVEL0) {
        levelThreadsAG_[OMNIPIPE_LEVEL0].assign(threads_.begin() + ThreadsNumStart + 1,
                                                threads_.begin() + ThreadsNumStart + 1 + levelThreadNum);
        tempMainThreadsLevel01AG_.push_back(levelThreadsAG_[0].at(0));
    } else if (level == OMNIPIPE_LEVEL1) {
        levelThreadsAG_[OMNIPIPE_LEVEL1].assign(
            threads_.begin() + ThreadsNumStart + 1 + levelThreadsAG_[0].size(),
            threads_.begin() + ThreadsNumStart + 1 + levelThreadsAG_[0].size() + levelThreadNum);
        tempMainThreadsLevel01AG_.push_back(levelThreadsAG_[1].at(0));
    } else if (level == OMNIPIPE_LEVEL2) {
        levelThreadsAG_[OMNIPIPE_LEVEL2].assign(
            threads_.begin() + ThreadsNumStart + 1 + levelThreadsAG_[0].size() + levelThreadsAG_[1].size(),
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

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<
    AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ, InsAgAlgTemplateX, InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::RestoreChannelMap(const AlgResourceCtxSerializable& resCtx,
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

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<
    AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ, InsAgAlgTemplateX, InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::InitOmniPipeScratchParam(OmniPipeScratchParam& scratchParam, const OpParam& param,
                                                 const std::vector<double>& endpointAttrBwNew,
                                                 std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap) const
{
    std::vector<u64> levelRankSizeVec = {rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_};
    std::vector<u64> levelRankIdVec = {rankIdxLevel0_, rankIdxLevel1_, rankIdxLevel2_};
    scratchParam.levelRankSize = levelRankSizeVec;
    scratchParam.endpointAttrBw = endpointAttrBwNew;
    std::vector<u64> levelAlgType;

    (tempMap.count(OMNIPIPE_RS_LEVEL0) > 0) ? levelAlgType.push_back(tempMap[OMNIPIPE_RS_LEVEL0]->CalcScratchMultiple(
                                                  BufferType::DEFAULT, BufferType::DEFAULT)) :
                                              levelAlgType.push_back(0);

    (tempMap.count(OMNIPIPE_RS_LEVEL1) > 0) ? levelAlgType.push_back(tempMap[OMNIPIPE_RS_LEVEL1]->CalcScratchMultiple(
                                                  BufferType::DEFAULT, BufferType::DEFAULT)) :
                                              levelAlgType.push_back(0);

    (tempMap.count(OMNIPIPE_RS_LEVEL2) > 0) ? levelAlgType.push_back(tempMap[OMNIPIPE_RS_LEVEL2]->CalcScratchMultiple(
                                                  BufferType::DEFAULT, BufferType::DEFAULT)) :
                                              levelAlgType.push_back(0);

    // scratchParam.dataSize在外部赋值
    scratchParam.levelAlgType = levelAlgType;
    scratchParam.dataTypeSize = dataTypeSize_;
    scratchParam.opMode = param.opMode;
    scratchParam.engine = param.engine;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<
    AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ, InsAgAlgTemplateX, InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::InitOmniPipeSliceParam(OmniPipeSliceParam& sliceParam, const OpParam& param,
                                               const std::vector<double>& endpointAttrBwNew,
                                               std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap,
                                               u64 maxCountPerLoop) const
{
    std::vector<u64> levelRankSizeVec = {rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_};
    std::vector<u64> levelRankIdVec = {rankIdxLevel0_, rankIdxLevel1_, rankIdxLevel2_};
    std::vector<u64> levelAlgType;

    (tempMap.count(OMNIPIPE_RS_LEVEL0) > 0) ? levelAlgType.push_back(tempMap[OMNIPIPE_RS_LEVEL0]->CalcScratchMultiple(
                                                  BufferType::DEFAULT, BufferType::DEFAULT)) :
                                              levelAlgType.push_back(0);

    (tempMap.count(OMNIPIPE_RS_LEVEL1) > 0) ? levelAlgType.push_back(tempMap[OMNIPIPE_RS_LEVEL1]->CalcScratchMultiple(
                                                  BufferType::DEFAULT, BufferType::DEFAULT)) :
                                              levelAlgType.push_back(0);

    (tempMap.count(OMNIPIPE_RS_LEVEL2) > 0) ? levelAlgType.push_back(tempMap[OMNIPIPE_RS_LEVEL2]->CalcScratchMultiple(
                                                  BufferType::DEFAULT, BufferType::DEFAULT)) :
                                              levelAlgType.push_back(0);

    //sliceParam.dataSizePerLoop\ sliceParam.dataWholeSize 在外部赋值
    sliceParam.endpointAttrBw = endpointAttrBwNew;
    sliceParam.levelRankSize = levelRankSizeVec;
    sliceParam.levelRankId = levelRankIdVec;
    sliceParam.levelAlgType = levelAlgType;
    sliceParam.dataTypeSize = dataTypeSize_;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = param.engine;
    sliceParam.needSetStepNum = omniNeedSetStepNum_;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<
    AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ, InsAgAlgTemplateX, InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::InitTemplate(const OpParam& param, std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap,
                                     const std::vector<std::vector<u32>>& subCommRanks0,
                                     const std::vector<std::vector<u32>>& subCommRanks1,
                                     const std::vector<std::vector<u32>>& subCommRanks2)
{
    if (rankSizeLevel0_ > 1) {
        tempMap[OMNIPIPE_RS_LEVEL0] = std::make_shared<InsRsAlgTemplateX>(param, myRank_, subCommRanks0);
        tempMap[OMNIPIPE_AG_LEVEL0] = std::make_shared<InsAgAlgTemplateX>(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel1_ > 1) {
        tempMap[OMNIPIPE_RS_LEVEL1] = std::make_shared<InsRsAlgTemplateY>(param, myRank_, subCommRanks1);
        tempMap[OMNIPIPE_AG_LEVEL1] = std::make_shared<InsAgAlgTemplateY>(param, myRank_, subCommRanks1);
    }
    if (rankSizeLevel2_ > 1) {
        tempMap[OMNIPIPE_RS_LEVEL2] = std::make_shared<InsRsAlgTemplateZ>(param, myRank_, subCommRanks2);
        tempMap[OMNIPIPE_AG_LEVEL2] = std::make_shared<InsAgAlgTemplateZ>(param, myRank_, subCommRanks2);
    }

    levelThreadsRS_.resize(OMNIPIPE_LEVEL_NUM);
    levelThreadsAG_.resize(OMNIPIPE_LEVEL_NUM);

    HCCL_DEBUG("[InsV2AllReduceOmniPipeExecutor][InitTemplate] tempMap.size()[%u]", tempMap.size());
    controlThread_ = threads_.at(0);

    for (int level = 0; level < OMNIPIPE_AR_LEVEL_NUM; level++) {
        if (tempMap.count(level) > 0) {
            if (level < OMNIPIPE_AG_LEVEL0) {
                CHK_RET(PrepareResForTemplateLevelRS(level, tempMap[level]));
            } else {
                CHK_RET(PrepareResForTemplateLevelAG(level - OMNIPIPE_AG_LEVEL0, tempMap[level]));
            }
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<
    AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ, InsAgAlgTemplateX, InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::InitTemplateParams(const OpParam& param, const AlgResourceCtxSerializable& resCtx,
                                           const std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap,
                                           std::map<u32, TemplateResource>& tempResMap,
                                           std::map<u32, TemplateDataParams>& tempAlgParamMap)
{
    for (int level = 0; level < OMNIPIPE_AR_LEVEL_NUM; level++) {
        if (tempMap.count(level) > 0) {
            if (level < OMNIPIPE_AG_LEVEL0) {
                // [RS-level0, RS-level2]
                tempResMap[level].threads = levelThreadsRS_[level];
                tempResMap[level].channels = remoteRankToChannelInfo_[level];
            } else {
                // [AG-level0, AG-level2]
                tempResMap[level].threads = levelThreadsAG_[level - OMNIPIPE_AG_LEVEL0];
                tempResMap[level].channels = remoteRankToChannelInfo_[level - OMNIPIPE_AG_LEVEL0];
            }
            tempResMap[level].npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
            tempResMap[level].dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;

            tempAlgParamMap[level].buffInfo.inputPtr = param.inputPtr;
            tempAlgParamMap[level].buffInfo.outputPtr = param.outputPtr;
            tempAlgParamMap[level].buffInfo.hcclBuff = resCtx.cclMem;
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<
    AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ, InsAgAlgTemplateX, InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::InitSubCommRanks(std::vector<std::vector<u32>>& subCommRanks0,
                                         std::vector<std::vector<u32>>& subCommRanks1,
                                         std::vector<std::vector<u32>>& subCommRanks2,
                                         const TopoInfoWithNetLayerDetails* topoInfo)
{
    subCommRanks0.clear();
    subCommRanks1.clear();
    subCommRanks2.clear();

    if(topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        subCommRanks0 = {algHierarchyInfo_.infos[0][0]};
        std::vector<u32> closRanks;
        u32 meshSize = algHierarchyInfo_.infos[0][0].size();
        for(auto rank : algHierarchyInfo_.infos[0][1]) {
            if(rank % meshSize == topoInfo->userRank % meshSize) {
                closRanks.push_back(rank);
            }
        }
        subCommRanks1 = {closRanks};
        omniNeedSetStepNum_ = (subCommRanks1[0].size() == 4) ? OmniNeedSetStepNum::OMNIPIPE_UBX_16P
                                                             : OmniNeedSetStepNum::OMNIPIPE_DEFAULT;
        subCommRanks2.emplace_back(std::vector<u32>{myRank_});
    } else {
        subCommRanks0 = algHierarchyInfo_.infos[0];
        subCommRanks1 = algHierarchyInfo_.infos[1];
        subCommRanks2.emplace_back(std::vector<u32>{myRank_});
    }
    HCCL_INFO("[InsV2AllReduceOmniPipeExecutor]algHierarchyInfo_.info[%s]", ThreeDVecToStrOmni(algHierarchyInfo_.infos).c_str());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<
    AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ, InsAgAlgTemplateX, InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::ClacOmniBandwidthInSever(const AlgResourceCtxSerializable &resCtx, std::vector<double>& bdvec)
{
    bdvec.clear();
    double bw_ag_l0 = BW_OMNI_DEFAULT;
    double bw_ag_l1 = BW_OMNI_DEFAULT;
    double bw_ag_l2 = BW_OMNI_DEFAULT;
    double bw_rs_l0 = BW_OMNI_DEFAULT;
    double bw_rs_l1 = BW_OMNI_DEFAULT;
    double bw_rs_l2 = BW_OMNI_DEFAULT;

    if (resCtx.topoInfo.level0PcieMix) {
        if (rankSizeLevel1_ == RANK_SIZE_LEVEL1_2) {
            bw_ag_l1 = BW_OMNI_PCIE_EIGHT_AG_CLOS;
            bw_rs_l1 = BW_OMNI_PCIE_EIGHT_RS_CLOS;
        } else if (rankSizeLevel1_ == RANK_SIZE_LEVEL1_4) {
            bw_ag_l1 = BW_OMNI_PCIE_SIXTEEN_AG_CLOS;
            bw_rs_l1 = BW_OMNI_PCIE_SIXTEEN_RS_CLOS;
        }
    }
    bdvec = {bw_ag_l0, bw_ag_l1, bw_ag_l2, bw_rs_l0, bw_rs_l1, bw_rs_l2};
    HCCL_INFO("[ClacOmniBandwidthInSever]{bw_ag_l0[%f], bw_ag_l1[%f], bw_ag_l2[%f], bw_rs_l0[%f], bw_rs_l1[%f], "
              "bw_rs_l2[%f]}",
              bw_ag_l0, bw_ag_l1, bw_ag_l2, bw_rs_l0, bw_rs_l1, bw_rs_l2);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ,
                                          InsAgAlgTemplateX, InsAgAlgTemplateY,
                                          InsAgAlgTemplateZ>::BuildSubCommAndTempMap(
    const OpParam& param,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    std::vector<std::vector<u32>>& subCommRanks0,
    std::vector<std::vector<u32>>& subCommRanks1,
    std::vector<std::vector<u32>>& subCommRanks2,
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    subCommRanks0.clear();
    subCommRanks1.clear();
    subCommRanks2.clear();
    tempMap.clear();

    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        std::vector<u32> closRanks;
        if (!algHierarchyInfo_.infos[0].empty() && !algHierarchyInfo_.infos[0][0].empty()) {
            subCommRanks0 = {algHierarchyInfo_.infos[0][0]};
            u32 meshSize = algHierarchyInfo_.infos[0][0].size();
            if (!algHierarchyInfo_.infos[0][1].empty()) {
                for (auto rank : algHierarchyInfo_.infos[0][1]) {
                    if (rank % meshSize == topoInfo->userRank % meshSize) {
                        closRanks.push_back(rank);
                    }
                }
            }
        }
        subCommRanks1 = {closRanks};
        omniNeedSetStepNum_ = (subCommRanks1[0].size() == 4) ? OmniNeedSetStepNum::OMNIPIPE_UBX_16P
                                                        : OmniNeedSetStepNum::OMNIPIPE_DEFAULT;
        if (!algHierarchyInfo_.infos[1].empty()) {
            subCommRanks2 = algHierarchyInfo_.infos[1];
        } else {
            subCommRanks2.emplace_back(std::vector<u32>{myRank_});
        }
    } else if (topoType_ == TopoType::THREE_LEVEL) {
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
    } else {
        if (!algHierarchyInfo_.infos[0].empty()) {
            subCommRanks0 = algHierarchyInfo_.infos[0];
        }
        if (!algHierarchyInfo_.infos[1].empty()) {
            subCommRanks1 = algHierarchyInfo_.infos[1];
        }
        subCommRanks2.emplace_back(std::vector<u32>{myRank_});
    }

    rankSizeLevel0_ = subCommRanks0[0].size();
    rankSizeLevel1_ = subCommRanks1[0].size();
    rankSizeLevel2_ = subCommRanks2[0].size();

    uint32_t intraSuperpodDeviceNum = rankSizeLevel0_ * rankSizeLevel1_;
    rankIdxLevel0_ = (myRank_ % intraSuperpodDeviceNum) % rankSizeLevel0_;
    rankIdxLevel1_ = (myRank_ % intraSuperpodDeviceNum) / rankSizeLevel0_;
    rankIdxLevel2_ = myRank_ / intraSuperpodDeviceNum;

    if (rankSizeLevel0_ > 1) {
        tempMap[OMNIPIPE_RS_LEVEL0] = std::make_shared<InsRsAlgTemplateX>(param, myRank_, subCommRanks0);
        tempMap[OMNIPIPE_AG_LEVEL0] = std::make_shared<InsAgAlgTemplateX>(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel1_ > 1) {
        tempMap[OMNIPIPE_RS_LEVEL1] = std::make_shared<InsRsAlgTemplateY>(param, myRank_, subCommRanks1);
        tempMap[OMNIPIPE_AG_LEVEL1] = std::make_shared<InsAgAlgTemplateY>(param, myRank_, subCommRanks1);
    }
    if (rankSizeLevel2_ > 1) {
        tempMap[OMNIPIPE_RS_LEVEL2] = std::make_shared<InsRsAlgTemplateZ>(param, myRank_, subCommRanks2);
        tempMap[OMNIPIPE_AG_LEVEL2] = std::make_shared<InsAgAlgTemplateZ>(param, myRank_, subCommRanks2);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ,
                                          InsAgAlgTemplateX, InsAgAlgTemplateY,
                                          InsAgAlgTemplateZ>::OrchestrateLoop(const OpParam& param,
                                                                              const AlgResourceCtxSerializable& resCtx)
{
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;

    // 初始化通信域和template
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;
    CHK_RET(BuildSubCommAndTempMap(param, algHierarchyInfo_,
            subCommRanks0, subCommRanks1, subCommRanks2, tempMap, &(resCtx.topoInfo)));

    if (rankSizeLevel1_ > 1) {
        tempMap[OMNIPIPE_RS_LEVEL1]->SetchannelsPerRank(remoteRankToChannelInfo_[1]);
        tempMap[OMNIPIPE_AG_LEVEL1]->SetchannelsPerRank(remoteRankToChannelInfo_[1]);
    }
    // 为temp分配thread
    levelThreadsRS_.resize(OMNIPIPE_LEVEL_NUM);
    levelThreadsAG_.resize(OMNIPIPE_LEVEL_NUM);
    controlThread_ = threads_.at(0);
    for (int level = 0; level < OMNIPIPE_AR_LEVEL_NUM; level++) {
        if (tempMap.count(level) > 0) {
            if (level < OMNIPIPE_AG_LEVEL0) {
                CHK_RET(PrepareResForTemplateLevelRS(level, tempMap[level]));
            } else {
                CHK_RET(PrepareResForTemplateLevelAG(level - OMNIPIPE_AG_LEVEL0, tempMap[level]));
            }
        }
    }

    // 初始化资源TemplateResource\TemplateDataParams
    std::map<u32, TemplateResource> tempResMap;
    std::map<u32, TemplateDataParams> tempAlgParamMap;
    CHK_RET(InitTemplateParams(param, resCtx, tempMap, tempResMap, tempAlgParamMap));


    double bw_ag_l0 = BW_OMNI_DEFAULT;
    double bw_ag_l1 = BW_OMNI_DEFAULT;
    double bw_ag_l2 = BW_OMNI_DEFAULT;
    double bw_rs_l0 = BW_OMNI_DEFAULT;
    double bw_rs_l1 = BW_OMNI_DEFAULT;
    double bw_rs_l2 = BW_OMNI_DEFAULT;

    if (resCtx.topoInfo.level0PcieMix) {
        if (rankSizeLevel1_ == 2) {
            bw_ag_l1 = BW_OMNI_PCIE_EIGHT_AG_CLOS;
            bw_rs_l1=BW_OMNI_PCIE_EIGHT_RS_CLOS;
        } else if (rankSizeLevel1_ == 4) {
            bw_ag_l1 = BW_OMNI_PCIE_SIXTEEN_AG_CLOS;
            bw_rs_l1 = BW_OMNI_PCIE_SIXTEEN_RS_CLOS;
        }
    }  else if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS) {
        bw_ag_l1 = BW_OMNI_UBX_AG_CLOS;
        bw_rs_l1 = BW_OMNI_UBX_RS_CLOS;
    }

    //计算等价带宽
    double eqBw0 = bw_ag_l0;  // L0 mesh
    double eqBw1 = bw_ag_l1;  // L1 NHR
    double eqBw2 = bw_ag_l2;  // L2 NHR

    //level0为mesh,等价mesh为其本身
    //level1为nhr
    //level2, ranksize = 1
    eqBw1 = rankSizeLevel1_ > 1 ? eqBw1 / (rankSizeLevel1_ - 1) : eqBw1;
    eqBw2 = rankSizeLevel2_ > 1 ? eqBw2 / (rankSizeLevel2_ - 1) : eqBw2;
    std::vector<double> endpointAttrBwAG{eqBw0, eqBw1, eqBw2};

    double eqBw3 = bw_rs_l0;
    double eqBw4 = bw_rs_l1;
    double eqBw5 = bw_rs_l2;
    eqBw4 = rankSizeLevel1_ > 1 ? eqBw4 / (rankSizeLevel1_ - 1) : eqBw4;
    eqBw5 = rankSizeLevel2_ > 1 ? eqBw5 / (rankSizeLevel2_ - 1) : eqBw5;
    std::vector<double> endpointAttrBwNew{eqBw3, eqBw4, eqBw5};

    // 2.1 计算scratch
    OmniPipeScratchParam scratchParam;
    CHK_RET(InitOmniPipeScratchParam(scratchParam, param, endpointAttrBwNew, tempMap));
    scratchParam.maxTmpMemSize = resCtx.cclMem.size;

    // 2.2 获取每个rank切分的数据量count
    auto allRankSplitData = OmniPipeSplitData(rankSize_, dataCount_, dataTypeSize_);

    // 2.3 将数据量切分count转化为dataSize，传给scratchParam
    scratchParam.dataSize = CalcCountToDataSize(allRankSplitData, dataTypeSize_);

    std::vector<u64> loopInfo = CalcOmniPipeScratchInfo(scratchParam);

    // 中转内存单次最多能够接受的output count，注意是count不是size
    u64 maxCountPerLoop = loopInfo[0];
    u64 loopTimes = loopInfo[1];

    HCCL_DEBUG("maxCountPerLoop[%u], loopTimes[%u]", maxCountPerLoop, loopTimes);

    // 2.4 获取每个rank，每个loop切分的数据量count
    auto multiLoopAllRankSplitData =
        OmniPipeSplitRankDataLoop(allRankSplitData, maxCountPerLoop, loopTimes, dataTypeSize_);

    // 3. 计算loop的slice信息
    OmniPipeSliceParam sliceParam;
    CHK_RET(InitOmniPipeSliceParam(sliceParam, param, endpointAttrBwNew, tempMap, maxCountPerLoop));

    u64 processedDataCount = 0;
    OmniPipeSliceInfo OmniPipeSliceInfoRS;
    OmniPipeSliceInfo OmniPipeSliceInfoAG;

    TemplateDataParams tempParamLocalcopy;
    tempParamLocalcopy.buffInfo.hcclBuff = resCtx.cclMem;
    tempParamLocalcopy.buffInfo.inputPtr = param.inputPtr;
    tempParamLocalcopy.buffInfo.outputPtr = param.outputPtr;

    // 进行一次loop的数据处理
    for (u64 loop = 0; loop < loopTimes; loop++) {
        CHK_PRT_RET(
            multiLoopAllRankSplitData.size() <= loop,
            HCCL_ERROR("[InsV2AllReduceOmniPipeExecutor][Orchestrate] multiLoopAllRankSplitData.size() <= loop"),
            HCCL_E_PARA);

        // 4.1首轮计算, 或者与上轮不同loop重新计算OmniPipeSliceInfoRS、OmniPipeSliceInfoAG
        if (loop == 0 || !isSameLoop(multiLoopAllRankSplitData[loop - 1], multiLoopAllRankSplitData[loop])) {
            sliceParam.dataSizePerLoop = CalcCountToDataSize(multiLoopAllRankSplitData[loop], dataTypeSize_);
            sliceParam.dataWholeSize = sliceParam.dataSizePerLoop;

            sliceParam.endpointAttrBw=endpointAttrBwNew;
            OmniPipeSliceInfoRS = CalcRSOmniPipeSliceInfo(sliceParam);
            sliceParam.endpointAttrBw=endpointAttrBwAG;
            OmniPipeSliceInfoAG = CalcAGOmniPipeSliceInfo(sliceParam);
        }

        u64 currDataCount = multiLoopAllRankSplitData[loop][myRank_];

        // 4.2 RS在每次loop进行之前先将所有数据从usrin拷贝到ccl
        // intput -> ccl
        // ccl : 从每个rank-intputmem 上面拿到的数据放到 ccl
        tempParamLocalcopy.buffInfo.inBuffType = BufferType::INPUT;
        tempParamLocalcopy.buffInfo.inBuffBaseOff =
            processedDataCount * dataTypeSize_;  //每轮loop对应每个rank的搬运起始地址
        tempParamLocalcopy.buffInfo.outBuffBaseOff = 0;
        tempParamLocalcopy.repeatNum = rankSize_;

        CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01RS_, ntfIdxCtrlToTempLevel01RS_));
        CHK_RET(DoLocalCopy(tempParamLocalcopy, controlThread_, allRankSplitData, multiLoopAllRankSplitData[loop]));
        CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01RS_, ntfIdxTempToCtrlLevel01RS_));

        u32 interPodStepNum = OmniPipeSliceInfoRS.dataSliceLevel2.size();
        u32 intraPodStepNum = OmniPipeSliceInfoRS.dataSliceLevel0.size() / OmniPipeSliceInfoRS.dataSliceLevel2.size();

        // 4.3 RS for循环2层
        for (int stepZ = 0; stepZ < interPodStepNum; stepZ++) {
            if (rankSizeLevel2_ > 1) {
                HCCL_INFO("rankSizeLevel2_ > 1-----RS-Z");
                GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_RS_LEVEL2],
                                              OmniPipeSliceInfoRS.dataSliceLevel2[stepZ]);
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel2RS_, ntfIdxCtrlToTempLevel2RS_));
                CHK_RET(tempMap[OMNIPIPE_RS_LEVEL2]->KernelRun(param, tempAlgParamMap[OMNIPIPE_RS_LEVEL2],
                                                               tempResMap[OMNIPIPE_RS_LEVEL2]));
            }

            for (int stepXY = 0; stepXY < intraPodStepNum; stepXY++) {
                // XY前同步
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01RS_, ntfIdxCtrlToTempLevel01RS_));
                if (rankSizeLevel0_ > 1) {
                    HCCL_INFO("rankSizeLevel0_ > 1-----RS-X");
                    CHK_RET(GenTemplateAlgParamsByDimData(
                        tempAlgParamMap[OMNIPIPE_RS_LEVEL0],
                        OmniPipeSliceInfoRS.dataSliceLevel0[stepZ * intraPodStepNum + stepXY]));
                    CHK_RET(tempMap[OMNIPIPE_RS_LEVEL0]->KernelRun(param, tempAlgParamMap[OMNIPIPE_RS_LEVEL0],
                                                                   tempResMap[OMNIPIPE_RS_LEVEL0]));
                }
                if (rankSizeLevel1_ > 1) {
                    HCCL_INFO("rankSizeLevel1_ > 1-----RS-Y");
                    CHK_RET(GenTemplateAlgParamsByDimData(
                        tempAlgParamMap[OMNIPIPE_RS_LEVEL1],
                        OmniPipeSliceInfoRS.dataSliceLevel1[stepZ * intraPodStepNum + stepXY]));
                    CHK_RET(tempMap[OMNIPIPE_RS_LEVEL1]->KernelRun(param, tempAlgParamMap[OMNIPIPE_RS_LEVEL1],
                                                                   tempResMap[OMNIPIPE_RS_LEVEL1]));
                }
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01RS_, ntfIdxTempToCtrlLevel01RS_));
            }
            if (rankSizeLevel2_ > 1) {
                // Z后同步
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel2RS_, ntfIdxTempToCtrlLevel2RS_));
                HCCL_INFO("PostSyncInterThreads z success.");
            }
        }

        interPodStepNum = OmniPipeSliceInfoAG.dataSliceLevel2.size();
        intraPodStepNum = OmniPipeSliceInfoAG.dataSliceLevel0.size() / OmniPipeSliceInfoAG.dataSliceLevel2.size();

        // 5.1 AG for循环2层
        for (int stepZ = 0; stepZ < interPodStepNum; stepZ++) {
            if (rankSizeLevel2_ > 1) {
                HCCL_INFO("rankSizeLevel2_ > 1-----AG-Z");
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel2AG_, ntfIdxCtrlToTempLevel2AG_));
                CHK_RET(GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_AG_LEVEL2],
                                                      OmniPipeSliceInfoAG.dataSliceLevel2[stepZ]));
                CHK_RET(tempMap[OMNIPIPE_AG_LEVEL2]->KernelRun(param, tempAlgParamMap[OMNIPIPE_AG_LEVEL2],
                                                               tempResMap[OMNIPIPE_AG_LEVEL2]));
            }

            for (int stepXY = 0; stepXY < intraPodStepNum; stepXY++) {
                // XY前同步
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01AG_, ntfIdxCtrlToTempLevel01AG_));
                if (rankSizeLevel0_ > 1) {
                    HCCL_INFO("rankSizeLevel0_ > 1-----AG-X");
                    GenTemplateAlgParamsByDimData(
                        tempAlgParamMap[OMNIPIPE_AG_LEVEL0],
                        OmniPipeSliceInfoAG.dataSliceLevel0[stepZ * intraPodStepNum + stepXY]);
                    CHK_RET(tempMap[OMNIPIPE_AG_LEVEL0]->KernelRun(param, tempAlgParamMap[OMNIPIPE_AG_LEVEL0],
                                                                   tempResMap[OMNIPIPE_AG_LEVEL0]));
                }
                if (rankSizeLevel1_ > 1) {
                    HCCL_INFO("rankSizeLevel1_ > 1-----AG-Y");
                    GenTemplateAlgParamsByDimData(
                        tempAlgParamMap[OMNIPIPE_AG_LEVEL1],
                        OmniPipeSliceInfoAG.dataSliceLevel1[stepZ * intraPodStepNum + stepXY]);
                    CHK_RET(tempMap[OMNIPIPE_AG_LEVEL1]->KernelRun(param, tempAlgParamMap[OMNIPIPE_AG_LEVEL1],
                                                                   tempResMap[OMNIPIPE_AG_LEVEL1]));
                }
                // XY后同步
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01AG_, ntfIdxTempToCtrlLevel01AG_));
            }
            // Z后同步
            if (rankSizeLevel2_ > 1) {
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel2AG_, ntfIdxTempToCtrlLevel2AG_));
            }
        }
        // --------------------------------
        // 5.2将当前这个loop在ccl中的数据一次性拷贝到userout中
        tempParamLocalcopy.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        tempParamLocalcopy.buffInfo.inBuffBaseOff = 0;
        tempParamLocalcopy.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempParamLocalcopy.repeatNum = rankSize_;

        CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01AG_, ntfIdxCtrlToTempLevel01AG_));
        CHK_RET(DoLocalCopy(tempParamLocalcopy, controlThread_, allRankSplitData, multiLoopAllRankSplitData[loop]));
        CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01AG_, ntfIdxTempToCtrlLevel01AG_));

        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2AllReduceOmniPipeExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
          typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
HcclResult InsV2AllReduceOmniPipeExecutor<
    AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, InsRsAlgTemplateZ, InsAgAlgTemplateX, InsAgAlgTemplateY,
    InsAgAlgTemplateZ>::DoLocalCopy(const TemplateDataParams& tempAlgParams, const ThreadHandle& thread,
                                    const std::vector<u64>& allRankSplitData,
                                    const std::vector<u64>& curLoopAllRankSplitData) const
{
    std::vector<DataSlice> srcDataSlice;
    std::vector<DataSlice> dstDataSlice;

    CHK_RET(CalLocalCopySlice(tempAlgParams, allRankSplitData, curLoopAllRankSplitData, srcDataSlice, dstDataSlice,
                              dataTypeSize_));

    CHK_PRT_RET(srcDataSlice.size() != dstDataSlice.size(),
                HCCL_ERROR("[InsV2AllReduceOmniPipeExecutor][DoLocalCopy] srcDataSlice.size != dstDataSlice.size"),
                HCCL_E_PARA);

    for (auto i = 0; i < srcDataSlice.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(LocalCopy(thread, srcDataSlice[i], dstDataSlice[i])));
    }
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLREDUCE, InsV2AllReduceOmniPipeMultilevel, InsV2AllReduceOmniPipeExecutor,
                       TopoMatchMultilevel, InsTempReduceScatterOmniPipeMesh1D, InsTempReduceScatterOmniPipeNHR,
                       InsTempReduceScatterOmniPipeMesh1dDpu, InsTempAllGatherOmniPipeMesh1D, InsTempAllGatherOmniPipeNHR,
                       InsTempAllGatherOmniPipeNHRDPU);
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLREDUCE, InsV2AllReduceOmniPipePcie, InsV2AllReduceOmniPipeExecutor,
                       TopoMatchPcieMix, InsTempReduceScatterOmniPipeMesh1D, InsTempReduceScatterOmniPipeNHR,
                       InsTempReduceScatterOmniPipeMesh1dDpu, InsTempAllGatherOmniPipeMesh1D, InsTempAllGatherOmniPipeNHR,
                       InsTempAllGatherOmniPipeNHRDPU);
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLREDUCE, InsV2AllReduceOmniPipe, InsV2AllReduceOmniPipeExecutor,
                       TopoMatchUBX, InsTempReduceScatterOmniPipeMesh1D, InsTempReduceScatterOmniPipeNHR,
                       InsTempReduceScatterOmniPipeMesh1dDpu, InsTempAllGatherOmniPipeMesh1D, InsTempAllGatherOmniPipeNHR,
                       InsTempAllGatherOmniPipeNHRDPU);
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLREDUCE, InsV2AllReduceOmniPipeUboe, InsV2AllReduceOmniPipeExecutor,
                       TopoMatch3Level, InsTempReduceScatterOmniPipeMesh1D, InsTempReduceScatterOmniPipeNHR,
                       InsTempReduceScatterOmniPipeMesh1D, InsTempAllGatherOmniPipeMesh1D, InsTempAllGatherOmniPipeNHR,
                       InsTempAllGatherOmniPipeNHR);

}  // namespace ops_hccl
