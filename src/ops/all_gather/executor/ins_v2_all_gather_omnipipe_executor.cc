/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_omnipipe_executor.h"
#include <algorithm>
#include <sstream>
#include "alg_data_trans_wrapper.h"
#include "alg_param.h"
#include "topo_match_ubx.h"
#include "topo_match_multilevel.h"
#include "topo_match_pcie_mix.h"
#include "ins_temp_all_gather_omnipipe_mesh_1D.h"
#include "ins_temp_all_gather_omnipipe_nhr_dpu.h"
#include "ins_temp_all_gather_omnipipe_nhr.h"
#include "topo_match_3_level.h"
#include "omnipipe_template_utils.h"

namespace ops_hccl {
constexpr u32 ALG_HIERARCHY_NUM3 = 3;
constexpr u32 RANK_LEVEL_2 = 2;
constexpr u32 RANK_LEVEL_4 = 4;
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                               InsAlgTemplate2>::InsV2AllGatherOmniPipeExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InitCommInfo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    opMode_ = param.opMode;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][InitCommInfo] myRank [%u], rankSize [%u], devType [%u], "
              "dataType [%u] dataTypeSize [%u]",
              myRank_, rankSize_, devType_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::BuildSubCommAndTempMap(
    const OpParam& param,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    std::vector<std::vector<u32>>& subCommRanks0,
    std::vector<std::vector<u32>>& subCommRanks1,
    std::vector<std::vector<u32>>& subCommRanks2,
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    HCCL_INFO("[BuildSubCommAndTempMap]infos,%s", ThreeDVecToStrOmni(algHierarchyInfo_.infos).c_str());
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        HCCL_INFO("[BuildSubCommAndTempMap] topoUBX");
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
        omniUbxLastStepRead_ = true;
        if (!algHierarchyInfo_.infos[1].empty()){
            subCommRanks2 = algHierarchyInfo_.infos[1];
        } else {
            subCommRanks2.emplace_back(std::vector<u32>{myRank_});
        }
    } else if(topoType_ == TopoType::THREE_LEVEL) {
        HCCL_INFO("[BuildSubCommAndTempMap] treeLevel");
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
    }
    else { 
        HCCL_INFO("[BuildSubCommAndTempMap] MutiLevel_Pcie");
        if (!algHierarchyInfo_.infos[0].empty()) {
            subCommRanks0 = algHierarchyInfo_.infos[0];
        }
        if (!algHierarchyInfo_.infos[1].empty()) {
            subCommRanks1 = algHierarchyInfo_.infos[1];
        }
        subCommRanks2.emplace_back(std::vector<u32>{myRank_});
    }
    rankSizeLevel_[OMNIPIPE_LEVEL0] = subCommRanks0[0].size();
    rankSizeLevel_[OMNIPIPE_LEVEL1] = subCommRanks1[0].size();
    rankSizeLevel_[OMNIPIPE_LEVEL2] = subCommRanks2[0].size();
    tempMap.clear();
    if (rankSizeLevel_[OMNIPIPE_LEVEL0] > 1) {
        tempMap[OMNIPIPE_LEVEL0] = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
        tempMap[OMNIPIPE_LEVEL1] = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1);
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
        tempMap[OMNIPIPE_LEVEL2] = std::make_shared<InsAlgTemplate2>(param, myRank_, subCommRanks2);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{ 
    // 初始化一些基本成员变量
    InitCommInfo(param, topoInfo, algHierarchyInfo);
     if (algHierarchyInfo_.infos.size() == ALG_HIERARCHY_NUM3 &&
 	        !algHierarchyInfo_.infos[2].empty() && !algHierarchyInfo_.infos[2][0].empty()) {
 	        topoType_ = TopoType::THREE_LEVEL;
 	    } else {
 	        topoType_ = TopoType::UBX_2LEVEL;
 	    }
    // 计算subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;    
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;
    rankSizeLevel_.resize(OMNIPIPE_LEVEL_NUM);
    rankIdxLevel_.resize(OMNIPIPE_LEVEL_NUM);
    
    CHK_RET(BuildSubCommAndTempMap(param, algHierarchyInfo,
            subCommRanks0, subCommRanks1, subCommRanks2, tempMap, topoInfo));

    rankIdxLevel_[OMNIPIPE_LEVEL0] = myRank_ % rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL1] = myRank_ % (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]) /
                                      rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL2] = myRank_ / (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]);
    
    for (auto& temp : tempMap) {
        CHK_RET(CalcResLevel(comm, param, topoInfo, temp.second, resourceRequest));
    }
    HCCL_INFO("[InInsV2AllGatherOmniPipeExecutor][CalcRes]");

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcResLevel(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    std::shared_ptr<InsAlgTemplateBase> tempAlg, AlgResourceRequest& resourceRequest) const
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
    resourceRequest.channels.emplace_back(resReqlevel.channels[0]);
    return HCCL_SUCCESS;
}

// 该函数必须按照level0、level1、level2的顺序调用
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    PrepareResForTemplateLevel(u32 level, std::shared_ptr<InsAlgTemplateBase>& tempBase)
{  
    u32 levelThreadNum = tempBase->GetThreadNum();
    if (level == OMNIPIPE_LEVEL0) {
        levelThreads_[OMNIPIPE_LEVEL0].assign(threads_.begin() + 1, threads_.begin() + 1 + levelThreadNum);
        tempMainThreadsXY_.push_back(levelThreads_[OMNIPIPE_LEVEL0].at(0));
    } else if (level == OMNIPIPE_LEVEL1) {
        levelThreads_[OMNIPIPE_LEVEL1].assign(threads_.begin() + 1 + levelThreads_[OMNIPIPE_LEVEL0].size(),
                                              threads_.begin() + 1 + levelThreads_[0].size() + levelThreadNum);
        tempMainThreadsXY_.push_back(levelThreads_[OMNIPIPE_LEVEL1].at(0));
    } else if (level == OMNIPIPE_LEVEL2) {
        levelThreads_[OMNIPIPE_LEVEL2].assign(
            threads_.begin() + 1 + levelThreads_[OMNIPIPE_LEVEL0].size() + levelThreads_[OMNIPIPE_LEVEL1].size(),
            threads_.end());
        tempMainThreadsZ_.push_back(levelThreads_[OMNIPIPE_LEVEL2].at(0));
    }

    // 获取当前template各自的主thread上有多少notify
    AlgResourceRequest levelTempRequest;
    CHK_RET(tempBase->GetRes(levelTempRequest));
    if (level < OMNIPIPE_LEVEL2) {
        ntfIdxCtrlToTempXY_.push_back(levelTempRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlXY_.push_back(tempMainThreadsXY_.size() + tempMainThreadsZ_.size() - 1);
    } else {
        ntfIdxCtrlToTempZ_.push_back(levelTempRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlZ_.push_back(tempMainThreadsXY_.size() + tempMainThreadsZ_.size() - 1);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    if (algHierarchyInfo_.infos.size() == ALG_HIERARCHY_NUM3 &&
        !algHierarchyInfo_.infos[2].empty() && !algHierarchyInfo_.infos[2][0].empty()) {
            topoType_ = TopoType::THREE_LEVEL;
 	    } else {
            topoType_ = TopoType::UBX_2LEVEL;
        }
    maxTmpMemSize_ = resCtx.cclMem.size;  // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn

    // 计算subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;    
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;

    rankSizeLevel_.resize(OMNIPIPE_LEVEL_NUM);
    rankIdxLevel_.resize(OMNIPIPE_LEVEL_NUM);

    CHK_RET(BuildSubCommAndTempMap(param, algHierarchyInfo_,
            subCommRanks0, subCommRanks1, subCommRanks2, tempMap, &resCtx.topoInfo));

    rankIdxLevel_[OMNIPIPE_LEVEL0] = myRank_ % rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL1] = myRank_ % (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]) /
                                      rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL2] = myRank_ / (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]);
    
    // 为temp分配thread
    threads_ = resCtx.threads;
    controlThread_ = threads_.at(0);
    levelThreads_.resize(OMNIPIPE_LEVEL_NUM);

    // 先初始化remoteRankToChannelInfo_，然后为nhr赋值多channel，最后再计算资源，这样计算线程资源的时候就能获取到多channel需要的线程数
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    // todo 这边写死了
    if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
        tempMap[OMNIPIPE_LEVEL1]->SetchannelsPerRank(remoteRankToChannelInfo_[1]);
    }
    for (auto& temp : tempMap) {
        CHK_RET(PrepareResForTemplateLevel(temp.first, temp.second));
    }
    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx, tempMap);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllGatherOmniPipeExecutor][Orchestrate][rank:%u] errNo[0x%016llx] AllGather excutor kernel run failed",
                   myRank_, HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                               InsAlgTemplate2>::GenTemplateAlgParamsByDimData(TemplateDataParams& tempAlgParams,
                                                                               StepSliceInfo& stepSliceInfo) const
{
    CHK_RET(FillOmniPipeTemplateAlgParams(tempAlgParams, stepSliceInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx,
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap)
{
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop] Start");
    //带宽赋值
    double bw_ag_l0 = BW_OMNI_DEFAULT;
    double bw_ag_l1 = BW_OMNI_DEFAULT;
    double bw_ag_l2 = BW_OMNI_DEFAULT;

    if (resCtx.topoInfo.level0PcieMix) {//PCIE
        if (rankSizeLevel_[OMNIPIPE_LEVEL1] == RANK_LEVEL_2) {
            bw_ag_l1 = BW_OMNI_PCIE_EIGHT_AG_CLOS;
        } else if (rankSizeLevel_[OMNIPIPE_LEVEL1] == RANK_LEVEL_4) {
            bw_ag_l1 = BW_OMNI_PCIE_SIXTEEN_AG_CLOS;
        }
        //UBX
    } else if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS) {
        bw_ag_l1 = BW_OMNI_UBX_AG_CLOS;
    }
    std::vector<double> endpointAttrBw{bw_ag_l0, bw_ag_l1, bw_ag_l2};
    
    //计算等价带宽
    double eqBw0 = endpointAttrBw[0];//L0 mesh
    double eqBw1 = endpointAttrBw[1];//L1 NHR
    double eqBw2 = endpointAttrBw[2];//L2 NHR

    HCCL_DEBUG("eqBw0[%f], eqBw1[%f], eqBw2[%f]", eqBw0, eqBw1, eqBw2);

    //level0为mesh,等价mesh为其本身
    //level1为nhr
    //level2, ranksize = 1
    eqBw1 = rankSizeLevel_[OMNIPIPE_LEVEL1] > 1 ? eqBw1 / (rankSizeLevel_[OMNIPIPE_LEVEL1] - 1) : eqBw1;
    eqBw2 = rankSizeLevel_[OMNIPIPE_LEVEL2] > 1 ? eqBw2 / (rankSizeLevel_[OMNIPIPE_LEVEL2] - 1) : eqBw2;

    std::vector<double> endpointAttrBwNew{eqBw0, eqBw1, eqBw2};
    u64 scratchBoundDataSize = maxTmpMemSize_ / rankSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 maxCountPerLoop = std::min(scratchBoundDataSize, transportBoundDataSize);
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);

    u64 perLoopSize = maxCountPerLoop * dataTypeSize_;
    std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
    std::vector<u64> dataWholeSize(rankSize_, perLoopSize);

    for (int i = 0; i < rankSize_; i++) {
        dataSizePerLoop.push_back(perLoopSize);
        dataWholeSize.push_back(perLoopSize);
    }

    OmniPipeSliceParam omniPipeSliceParam;
    omniPipeSliceParam.levelRankSize = {rankSizeLevel_[OMNIPIPE_LEVEL0], rankSizeLevel_[OMNIPIPE_LEVEL1],
                                        rankSizeLevel_[OMNIPIPE_LEVEL2]};
    omniPipeSliceParam.endpointAttrBw = endpointAttrBwNew;
    omniPipeSliceParam.dataSizePerLoop = dataSizePerLoop;
    omniPipeSliceParam.dataTypeSize = dataTypeSize_;
    omniPipeSliceParam.levelRankId = {rankIdxLevel_[OMNIPIPE_LEVEL0], rankIdxLevel_[OMNIPIPE_LEVEL1],
                                      rankIdxLevel_[OMNIPIPE_LEVEL2]};
    omniPipeSliceParam.opMode = opMode_;
    omniPipeSliceParam.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    omniPipeSliceParam.dataWholeSize = dataWholeSize;

    OmniPipeSliceInfo alignSliceInfo = CalcAGOmniPipeSliceInfo(omniPipeSliceParam);

    // localcopy使用
    OmniPipeSliceParam localcopySliceParam;
    localcopySliceParam.levelRankSize
        = {rankSizeLevel_[OMNIPIPE_LEVEL0], rankSizeLevel_[OMNIPIPE_LEVEL1], rankSizeLevel_[OMNIPIPE_LEVEL2]};
    localcopySliceParam.endpointAttrBw = endpointAttrBwNew;
    localcopySliceParam.dataSizePerLoop = dataSizePerLoop;
    localcopySliceParam.dataTypeSize = dataTypeSize_;
    localcopySliceParam.levelRankId
        = {rankIdxLevel_[OMNIPIPE_LEVEL0], rankIdxLevel_[OMNIPIPE_LEVEL1], rankIdxLevel_[OMNIPIPE_LEVEL2]};
    localcopySliceParam.opMode = opMode_;
    localcopySliceParam.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    std::vector<u64> dataWholeSizeLocalcopy(rankSize_, dataSize_);
    localcopySliceParam.dataWholeSize = dataWholeSizeLocalcopy; // 这里用整体数据量算一遍
    localcopySliceParam.needSetStepNum = omniNeedSetStepNum_;

    OmniPipeSliceInfo localcopySliceInfo = CalcAGOmniPipeSliceInfo(localcopySliceParam);

    // 4、计算第n次的loop的slice信息
    OmniPipeSliceInfo tailSliceInfo;
    OmniPipeSliceInfo localcopyTailSliceInfo;
    if (dataCount_ % maxCountPerLoop != 0) {
        u64 perLoopSize = (dataCount_ % maxCountPerLoop) * dataTypeSize_;
        std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
        std::vector<u64> dataWholeSize(rankSize_, perLoopSize);
        omniPipeSliceParam.dataSizePerLoop = dataSizePerLoop;
        omniPipeSliceParam.dataWholeSize = dataWholeSize;
        tailSliceInfo = CalcAGOmniPipeSliceInfo(omniPipeSliceParam);
        // // 尾块也一样
        localcopySliceParam.dataSizePerLoop = dataSizePerLoop;
        localcopySliceParam.dataWholeSize = dataWholeSizeLocalcopy;
        localcopyTailSliceInfo = CalcAGOmniPipeSliceInfo(localcopySliceParam);
    }

    u64 processedDataCount = 0;
    OmniPipeSliceInfo omniPipeSliceInfo;
    OmniPipeSliceInfo omniPipeSliceLocalcopyInfo;

    std::map<u32, TemplateResource> tempResMap;
    std::map<u32, TemplateDataParams> tempAlgParamMap;

    for (auto& temp : tempMap) {
        tempResMap[temp.first].channels = remoteRankToChannelInfo_[temp.first];
        tempResMap[temp.first].threads = levelThreads_[temp.first];
        tempAlgParamMap[temp.first].buffInfo.hcclBuff = resCtx.cclMem;
    }
    HCCL_DEBUG("loopTimes[%d]", loopTimes);
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        DataSlice src(param.inputPtr, processedDataCount * dataTypeSize_, currDataCount * dataTypeSize_, currDataCount);
        DataSlice dst(resCtx.cclMem.addr, myRank_ * currDataCount * dataTypeSize_, currDataCount * dataTypeSize_,
                        currDataCount);
        CHK_RET(LocalCopy(controlThread_, src, dst));

        if (loop == loopTimes - 1 && dataCount_ % maxCountPerLoop != 0) {
            omniPipeSliceInfo = tailSliceInfo;
            omniPipeSliceLocalcopyInfo = localcopyTailSliceInfo;
        } else {
            omniPipeSliceInfo = alignSliceInfo;
            omniPipeSliceLocalcopyInfo = localcopySliceInfo;
        }

        CHK_PRT_RET(omniPipeSliceInfo.dataSliceLevel2.size() == 0,
                    HCCL_ERROR("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] omniPipeSliceInfo Level2 slice size is 0.", myRank_),
                    HCCL_E_PARA);

        u32 level2StepCount = omniPipeSliceInfo.dataSliceLevel2.size();
        u32 level0StepCount = omniPipeSliceInfo.dataSliceLevel0.size() / omniPipeSliceInfo.dataSliceLevel2.size();

        for (int i = 0; i < level2StepCount; i++) {
            if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
                CHK_RET(GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_LEVEL2],
                                                      omniPipeSliceInfo.dataSliceLevel2[i]));
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsZ_, ntfIdxCtrlToTempZ_));
                CHK_RET(tempMap[OMNIPIPE_LEVEL2]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL2],
                                                             tempResMap[OMNIPIPE_LEVEL2]));
            }
            for (int j = 0; j < level0StepCount; j++) {
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsXY_, ntfIdxCtrlToTempXY_));
                if (omniUbxLastStepRead_ == true && j == level0StepCount - 1) {
                    tempAlgParamMap[OMNIPIPE_LEVEL0].omniLastStepRead_=true;
                    tempAlgParamMap[OMNIPIPE_LEVEL0].omniReadDstStepSliceInfo=omniPipeSliceLocalcopyInfo.dataSliceLevel0[i * level0StepCount + j];
                    tempAlgParamMap[OMNIPIPE_LEVEL0].processedDataCount=processedDataCount;
                    tempAlgParamMap[OMNIPIPE_LEVEL1].omniLastStepRead_=true;
                    tempAlgParamMap[OMNIPIPE_LEVEL1].omniReadDstStepSliceInfo=omniPipeSliceLocalcopyInfo.dataSliceLevel1[i * level0StepCount + j];
                    tempAlgParamMap[OMNIPIPE_LEVEL1].processedDataCount=processedDataCount;
                } else {
                    tempAlgParamMap[OMNIPIPE_LEVEL0].omniLastStepRead_=false;
                    tempAlgParamMap[OMNIPIPE_LEVEL0].omniReadDstStepSliceInfo=omniPipeSliceLocalcopyInfo.dataSliceLevel0[i * level0StepCount + j];
                    tempAlgParamMap[OMNIPIPE_LEVEL0].processedDataCount=processedDataCount;
                    tempAlgParamMap[OMNIPIPE_LEVEL1].omniLastStepRead_=false;
                    tempAlgParamMap[OMNIPIPE_LEVEL1].omniReadDstStepSliceInfo=omniPipeSliceLocalcopyInfo.dataSliceLevel1[i * level0StepCount + j];
                    tempAlgParamMap[OMNIPIPE_LEVEL1].processedDataCount=processedDataCount;
                }
                if (rankSizeLevel_[OMNIPIPE_LEVEL0] > 1) {
                    CHK_RET(GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_LEVEL0],
                                                          omniPipeSliceInfo.dataSliceLevel0[i * level0StepCount + j]));
                    CHK_RET(tempMap[OMNIPIPE_LEVEL0]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL0],
                                                                 tempResMap[OMNIPIPE_LEVEL0]));
                }
                if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
                    CHK_RET(GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_LEVEL1],
                                                          omniPipeSliceInfo.dataSliceLevel1[i * level0StepCount + j]));
                    CHK_RET(tempMap[OMNIPIPE_LEVEL1]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL1],
                                                                 tempResMap[OMNIPIPE_LEVEL1]));
                }
                // -----------------------------UBX才做这个-------------------------
                // 从第二次开始做localcopy，上一步接受的数据是这一步需要做本地拷贝的数据
                if (omniUbxLastStepRead_ && j != 0) {
                    CHK_RET(UbxLastStepLocalCopy(param, omniPipeSliceInfo, omniPipeSliceLocalcopyInfo, 
                        tempAlgParamMap, processedDataCount, j));
                }
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsXY_, ntfIdxTempToCtrlXY_));
            }
            if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsZ_, ntfIdxTempToCtrlZ_));
            }
        }

        if (omniUbxLastStepRead_) {
            CHK_RET(UbxLocalCopy(param, omniPipeSliceInfo, omniPipeSliceLocalcopyInfo, 
                        tempAlgParamMap, processedDataCount, level0StepCount));
        }else {
            for (u32 rank = 0; rank < rankSize_; rank++) {
                DataSlice dst(param.outputPtr, (rank * dataCount_ + processedDataCount) * dataTypeSize_,
                                currDataCount * dataTypeSize_, currDataCount);
                DataSlice src(resCtx.cclMem.addr, rank * currDataCount * dataTypeSize_, currDataCount * dataTypeSize_,
                                currDataCount);
                CHK_RET(LocalCopy(controlThread_, src, dst));
            }
        }
        processedDataCount += currDataCount;
    }
    return HCCL_SUCCESS;
}
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::RestoreChannelMap(
    const AlgResourceCtxSerializable& resCtx,
    std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo) const
{
    rankIdToChannelInfo.resize(OMNIPIPE_LEVEL_NUM);
    u32 level = 0;
    for (u32 i = 0; i < OMNIPIPE_LEVEL_NUM; i++) {
        if (rankSizeLevel_[i] > 1) {
            for (auto& channel : resCtx.channels[level]) {
                u32 remoteRank = channel.remoteRank;
                rankIdToChannelInfo[i][remoteRank].push_back(channel);
            }
            level++;
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::UbxLastStepLocalCopy(
    const OpParam& param, 
    const OmniPipeSliceInfo& omniPipeSliceInfo, const OmniPipeSliceInfo& omniPipeSliceLocalcopyInfo,
    std::map<u32, TemplateDataParams>& tempAlgParamMap, const u64 processedDataCount, int step) const
{
    HCCL_DEBUG("do localcopy, parallel with step %u", step);
    // 做j-1这一步的localcopy 外面是每个rank遍历，里面是每个rank的多片
    for (int k = 0; k < rankSizeLevel_[OMNIPIPE_LEVEL0]; k++) {
        for (int rpt = 0;
                rpt < omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].inputOmniPipeSliceStride[k].size();
                rpt++) {
            // level0的localcopy
            void *txSrcPtr0 = tempAlgParamMap[OMNIPIPE_LEVEL0].buffInfo.hcclBuff.addr;
            void *txDstPtr0 = param.outputPtr;
            u64 txBaseOff0  = tempAlgParamMap[OMNIPIPE_LEVEL0].buffInfo.inBuffBaseOff
                    + omniPipeSliceInfo.dataSliceLevel0[step - 1].inputOmniPipeSliceStride[k][rpt];
            u64 txOffset0   = omniPipeSliceInfo.dataSliceLevel0[step - 1].stepInputSliceStride[k] + txBaseOff0;
            u64 txBaseOffDst0 = tempAlgParamMap[OMNIPIPE_LEVEL0].buffInfo.inBuffBaseOff
                    + omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].inputOmniPipeSliceStride[k][rpt];
            u64 txOffsetDst0 = omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepInputSliceStride[k]+ txBaseOffDst0;
            txBaseOffDst0 = txOffsetDst0 + processedDataCount * dataTypeSize_;
            // src用ccl的
            DataSlice txSrcSlice0 = DataSlice(txSrcPtr0, txOffset0,
                omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepSliceSize[k][rpt],
                omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepCount[k][rpt]);
            // dst用localcopy的
            DataSlice txDstSlice0 = DataSlice(txDstPtr0, txBaseOffDst0,
                omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepSliceSize[k][rpt],
                omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepCount[k][rpt]);
            CHK_RET(LocalCopy(controlThread_, txSrcSlice0, txDstSlice0));
        }
    }
    for (int k = 0; k < rankSizeLevel_[OMNIPIPE_LEVEL1]; k++) {
        for (int rpt = 0;
                rpt < omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].inputOmniPipeSliceStride[k].size();
                rpt++) {
            // level1的localcopy
            void *txSrcPtr1 = tempAlgParamMap[OMNIPIPE_LEVEL1].buffInfo.hcclBuff.addr;
            void *txDstPtr1 = param.outputPtr;
            u64 txBaseOff1  = tempAlgParamMap[OMNIPIPE_LEVEL1].buffInfo.inBuffBaseOff
                    + omniPipeSliceInfo.dataSliceLevel1[step - 1].inputOmniPipeSliceStride[k][rpt];
            u64 txOffset1 = omniPipeSliceInfo.dataSliceLevel1[step - 1].stepInputSliceStride[k] + txBaseOff1;
            u64 txBaseOffDst1 = tempAlgParamMap[OMNIPIPE_LEVEL1].buffInfo.inBuffBaseOff
                    + omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].inputOmniPipeSliceStride[k][rpt];
            u64 txOffsetDst1 = omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepInputSliceStride[k]+ txBaseOffDst1;
            txBaseOffDst1 = txOffsetDst1 + processedDataCount * dataTypeSize_;
            // src用ccl的
            DataSlice txSrcSlice1 = DataSlice(txSrcPtr1, txOffset1,
                omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepSliceSize[k][rpt],
                omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepCount[k][rpt]);
            // dst用localcopy的
            DataSlice txDstSlice1 = DataSlice(txDstPtr1, txBaseOffDst1,
                omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepSliceSize[k][rpt],
                omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepCount[k][rpt]);
            CHK_RET(LocalCopy(controlThread_, txSrcSlice1, txDstSlice1));
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::UbxLocalCopy(
    const OpParam& param, 
    const OmniPipeSliceInfo& omniPipeSliceInfo, const OmniPipeSliceInfo& omniPipeSliceLocalcopyInfo,
    std::map<u32, TemplateDataParams>& tempAlgParamMap, const u64 processedDataCount, int step) const
{
    // 处理最后一步发的数据，做本地拷贝
    int k = rankIdxLevel_[OMNIPIPE_LEVEL0];
    for (int rpt = 0;
            rpt
            < omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].inputOmniPipeSliceStride[k].size();
            rpt++) {
        // level0的localcopy
        void *txSrcPtr0 = tempAlgParamMap[OMNIPIPE_LEVEL0].buffInfo.hcclBuff.addr;
        void *txDstPtr0 = param.outputPtr;
        u64 txBaseOff0  = tempAlgParamMap[OMNIPIPE_LEVEL0].buffInfo.inBuffBaseOff
                    + omniPipeSliceInfo.dataSliceLevel0[step - 1].inputOmniPipeSliceStride[k][rpt];
        u64 txOffset0   = omniPipeSliceInfo.dataSliceLevel0[step - 1].stepInputSliceStride[k] 
                    + txBaseOff0;
        u64 txBaseOffDst0 = tempAlgParamMap[OMNIPIPE_LEVEL0].buffInfo.inBuffBaseOff
                    + omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1]
                                    .inputOmniPipeSliceStride[k][rpt];
        u64 txOffsetDst0 = omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepInputSliceStride[k]
                    + txBaseOffDst0;
        txBaseOffDst0 = txOffsetDst0 + processedDataCount * dataTypeSize_;
        // src用ccl的
        DataSlice txSrcSlice0 = DataSlice(txSrcPtr0, txOffset0,
            omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepSliceSize[k][rpt],
            omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepCount[k][rpt]);
        // dst用localcopy的
        DataSlice txDstSlice0 = DataSlice(txDstPtr0, txBaseOffDst0,
            omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepSliceSize[k][rpt],
            omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepCount[k][rpt]);
        CHK_RET(LocalCopy(controlThread_, txSrcSlice0, txDstSlice0));
    }

    k = rankIdxLevel_[OMNIPIPE_LEVEL1];
    for (int rpt = 0;
            rpt
            < omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].inputOmniPipeSliceStride[k].size();
            rpt++) {
        // level1的localcopy
        void *txSrcPtr1 = tempAlgParamMap[OMNIPIPE_LEVEL1].buffInfo.hcclBuff.addr;
        void *txDstPtr1 = param.outputPtr;
        u64 txBaseOff1  = tempAlgParamMap[OMNIPIPE_LEVEL1].buffInfo.inBuffBaseOff
                + omniPipeSliceInfo.dataSliceLevel1[step - 1].inputOmniPipeSliceStride[k][rpt];
        u64 txOffset1 = omniPipeSliceInfo.dataSliceLevel1[step - 1].stepInputSliceStride[k] + txBaseOff1;
        u64 txBaseOffDst1 = tempAlgParamMap[OMNIPIPE_LEVEL1].buffInfo.inBuffBaseOff
                + omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].inputOmniPipeSliceStride[k][rpt];
        u64 txOffsetDst1 = omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepInputSliceStride[k]
                + txBaseOffDst1;
        txBaseOffDst1 = txOffsetDst1 + processedDataCount * dataTypeSize_;
        // src用ccl的
        DataSlice txSrcSlice1 = DataSlice(txSrcPtr1, txOffset1,
            omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepSliceSize[k][rpt],
            omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepCount[k][rpt]);
        // dst用localcopy的
        DataSlice txDstSlice1 = DataSlice(txDstPtr1, txBaseOffDst1,
            omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepSliceSize[k][rpt],
            omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepCount[k][rpt]);
        CHK_RET(LocalCopy(controlThread_, txSrcSlice1, txDstSlice1));
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER, InsV2AllGatherOmniPipeMultilevel,
                       InsV2AllGatherOmniPipeExecutor, TopoMatchMultilevel, InsTempAllGatherOmniPipeMesh1D,
                       InsTempAllGatherOmniPipeNHR, InsTempAllGatherOmniPipeNHRDPU);
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER, InsV2AllGatherOmniPipePcie,
                       InsV2AllGatherOmniPipeExecutor, TopoMatchPcieMix, InsTempAllGatherOmniPipeMesh1D,
                       InsTempAllGatherOmniPipeNHR, InsTempAllGatherOmniPipeNHRDPU);
 REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER, InsV2AllGatherOmniPipe,	 
                        InsV2AllGatherOmniPipeExecutor, TopoMatchUBX, InsTempAllGatherOmniPipeMesh1D,	 
                        InsTempAllGatherOmniPipeNHR, InsTempAllGatherOmniPipeNHRDPU);

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER, InsV2AllGatherOmniPipeUboe,
                       InsV2AllGatherOmniPipeExecutor, TopoMatch3Level, InsTempAllGatherOmniPipeMesh1D,
                       InsTempAllGatherOmniPipeNHR, InsTempAllGatherOmniPipeNHR);
}  // namespace ops_hccl