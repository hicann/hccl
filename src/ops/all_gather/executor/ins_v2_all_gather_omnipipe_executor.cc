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
#include "ins_temp_all_gather_omnipipe_mesh_1D.h"
#include "ins_temp_all_gather_omnipipe_nhr_dpu.h"
#include "ins_temp_all_gather_omnipipe_nhr.h"

namespace ops_hccl {
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
    CHK_RET(InitExectorInfo(param));

    // 计算subCommRanks
    int index = 0;
    std::vector<std::vector<u32>> subCommRanks0{algHierarchyInfo.infos[0][0]};
    std::vector<std::vector<u32>> subCommRanks2;
    std::vector<std::vector<u32>> subCommRanks1;

    if (!algHierarchyInfo_.infos[1].empty() && !algHierarchyInfo_.infos[1][0].empty()) {
        subCommRanks2.push_back(algHierarchyInfo.infos[1][0]);
    } else {
        subCommRanks2.emplace_back(std::vector<u32>{myRank_});
    }
    subCommRanks1.resize(1);

    for (int i = myRank_ % rankSizeLevel_[0]; i < algHierarchyInfo.infos[0][1].size(); i += rankSizeLevel_[0]) {
        subCommRanks1[0].push_back(algHierarchyInfo.infos[0][1][i]);
    }
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;

    if (rankSizeLevel_[OMNIPIPE_LEVEL0] > 1) {
        tempMap[OMNIPIPE_LEVEL0] = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
        tempMap[OMNIPIPE_LEVEL1] = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1);
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
        tempMap[OMNIPIPE_LEVEL2] = std::make_shared<InsAlgTemplate2>(param, myRank_, subCommRanks2);
    }

    for (auto& temp : tempMap) {
        CHK_RET(CalcResLevel(comm, param, topoInfo, temp.second, resourceRequest));
    }
    HCCL_DEBUG("[InInsV2AllGatherOmniPipeExecutor][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u], "
               "channels[%u]",
        myRank_, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
        resourceRequest.channels.size());

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
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][Orchestrate] Orchestrate Start");

    maxTmpMemSize_ = resCtx.cclMem.size;  // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    CHK_RET(InitExectorInfo(param));

    // 计算subCommRanks
    int index = 0;
    std::vector<std::vector<u32>> subCommRanks0{resCtx.algHierarchyInfo.infos[0][0]};
    std::vector<std::vector<u32>> subCommRanks2;
    std::vector<std::vector<u32>> subCommRanks1;

    if (!algHierarchyInfo_.infos[1].empty() && !algHierarchyInfo_.infos[1][0].empty()) {
        subCommRanks2.push_back(algHierarchyInfo_.infos[1][0]);
    } else {
        subCommRanks2.emplace_back(std::vector<u32>{myRank_});
    }
    subCommRanks1.resize(1);
    for (int i = myRank_ % rankSizeLevel_[OMNIPIPE_LEVEL0]; i < resCtx.algHierarchyInfo.infos[0][1].size();
         i += rankSizeLevel_[0]) {
        subCommRanks1[0].push_back(resCtx.algHierarchyInfo.infos[0][1][i]);
    }
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;

    if (rankSizeLevel_[OMNIPIPE_LEVEL0] > 1) {
        tempMap[OMNIPIPE_LEVEL0] = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
        tempMap[OMNIPIPE_LEVEL1] = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1);
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
        tempMap[OMNIPIPE_LEVEL2] = std::make_shared<InsAlgTemplate2>(param, myRank_, subCommRanks2);
    }

    // 为temp分配thread
    threads_ = resCtx.threads;
    controlThread_ = threads_.at(0);
    levelThreads_.resize(OMNIPIPE_LEVEL_NUM);
    for (auto& temp : tempMap) {
        CHK_RET(PrepareResForTemplateLevel(temp.first, temp.second));
    }

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx, tempMap);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllGatherOmniPipeExecutor][Orchestrate]errNo[0x%016llx] AllGather excutor kernel run failed",
                   HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                          InsAlgTemplate2>::InitExectorInfo(const OpParam& param)
{
    (void) param;
    rankSizeLevel_.resize(OMNIPIPE_LEVEL_NUM);
    rankIdxLevel_.resize(OMNIPIPE_LEVEL_NUM);
    // 处理分层
    rankSizeLevel_[OMNIPIPE_LEVEL0] = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel_[OMNIPIPE_LEVEL1] = algHierarchyInfo_.infos[0][1].size() / algHierarchyInfo_.infos[0][0].size();

    if (!algHierarchyInfo_.infos[1].empty() && !algHierarchyInfo_.infos[1][0].empty()) {
        rankSizeLevel_[OMNIPIPE_LEVEL2] = algHierarchyInfo_.infos[1][0].size();
    } else {
        rankSizeLevel_[OMNIPIPE_LEVEL2] = 1;
    }

    rankIdxLevel_[OMNIPIPE_LEVEL0] = myRank_ % rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL1] = myRank_ % (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]) /
                                      rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL2] = myRank_ / (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                               InsAlgTemplate2>::GenTemplateAlgParamsByDimData(TemplateDataParams& tempAlgParams,
                                                                               StepSliceInfo& stepSliceInfo) const
{
    // tempAlgParams.buffInfo.hcclBuff 已在外部赋值
    tempAlgParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;

    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.buffInfo.hcclBuffBaseOff = stepSliceInfo.buffInfo.hcclBuffBaseOff;  // 实际上是空值
    tempAlgParams.stepSliceInfo = stepSliceInfo;

    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor] tempAlgParams.buffInfo.inBuffBaseOff [%u],"
              "tempAlgParams.buffInfo.outBuffBaseOff [%u]",
              tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.outBuffBaseOff);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx,
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap)
{
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop] Start");
    // 1、计算带宽
    std::vector<std::vector<EndpointAttrBwCoeff>> endpointAttrBw;
    u32 levelSize = 3;
    CHK_RET(CalAllLevelEndpointAttrBwCoeff(param.hcclComm, myRank_, levelSize, endpointAttrBw));
    std::vector<EndpointAttrBwCoeff> endpointAttrBwNew;
    endpointAttrBwNew.resize(endpointAttrBw.size());
    u64 index = 0;
    //转成平均带宽
    for (u64 i = 0; i < endpointAttrBw.size(); i++) {
        for (u64 j = 0; j < endpointAttrBw[i].size(); j++) {
            if ((i == 0) && (j != 0)) {
                endpointAttrBw[i][j] /= rankSizeLevel_[1] - 1;
            } else if (i!= 0) {
                endpointAttrBw[i][j] /= algHierarchyInfo_.infos[i][j].size() - 1;
            }
            endpointAttrBwNew[index++] = endpointAttrBw[i][j];
        }
    }
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
    // 4、计算第n次的loop的slice信息
    OmniPipeSliceInfo tailSliceInfo;
    if (dataCount_ % maxCountPerLoop != 0) {
        u64 perLoopSize = (dataCount_ % maxCountPerLoop) * dataTypeSize_;
        std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
        std::vector<u64> dataWholeSize(rankSize_, perLoopSize);
        omniPipeSliceParam.dataSizePerLoop = dataSizePerLoop;
        omniPipeSliceParam.dataWholeSize = dataWholeSize;
        tailSliceInfo = CalcAGOmniPipeSliceInfo(omniPipeSliceParam);
    }

    u64 processedDataCount = 0;
    OmniPipeSliceInfo omniPipeSliceInfo;

    std::map<u32, TemplateResource> tempResMap;
    std::map<u32, TemplateDataParams> tempAlgParamMap;

    for (auto& temp : tempMap) {
        tempResMap[temp.first].channels = remoteRankToChannelInfo_[temp.first];
        tempResMap[temp.first].threads = levelThreads_[temp.first];
        tempAlgParamMap[temp.first].buffInfo.hcclBuff = resCtx.cclMem;
    }
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        DataSlice src(param.inputPtr, processedDataCount * dataTypeSize_, currDataCount * dataTypeSize_, currDataCount);
        DataSlice dst(resCtx.cclMem.addr, myRank_ * currDataCount * dataTypeSize_, currDataCount * dataTypeSize_,
                        currDataCount);
        CHK_RET(LocalCopy(controlThread_, src, dst));

        if (loop == loopTimes - 1 && dataCount_ % maxCountPerLoop != 0) {
            omniPipeSliceInfo = tailSliceInfo;
        } else {
            omniPipeSliceInfo = alignSliceInfo;
        }

        CHK_PRT_RET(omniPipeSliceInfo.dataSliceLevel2.size() == 0,
                    HCCL_ERROR("[InsV2AllGatherOmniPipeExecutor] omniPipeSliceInfo Level2 slice size is 0."),
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
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsXY_, ntfIdxTempToCtrlXY_));
            }
            if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsZ_, ntfIdxTempToCtrlZ_));
            }
        }

        for (u32 rank = 0; rank < rankSize_; rank++) {
            DataSlice dst(param.outputPtr, (rank * dataCount_ + processedDataCount) * dataTypeSize_,
                            currDataCount * dataTypeSize_, currDataCount);
            DataSlice src(resCtx.cclMem.addr, rank * currDataCount * dataTypeSize_, currDataCount * dataTypeSize_,
                            currDataCount);
            CHK_RET(LocalCopy(controlThread_, src, dst));
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

#ifndef AICPU_COMPILE
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER, InsV2AllGatherOmniPipe,
                       InsV2AllGatherOmniPipeExecutor, TopoMatchUBX, InsTempAllGatherOmniPipeMesh1D,
                       InsTempAllGatherOmniPipeNHR, InsTempAllGatherOmniPipeNHRDPU);
#endif
}  // namespace ops_hccl