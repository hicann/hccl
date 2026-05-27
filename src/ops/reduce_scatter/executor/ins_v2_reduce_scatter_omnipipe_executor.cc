/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_scatter_omnipipe_executor.h"
#include "ins_temp_reduce_scatter_omnipipe_mesh_1D.h"
#include "ins_temp_reduce_scatter_omnipipe_mesh_1d_dpu.h"
#include "ins_temp_reduce_scatter_omnipipe_nhr.h"
#include "topo_match_pcie_mix.h"
namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                   InsAlgTemplate2>::InsV2ReduceScatterOmniPipeExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InitCommInfo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
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
    HCCL_INFO("[%s]myRank[%u] userRankSize[%u] devType[%u] redOp[%u] dataType[%u] dataTypeSize[%u]", __func__, myRank_,
              rankSize_, devType_, reduceOp_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

// 实例化实际执行以来AutoMatchMeshNhr这个类的实现
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                   InsAlgTemplate2>::CalcAlgHierarchyInfo(HcclComm comm,
                                                                          TopoInfoWithNetLayerDetails* topoInfo,
                                                                          AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    HCCL_DEBUG("CalcRes");
    // 初始化一些基本成员变量
    InitCommInfo(param, topoInfo, algHierarchyInfo);

    // 处理分层
    // 计算subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;  // 这个数组就算x轴为1也有值    
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;

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
        subCommRanks2 = algHierarchyInfo_.infos[1];
    } else {
        subCommRanks0 = algHierarchyInfo_.infos[0];
        subCommRanks1 = algHierarchyInfo_.infos[1];
        subCommRanks2.emplace_back(std::vector<u32>{myRank_});
    }

    rankSizeLevel0_ = subCommRanks0[0].size();
    rankSizeLevel1_ = subCommRanks1[0].size();
    rankSizeLevel2_ = subCommRanks2[0].size();

    // 初始化resourceRequest，思路是只要rankSize>1就给他初始化，注意下后面会不会有空指针
    AlgResourceRequest resReqlevel0;
    AlgResourceRequest resReqlevel1;
    AlgResourceRequest resReqlevel2;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    if (rankSizeLevel0_ > 1) {
        HCCL_INFO("rankSizeLevel0_ > 1, generate resource for temp0");
        std::shared_ptr<InsAlgTemplate0> level0TempAlg =
            std::make_shared<InsAlgTemplate0>(param, topoInfo->userRank, subCommRanks0);
        CHK_RET(level0TempAlg->CalcRes(comm, param, topoInfo, resReqlevel0));
        resourceRequest.slaveThreadNum += 1 + resReqlevel0.slaveThreadNum;
        resourceRequest.notifyNumPerThread.emplace_back(resReqlevel0.notifyNumOnMainThread + 1);
        resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                                  resReqlevel0.notifyNumPerThread.begin(),
                                                  resReqlevel0.notifyNumPerThread.end());
        resourceRequest.notifyNumOnMainThread++;
        resourceRequest.channels.push_back(resReqlevel0.channels[0]);
    }

    if (rankSizeLevel1_ > 1) {
        HCCL_INFO("rankSizeLevel1_ > 1, generate resource for temp1");
        std::shared_ptr<InsAlgTemplate1> level1TempAlg =
            std::make_shared<InsAlgTemplate1>(param, topoInfo->userRank, subCommRanks1);
        CHK_RET(level1TempAlg->CalcRes(comm, param, topoInfo, resReqlevel1));
        resourceRequest.slaveThreadNum += 1 + resReqlevel1.slaveThreadNum;
        resourceRequest.notifyNumPerThread.emplace_back(resReqlevel1.notifyNumOnMainThread + 1);
        resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                                  resReqlevel1.notifyNumPerThread.begin(),
                                                  resReqlevel1.notifyNumPerThread.end());
        resourceRequest.notifyNumOnMainThread++;
        resourceRequest.channels.push_back(resReqlevel1.channels[0]);
    }

    if (rankSizeLevel2_ > 1) {
        HCCL_INFO("rankSizeLevel2_ > 1, generate resource for temp2");
        std::shared_ptr<InsAlgTemplate2> level2TempAlg =
            std::make_shared<InsAlgTemplate2>(param, topoInfo->userRank, subCommRanks2);
        CHK_RET(level2TempAlg->CalcRes(comm, param, topoInfo, resReqlevel2));
        resourceRequest.slaveThreadNum += 1 + resReqlevel2.slaveThreadNum;
        resourceRequest.notifyNumPerThread.emplace_back(resReqlevel2.notifyNumOnMainThread + 1);
        resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                                  resReqlevel2.notifyNumPerThread.begin(),
                                                  resReqlevel2.notifyNumPerThread.end());
        resourceRequest.notifyNumOnMainThread++;
        resourceRequest.channels.push_back(resReqlevel2.channels[0]);
    }

    return HCCL_SUCCESS;
}

// 该函数必须按照level0、level1、level2的顺序调用
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    PrepareResForTemplateLevel(u32 level, std::shared_ptr<InsAlgTemplateBase>& tempBase)
{
    u32 levelThreadNum = tempBase->GetThreadNum();
    if (level == OMNIPIPE_LEVEL0) {
        levelThreads_[OMNIPIPE_LEVEL0].assign(threads_.begin() + 1, threads_.begin() + 1 + levelThreadNum);
        tempMainThreadsLevel01_.push_back(levelThreads_[0].at(0));
    } else if (level == OMNIPIPE_LEVEL1) {
        levelThreads_[OMNIPIPE_LEVEL1].assign(threads_.begin() + 1 + levelThreads_[0].size(),
                                              threads_.begin() + 1 + levelThreads_[0].size() + levelThreadNum);
        tempMainThreadsLevel01_.push_back(levelThreads_[1].at(0));
    } else if (level == OMNIPIPE_LEVEL2) {
        levelThreads_[OMNIPIPE_LEVEL2].assign(threads_.begin() + 1 + levelThreads_[0].size() + levelThreads_[1].size(),
                                              threads_.end());
        tempMainThreadsLevel2_.push_back(levelThreads_[OMNIPIPE_LEVEL2].at(0));
    }

    // 获取当前template各自的主thread上有多少notify
    AlgResourceRequest levelTempRequest;
    CHK_RET(tempBase->GetRes(levelTempRequest));
    if (level < OMNIPIPE_LEVEL2) {
        notifyIdxCtrlToTempLevel01_.push_back(levelTempRequest.notifyNumOnMainThread);
        notifyIdxTempToCtrlLevel01_.push_back(tempMainThreadsLevel01_.size() + tempMainThreadsLevel2_.size() - 1);
    } else {
        HCCL_DEBUG("levelTempRequest.notifyNumOnMainThread=%u, tempMainThreadsLevel01_.size()=%u, "
                   "tempMainThreadsLevel2_.size()=%u",
                   levelTempRequest.notifyNumOnMainThread, tempMainThreadsLevel01_.size(),
                   tempMainThreadsLevel2_.size());
        notifyIdxCtrlToTempLevel2_.push_back(levelTempRequest.notifyNumOnMainThread);
        notifyIdxTempToCtrlLevel2_.push_back(tempMainThreadsLevel01_.size() + tempMainThreadsLevel2_.size() - 1);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][Orchestrate] Orchestrate Start");
    // 参数填充
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    threads_ = resCtx.threads;
    
    // 计算subCommRanks
    int index = 0;
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;    

    subCommRanks0.clear();
    subCommRanks1.clear();
    subCommRanks2.clear();

    if(resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix) {
        subCommRanks0 = {algHierarchyInfo_.infos[0][0]};
        std::vector<u32> closRanks;
        u32 meshSize = algHierarchyInfo_.infos[0][0].size();
        for(auto rank : algHierarchyInfo_.infos[0][1]) {
            if(rank % meshSize == resCtx.topoInfo.userRank % meshSize) {
                closRanks.push_back(rank);
            }
        }
        subCommRanks1 = {closRanks};
        subCommRanks2 = algHierarchyInfo_.infos[1];
    } else {
        subCommRanks0 = algHierarchyInfo_.infos[0];
        subCommRanks1 = algHierarchyInfo_.infos[1];
        subCommRanks2.emplace_back(std::vector<u32>{myRank_});
    }

    // 处理分层
    rankSizeLevel0_ = subCommRanks0[0].size();
    rankSizeLevel1_ = subCommRanks1[0].size();
    rankSizeLevel2_ = subCommRanks2[0].size();

    HCCL_DEBUG("algHierarchyInfo_.infos[1].size() = %u, algHierarchyInfo_.infos[1][0].size() = %u",
               algHierarchyInfo_.infos[1].size(), algHierarchyInfo_.infos[1][0].size());
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ % (rankSizeLevel0_ * rankSizeLevel1_) / rankSizeLevel0_;
    rankIdxLevel2_ = myRank_ / (rankSizeLevel0_ * rankSizeLevel1_);

    // 改成智能指针，初始化
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;
    if (rankSizeLevel0_ > 1) {
        tempMap[OMNIPIPE_LEVEL0] = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel1_ > 1) {
        tempMap[OMNIPIPE_LEVEL1] = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1);
    }
    if (rankSizeLevel2_ > 1) {
        tempMap[OMNIPIPE_LEVEL2] = std::make_shared<InsAlgTemplate2>(param, myRank_, subCommRanks2);
    }
    HCCL_DEBUG("tempMap.size()=%u", tempMap.size());
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
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[InsV2ReduceScatterOmniPipeExecutor][Orchestrate]errNo[0x%016llx] Reduce scatter excutor "
                           "kernel run failed",
                           HCCL_ERROR_CODE(ret)),
                ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::RestoreChannelMap(
    const AlgResourceCtxSerializable& resCtx, std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo)
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

// 将计算出的单步slice信息初始化到templateParam中
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                   InsAlgTemplate2>::GenTemplateAlgParamsByDimData(TemplateDataParams& tempAlgParams,
                                                                                   const StepSliceInfo& stepSliceInfo)
{
    // rs特殊处理，过程中的所有step都在ccl中进行数据搬运，在template中只使用ccl的起始地址就可以了，in和out不用赋值
    tempAlgParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    // 这三个值使用时，第一步从in->hccl，第二步确定性计算按序ccl->out进行规约
    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.buffInfo.hcclBuffBaseOff = stepSliceInfo.buffInfo.hcclBuffBaseOff;
    // 统一赋值，直接透传
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor] tempAlgParams.buffInfo.inBuffBaseOff [%u],"
              "tempAlgParams.buffInfo.outBuffBaseOff [%u]",
              tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.outBuffBaseOff);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx,
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap)
{
    HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop] Start");

    HCCL_DEBUG("Start Step Calc!");
    // 1.计算带宽
    double bw_ag_l0=BW_OMNI_DEFAULT;
    double bw_ag_l1=BW_OMNI_DEFAULT;
    double bw_ag_l2=BW_OMNI_DEFAULT;
    double bw_rs_l0=BW_OMNI_DEFAULT;
    double bw_rs_l1=BW_OMNI_DEFAULT;
    double bw_rs_l2=BW_OMNI_DEFAULT;

    if (resCtx.topoInfo.level0PcieMix) {
        if (rankSizeLevel1_==2) {
            bw_ag_l1=BW_OMNI_PCIE_EIGHT_AG_CLOS;
            bw_rs_l1=BW_OMNI_PCIE_EIGHT_RS_CLOS;
        } else if (rankSizeLevel1_==4) {
            bw_ag_l1=BW_OMNI_PCIE_SIXTEEN_AG_CLOS;
            bw_rs_l1=BW_OMNI_PCIE_SIXTEEN_RS_CLOS;
        }
    }

    //计算等价带宽
    double eqBw0 = bw_rs_l0;//L0 mesh
    double eqBw1 = bw_rs_l1;//L1 NHR
    double eqBw2 = bw_rs_l2;//L2 NHR

    //level0为mesh,等价mesh为其本身
    //level1为nhr
    //level2, ranksize = 1
    eqBw1 = rankSizeLevel1_ > 1 ? eqBw1 / (rankSizeLevel1_ - 1) : eqBw1;
    eqBw2 = rankSizeLevel1_ > 1 ? eqBw2 / (rankSizeLevel1_ - 1) : eqBw2;

    std::vector<double> endpointAttrBwNew{eqBw0, eqBw1, eqBw2};

    // 2、计算scratch 返回的数组0是maxCountPerloop, 1是loopTimes
    OmniPipeScratchParam scratchParam;
    scratchParam.endpointAttrBw = endpointAttrBwNew;
    scratchParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_};
    std::vector<u64> levelAlgType;
    (tempMap.count(OMNIPIPE_LEVEL0) > 0) ?
        levelAlgType.push_back(tempMap[OMNIPIPE_LEVEL0]->CalcScratchMultiple(BufferType::DEFAULT, BufferType::DEFAULT)) :
        levelAlgType.push_back(0);
    (tempMap.count(OMNIPIPE_LEVEL1) > 0) ?
        levelAlgType.push_back(tempMap[OMNIPIPE_LEVEL1]->CalcScratchMultiple(BufferType::DEFAULT, BufferType::DEFAULT)) :
        levelAlgType.push_back(0);
    (tempMap.count(OMNIPIPE_LEVEL2) > 0) ?
        levelAlgType.push_back(tempMap[OMNIPIPE_LEVEL2]->CalcScratchMultiple(BufferType::DEFAULT, BufferType::DEFAULT)) :
        levelAlgType.push_back(0);
    scratchParam.levelAlgType = levelAlgType;
    // 手动转成数组，这边只给reducescatter用
    std::vector<u64> dataSizeVec;
    for (int i = 0; i < rankSize_; i++) {
        dataSizeVec.push_back(dataSize_);
    }
    scratchParam.dataSize = dataSizeVec;
    scratchParam.dataTypeSize = dataTypeSize_;
    scratchParam.maxTmpMemSize = resCtx.cclMem.size;
    scratchParam.opMode = param.opMode;
    scratchParam.engine = param.engine;
    std::vector<u64> loopInfo = CalcOmniPipeScratchInfo(scratchParam);
    u64 maxCountPerLoop = loopInfo[0];
    u64 loopTimes = loopInfo[1];
    HCCL_DEBUG("loopInfo[0](maxCountPerLoop)=%u, loopInfo[1](loopTimes)=%u", maxCountPerLoop, loopTimes);

    // 3、计算n-1次loop的slice信息
    OmniPipeSliceParam sliceParam;
    std::vector<u64> dataSizePerLoop;
    std::vector<u64> dataWholeSize;
    u64 perLoopSize = maxCountPerLoop * dataTypeSize_;

    for (int i = 0; i < rankSize_; i++) {
        dataSizePerLoop.push_back(perLoopSize);
        dataWholeSize.push_back(perLoopSize);
    }
    sliceParam.dataSizePerLoop = dataSizePerLoop;
    sliceParam.dataWholeSize = dataWholeSize;  // rs这个值和peerloop一致，已对齐
    sliceParam.endpointAttrBw = endpointAttrBwNew;
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, rankIdxLevel2_};
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_};
    sliceParam.levelAlgType = levelAlgType;
    sliceParam.dataTypeSize = dataTypeSize_;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = param.engine;
    OmniPipeSliceInfo alignSliceInfo = CalcRSOmniPipeSliceInfo(sliceParam);

    // 4、计算第n次的loop的slice信息
    OmniPipeSliceInfo tailSliceInfo;
    if (dataCount_ % maxCountPerLoop != 0) {
        std::vector<u64> dataSizePerLoop;
        std::vector<u64> dataWholeSize;
        u64 perLoopSize = (dataCount_ % maxCountPerLoop) * dataTypeSize_;
        for (int i = 0; i < rankSize_; i++) {
            dataSizePerLoop.push_back(perLoopSize);
            dataWholeSize.push_back(perLoopSize);
        }
        sliceParam.dataSizePerLoop = dataSizePerLoop;
        sliceParam.dataWholeSize = dataWholeSize;
        tailSliceInfo = CalcRSOmniPipeSliceInfo(sliceParam);
    }

    u64 processedDataCount = 0;
    OmniPipeSliceInfo omnipipeSliceInfo;
    HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop]loopTimes = [%u]", loopTimes);
    std::map<u32, TemplateResource> tempResMap;
    std::map<u32, TemplateDataParams> tempAlgParamMap;
    for (auto& temp : tempMap) {
        tempResMap[temp.first].channels = remoteRankToChannelInfo_[temp.first];
        tempResMap[temp.first].threads = levelThreads_[temp.first];
        tempResMap[temp.first].npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
        tempResMap[temp.first].dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
        tempAlgParamMap[temp.first].buffInfo.hcclBuff = resCtx.cclMem;
    }

    TemplateDataParams tempParamLocalcopy;
    tempParamLocalcopy.buffInfo.hcclBuff = resCtx.cclMem;
    tempParamLocalcopy.buffInfo.inputPtr = param.inputPtr;
    tempParamLocalcopy.buffInfo.outputPtr = param.outputPtr;
    // 5、进行一次loop的数据处理
    for (u64 loop = 0; loop < loopTimes; loop++) {
        // localcopy前同步，这里默认xy不会同时为1，因此使用01的线程组处理localcopy的前后同步
        CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxCtrlToTempLevel01_));
        // 5.1 RS在每次loop进行之前先将所有数据从usrin拷贝到ccl，2.3已修改下列参数
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        tempParamLocalcopy.buffInfo.inBuffType = BufferType::INPUT;
        tempParamLocalcopy.count = currDataCount;
        tempParamLocalcopy.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempParamLocalcopy.inputSliceStride = dataCount_ * dataTypeSize_;
        auto loopSize = currDataCount * dataTypeSize_;
        tempParamLocalcopy.buffInfo.outBuffBaseOff = 0;
        tempParamLocalcopy.outputSliceStride = loopSize;
        tempParamLocalcopy.repeatNum = rankSize_;
        tempParamLocalcopy.sliceSize = loopSize;
        // 这边不论三层为什么拓扑，都使用第一个template去做localcopy
        if (rankSizeLevel0_ > 1) {
            auto temp0 = std::dynamic_pointer_cast<InsAlgTemplate0>(tempMap.begin()->second);
            CHK_RET(temp0->DoLocalCopy(tempParamLocalcopy, tempResMap.begin()->second.threads));
        } else {
            auto temp1 = std::dynamic_pointer_cast<InsAlgTemplate1>(tempMap.begin()->second);
            CHK_RET(temp1->DoLocalCopy(tempParamLocalcopy, tempResMap.begin()->second.threads));
        }
        // localcopy后同步
        CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxTempToCtrlLevel01_));

        // 5.2 确定当前是前n-1次loop的slice结果，还是存在尾块时最后一次loop的slice结果
        if (loop == loopTimes - 1 && !tailSliceInfo.isEmpty()) {
            omnipipeSliceInfo = tailSliceInfo;
        } else {
            omnipipeSliceInfo = alignSliceInfo;
        }
        u32 level2StepCount = omnipipeSliceInfo.dataSliceLevel2.size();
        u32 level0StepCount = omnipipeSliceInfo.dataSliceLevel0.size() / omnipipeSliceInfo.dataSliceLevel2.size();
        HCCL_INFO(
            "[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop]level2 step count = [%u], level0 step count = [%u]",
            level2StepCount, level0StepCount);

        // 5.3 for外层2d
        u32 axisReduceId = 0;  // 轴间reduce从计算slice的结果中获取
        std::vector<TemplateDataParams> axisReduceTempParams;
        for (int i = 0; i < level2StepCount; i++) {
            HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop]Step [%u] in level2", i);
            if (rankSizeLevel2_ > 1) {
                HCCL_DEBUG("rankSizeLevel2_ > 1");
                // 初始化机间template param
                GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_LEVEL2], omnipipeSliceInfo.dataSliceLevel2[i]);
                // level2前同步
                HCCL_DEBUG("tempMainThreadsLevel2_.size=%u, notifyIdxCtrlToTempLevel2_.size=%u, "
                           "notifyIdxCtrlToTempLevel2_.size=%u",
                           tempMainThreadsLevel2_.size(), notifyIdxCtrlToTempLevel2_.size(),
                           notifyIdxCtrlToTempLevel2_.size());
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel2_, notifyIdxCtrlToTempLevel2_));
                // 执行机间template任务
                HCCL_DEBUG("PreSyncInterThreads z success?");
                CHK_RET(tempMap[OMNIPIPE_LEVEL2]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL2], tempResMap[OMNIPIPE_LEVEL2]));
            }
            // 5.4 for内层2d
            for (int j = 0; j < level0StepCount; j++) {
                // level0、1前同步
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxCtrlToTempLevel01_));
                // 初始化并执行机内template任务
                if (rankSizeLevel0_ > 1) {
                    HCCL_DEBUG("rankSizeLevel0_ > 1");
                    CHK_RET(GenTemplateAlgParamsByDimData(tempAlgParamMap[0],
                                                          omnipipeSliceInfo.dataSliceLevel0[i * level0StepCount + j]));
                    CHK_RET(
                        tempMap[0]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL0], tempResMap[OMNIPIPE_LEVEL0]));
                }
                if (rankSizeLevel1_ > 1) {
                    HCCL_DEBUG("rankSizeLevel1_ > 1");
                    CHK_RET(GenTemplateAlgParamsByDimData(tempAlgParamMap[1],
                                                          omnipipeSliceInfo.dataSliceLevel1[i * level0StepCount + j]));
                    CHK_RET(
                        tempMap[1]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL1], tempResMap[OMNIPIPE_LEVEL1]));
                }
                // level0、1尾同步
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxTempToCtrlLevel01_));
            }
            if (rankSizeLevel2_ > 1) {
                // z轴尾同步
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel2_, notifyIdxTempToCtrlLevel2_));
                HCCL_DEBUG("PostSyncInterThreads z success.");
            }
        }
        // localcopy前同步
        CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxCtrlToTempLevel01_));
        // 5.5 将当前这个loop在ccl中的数据一次性拷贝到userout中
        tempParamLocalcopy.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        tempParamLocalcopy.buffInfo.inBuffBaseOff = loopSize * myRank_;
        tempParamLocalcopy.inputSliceStride = 0;
        tempParamLocalcopy.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempParamLocalcopy.outputSliceStride = 0;
        tempParamLocalcopy.sliceSize = loopSize;  // 尾拷贝数据量变成1/rankSize
        // repeat=1，temp内部已经没有和rankid相关的处理
        tempParamLocalcopy.repeatNum = 1;
        if (rankSizeLevel0_ > 1) {
            auto temp0 = std::dynamic_pointer_cast<InsAlgTemplate0>(tempMap.begin()->second);
            CHK_RET(temp0->DoLocalCopy(tempParamLocalcopy, tempResMap.begin()->second.threads));
        } else {
            auto temp1 = std::dynamic_pointer_cast<InsAlgTemplate1>(tempMap.begin()->second);
            CHK_RET(temp1->DoLocalCopy(tempParamLocalcopy, tempResMap.begin()->second.threads));
        }
        // localcopy后同步
        CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxTempToCtrlLevel01_));
        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, InsV2ReduceScatterOmniPipeMultilevel,
                       InsV2ReduceScatterOmniPipeExecutor, TopoMatchMultilevel, InsTempReduceScatterOmniPipeMesh1D,
                       InsTempReduceScatterOmniPipeMesh1D, InsTempReduceScatterOmniPipeMesh1dDpu);
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, InsV2ReduceScatterOmniPipePcie,
                       InsV2ReduceScatterOmniPipeExecutor, TopoMatchPcieMix, InsTempReduceScatterOmniPipeMesh1D,
                       InsTempReduceScatterOmniPipeNHR, InsTempReduceScatterOmniPipeMesh1dDpu);
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, InsV2ReduceScatterOmniPipe,
                       InsV2ReduceScatterOmniPipeExecutor, TopoMatchUBX, InsTempReduceScatterOmniPipeMesh1D,
                       InsTempReduceScatterOmniPipeNHR, InsTempReduceScatterOmniPipeMesh1dDpu);
}  // namespace ops_hccl