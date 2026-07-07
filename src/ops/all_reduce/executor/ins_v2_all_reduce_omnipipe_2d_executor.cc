/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "ins_v2_all_reduce_omnipipe_2d_executor.h"
#include "alg_data_trans_wrapper.h"
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_reduce_scatter_omnipipe_mesh1d_mem2mem.h"
#include "ccu_temp_reduce_scatter_omnipipe_nhr1d_mem2mem.h"
#include "ccu_temp_all_gather_omnipipe_mesh1d_mem2mem.h"
#include "ccu_temp_all_gather_omnipipe_nhr1d_mem2mem.h"
#include "ccu_temp_reduce_scatter_omnipipe_mesh1d.h"
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#endif

namespace ops_hccl {

constexpr u32 CCU_OMNIPIPE_LEVEL0 = 0;
constexpr u32 CCU_OMNIPIPE_LEVEL1 = 1;
constexpr u32 CCU_OMNIPIPE_LEVEL_NUM = 2;

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
InsV2AllReduceOmniPipe2dExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InsV2AllReduceOmniPipe2dExecutor()
{
}


template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2AllReduceOmniPipe2dExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::CalcAlgHierarchyInfo(HcclComm comm, 
            TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    u32 userrank = topoInfo->userRank;
    HCCL_DEBUG("[%s] myRank[%u]", __func__, userrank);

    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));

    for (auto i = 0; i < algHierarchyInfo.infos.size(); ++i) {
        for (auto j = 0; j < algHierarchyInfo.infos[i].size(); ++j) {
            for (auto k = 0; k < algHierarchyInfo.infos[i][j].size(); ++k) {
                HCCL_INFO("[%s] myRank[%u] (%d, %d, %d) %u", __func__, topoInfo->userRank, i, j, k,
                    algHierarchyInfo.infos[i][j][k]);
            }
        }
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2AllReduceOmniPipe2dExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InitCommInfo(
            const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    rankSize_ = topoInfo->userRankSize;
    myRank_ = topoInfo->userRank;
    reduceOp_ = param.reduceType;
    devType_ = topoInfo->deviceType;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];
    dataCount_ = param.DataDes.count;
    dataSize_ = dataCount_ * dataTypeSize_;

    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[InitCommInfo] rankSizeLevel0 is 0, should be greater than  0");
        return HcclResult::HCCL_E_PARA;
    }

    rankSizeLevel1_ = algHierarchyInfo.infos[0][1].size() / rankSizeLevel0_;
    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[InitCommInfo] rankSizeLevel1 is 0, should be greater than 0");
        return HcclResult::HCCL_E_PARA;
    }

    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;

    HCCL_DEBUG("[%s]myRank[%u] rankSize[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] "
        "rankIdxLevel1[%u] devType[%u] dataCount[%u] dataType[%u] dataTypeSize[%u]",
        __func__, myRank_, rankSize_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_, devType_,
        dataCount_, dataType_, dataTypeSize_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2AllReduceOmniPipe2dExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::CalcResLevel(
            HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
            std::shared_ptr<CcuAlgTemplateBase> tempAlg, AlgResourceRequest& resourceReq, const int& curLevel)
{
    AlgResourceRequest resReqlevel;
    CHK_RET(tempAlg->CalcRes(comm, param, topoInfo, resReqlevel));

    resourceReq.slaveThreadNum += resReqlevel.slaveThreadNum;
    resourceReq.notifyNumOnMainThread += resReqlevel.notifyNumOnMainThread;
    resourceReq.notifyNumPerThread.insert(resourceReq.notifyNumPerThread.end(),
                                            resReqlevel.notifyNumPerThread.begin(),
                                            resReqlevel.notifyNumPerThread.end());
    
    HCCL_DEBUG("[%s] currTemplate has [%d] kernels.", __func__, resReqlevel.ccuKernelNum[0]);
    if (curLevel == OMNIPIPE_RS_LEVEL0 || curLevel == OMNIPIPE_RS_LEVEL1) {
        std::for_each(resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end(), [](CcuKernelInfo &info) {
            info.resGroup = 0;
        });
    } else {
        std::for_each(resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end(), [](CcuKernelInfo &info) {
            info.resGroup = 1;
        });
    }
    resourceReq.ccuKernelInfos.insert(resourceReq.ccuKernelInfos.end(), resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end());
    resourceReq.ccuKernelNum.insert(resourceReq.ccuKernelNum.end(), resReqlevel.ccuKernelNum.begin(), resReqlevel.ccuKernelNum.end());

    return HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2AllReduceOmniPipe2dExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, topoInfo, algHierarchyInfo));

    // 初始化通信域subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    subCommRanks0.push_back(algHierarchyInfo.infos[0][0]);
    subCommRanks1.resize(1);
    for (auto i = myRank_ % rankSizeLevel0_; i < algHierarchyInfo.infos[0][1].size(); i += rankSizeLevel0_) {
        subCommRanks1[0].push_back(algHierarchyInfo.infos[0][1][i]);
    }

    // 初始化template
    std::map<u32, std::shared_ptr<CcuAlgTemplateBase>> tempMap;
    tempMap[OMNIPIPE_RS_LEVEL0] = std::make_shared<CcuRsAlgTemplateX>(param, myRank_, subCommRanks0);
    tempMap[OMNIPIPE_RS_LEVEL1] = std::make_shared<CcuRsAlgTemplateY>(param, myRank_, subCommRanks1);
    tempMap[OMNIPIPE_AG_LEVEL0] = std::make_shared<CcuAgAlgTemplateX>(param, myRank_, subCommRanks0);
    tempMap[OMNIPIPE_AG_LEVEL1] = std::make_shared<CcuAgAlgTemplateY>(param, myRank_, subCommRanks1);

    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    for (int level = 0; level < OMNIPIPE_AR_LEVEL_NUM; level++) {
        if (tempMap.count(level) > 0) {
            CHK_RET(CalcResLevel(comm, param, topoInfo, tempMap[level], resourceRequest, level));
        }
    }
    resourceRequest.slaveThreadNum += 1; // 需要一个主流和一个从流来并行2d
    resourceRequest.notifyNumOnMainThread += 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG("[%s] slaveThreadNum:%d, notifyNumOnMainThread:%d", __func__, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);

    return HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2AllReduceOmniPipe2dExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_DEBUG("[%s] start", __func__);
    threads_ = resCtx.threads;
    HCCL_DEBUG("[%s]threads size: %u", __func__, threads_.size());
    rankSize_ = resCtx.topoInfo.userRankSize;
    myRank_ = resCtx.topoInfo.userRank;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    maxTmpMemSize_ = resCtx.cclMem.size;
    rankSizeLevel0_ = resCtx.algHierarchyInfo.infos[0][0].size();
    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[Orchestrate] rankSizeLevel0 is 0");
        return HcclResult::HCCL_E_PARA;
    }

    rankSizeLevel1_ = resCtx.algHierarchyInfo.infos[0][1].size() / rankSizeLevel0_;
    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel1 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    
    HCCL_DEBUG("[Orchestrate] myRank[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]",
        myRank_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllReduceOmniPipe2dExecutor][Orchestrate]errNo[0x%016llx] excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2AllReduceOmniPipe2dExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InitTemplate(
            const OpParam& param, std::map<u32, std::shared_ptr<CcuAlgTemplateBase>>& tempMap,
            const std::vector<std::vector<u32>>& subCommRanks0, const std::vector<std::vector<u32>>& subCommRanks1)
{
    if (rankSizeLevel0_ > 1) {
        tempMap[OMNIPIPE_RS_LEVEL0] = std::make_shared<CcuRsAlgTemplateX>(param, myRank_, subCommRanks0);
        tempMap[OMNIPIPE_AG_LEVEL0] = std::make_shared<CcuAgAlgTemplateX>(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel1_ > 1) {
        tempMap[OMNIPIPE_RS_LEVEL1] = std::make_shared<CcuRsAlgTemplateY>(param, myRank_, subCommRanks1);
        tempMap[OMNIPIPE_AG_LEVEL1] = std::make_shared<CcuAgAlgTemplateY>(param, myRank_, subCommRanks1);
    }
    HCCL_DEBUG("[InsV2AllReduceOmniPipe2dExecutor][%s] tempMap.size:%u", __func__, tempMap.size());
    
    HCCL_DEBUG("[InsV2AllReduceOmniPipe2dExecutor][%s] threads_.size:%d", __func__, threads_.size());
    levelThreads_.resize(CCU_OMNIPIPE_LEVEL_NUM);

    levelThreads_[CCU_OMNIPIPE_LEVEL0].push_back(threads_[0]);
    levelThreads_[CCU_OMNIPIPE_LEVEL1].push_back(threads_[1]);

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2AllReduceOmniPipe2dExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InitTemplateParams(
            const OpParam& param, const AlgResourceCtxSerializable& resCtx,
            const std::map<u32, std::shared_ptr<CcuAlgTemplateBase>>& tempMap,
            std::map<u32, TemplateResource>& tempResMap,
            std::map<u32, TemplateDataParams>& tempAlgParamMap)
{
    // CCU不用在TemplateResource填充channel（填充在kernelInfo中）
    u32 kernelOffset = 0;
    for (int level = 0; level < OMNIPIPE_AR_LEVEL_NUM; level++) {
        if (tempMap.count(level) > 0) {
            tempResMap[level].ccuKernels.insert(tempResMap[level].ccuKernels.end(),
                    resCtx.ccuKernels.begin() + kernelOffset,
                    resCtx.ccuKernels.begin() + kernelOffset + resCtx.ccuKernelNum[level]);
            kernelOffset += resCtx.ccuKernelNum[level];

            tempAlgParamMap[level].buffInfo.inputPtr = param.inputPtr;
            tempAlgParamMap[level].buffInfo.outputPtr = param.outputPtr;
            tempAlgParamMap[level].buffInfo.inputSize = param.inputSize;
            tempAlgParamMap[level].buffInfo.outputSize = param.outputSize;
            tempAlgParamMap[level].buffInfo.hcclBuff = resCtx.cclMem;
            tempAlgParamMap[level].buffInfo.hcclBuffBaseOff = 0;
            tempAlgParamMap[level].inputSliceStride = dataSize_;
            tempAlgParamMap[level].outputSliceStride = dataSize_;
            tempAlgParamMap[level].localCopyFlag = 0;
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2AllReduceOmniPipe2dExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InitOmniPipeSliceParam(
            OmniPipeSliceParam& sliceParam, const OpParam& param,
            const std::vector<double>& endpointAttrBwAvg,
            std::map<u32, std::shared_ptr<CcuAlgTemplateBase>>& tempMap)
{
    //sliceParam.dataSizePerLoop\ sliceParam.dataWholeSize 在外部赋值
    sliceParam.endpointAttrBw = endpointAttrBwAvg;
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1};
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, 0};
    sliceParam.levelAlgType = {1, 1, 1};
    sliceParam.dataTypeSize = dataTypeSize_;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = param.engine;
    return HCCL_SUCCESS;
}

// 将计算出的单步slice信息初始化到templateParam中
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2AllReduceOmniPipe2dExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::GenTemplateAlgParamsByDimData(
            TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount)
{
    tempAlgParams.count = 0;

    tempAlgParams.stepSliceInfo = stepSliceInfo;

    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff + processedDataCount * dataTypeSize_;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff + processedDataCount * dataTypeSize_;

    HCCL_DEBUG("myrank is %u, inBuffBaseOff is %llu, processedDataCount is %llu, end inBuffBaseOff is %llu", myRank_,
        stepSliceInfo.buffInfo.inBuffBaseOff, processedDataCount, tempAlgParams.buffInfo.inBuffBaseOff);

    HCCL_DEBUG("myrank is %u, outBuffBaseOff is %llu, processedDataCount is %llu, end outBuffBaseOff is %llu", myRank_,
        stepSliceInfo.buffInfo.outBuffBaseOff, processedDataCount, tempAlgParams.buffInfo.outBuffBaseOff);

    HCCL_DEBUG("[%s] inputSliceStrie.size[%u]", __func__, tempAlgParams.stepSliceInfo.stepInputSliceStride.size());

    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.localCopyFlag = 0;
    return HcclResult::HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult InsV2AllReduceOmniPipe2dExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::OrchestrateLoop(
            const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[%s] Start", __func__);

    // 初始化通信域subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    subCommRanks0.push_back(resCtx.algHierarchyInfo.infos[0][0]);
    subCommRanks1.resize(1);
    for (auto i = myRank_ % rankSizeLevel0_; i < resCtx.algHierarchyInfo.infos[0][1].size(); i += rankSizeLevel0_) {
        subCommRanks1[0].push_back(resCtx.algHierarchyInfo.infos[0][1][i]);
    }

    // 初始化template
    std::map<u32, std::shared_ptr<CcuAlgTemplateBase>> tempMap;
    CHK_RET(InitTemplate(param, tempMap, subCommRanks0, subCommRanks1));

    // 初始化资源TemplateResource\TemplateDataParams
    std::map<u32, TemplateResource> tempResMap;
    std::map<u32, TemplateDataParams> tempAlgParamMap;
    CHK_RET(InitTemplateParams(param, resCtx, tempMap, tempResMap, tempAlgParamMap));
    tempResMap[OMNIPIPE_RS_LEVEL0].threads.emplace_back(threads_[0]);
    tempResMap[OMNIPIPE_RS_LEVEL1].threads.emplace_back(threads_[1]);
    tempResMap[OMNIPIPE_AG_LEVEL0].threads.emplace_back(threads_[0]);
    tempResMap[OMNIPIPE_AG_LEVEL1].threads.emplace_back(threads_[1]);

    // 1、计算带宽
    double eqBwLevel0RS = BW_OMNI_DEFAULT;
    double eqBwLevel1RS = BW_OMNI_DEFAULT;
    if (param.opExecuteConfig == OpExecuteConfig::CCU_SCHED) {
        eqBwLevel0RS = BW_OMNI_UBX_CCU_SCHED_RS_MESH;
        eqBwLevel1RS = BW_OMNI_UBX_CCU_SCHED_RS_CLOS;
    } else if (param.opExecuteConfig == OpExecuteConfig::CCU_MS) {
        eqBwLevel0RS = BW_OMNI_UBX_CCU_MS_RS_MESH;
        eqBwLevel1RS = BW_OMNI_UBX_CCU_MS_RS_CLOS;
    }
    eqBwLevel1RS = rankSizeLevel1_ > 1 ? eqBwLevel1RS / (rankSizeLevel1_ - 1) : eqBwLevel1RS;
    std::vector<double> endpointAttrBwAvgRS = {eqBwLevel0RS, eqBwLevel1RS, 1.0};
    double eqBwLevel0AG = BW_OMNI_UBX_CCU_SCHED_AG_MESH;
    double eqBwLevel1AG = BW_OMNI_UBX_CCU_SCHED_AG_CLOS;
    eqBwLevel1AG = rankSizeLevel1_ > 1 ? eqBwLevel1AG / (rankSizeLevel1_ - 1) : eqBwLevel1AG;
    std::vector<double> endpointAttrBwAvgAG = {eqBwLevel0AG, eqBwLevel1AG, 1.0};
    HCCL_INFO("[%s] param.opExecuteConfig:%d, eqBwLevel0RS:%f, eqBwLevel1RS:%f, eqBwLevel0AG:%f, eqBwLevel1AG:%f",
                __func__, param.opExecuteConfig, eqBwLevel0RS, eqBwLevel1RS, eqBwLevel0AG, eqBwLevel1AG);

    // 2.1 获取每个rank切分的数据量count
    auto allRankSplitData = OmniPipeSplitData(rankSize_, dataCount_, dataTypeSize_);
    for (int i=0;i<allRankSplitData.size();i++){
        HCCL_DEBUG("[%s] rankId[%d], allRankSplitData[%d]:%d", __func__, myRank_, i, allRankSplitData[i]);
    }

    // 2.2 计算loop次数
    u64 templateScratchMultiplier = tempMap[OMNIPIPE_RS_LEVEL0]->CalcScratchMultiple(BufferType::DEFAULT, BufferType::DEFAULT);;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplier; // / HCCL_MIN_SLICE_ALIGN* HCCL_MIN_SLICE_ALIGN
    u64 maxCountPerLoop = std::min(transportBoundDataSize, scratchBoundDataSize) / dataTypeSize_;

    u64 loopTimes = allRankSplitData[0] / maxCountPerLoop + ((allRankSplitData[0] % maxCountPerLoop == 0) ? 0 : 1);

    // 2.3 获取每个rank，每个loop切分的数据量count
    auto multiLoopAllRankSplitData =
        OmniPipeSplitRankDataLoop(allRankSplitData, maxCountPerLoop, loopTimes, dataTypeSize_);
    for (int i=0;i<multiLoopAllRankSplitData.size();i++){
        for(int j=0;j<multiLoopAllRankSplitData[i].size();j++){
            HCCL_DEBUG("[%s] rankId[%d], multiLoopAllRankSplitData[%d][%d]:%d", __func__, myRank_, i, j, multiLoopAllRankSplitData[i][j]);
        }
    }

    // 3.1 计算n-1次loop的slice信息
    u64 perLoopSize = multiLoopAllRankSplitData[0][0] * dataTypeSize_;
    perLoopSize = dataSize_ > perLoopSize ? perLoopSize : dataSize_;
    HCCL_DEBUG("[%s] myRank[%u] loopTimes[%llu] perLoopSize[%u] dataSize_[%u] rankSize_[%u]",
                    __func__, myRank_, loopTimes, perLoopSize, dataSize_, rankSize_);
    std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
    std::vector<u64> dataWholeSize(rankSize_, allRankSplitData[myRank_] * dataTypeSize_);
    OmniPipeSliceParam sliceParam;
    sliceParam.dataSizePerLoop = dataSizePerLoop;
    sliceParam.dataWholeSize = dataWholeSize;
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, 0};
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1};
    std::vector<u64> levelAlgType{1, 1, 1};
    sliceParam.levelAlgType = levelAlgType;
    sliceParam.dataTypeSize = dataTypeSize_;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = CommEngine::COMM_ENGINE_CCU;

    // 4 进行一次loop的数据处理
    u64 processedDataCount = 0;
    OmniPipeSliceInfo omniPipeSliceInfoRS;
    OmniPipeSliceInfo omniPipeSliceInfoAG;
    for (u64 loop = 0; loop < loopTimes; loop++) {//loopTimes
        CHK_PRT_RET(
            multiLoopAllRankSplitData.size() <= loop,
            HCCL_ERROR("[InsV2AllReduceOmniPipe2dExecutor][Orchestrate] multiLoopAllRankSplitData.size() <= loop"),
            HCCL_E_PARA);

        // 4.1首轮计算, 或者与上轮不同loop重新计算OmniPipeSliceInfoRS、OmniPipeSliceInfoAG
        if (loop == 0 || !isSameLoop(multiLoopAllRankSplitData[loop - 1], multiLoopAllRankSplitData[loop])) {
            sliceParam.dataSizePerLoop = CalcCountToDataSize(multiLoopAllRankSplitData[loop], dataTypeSize_);
            sliceParam.dataWholeSize = CalcCountToDataSize(allRankSplitData, dataTypeSize_);
            sliceParam.endpointAttrBw = endpointAttrBwAvgRS;
            omniPipeSliceInfoRS = CalcRSOmniPipeSliceInfo(sliceParam);
            sliceParam.endpointAttrBw = endpointAttrBwAvgAG;
            omniPipeSliceInfoAG = CalcAGOmniPipeSliceInfo(sliceParam);
        }

        u64 currDataCount = multiLoopAllRankSplitData[loop][myRank_];
        
        HCCL_DEBUG("[%s] myRank:%d dataCount_ %llu, processedDataCount %llu, maxCountPerLoop %llu, currDataCount %llu",
                        __func__, myRank_, dataCount_, processedDataCount, maxCountPerLoop, currDataCount);

        // 4.2 RS的通信步数
        auto level0StepCountRS = omniPipeSliceInfoRS.dataSliceLevel0.size();
        HCCL_DEBUG("[%s] myRank[%u] level0StepCountRS[%u]", __func__, myRank_, level0StepCountRS);

        if (omniPipeSliceInfoRS.isEmpty()) {
            HCCL_DEBUG("[%s] myRank[%u] omniPipeSliceInfo is Empty!", __func__, myRank_);
        } else {
            auto l0StepNum = omniPipeSliceInfoRS.dataSliceLevel0;
            auto l1StepNum = omniPipeSliceInfoRS.dataSliceLevel1;
            HCCL_DEBUG("[%s] myRank[%u] L0 stepNum[%u]", __func__, myRank_, l0StepNum.size());
            HCCL_DEBUG("[%s] myRank[%u] L1 stepNum[%u]", __func__, myRank_, l1StepNum.size());
        }

        // template间同步所需信息计算
        ThreadHandle mainThread = threads_[0];
        std::vector<ThreadHandle> syncThreads{threads_[1]};
        std::vector<u32> notifyIdxesMainToSub{0};
        std::vector<u32> notifyIdxesSubToMain{0};

        // 4.3 RS for内层2d
        for (auto i = 0; i < level0StepCountRS; ++i) {
            // 第一步开始前同步
            CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));
            
            // level0
            GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_RS_LEVEL0], omniPipeSliceInfoRS.dataSliceLevel0[i], processedDataCount);
            CHK_RET(tempMap[OMNIPIPE_RS_LEVEL0]->KernelRun(param, tempAlgParamMap[OMNIPIPE_RS_LEVEL0], tempResMap[OMNIPIPE_RS_LEVEL0]));
            
            // level1
            GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_RS_LEVEL1], omniPipeSliceInfoRS.dataSliceLevel1[i], processedDataCount);
            CHK_RET(tempMap[OMNIPIPE_RS_LEVEL1]->KernelRun(param, tempAlgParamMap[OMNIPIPE_RS_LEVEL1], tempResMap[OMNIPIPE_RS_LEVEL1]));
            CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
        }

        // 4.4 AG本地拷贝
        HCCL_DEBUG("[%s] AG local copy start, myRank[%d], currDataCount %llu, processedDataCount %llu",
                        __func__, myRank_, currDataCount, processedDataCount);
        CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));
        // 本地拷贝
        TemplateDataParams tempAlgParamLocalCopy;
        tempAlgParamLocalCopy.buffInfo.inputPtr = param.inputPtr;
        tempAlgParamLocalCopy.buffInfo.outputPtr = param.outputPtr;
        tempAlgParamLocalCopy.buffInfo.inputSize = param.inputSize;
        tempAlgParamLocalCopy.buffInfo.outputSize = param.outputSize;
        tempAlgParamLocalCopy.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamLocalCopy.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParamLocalCopy.count = currDataCount;
        std::vector<u64> perRankOffset;
        perRankOffset.resize(rankSize_);
        perRankOffset[0] = 0;
        for (u32 i = 1; i < rankSize_; i++) {
            perRankOffset[i] = perRankOffset[i - 1] + allRankSplitData[i - 1];
        }
        tempAlgParamLocalCopy.buffInfo.inBuffBaseOff = perRankOffset[myRank_] * dataTypeSize_ + processedDataCount * dataTypeSize_;
        tempAlgParamLocalCopy.buffInfo.outBuffBaseOff = perRankOffset[myRank_] * dataTypeSize_ + processedDataCount * dataTypeSize_;
        tempAlgParamLocalCopy.inputSliceStride = 0;
        tempAlgParamLocalCopy.outputSliceStride = 0;
        tempAlgParamLocalCopy.repeatNum = rankSize_;
        tempAlgParamLocalCopy.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParamLocalCopy.localCopyFlag = 1;

        tempResMap[OMNIPIPE_AG_LEVEL0].threads.clear();
        tempResMap[OMNIPIPE_AG_LEVEL0].threads.emplace_back(threads_[0]);
        CHK_RET(tempMap[OMNIPIPE_AG_LEVEL0]->KernelRun(param, tempAlgParamLocalCopy, tempResMap[OMNIPIPE_AG_LEVEL0]));
        CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
        HCCL_DEBUG("[%s] AG local copy end", __func__);

        // 4.5 AG for内层2d
        u32 level0StepCountAG = omniPipeSliceInfoAG.dataSliceLevel0.size();
        HCCL_DEBUG("[%s] level0StepCountAG %u", __func__, level0StepCountAG);
        for (u32 i = 0; i < level0StepCountAG; i++) {
            // 初始化机内template param
            GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_AG_LEVEL0], omniPipeSliceInfoAG.dataSliceLevel0[i], processedDataCount);
            GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_AG_LEVEL1], omniPipeSliceInfoAG.dataSliceLevel1[i], processedDataCount);

            //第一步开始前同步
            CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));

            // 执行机内template任务
            // level0
            CHK_RET(tempMap[OMNIPIPE_AG_LEVEL0]->KernelRun(param, tempAlgParamMap[OMNIPIPE_AG_LEVEL0], tempResMap[OMNIPIPE_AG_LEVEL0]));
            // level1
            CHK_RET(tempMap[OMNIPIPE_AG_LEVEL1]->KernelRun(param, tempAlgParamMap[OMNIPIPE_AG_LEVEL1], tempResMap[OMNIPIPE_AG_LEVEL1]));

            //第一步做完后回到主流做尾同步
            CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
        }
        
        processedDataCount += currDataCount;
    }

    HCCL_INFO("[%s][OrchestrateLoop] End.", __func__);
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLREDUCE, CcuV2AllReduceOmniPipe2D,
                                InsV2AllReduceOmniPipe2dExecutor, TopoMatchUBX, 
                                CcuTempReduceScatterOmniPipeMesh1DMem2Mem, CcuTempReduceScatterOmniPipeNHR1DMem2Mem, 
                                CcuTempAllGatherOmniPipeMesh1DMem2Mem, CcuTempAllGatherOmniPipeNHR1DMem2Mem);
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLREDUCE, CcuV2AllReduceOmniPipe2DMs,
                                InsV2AllReduceOmniPipe2dExecutor, TopoMatchUBX, 
                                CcuTempReduceScatterOmniPipeMesh1D, CcuTempReduceScatterOmniPipeNHR1DMem2Mem, 
                                CcuTempAllGatherOmniPipeMesh1DMem2Mem, CcuTempAllGatherOmniPipeNHR1DMem2Mem);
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#endif 
}
