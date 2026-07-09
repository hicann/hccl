/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_scatter_omnipipe_2d_executor.h"
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_reduce_scatter_omnipipe_mesh1d_mem2mem.h"
#include "ccu_temp_reduce_scatter_omnipipe_nhr1d_mem2mem.h"
#include "ccu_temp_reduce_scatter_omnipipe_mesh1d.h"
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#endif

namespace ops_hccl {
constexpr u32 DUAL_TEMPLATE_NUM = 2;

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
InsV2ReduceScatterOmniPipe2dExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::
    InsV2ReduceScatterOmniPipe2dExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult InsV2ReduceScatterOmniPipe2dExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::
    CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));

    for (auto i = 0; i < algHierarchyInfo.infos.size(); ++i) {
        for (auto j = 0; j < algHierarchyInfo.infos[i].size(); ++j) {
            for (auto k = 0; k < algHierarchyInfo.infos[i][j].size(); ++k) {
                HCCL_DEBUG("[CalcAlgHierarchyInfo] myRank[%u] (%d, %d, %d) %u", topoInfo->userRank, i, j, k,
                    algHierarchyInfo.infos[i][j][k]);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult InsV2ReduceScatterOmniPipe2dExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::InitCommInfo(
    const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;

    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[InitCommInfo] rankSizeLevel0 is 0");
        return HcclResult::HCCL_E_PARA;
    }

    rankSizeLevel1_ = algHierarchyInfo.infos[0][1].size() / rankSizeLevel0_;
    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[InitCommInfo] rankSizeLevel1 is 0");
        return HcclResult::HCCL_E_PARA;
    }
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;

    HCCL_DEBUG("[%s] myRank[%u] rankSize[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] "
        "rankIdxLevel1[%u] devType[%u] dataCount[%u] dataType[%u] dataTypeSize[%u]",
        __func__, myRank_, rankSize_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_, devType_,
        dataCount_, dataType_, dataTypeSize_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult InsV2ReduceScatterOmniPipe2dExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    HCCL_DEBUG("[%s] myRank[%u] CalcRes start", __func__, myRank_);
    CHK_RET(InitCommInfo(param, topoInfo, algHierarchyInfo));

    // 重复的template构造
    std::vector<std::vector<u32>> subCommRanks0{algHierarchyInfo.infos[0][0]};
    auto size = algHierarchyInfo.infos[0][1].size() / algHierarchyInfo.infos[0][0].size();
    std::vector<std::vector<u32>> subCommRanks1(1, std::vector<u32>(size, 0));
    u32 index = 0;
    for (auto i = myRank_ % rankSizeLevel0_; i < algHierarchyInfo.infos[0][1].size(); i += rankSizeLevel0_) {
        subCommRanks1[0][index++] = algHierarchyInfo.infos[0][1][i];
    }
    InsAlgTempLevel0 algTempLevel0(param, myRank_, subCommRanks0);
    InsAlgTempLevel1 algTempLevel1(param, myRank_, subCommRanks1);

    AlgResourceRequest resReqLevel0; // X
    CHK_RET(algTempLevel0.CalcRes(comm, param, topoInfo, resReqLevel0));
    AlgResourceRequest resReqLevel1; // Y
    CHK_RET(algTempLevel1.CalcRes(comm, param, topoInfo, resReqLevel1));

    resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                          resReqLevel0.ccuKernelInfos.begin(),
                                          resReqLevel0.ccuKernelInfos.end());
    resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                          resReqLevel1.ccuKernelInfos.begin(),
                                          resReqLevel1.ccuKernelInfos.end());
    resourceRequest.ccuKernelNum.insert(resourceRequest.ccuKernelNum.end(),
                                        resReqLevel0.ccuKernelNum.begin(),
                                        resReqLevel0.ccuKernelNum.end());   
    resourceRequest.ccuKernelNum.insert(resourceRequest.ccuKernelNum.end(),
                                        resReqLevel1.ccuKernelNum.begin(),
                                        resReqLevel1.ccuKernelNum.end());

    // 三条流，分别是主流（控制thread）+两条从流（template0的流和template1的流）
    // 申请一条控制thread作为主thread，该thread仅用于两个template之间同步
    resourceRequest.notifyNumOnMainThread = DUAL_TEMPLATE_NUM;
    // 由于主thread被单独作为控制thread，因此总的slaveThread需要额外加上两个template的thread
    resourceRequest.slaveThreadNum = resReqLevel0.slaveThreadNum + resReqLevel1.slaveThreadNum + DUAL_TEMPLATE_NUM;

    // template0的流需要的notify数量，+1是因为需要和控制thread做同步
    resourceRequest.notifyNumPerThread.emplace_back(resReqLevel0.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReqLevel0.notifyNumPerThread.begin(),
                                              resReqLevel0.notifyNumPerThread.end());
    // template1的流需要的notify数量，+1是因为需要和控制thread做同步
    resourceRequest.notifyNumPerThread.emplace_back(resReqLevel1.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReqLevel1.notifyNumPerThread.begin(),
                                              resReqLevel1.notifyNumPerThread.end());
    HCCL_DEBUG("[%s] slaveThreadNum[%u]", __func__, resourceRequest.slaveThreadNum);
    HCCL_DEBUG("[%s] myRank[%u] CalcRes end", __func__, myRank_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult InsV2ReduceScatterOmniPipe2dExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_DEBUG("[%s] myRank[%u] start", __func__, myRank_);
    threads_ = resCtx.threads;
    HCCL_DEBUG("[%s] threads_ size[%u]", __func__, threads_.size()); // 3: main+x+y
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    maxTmpMemSize_ = resCtx.cclMem.size;
    rankSizeLevel0_ = resCtx.algHierarchyInfo.infos[0][0].size();
    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[Orchestrate] rankSizeLevel0 is 0, expected to be greater than 0");
        return HcclResult::HCCL_E_PARA;
    }

    rankSizeLevel1_ = resCtx.algHierarchyInfo.infos[0][1].size() / rankSizeLevel0_;
    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[Orchestrate] rankSizeLevel1 is 0, expected to be greater than 0");
        return HcclResult::HCCL_E_PARA;
    }
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;

    HCCL_DEBUG("[%s] myRank[%u] rankSize[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]",
        __func__, myRank_, rankSize_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] executor kernel run failed", __func__, HCCL_ERROR_CODE(ret)), ret);
    HCCL_DEBUG("[%s] myRank[%u] end", __func__, myRank_);
    return HcclResult::HCCL_SUCCESS;
}

// 将计算出的单步slice信息初始化到templateParam中
template <typename M, typename X, typename Y>
HcclResult InsV2ReduceScatterOmniPipe2dExecutor<M, X, Y>::GenTemplateAlgParamsByDimData(
    TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount)
{
    tempAlgParams.count = 0;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.buffInfo.inBuffBaseOff
        = stepSliceInfo.buffInfo.inBuffBaseOff + processedDataCount * dataTypeSize_;
    tempAlgParams.buffInfo.outBuffBaseOff
        = stepSliceInfo.buffInfo.outBuffBaseOff + processedDataCount * dataTypeSize_;

    HCCL_DEBUG("[%s] myRank[%u] inBuffBaseOff[%llu] processedDataCount[%llu] end inBuffBaseOff[%llu]", __func__,
        myRank_, stepSliceInfo.buffInfo.inBuffBaseOff, processedDataCount, tempAlgParams.buffInfo.inBuffBaseOff);

    HCCL_DEBUG("[%s] myRank[%u] outBuffBaseOff[%llu] processedDataCount[%llu] end outBuffBaseOff[%llu]", __func__,
        myRank_, stepSliceInfo.buffInfo.outBuffBaseOff, processedDataCount, tempAlgParams.buffInfo.outBuffBaseOff);

    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.localCopyFlag = 0;

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult InsV2ReduceScatterOmniPipe2dExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::PrepareResForTemplate(
        const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTempLevel0 &algTempLevel0,
        InsAlgTempLevel1 &algTempLevel1)
{
    HCCL_DEBUG("[%s] start", __func__);
    // 获取每个temp的线程数
    u64 level0ThreadsNum = algTempLevel0.GetThreadNum();
    u64 level1ThreadsNum = algTempLevel1.GetThreadNum();
    HCCL_DEBUG("[%s] level0ThreasNum[%u] level1ThreadsNum[%u]", __func__, level0ThreadsNum, level1ThreadsNum);

    // 获取template各自的主thread上有多少notify
    AlgResourceRequest level0TempRequest;
    CHK_RET(algTempLevel0.GetRes(level0TempRequest));
    notifyIdxControlToTemplates_.push_back(level0TempRequest.notifyNumOnMainThread);
    AlgResourceRequest level1TempRequest;
    CHK_RET(algTempLevel1.GetRes(level1TempRequest));
    notifyIdxControlToTemplates_.push_back(level1TempRequest.notifyNumOnMainThread);
    notifyIdxTemplatesToControl_.push_back(0);
    notifyIdxTemplatesToControl_.push_back(1);

    level0Threads_.assign(threads_.begin() + 1, threads_.begin() + 1 + level0ThreadsNum);
    level1Threads_.assign(threads_.begin() + 1 + level0ThreadsNum, threads_.end());
    HCCL_DEBUG("[%s] level0Threads size[%u], level1Threads size[%u]", __func__, level0Threads_.size(), level1Threads_.size());

    // 控制线程 用于算法同步
    controlThread_ = threads_.at(0);
    // xy轴各自的主线程
    templateMainThreads_.push_back(level0Threads_.at(0));
    templateMainThreads_.push_back(level1Threads_.at(0));
    HCCL_DEBUG("[%s] templateMainThreads size[%u]", __func__, templateMainThreads_.size());

    // 单独本地拷贝使用
    templateLocalCopyThreads_.push_back(level0Threads_.at(0));

    HCCL_DEBUG("[%s] run success", __func__);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult InsV2ReduceScatterOmniPipe2dExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[%s] Start", __func__);

    auto algHierarchyInfo = resCtx.algHierarchyInfo;
    // 重复的template构造
    std::vector<std::vector<u32>> subCommRanks0{algHierarchyInfo.infos[0][0]};
    auto size = algHierarchyInfo.infos[0][1].size() / algHierarchyInfo.infos[0][0].size();
    std::vector<std::vector<u32>> subCommRanks1(1, std::vector<u32>(size, 0));
    u32 index = 0;
    for (auto i = myRank_ % rankSizeLevel0_; i < algHierarchyInfo.infos[0][1].size(); i += rankSizeLevel0_) {
        subCommRanks1[0][index++] = algHierarchyInfo.infos[0][1][i];
    }
    InsAlgTempLevel0 algTemplateLevel0(param, myRank_, subCommRanks0); // [[0,1]]
    InsAlgTempLevel1 algTemplateLevel1(param, myRank_, subCommRanks1); // [[0,2]]

    TemplateDataParams tempAlgParamsCommon;
    tempAlgParamsCommon.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsCommon.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsCommon.buffInfo.inputSize = param.inputSize;
    tempAlgParamsCommon.buffInfo.outputSize = param.outputSize;
    tempAlgParamsCommon.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsCommon.buffInfo.hcclBuffBaseOff = 0;
    tempAlgParamsCommon.inputSliceStride = dataSize_;
    tempAlgParamsCommon.outputSliceStride = dataSize_;
    tempAlgParamsCommon.localCopyFlag = 0;

    TemplateResource templateResourceCommon;
    if (param.engine == COMM_ENGINE_CCU) {
        HCCL_DEBUG("[%s] myRank[%u] param engine is CCU", __func__, myRank_);
    } else {
        HCCL_DEBUG("[%s] myRank[%u] param engine is not CCU", __func__, myRank_);
    }

    TemplateResource templateResourceLevel0 = templateResourceCommon;
    templateResourceLevel0.threads.push_back(resCtx.threads[1]);
    templateResourceLevel0.ccuKernels.insert(templateResourceLevel0.ccuKernels.end(),
                resCtx.ccuKernels.begin(),
                resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
    
    TemplateResource templateResourceLevel1 = templateResourceCommon;
    templateResourceLevel1.threads.push_back(resCtx.threads[2]);
    templateResourceLevel1.ccuKernels.insert(templateResourceLevel1.ccuKernels.end(),
                resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
                resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);

    PrepareResForTemplate(param, resCtx, algTemplateLevel0, algTemplateLevel1);

    // 1、计算带宽
    double eqBwLevel0 = BW_OMNI_DEFAULT;
    double eqBwLevel1 = BW_OMNI_DEFAULT;
    if (param.opExecuteConfig == OpExecuteConfig::CCU_SCHED) {
        eqBwLevel0 = BW_OMNI_UBX_CCU_SCHED_RS_MESH;
        eqBwLevel1 = BW_OMNI_UBX_CCU_SCHED_RS_CLOS;
    } else if (param.opExecuteConfig == OpExecuteConfig::CCU_MS) {
        eqBwLevel0 = BW_OMNI_UBX_CCU_MS_RS_MESH;
        eqBwLevel1 = BW_OMNI_UBX_CCU_MS_RS_CLOS;
    }
    eqBwLevel1 = rankSizeLevel1_ > 1 ? eqBwLevel1 / (rankSizeLevel1_ - 1) : eqBwLevel1;
    std::vector<double> endpointAttrBwAvg = {eqBwLevel0, eqBwLevel1, 1.0};
    HCCL_INFO("[%s] param.opExecuteConfig:%d, eqBwLevel0:%f, eqBwLevel1:%f",
                __func__, param.opExecuteConfig, eqBwLevel0, eqBwLevel1);

    // 2、计算loop
    u64 templateScratchMultiplier = algTemplateLevel0.CalcScratchMultiple(BufferType::DEFAULT, BufferType::DEFAULT);
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplier;
    u64 maxCountPerLoop = std::min(transportBoundDataSize, scratchBoundDataSize) / dataTypeSize_;
    u64 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);
    u64 perLoopSize = maxCountPerLoop * dataTypeSize_;
    perLoopSize = dataSize_ > perLoopSize ? perLoopSize : dataSize_;
    HCCL_DEBUG("[%s] myRank[%u] loopTimes[%llu] perLoopSize[%u] dataSize_[%u] rankSize_[%u]",
                    __func__, myRank_, loopTimes, perLoopSize, dataSize_, rankSize_);
    std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
    std::vector<u64> dataWholeSize(rankSize_, dataSize_);

    // 3、计算n-1次loop的slice信息
    OmniPipeSliceParam sliceParam;
    sliceParam.dataSizePerLoop = dataSizePerLoop;
    sliceParam.dataWholeSize = dataWholeSize;
    sliceParam.endpointAttrBw = endpointAttrBwAvg;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = CommEngine::COMM_ENGINE_CCU;
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, 0};
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1};
    std::vector<u64> levelAlgType = {1, 1, 1};
    sliceParam.levelAlgType = levelAlgType;
    sliceParam.dataTypeSize = dataTypeSize_;
    OmniPipeSliceInfo alignSliceInfo = CalcRSOmniPipeSliceInfo(sliceParam);

    // 4、计算第n次的loop的slice信息
    OmniPipeSliceInfo tailSliceInfo;
    if (dataCount_ % maxCountPerLoop != 0) {
        u64 lastLoopSize = (dataCount_ % maxCountPerLoop) * dataTypeSize_;
        HCCL_DEBUG("[%s] lastLoopSize:%d", __func__, lastLoopSize);
        std::vector<u64> dataSizePerLoop(rankSize_, lastLoopSize);
        std::vector<u64> dataWholeSize(rankSize_, dataSize_);
        sliceParam.dataSizePerLoop = dataSizePerLoop;
        sliceParam.dataWholeSize = dataWholeSize;
        tailSliceInfo = CalcRSOmniPipeSliceInfo(sliceParam);
    }

    // 5、一次loop的数据处理
    u64 processedDataCount = 0;
    OmniPipeSliceInfo omniPipeSliceInfo;
    TemplateDataParams tempAlgParamsLevel0 = tempAlgParamsCommon;
    TemplateDataParams tempAlgParamsLevel1 = tempAlgParamsCommon;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        // 5.1 确定当前是前n-1次loop的slice结果，还是存在尾块时最后一次loop的slice结果
        if (loop == loopTimes - 1 && !tailSliceInfo.isEmpty()) {
            omniPipeSliceInfo = tailSliceInfo;
        } else {
            omniPipeSliceInfo = alignSliceInfo;
        }

        auto innerServerStepNum = omniPipeSliceInfo.dataSliceLevel0.size();
        HCCL_DEBUG("[%s] myRank[%u] currDataCount[%llu] innerServerStepNum[%u]",
                    __func__, myRank_, currDataCount, innerServerStepNum);

        if (omniPipeSliceInfo.isEmpty()) {
            HCCL_DEBUG("[%s] myRank[%u] omniPipeSliceInfo is Empty!", __func__, myRank_);
        } else {
            auto l0si = omniPipeSliceInfo.dataSliceLevel0;
            auto l1si = omniPipeSliceInfo.dataSliceLevel1;
            HCCL_DEBUG("[%s] myRank[%u] L0 stepNum[%u]", __func__, myRank_, l0si.size());
            HCCL_DEBUG("[%s] myRank[%u] L1 stepNum[%u]", __func__, myRank_, l1si.size());
        }

        // 5.2 for内层2d
        for (auto i = 0; i < innerServerStepNum; ++i) {
            // 前同步
            CHK_RET(PreSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxControlToTemplates_));

            // 下发X轴
            GenTemplateAlgParamsByDimData(tempAlgParamsLevel0, omniPipeSliceInfo.dataSliceLevel0[i], processedDataCount);
            CHK_RET(algTemplateLevel0.KernelRun(param, tempAlgParamsLevel0, templateResourceLevel0));

            // 下发Y轴
            GenTemplateAlgParamsByDimData(tempAlgParamsLevel1, omniPipeSliceInfo.dataSliceLevel1[i], processedDataCount);
            CHK_RET(algTemplateLevel1.KernelRun(param, tempAlgParamsLevel1, templateResourceLevel1));

            // 尾同步
            CHK_RET(PostSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxTemplatesToControl_));
        }

        // 本地拷贝 userin->userout
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(controlThread_, templateLocalCopyThreads_, notifyIdxMainToSub));

        TemplateDataParams tempAlgParamsLocalCopy = tempAlgParamsCommon;
        tempAlgParamsLocalCopy.localCopyFlag = 1;
        tempAlgParamsLocalCopy.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamsLocalCopy.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParamsLocalCopy.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParamsLocalCopy.count = currDataCount;
        tempAlgParamsLocalCopy.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParamsLocalCopy.buffInfo.inBuffBaseOff = myRank_ * dataSize_ + processedDataCount * dataTypeSize_;

        HCCL_DEBUG("[%s] myRank[%u] localCopy inBuffBaseOff[%lu] outBuffBaseOff[%lu] sliceSize[%lu]", __func__,
            myRank_, tempAlgParamsLocalCopy.buffInfo.inBuffBaseOff, tempAlgParamsLocalCopy.buffInfo.outBuffBaseOff,
            tempAlgParamsLocalCopy.sliceSize);
        CHK_RET(algTemplateLevel0.KernelRun(param, tempAlgParamsLocalCopy, templateResourceLevel0));
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(controlThread_, templateLocalCopyThreads_, notifyIdxSubToMain));

        processedDataCount += currDataCount;
        HCCL_DEBUG("[%s] processedDataCount[%llu]", __func__, processedDataCount);
    }

    HCCL_INFO("[%s] End.", __func__);
    return HcclResult::HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CcuV2ReduceScatterOmniPipe,
    InsV2ReduceScatterOmniPipe2dExecutor, TopoMatchUBX, CcuTempReduceScatterOmniPipeMesh1DMem2Mem, CcuTempReduceScatterOmniPipeNHR1DMem2Mem);
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CcuV2ReduceScatterOmniPipeMs,
    InsV2ReduceScatterOmniPipe2dExecutor, TopoMatchUBX, CcuTempReduceScatterOmniPipeMesh1D, CcuTempReduceScatterOmniPipeNHR1DMem2Mem);
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#endif
}