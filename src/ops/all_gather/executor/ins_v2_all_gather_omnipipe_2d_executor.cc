/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_omnipipe_2d_executor.h"
#include "alg_env_config.h"
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_all_gather_omnipipe_mesh1d_mem2mem.h"
#include "ccu_temp_all_gather_omnipipe_nhr1d_mem2mem.h"
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#endif

namespace ops_hccl {
template <typename AlgTopoMatch, typename CcuAlgTempLevel0, typename CcuAlgTempLevel1>
InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, CcuAlgTempLevel0, CcuAlgTempLevel1>::InsV2AllGatherOmniPipe2DExecutor()
{
}

template <typename AlgTopoMatch, typename CcuAlgTempLevel0, typename CcuAlgTempLevel1>
HcclResult InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, CcuAlgTempLevel0, CcuAlgTempLevel1>::CalcAlgHierarchyInfo(
    HcclComm comm,
    TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    auto userrank = topoInfo->userRank;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));

    for (auto i = 0; i < algHierarchyInfo.infos.size(); ++i) {
        for (auto j = 0; j < algHierarchyInfo.infos[i].size(); ++j) {
            for (auto k = 0; k < algHierarchyInfo.infos[i][j].size(); ++k) {
                HCCL_DEBUG("[%s] myRank[%u] (%d, %d, %d) %u", __func__, topoInfo->userRank, i, j, k,
                    algHierarchyInfo.infos[i][j][k]);
            }
        }
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, CcuAlgTempLevel0, InsAlgTempLevel1>::InitCommInfo(
    const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel0 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    rankSizeLevel1_ = algHierarchyInfo.infos[0][1].size() / rankSizeLevel0_;
    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel1 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;

    HCCL_INFO("[%s] myRank[%u] rankSize[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] "
        "rankIdxLevel1[%u] devType[%u] dataCount[%u] dataType[%u] dataTypeSize[%u]",
        __func__, myRank_, rankSize_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_, devType_,
        dataCount_, dataType_, dataTypeSize_);

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuAlgTempLevel0, typename CcuAlgTempLevel1>
HcclResult InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, CcuAlgTempLevel0, CcuAlgTempLevel1>::CalcResLevel(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    CommonAlgTemplateBase &tempAlg, AlgResourceRequest &resourceRequest)
{
    AlgResourceRequest resReqlevel;
    CHK_RET(tempAlg.CalcRes(comm, param, topoInfo, resReqlevel));
    resourceRequest.slaveThreadNum += resReqlevel.slaveThreadNum + 1;
    resourceRequest.notifyNumOnMainThread += 1;
    resourceRequest.notifyNumPerThread.emplace_back(resReqlevel.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReqlevel.notifyNumPerThread.begin(),
                                              resReqlevel.notifyNumPerThread.end());
    resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(), resReqlevel.ccuKernelInfos.begin(),
                                          resReqlevel.ccuKernelInfos.end());
    resourceRequest.ccuKernelNum.insert(resourceRequest.ccuKernelNum.end(), resReqlevel.ccuKernelNum.begin(),
                                        resReqlevel.ccuKernelNum.end());

    return HcclResult::HCCL_SUCCESS;
}


template <typename AlgTopoMatch, typename CcuAlgTempLevel0, typename CcuAlgTempLevel1>
HcclResult InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, CcuAlgTempLevel0, CcuAlgTempLevel1>::CalcRes(
    HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    HCCL_DEBUG("[%s] ins0618 alleref start", __func__);
    CHK_RET(InitCommInfo(param, topoInfo, algHierarchyInfo));

    std::vector<std::vector<u32>> subCommRanks0{algHierarchyInfo.infos[0][0]};
    auto size = algHierarchyInfo.infos[0][1].size() / algHierarchyInfo.infos[0][0].size();
    std::vector<std::vector<u32>> subCommRanks1(1, std::vector<u32>(size, 0));
    u32 index = 0;
    for (auto i = myRank_ % rankSizeLevel0_; i < algHierarchyInfo.infos[0][1].size(); i += rankSizeLevel0_) {
        subCommRanks1[0][index++] = algHierarchyInfo.infos[0][1][i];
    }
    CcuAlgTempLevel0 algTempLevel0(param, myRank_, subCommRanks0);
    CcuAlgTempLevel1 algTempLevel1(param, myRank_, subCommRanks1);

    CHK_RET(CalcResLevel(comm, param, topoInfo, algTempLevel0, resourceRequest));
    CHK_RET(CalcResLevel(comm, param, topoInfo, algTempLevel1, resourceRequest));
    HCCL_DEBUG("[%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

// 将计算出的单步slice信息初始化到templateParam中
template <typename M, typename X, typename Y>
HcclResult InsV2AllGatherOmniPipe2DExecutor<M, X, Y>::GenTemplateAlgParamsByDimData(
    const OpParam &param, TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount) {
    tempAlgParams.buffInfo.inBuffType = stepSliceInfo.buffInfo.inBuffType;
    tempAlgParams.buffInfo.outBuffType = stepSliceInfo.buffInfo.outBuffType;

    tempAlgParams.count = 0;
    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff + processedDataCount * dataTypeSize_;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff + processedDataCount * dataTypeSize_;

    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;

    tempAlgParams.localCopyFlag = 0;

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuAlgTempLevel0, typename CcuAlgTempLevel1>
HcclResult InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, CcuAlgTempLevel0, CcuAlgTempLevel1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_DEBUG("[%s] start", __func__);
    threads_ = resCtx.threads;
    HCCL_DEBUG("[%s]threads size: %u", __func__, threads_.size()); // 3: main+x+y
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    rankSizeLevel0_ = resCtx.algHierarchyInfo.infos[0][0].size();
    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel0 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    rankSizeLevel1_ = resCtx.algHierarchyInfo.infos[0][1].size() / rankSizeLevel0_;
    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel1 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;

    HCCL_DEBUG("[%s] myRank[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]",
        __func__, myRank_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] executor kernel run failed", __func__, HCCL_ERROR_CODE(ret)), ret);
    HCCL_DEBUG("[%s] myRank[%u] end", __func__, myRank_);

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuAlgTempLevel0, typename CcuAlgTempLevel1>
HcclResult InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, CcuAlgTempLevel0, CcuAlgTempLevel1>::PrepareResForTemplate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, CcuAlgTempLevel0 &algTempLevel0,
    CcuAlgTempLevel1 &algTempLevel1)
{
    HCCL_DEBUG("[%s] start", __func__);
    // 获取每个temp的线程数
    u64 level0ThreadsNum = algTempLevel0.GetThreadNum();
    u64 level1ThreadsNum = algTempLevel1.GetThreadNum();
    HCCL_DEBUG("[%s] level0ThreasNum[%u] level1ThreadsNum[%u]", __func__, level0ThreadsNum, level1ThreadsNum);

    level0Threads_.assign(threads_.begin() + 1, threads_.begin() + 1 + level0ThreadsNum);
    level1Threads_.assign(threads_.begin() + 1 + level0ThreadsNum, threads_.end());
    HCCL_DEBUG("[%s] level0Threads size[%u] level1Threads size[%u]",
        __func__, level0Threads_.size(), level1Threads_.size());

    // 控制线程 用于算法同步
    controlThread_ = threads_.at(0);
    // xy轴各自的主线程
    templateMainThreads_.push_back(level0Threads_.at(0));
    templateMainThreads_.push_back(level1Threads_.at(0));
    HCCL_DEBUG("[%s] templateMainThreads size[%u]", __func__, templateMainThreads_.size());

    // 获取template各自的主thread上有多少notify
    AlgResourceRequest level0TempRequest;
    CHK_RET(algTempLevel0.GetRes(level0TempRequest));
    notifyIdxControlToTemplates_.push_back(level0TempRequest.notifyNumOnMainThread);
    AlgResourceRequest level1TempRequest;
    CHK_RET(algTempLevel1.GetRes(level1TempRequest));
    notifyIdxControlToTemplates_.push_back(level1TempRequest.notifyNumOnMainThread);
    notifyIdxTemplatesToControl_.push_back(0);
    notifyIdxTemplatesToControl_.push_back(1);
    HCCL_DEBUG("[%s] notifyIdxControlToTemplates_ size[%u]", __func__, notifyIdxControlToTemplates_.size());
    HCCL_DEBUG("[%s] notifyIdxTemplatesToControl_ size[%u]", __func__, notifyIdxTemplatesToControl_.size());

    // 单独本地拷贝使用
    templateLocalCopyThreads_.push_back(level0Threads_.at(0));

    HCCL_DEBUG("[%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuAlgTempLevel0, typename CcuAlgTempLevel1>
HcclResult InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, CcuAlgTempLevel0, CcuAlgTempLevel1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_DEBUG("[%s] Start", __func__);
    auto algHierarchyInfo = resCtx.algHierarchyInfo;
    std::vector<std::vector<u32>> subCommRanks0{algHierarchyInfo.infos[0][0]};
    auto size = algHierarchyInfo.infos[0][1].size() / algHierarchyInfo.infos[0][0].size();
    std::vector<std::vector<u32>> subCommRanks1(1, std::vector<u32>(size, 0));
    u32 index = 0;
    for (auto i = myRank_ % rankSizeLevel0_; i < algHierarchyInfo.infos[0][1].size(); i += rankSizeLevel0_) {
        subCommRanks1[0][index++] = algHierarchyInfo.infos[0][1][i];
    }
    CcuAlgTempLevel0 algTemplateLevel0(param, myRank_, subCommRanks0);
    CcuAlgTempLevel1 algTemplateLevel1(param, myRank_, subCommRanks1);

    TemplateDataParams tempAlgParamsCommon;
    tempAlgParamsCommon.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsCommon.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsCommon.buffInfo.inputSize = param.inputSize;
    tempAlgParamsCommon.buffInfo.outputSize = param.outputSize;
    tempAlgParamsCommon.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsCommon.inputSliceStride = dataSize_;
    tempAlgParamsCommon.outputSliceStride = dataSize_;
    tempAlgParamsCommon.localCopyFlag = 0;

    TemplateResource templateResourceCommon;
    TemplateResource templateResourceLevel0 = templateResourceCommon;
    templateResourceLevel0.threads.push_back(resCtx.threads[1]);
    templateResourceLevel0.ccuKernels.insert(templateResourceLevel0.ccuKernels.end(), resCtx.ccuKernels.begin(),
            resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
    TemplateResource templateResourceLevel1 = templateResourceCommon;
    templateResourceLevel1.threads.push_back(resCtx.threads[2]);
    templateResourceLevel1.ccuKernels.insert(templateResourceLevel1.ccuKernels.end(),
            resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
            resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);
    PrepareResForTemplate(param, resCtx, algTemplateLevel0, algTemplateLevel1);

    // 1. 计算带宽
    double eqBwLevel0 = BW_OMNI_UBX_CCU_SCHED_AG_MESH;
    double eqBwLevel1 = BW_OMNI_UBX_CCU_SCHED_AG_CLOS;
    eqBwLevel1 = rankSizeLevel1_ > 1 ? eqBwLevel1 / (rankSizeLevel1_ - 1) : eqBwLevel1;
    std::vector<double> endpointAttrBw = {eqBwLevel0, eqBwLevel1, 1.0};
    HCCL_INFO("[%s] eqBwLevel0:%f, eqBwLevel1:%f", __func__, eqBwLevel0, eqBwLevel1);

    // 2. 计算loop  ccu不用cclbuff, 根据UB_MAX_DATA_SIZE来计算
    u64 maxCountPerLoop = static_cast<u64>(UB_MAX_DATA_SIZE) / dataTypeSize_;
    u64 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);
    u64 perLoopSize = maxCountPerLoop * dataTypeSize_;
    perLoopSize = dataSize_ > perLoopSize ? perLoopSize : dataSize_;
    HCCL_DEBUG("[%s] myRank[%u], loopTimes[%llu], perLoopSize[%llu], dataSize_[%llu]", __func__, myRank_, loopTimes, perLoopSize, dataSize_);
    std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
    std::vector<u64> dataWholeSize(rankSize_, dataSize_);

    // 3. 计算n-1次loop的slice信息
    OmniPipeSliceParam sliceParam;
    sliceParam.dataSizePerLoop = dataSizePerLoop;
    sliceParam.dataWholeSize = dataWholeSize;
    sliceParam.endpointAttrBw = endpointAttrBw;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = CommEngine::COMM_ENGINE_CCU;
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, 0};
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1};
    std::vector<u64> levelAlgType{1, 1, 1};
    sliceParam.levelAlgType = levelAlgType;
    sliceParam.dataTypeSize = dataTypeSize_;
    OmniPipeSliceInfo alignSliceInfo = CalcAGOmniPipeSliceInfo(sliceParam);

    // 4. 计算第n次的loop的slice信息
    OmniPipeSliceInfo tailSliceInfo;
    if (dataCount_ % maxCountPerLoop != 0) {
        u64 perLoopSize = (dataCount_ % maxCountPerLoop) * dataTypeSize_;
        std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
        std::vector<u64> dataWholeSize(rankSize_, dataSize_);
        sliceParam.dataSizePerLoop = dataSizePerLoop;
        sliceParam.dataWholeSize = dataWholeSize;
        tailSliceInfo = CalcAGOmniPipeSliceInfo(sliceParam);
    }

    // 5、一次loop的数据处理
    u64 processedDataCount = 0;
    OmniPipeSliceInfo omniPipeSliceInfo;
    TemplateDataParams tempAlgParamsLevel0 = tempAlgParamsCommon;
    TemplateDataParams tempAlgParamsLevel1 = tempAlgParamsCommon;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        HCCL_DEBUG("[%s] currDataCount[%llu]", __func__, currDataCount);
        // 5.2 确定当前是前n-1次loop的slice结果，还是存在尾块时最后一次loop的slice结果
        if (loop == loopTimes - 1 && dataCount_ % maxCountPerLoop != 0) {
            omniPipeSliceInfo = tailSliceInfo;
            HCCL_INFO("[%s] loop[%lu] use tail sliceinfo", __func__, loop);
        } else {
            omniPipeSliceInfo = alignSliceInfo;
            HCCL_INFO("[%s] loop[%lu] use align sliceinfo", __func__, loop);
        }

        // 本地拷贝
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(controlThread_, templateLocalCopyThreads_, notifyIdxMainToSub));
        TemplateDataParams tempAlgParamsLocalCopy = tempAlgParamsCommon;
        tempAlgParamsLocalCopy.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParamsLocalCopy.count = currDataCount;
        tempAlgParamsLocalCopy.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamsLocalCopy.inputSliceStride = dataCount_ * dataTypeSize_;
        tempAlgParamsLocalCopy.outputSliceStride = dataCount_ * dataTypeSize_;
        tempAlgParamsLocalCopy.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamsLocalCopy.repeatNum = rankSize_;
        tempAlgParamsLocalCopy.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParamsLocalCopy.localCopyFlag = 1;
        HCCL_DEBUG("[%s] myRank[%u] localCopy inBuffBaseOff[%lu] outBuffBaseOff[%lu] sliceSize[%lu]", __func__, myRank_,
            tempAlgParamsLocalCopy.buffInfo.inBuffBaseOff, tempAlgParamsLocalCopy.buffInfo.outBuffBaseOff,
            tempAlgParamsLocalCopy.sliceSize);

        CHK_RET(algTemplateLevel0.KernelRun(param, tempAlgParamsLocalCopy, templateResourceLevel0));
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(controlThread_, templateLocalCopyThreads_, notifyIdxSubToMain));

        auto innerServerStepNum = omniPipeSliceInfo.dataSliceLevel0.size();
        HCCL_DEBUG("[%s] myRank[%u] innerServerStepNum[%u]", __func__, myRank_, innerServerStepNum);
        // 5.3 for内层2d
        for (auto i = 0; i < innerServerStepNum; ++i) {
            //第一步开始前同步
            CHK_RET(PreSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxControlToTemplates_));

            GenTemplateAlgParamsByDimData(param, tempAlgParamsLevel0, omniPipeSliceInfo.dataSliceLevel0[i],
                processedDataCount);
            CHK_RET(algTemplateLevel0.KernelRun(param, tempAlgParamsLevel0, templateResourceLevel0));

            GenTemplateAlgParamsByDimData(param, tempAlgParamsLevel1, omniPipeSliceInfo.dataSliceLevel1[i],
                processedDataCount);
            tempAlgParamsLevel1.inputSliceStride = dataSize_;
            CHK_RET(algTemplateLevel1.KernelRun(param, tempAlgParamsLevel1, templateResourceLevel1));

            //第一步做完后回到主流做尾同步
            CHK_RET(PostSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxTemplatesToControl_));
        }

        processedDataCount += currDataCount;
        HCCL_DEBUG("[%s] processedDataCount[%llu]", __func__, processedDataCount);
    }

    HCCL_DEBUG("[%s]End.", __func__);
    return HcclResult::HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, CcuAllGatherOmniPipe2D,
    InsV2AllGatherOmniPipe2DExecutor, TopoMatchUBX,
    CcuTempAllGatherOmniPipeMesh1DMem2Mem, CcuTempAllGatherOmniPipeNHR1DMem2Mem);
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#endif
}