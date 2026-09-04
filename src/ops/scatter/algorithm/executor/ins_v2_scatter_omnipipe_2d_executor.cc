/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_scatter_omnipipe_2d_executor.h"

#include <algorithm>

#include "omnipipe_scatter_data_slice_calc.h"
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_scatter_omnipipe_mesh1d_mem2mem.h"
#include "ccu_temp_scatter_omnipipe_nhr1d_mem2mem.h"
#endif
#endif
#include "alg_data_trans_wrapper.h"
#include "coll_alg_v2_exec_registry.h"
#include "alg_attrs_registry.h"
#include "auto_selector_base.h"

namespace ops_hccl {
constexpr u64 OMNI2D_UBX_SC_DATA_SIZE = 16 * 1024 * 1024; // UBX机型ccu并行/流水算法数据量分界，与selector保持一致
constexpr u32 OMNIPIPE_2D_MIN_THREAD_NUM = 3;
constexpr u32 OMNIPIPE_2D_MIN_CCU_KERNEL_NUM = 2;
constexpr u32 OMNIPIPE_2D_TEMPLATE_NUM = 2;    // 两个template(intra+inter)
constexpr u32 OMNIPIPE_2D_MAIN_NOTIFY_NUM = 2; // 两个template各自的notify数
namespace {
    constexpr double OMNIPIPE_FIXED_UB_UTILIZATION = 0.85;
    constexpr double GBPS_TO_BYTES_PER_SECOND = 1000.0 * 1000.0 * 1000.0;

    bool CalcOmniPipe2dCostAxes(const TopoInfoWithNetLayerDetails* topoInfo, u64& meshRankSize, u64& closRankSize)
    {
        if (topoInfo == nullptr || topoInfo->topoInstDetailsOfLayer.empty()) {
            return false;
        }
        const auto& rankNumForTopoType = topoInfo->topoInstDetailsOfLayer[0].rankNumForTopoType;
        auto meshIt = rankNumForTopoType.find(CommTopo::COMM_TOPO_1DMESH);
        auto closIt = rankNumForTopoType.find(CommTopo::COMM_TOPO_CLOS);
        if (meshIt == rankNumForTopoType.end() || meshIt->second.empty() || closIt == rankNumForTopoType.end()
            || closIt->second.empty() || meshIt->second[0] == 0 || closIt->second[0] % meshIt->second[0] != 0) {
            return false;
        }
        meshRankSize = meshIt->second[0];
        closRankSize = closIt->second[0] / meshRankSize;
        return closRankSize > 0 && meshRankSize * closRankSize == topoInfo->userRankSize;
    }

    u64 CalcScatterStepNum(
        double meshBandwidth, double closPlanBandwidth, u64 meshRankSize, u64 closRankSize, u64 maxStepNum)
    {
        if (meshBandwidth <= closPlanBandwidth) {
            return CalcAllgatherStepNum2D(meshBandwidth, closPlanBandwidth, meshRankSize, closRankSize, maxStepNum);
        }
        return CalcAllgatherStepNum2D(closPlanBandwidth, meshBandwidth, closRankSize, meshRankSize, maxStepNum);
    }

    float CalcTemplateLatency(u32 taskNum)
    {
        float latency = 0.0f;
        CostModelManager::Global()->CalcLatencyParams(taskNum, EngineType::CCU, latency);
        return latency;
    }
} // namespace

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
InsV2ScatterOmniPipe2DExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::InsV2ScatterOmniPipe2DExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult InsV2ScatterOmniPipe2DExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void)comm;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, AlgAttrs{}));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult InsV2ScatterOmniPipe2DExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
std::vector<CostModelParam>
InsV2ScatterOmniPipe2DExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)param;
    if (topoInfo == nullptr || algName == nullptr) {
        HCCL_ERROR("[%s] topoInfo or algName is null.", __func__);
        return {};
    }
    u64 meshRankSize = 1;
    u64 closRankSize = 1;
    if (!CalcOmniPipe2dCostAxes(topoInfo, meshRankSize, closRankSize)) {
        HCCL_WARNING("[%s] unable to derive OmniPipe axes for algName[%s].", __func__, algName);
        return {};
    }

    const double meshBandwidth = BW_OMNI_UBX_CCU_SCHED_SC_MESH / OMNIPIPE_FIXED_UB_UTILIZATION;
    const double closBandwidth = BW_OMNI_UBX_CCU_SCHED_SC_CLOS / OMNIPIPE_FIXED_UB_UTILIZATION;
    const double closPlanBandwidth = closRankSize > 1 ? closBandwidth / (closRankSize - 1) : closBandwidth;
    const u64 maxStepNum = MAX_STEP_NUM;
    const u64 stepNum = CalcScatterStepNum(meshBandwidth, closPlanBandwidth, meshRankSize, closRankSize, maxStepNum);

    const bool meshActive = meshRankSize > 1;
    const bool closActive = closRankSize > 1;
    double transferCoeff = 0.0;
    if (meshActive && closActive) {
        if (stepNum == maxStepNum) {
            transferCoeff = meshBandwidth <= closPlanBandwidth ? (meshRankSize - 1) / meshBandwidth :
                                                                 (closRankSize - 1) / closBandwidth;
        } else {
            transferCoeff = (topoInfo->userRankSize - 1) / (meshBandwidth + closBandwidth);
        }
    } else if (meshActive) {
        transferCoeff = (meshRankSize - 1) / meshBandwidth;
    } else if (closActive) {
        transferCoeff = (closRankSize - 1) / closBandwidth;
    }

    CostModelParam costParam{};
    costParam.A = static_cast<float>(transferCoeff / GBPS_TO_BYTES_PER_SECOND);
    CostModelManager::Global()->CalcLocalCopyParams(1.0f, EngineType::CCU, costParam.B);
    const float meshLatency = meshActive ? CalcTemplateLatency(1) : 0.0f;
    const float nhrLatency = closActive ? CalcTemplateLatency(GetNHRStepNum(static_cast<u32>(closRankSize))) : 0.0f;
    costParam.C = 2.0f * static_cast<float>(stepNum) * std::max(meshLatency, nhrLatency);

    HCCL_INFO(
        "[%s] algName[%s] axes[%llu,%llu] step[%llu] maxStep[%d] Ufixed[%f] A[%e] B[%e] C[%e].", __func__, algName,
        meshRankSize, closRankSize, stepNum, stepNum == maxStepNum, OMNIPIPE_FIXED_UB_UTILIZATION, costParam.A,
        costParam.B, costParam.C);
    return {costParam};
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
AlgNetMeta InsV2ScatterOmniPipe2DExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const
{
    (void)topoInfo;
    AlgNetMeta meta;
    meta.netTypes = {CommTopo::COMM_TOPO_1DMESH};
    meta.groupSizes = {1};
    return meta;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult InsV2ScatterOmniPipe2DExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::InitCommInfo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;

    if (algHierarchyInfo.infos.empty()) {
        HCCL_ERROR("[%s] algHierarchyInfo.infos[0] is invalid (empty or size < 2).", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel0 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    rankSizeLevel1_ = algHierarchyInfo.infos[1][0].size();
    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel1 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    u64 rootx = param.root % rankSizeLevel0_;
    u64 rooty = param.root / rankSizeLevel0_;

    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;

    bool isRoot = (myRank_ == param.root);
    isSameXAxisAsRoot = (rankIdxLevel1_ == rooty) && !isRoot;
    isSameYAxisAsRoot = (rankIdxLevel0_ == rootx) && !isRoot;

    HCCL_DEBUG(
        "[%s] myRank[%u] rankSize[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] "
        "rankIdxLevel1[%u] devType[%u] dataCount[%u] dataType[%u] dataTypeSize[%u]",
        __func__, myRank_, rankSize_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_, devType_,
        dataCount_, dataType_, dataTypeSize_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult InsV2ScatterOmniPipe2DExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    CHK_RET(InitCommInfo(param, topoInfo, algHierarchyInfo));
    HCCL_DEBUG("[%s] myRank[%u] start", __func__, myRank_);

    // 重复的template构造
    if (algHierarchyInfo.infos.empty()) {
        HCCL_ERROR("[%s] algHierarchyInfo.infos[0] is invalid (empty or size < 2).", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    std::vector<std::vector<u32>> subCommRanks0{algHierarchyInfo.infos[0][0]};
    std::vector<std::vector<u32>> subCommRanks1{algHierarchyInfo.infos[1][0]};

    // 申请一条控制thread作为主thread，该thread仅用于两个template之间同步
    resourceRequest.notifyNumOnMainThread = OMNIPIPE_2D_MAIN_NOTIFY_NUM;
    // 由于主thread被单独作为控制thread，因此总的slaveThread需要额外加上两个template的主thread
    resourceRequest.slaveThreadNum = 0;
    AlgResourceRequest resReqLevel0; // X
    if (rankSizeLevel0_ > 1) {
        InsAlgTempLevel0 algTempLevel0(param, myRank_, subCommRanks0);
        CHK_RET(algTempLevel0.CalcRes(comm, param, topoInfo, resReqLevel0));
        resourceRequest.ccuKernelInfos.insert(
            resourceRequest.ccuKernelInfos.end(), resReqLevel0.ccuKernelInfos.begin(),
            resReqLevel0.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.insert(
            resourceRequest.ccuKernelNum.end(), resReqLevel0.ccuKernelNum.begin(), resReqLevel0.ccuKernelNum.end());
        resourceRequest.slaveThreadNum += resReqLevel0.slaveThreadNum + 1;
        // 第一个template的zhuthread需要的notify数量，+1是因为需要和控制thread做同步
        resourceRequest.notifyNumPerThread.emplace_back(resReqLevel0.notifyNumOnMainThread + 1);
        resourceRequest.notifyNumPerThread.insert(
            resourceRequest.notifyNumPerThread.end(), resReqLevel0.notifyNumPerThread.begin(),
            resReqLevel0.notifyNumPerThread.end());
    }

    AlgResourceRequest resReqLevel1; // Y
    if (rankSizeLevel1_ > 1) {
        InsAlgTempLevel1 algTempLevel1(param, myRank_, subCommRanks1);
        // 与root同框的同列rank作为新server间模板的root
        algTempLevel1.SetRoot(param.root / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_);
        CHK_RET(algTempLevel1.CalcRes(comm, param, topoInfo, resReqLevel1));
        resourceRequest.ccuKernelInfos.insert(
            resourceRequest.ccuKernelInfos.end(), resReqLevel1.ccuKernelInfos.begin(),
            resReqLevel1.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.insert(
            resourceRequest.ccuKernelNum.end(), resReqLevel1.ccuKernelNum.begin(), resReqLevel1.ccuKernelNum.end());
        resourceRequest.slaveThreadNum += resReqLevel1.slaveThreadNum + 1;
        // 这一条是interTemplate的主thread，需要+1是为了和控制thread进行同步
        resourceRequest.notifyNumPerThread.emplace_back(resReqLevel1.notifyNumOnMainThread + 1);
        resourceRequest.notifyNumPerThread.insert(
            resourceRequest.notifyNumPerThread.end(), resReqLevel1.notifyNumPerThread.begin(),
            resReqLevel1.notifyNumPerThread.end());
    }

    HCCL_DEBUG("[%s] slaveThreadNum[%u]", __func__, resourceRequest.slaveThreadNum);
    HCCL_DEBUG("[%s] myRank[%u] end", __func__, myRank_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult InsV2ScatterOmniPipe2DExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    maxTmpMemSize_ = resCtx.cclMem.size;
    HCCL_DEBUG("[%s] myRank[%u] start", __func__, myRank_);
    localThreads_ = resCtx.threads;
    HCCL_DEBUG(
        "[%s]localThreads_ size[%u] dataSize_[%u] dataCount_[%u]", __func__, localThreads_.size(), dataSize_,
        dataCount_); // 3 main+x+y

    if (resCtx.algHierarchyInfo.infos.empty() || resCtx.algHierarchyInfo.infos[0].size() < MIN_SUBGROUP_NUM) {
        HCCL_ERROR("[%s] algHierarchyInfo.infos[0] is invalid (empty or size < 2).", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    rankSizeLevel0_ = resCtx.algHierarchyInfo.infos[0][0].size();
    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel0 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    rankSizeLevel1_ = resCtx.algHierarchyInfo.infos[1][0].size();
    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel1 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;

    u64 rootx = param.root % rankSizeLevel0_;
    u64 rooty = param.root / rankSizeLevel0_;

    bool isRoot = (myRank_ == param.root);
    isSameYAxisAsRoot = (rankIdxLevel0_ == rootx) && !isRoot;
    isSameXAxisAsRoot = (rankIdxLevel1_ == rooty) && !isRoot;

    HCCL_DEBUG(
        "[%s]myRank[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]", __func__, myRank_,
        rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);
    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] executor kernel run failed", __func__, HCCL_ERROR_CODE(ret)), ret);
    HCCL_DEBUG("[%s] myRank[%u] end", __func__, myRank_);
    return HcclResult::HCCL_SUCCESS;
}

// 将计算出的单步slice信息初始化到templateParam中
template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult
InsV2ScatterOmniPipe2DExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::GenTempAlgParamsIn2HCCLBuff(
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

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult
InsV2ScatterOmniPipe2DExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::GenTempAlgParamsHCCLBuff2HCCLBuff(
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

// 为模板准备资源
template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult InsV2ScatterOmniPipe2DExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::PrepareResForTemplate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, InsAlgTempLevel0& algTempLevel0,
    InsAlgTempLevel1& algTempLevel1)
{
    HCCL_DEBUG("[%s] start localThreads_ size[%u]", __func__, localThreads_.size());
    // 获取每个template的线程数
    u64 level0ThreadsNum = 0;
    u64 level1ThreadsNum = 0;
    AlgResourceRequest level0TempRequest;
    AlgResourceRequest level1TempRequest;
    if (rankSizeLevel0_ > 1) {
        level0ThreadsNum = algTempLevel0.GetThreadNum();
        level0Threads_.assign(localThreads_.begin() + 1, localThreads_.begin() + 1 + level0ThreadsNum);
        templateMainThreads_.push_back(level0Threads_.at(0));
        CHK_RET(algTempLevel0.GetRes(level0TempRequest));
        notifyIdxControlToTemplates_.push_back(level0TempRequest.notifyNumOnMainThread);
        notifyIdxTemplatesToControl_.push_back(0);
    }
    if (rankSizeLevel1_ > 1) {
        level1ThreadsNum = algTempLevel1.GetThreadNum();
        level1Threads_.assign(localThreads_.begin() + 1 + level0ThreadsNum, localThreads_.end());
        templateMainThreads_.push_back(level1Threads_.at(0));
        CHK_RET(algTempLevel1.GetRes(level1TempRequest));
        notifyIdxControlToTemplates_.push_back(level1TempRequest.notifyNumOnMainThread);
        notifyIdxTemplatesToControl_.push_back(1);
    }
    HCCL_DEBUG("[%s]level0ThreadsNum[%u] level1ThreadsNum[%u]", __func__, level0ThreadsNum, level1ThreadsNum);
    HCCL_DEBUG(
        "[%s]level0Threads size[%u] level1Threads size[%u]", __func__, level0Threads_.size(), level1Threads_.size());
    HCCL_DEBUG("[%s]templateMainThreads size[%u]", __func__, templateMainThreads_.size());

    // 控制线程用于算法同步
    controlThread_ = localThreads_.at(0);

    // 获取template各自的主thread上有多少notify
    HCCL_DEBUG("[%s]notifyIdxControlToTemplates_ size[%u]", __func__, notifyIdxControlToTemplates_.size());
    HCCL_DEBUG("[%s]notifyIdxTemplatesToControl_ size[%u]", __func__, notifyIdxTemplatesToControl_.size());

    // 单独本地拷贝使用
    if (!level0Threads_.empty()) {
        templateLocalCopyThreads_.push_back(level0Threads_.at(0));
    } else if (!level1Threads_.empty()) {
        templateLocalCopyThreads_.push_back(level1Threads_.at(0));
    }

    HCCL_DEBUG("[%s] run success", __func__);
    return HcclResult::HCCL_SUCCESS;
}

// 主执行函数
template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult InsV2ScatterOmniPipe2DExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    HCCL_DEBUG("[%s] myRank_[%u] Start", __func__, myRank_);
    auto algHierarchyInfo = resCtx.algHierarchyInfo;
    bool isRoot = (myRank_ == param.root);

    // 构造subCommRanks
    if (algHierarchyInfo.infos.empty() || algHierarchyInfo.infos[0][0].empty()) {
        HCCL_ERROR("[%s] algHierarchyInfo.infos[0] is invalid (empty or size < 2 or infos[0][0] empty).", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    std::vector<std::vector<u32>> subCommRanks0{algHierarchyInfo.infos[0][0]};
    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[1][0].size();
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    std::vector<std::vector<u32>> subCommRanks1;
    subCommRanks1.push_back(algHierarchyInfo.infos[1][0]);

    HCCL_DEBUG(
        "[%s]myRank[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]", __func__, myRank_,
        rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);

    // 打印子通信组信息
    for (size_t i = 0; i < subCommRanks0.size(); ++i) {
        std::stringstream ss;
        for (size_t j = 0; j < subCommRanks0[i].size(); ++j) {
            ss << subCommRanks0[i][j] << " ";
        }
        HCCL_DEBUG("[%s] subCommRanks0[%zu] content: %s", __func__, i, ss.str().c_str());
    }

    for (size_t i = 0; i < subCommRanks1.size(); ++i) {
        std::stringstream ss;
        for (size_t j = 0; j < subCommRanks1[i].size(); ++j) {
            ss << subCommRanks1[i][j] << " ";
        }
        HCCL_DEBUG("[%s] subCommRanks1[%zu] content: %s", __func__, i, ss.str().c_str());
    }

    // 创建template实例
    InsAlgTempLevel0 algTempX;
    InsAlgTempLevel1 algTempY;
    if (rankSizeLevel0_ > 1) {
        algTempX = InsAlgTempLevel0(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel1_ > 1) {
        algTempY = InsAlgTempLevel1(param, myRank_, subCommRanks1);
        algTempY.SetRoot(param.root / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_);
    }

    // 公共参数初始化
    TemplateDataParams tempAlgParamsCommon;
    tempAlgParamsCommon.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsCommon.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsCommon.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsCommon.buffInfo.hcclBuffSize = resCtx.cclMem.size;

    // 资源模板初始化
    TemplateResource templateResourceCommon;
    // 通道处理
    TemplateResource templateResourceX = templateResourceCommon;
    CHK_PRT_RET(
        resCtx.threads.size() < OMNIPIPE_2D_MIN_THREAD_NUM
            || resCtx.ccuKernelNum.size() < OMNIPIPE_2D_MIN_CCU_KERNEL_NUM
            || resCtx.ccuKernels.size() < static_cast<size_t>(resCtx.ccuKernelNum[0]) + resCtx.ccuKernelNum[1],
        HCCL_ERROR(
            "[%s] resCtx resource not enough. threads.size[%zu], ccuKernelNum.size[%zu], ccuKernels.size[%zu].",
            __func__, resCtx.threads.size(), resCtx.ccuKernelNum.size(), resCtx.ccuKernels.size()),
        HCCL_E_INTERNAL);
    templateResourceX.threads.push_back(resCtx.threads[1]);
    TemplateResource templateResourceY = templateResourceCommon;
    templateResourceY.threads.push_back(resCtx.threads[2]);

    // 为template准备资源
    CHK_RET(PrepareResForTemplate(param, resCtx, algTempX, algTempY));
    // 1. 计算带宽
    double eqBwLevel0 = BW_OMNI_UBX_CCU_SCHED_SC_MESH;
    double eqBwLevel1 = BW_OMNI_UBX_CCU_SCHED_SC_CLOS;
    eqBwLevel1 = rankSizeLevel1_ > 1 ? eqBwLevel1 / (rankSizeLevel1_ - 1) : eqBwLevel1;
    std::vector<double> endpointAttrBwAvg = {eqBwLevel0, eqBwLevel1, 1.0};
    HCCL_DEBUG(
        "[%s] myRank[%u] endpointAttrBwAvg[%.3f, %.3f, %.3f]", __func__, myRank_, endpointAttrBwAvg[0],
        endpointAttrBwAvg[1], endpointAttrBwAvg[2]);

    // 2.计算loop相关信息
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

    // 3.计算对齐数据的切片信息
    OmniPipeSliceInfo alignSliceInfo;
    std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
    std::vector<u64> dataWholeSize(rankSize_, dataSize_);

    // 填充参数
    OmniPipeSliceParam sliceParam;
    sliceParam.dataSizePerLoop = dataSizePerLoop;
    sliceParam.dataWholeSize = dataWholeSize;
    sliceParam.endpointAttrBw = endpointAttrBwAvg; // 默认带宽系数
    sliceParam.opMode = param.opMode;
    sliceParam.engine = CommEngine::COMM_ENGINE_CCU;
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, 0};     // z轴默认为0
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1}; // z轴默认为1
    std::vector<u64> levelAlgType{1, 0, 1};                           // MESH算法
    sliceParam.levelAlgType = levelAlgType;
    sliceParam.dataTypeSize = dataTypeSize_;

    alignSliceInfo = CalcScatterOmniPipeSliceInfo(sliceParam, param.root);
    CHK_PRT_RET(alignSliceInfo.isEmpty(), HCCL_ERROR("[%s] alignSliceInfo is empty", __func__), HCCL_E_INTERNAL);

    // 4.计算尾数据的切片信息
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

    // 5.以下是处理所有loop
    u64 processedDataCount = 0;
    TemplateDataParams tempAlgParamsX = tempAlgParamsCommon;
    TemplateDataParams tempAlgParamsY = tempAlgParamsCommon;
    // 确定当前使用的切片信息
    OmniPipeSliceInfo currentSliceInfo;

    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        HCCL_DEBUG("[%s] myRank[%u] currDataCount[%llu]", __func__, myRank_, currDataCount);
        if (loop == loopTimes - 1 && !tailSliceInfo.isEmpty()) {
            perLoopSize = tailLoopSize;
            currentSliceInfo = tailSliceInfo;
        } else {
            currentSliceInfo = alignSliceInfo;
        }

        // 获取步骤数量
        auto innerServerStepNum = currentSliceInfo.dataSliceLevel0.size();
        HCCL_DEBUG("[%s] myRank[%u] innerServerStepNum[%u]", __func__, myRank_, innerServerStepNum);

        // 处理所有步骤
        for (auto i = 0; i < innerServerStepNum; ++i) {
            // 开始前同步
            CHK_RET(PreSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxControlToTemplates_));

            // 清空之前的kernel列表
            if (rankSizeLevel0_ > 1) {
                templateResourceX.ccuKernels.clear();
                // 添加当前步骤需要的kernel
                templateResourceX.ccuKernels.insert(
                    templateResourceX.ccuKernels.end(), resCtx.ccuKernels.begin(),
                    resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
            }
            if (rankSizeLevel1_ > 1) {
                templateResourceY.ccuKernels.clear();
                templateResourceY.ccuKernels.insert(
                    templateResourceY.ccuKernels.end(), resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
                    resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);
            }

            // 统一设置从userIn发到cclbuff
            CHK_RET(GenTempAlgParamsIn2HCCLBuff(
                tempAlgParamsX, currentSliceInfo.dataSliceLevel0[i], processedDataCount, resCtx, param));
            CHK_RET(GenTempAlgParamsIn2HCCLBuff(
                tempAlgParamsY, currentSliceInfo.dataSliceLevel1[i], processedDataCount, resCtx, param));

            // NHR算法时，root的同y轴都需要执行y轴任务
            if (isSameYAxisAsRoot || myRank_ == param.root) {
                algTempY.ifDoTask_ = true;
            }
            // 最后一步
            if (i == innerServerStepNum - 1) {
                HCCL_DEBUG("[%s] myRank[%u] StepNum[%u]", __func__, myRank_, i);
                // 同慢轴(x轴)的非root节点：沿y轴方向发送转发数据（使用NHR算法,走templateY）
                // 数据流向：hcclbuff --> hcclbuff（使用NHR算法）
                // NHR算法时，最后一步时，所有节点都需要执行y轴任务
                algTempY.ifDoTask_ = true;
                if (isSameXAxisAsRoot && rankSizeLevel1_ > 1) {
                    HCCL_DEBUG("[%s] set myRank[%u] as root", __func__, myRank_);
                    CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
                        tempAlgParamsY, currentSliceInfo.dataSliceLevel1[i], processedDataCount, resCtx, param));
                }
                // 同快轴(y轴)的非root节点：沿x轴方向发送转发数据（使用mesh算法,走templateX）
                // 数据流向：hcclbuff --> hcclbuff
                if (isSameYAxisAsRoot && rankSizeLevel0_ > 1) {
                    HCCL_DEBUG("[%s] set myRank[%u] as root", __func__, myRank_);
                    algTempX.SetRoot(myRank_);

                    CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
                        tempAlgParamsX, currentSliceInfo.dataSliceLevel0[i], processedDataCount, resCtx, param));
                }
            } else if (i != 0 && i != innerServerStepNum - 1) {
                // 中间步骤：逐步转发阶段
                // - root节点：快轴发对角数据，慢轴发同轴数据
                // - 同x轴非root节点：不发送任何数据
                // - 同y轴非root节点：往x轴方向发送部分转发数据
                HCCL_DEBUG("[%s] myRank[%u] StepNum[%u]", __func__, myRank_, i);

                // 同快轴(y轴)的非root节点：往x轴方向发送部分转发数据（使用mesh算法,走templateX）
                // 数据流向：hcclbuff --> hcclbuff
                if (endpointAttrBwAvg[0] <= endpointAttrBwAvg[1]) {
                    if (isSameYAxisAsRoot && rankSizeLevel0_ > 1) {
                        HCCL_DEBUG("[%s] set myRank[%u] as root", __func__, myRank_);
                        algTempX.SetRoot(myRank_);

                        CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
                            tempAlgParamsX, currentSliceInfo.dataSliceLevel0[i], processedDataCount, resCtx, param));
                    }
                } else {
                    algTempY.ifDoTask_ = true;
                    if (isSameXAxisAsRoot && rankSizeLevel1_ > 1) {
                        CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
                            tempAlgParamsY, currentSliceInfo.dataSliceLevel1[i], processedDataCount, resCtx, param));
                    }
                }
            }
            // 执行X维度通信
            if (rankSizeLevel0_ > 1) {
                algTempX.isStepOne_ = (i == 0);
                algTempX.isLastStep_ = (i == innerServerStepNum - 1);
                CHK_RET(algTempX.KernelRun(param, tempAlgParamsX, templateResourceX));
            }
            // 执行Y维度通信
            if (rankSizeLevel1_ > 1) {
                algTempY.isStepOne_ = (i == 0);
                algTempY.isLastStep_ = (i == innerServerStepNum - 1);
                algTempY.xRankSize_ = rankSizeLevel0_;
                CHK_RET(algTempY.KernelRun(param, tempAlgParamsY, templateResourceY));
            }
            // 步骤完成后同步
            CHK_RET(PostSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxTemplatesToControl_));
        }

        // 分批copy到userout
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(controlThread_, templateLocalCopyThreads_, notifyIdxMainToSub));

        TemplateDataParams tempAlgParamsLocalCopy = tempAlgParamsCommon;
        tempAlgParamsLocalCopy.localCopyFlag = 1;
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
            "[%s] myRank[%u] localCopy inBuffBaseOff[%lu] outBuffBaseOff[%lu] sliceSize[%lu] outputSize[%u] "
            "inputSize[%u]",
            __func__, myRank_, tempAlgParamsLocalCopy.buffInfo.inBuffBaseOff,
            tempAlgParamsLocalCopy.buffInfo.outBuffBaseOff, tempAlgParamsLocalCopy.sliceSize,
            tempAlgParamsLocalCopy.buffInfo.outputSize, tempAlgParamsLocalCopy.buffInfo.inputSize);
        if (rankSizeLevel0_ > 1) {
            CHK_RET(algTempX.KernelRun(param, tempAlgParamsLocalCopy, templateResourceX));
        } else if (rankSizeLevel1_ > 1) {
            CHK_RET(algTempY.KernelRun(param, tempAlgParamsLocalCopy, templateResourceY));
        }
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(controlThread_, templateLocalCopyThreads_, notifyIdxSubToMain));

        processedDataCount += currDataCount;
        if (rankSizeLevel0_ > 1) {
            algTempX.UnsetRoot(myRank_);
        }
        algTempY.ifDoTask_ = false;
        HCCL_DEBUG("[%s] myRank[%u] processedDataCount[%llu]", __func__, myRank_, processedDataCount);
    }

    HCCL_DEBUG("[%s] myRank_[%u] End", __func__, myRank_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult InsV2ScatterOmniPipe2DExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::GetRes(
    AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = OMNIPIPE_2D_TEMPLATE_NUM;
    resourceRequest.notifyNumOnMainThread = OMNIPIPE_2D_MAIN_NOTIFY_NUM;
    return HcclResult::HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_SCATTER, CcuSchedScatterPipeLineMeshNHR, InsV2ScatterOmniPipe2DExecutor, TopoMatchTwoLevel,
    CcuTempScatterOmniPipeMesh1DMem2Mem, CcuTempScatterOmniPipeNHR1DMem2Mem);
REGISTER_ALG_ATTRS(
    CcuSchedScatterPipeLineMeshNHR, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS; topo.maxTopoLevelNum = 1;
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return !AutoSelectorBase::IsLayerAllConnetedWithTopo(topo, 0, CommTopo::COMM_TOPO_1DMESH);
    };);

#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif

} // namespace ops_hccl
