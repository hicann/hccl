/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_v_omnipipe_executor.h"

#include <algorithm>
#include <sstream>
#include <string>

#include "alg_data_trans_wrapper.h"
#include "alg_param.h"
#include "ins_temp_all_gather_omnipipe_mesh_1D.h"
#include "ins_temp_all_gather_omnipipe_nhr.h"
#include "omnipipe_template_utils.h"
#include "omnipipe_executor_utils.h"
#include "template_utils.h"
#include "alg_attrs_registry.h"
#include "auto_selector_base.h"

namespace ops_hccl {
constexpr u32 RANK_LEVEL_2 = 2;
constexpr u32 RANK_LEVEL_4 = 4;

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2AllGatherVOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2AllGatherVOmniPipeExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherVOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::ParseCountsAndDispls(
    const OpParam& param)
{
    // varData布局为[counts[rankSize], displs[rankSize]]
    const u64 expectVarMemSize = 2 * rankSize_ * sizeof(u64);
    CHK_PRT_RET(
        param.varMemSize < expectVarMemSize,
        HCCL_ERROR(
            "[InsV2AllGatherVOmniPipeExecutor][ParseCountsAndDispls] varMemSize[%llu] is less than "
            "expected[%llu], rankSize[%u].",
            param.varMemSize, expectVarMemSize, rankSize_),
        HCCL_E_PARA);
    const u64* varData = reinterpret_cast<const u64*>(param.varData);
    CHK_PRT_RET(
        varData == nullptr,
        HCCL_ERROR("[InsV2AllGatherVOmniPipeExecutor][ParseCountsAndDispls] varData is null, rankSize[%u].", rankSize_),
        HCCL_E_PARA);
    counts_.assign(varData, varData + rankSize_);
    displs_.assign(varData + rankSize_, varData + 2 * rankSize_);
    dataCount_ = 0;
    for (u32 rank = 0; rank < rankSize_; rank++) {
        dataCount_ = std::max(dataCount_, counts_[rank]);
    }
    dataSize_ = dataCount_ * dataTypeSize_;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
OpParam InsV2AllGatherVOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::MakeUniformParam(
    const OpParam& param) const
{
    OpParam uniformParam = param;
    uniformParam.DataDes.count = dataCount_;
    uniformParam.DataDes.dataType = dataType_;
    uniformParam.DataDes.outputType = dataType_;
    uniformParam.DataDes.strideCount = 0;
    return uniformParam;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherVOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitCommInfo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    opMode_ = param.opMode;
    dataType_ = param.vDataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.vDataDes.dataType];
    CHK_RET(ParseCountsAndDispls(param));
    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO(
        "[InsV2AllGatherVOmniPipeExecutor][InitCommInfo] initialize communication metadata, "
        "rank[%u], rankSize[%u], devType[%u], dataType[%u], dataTypeSize[%llu], uniformCount[%llu].",
        myRank_, rankSize_, devType_, dataType_, dataTypeSize_, dataCount_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherVOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::BuildSubCommAndTempMap(
    const OpParam& param, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    std::vector<std::vector<u32>>& subCommRanks0, std::vector<std::vector<u32>>& subCommRanks1,
    std::vector<std::vector<u32>>& subCommRanks2, std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    HCCL_INFO(
        "[InsV2AllGatherVOmniPipeExecutor][BuildSubCommAndTempMap] build sub-communicators from "
        "algorithm hierarchy, hierarchy[%s].",
        ThreeDVecToStrOmni(algHierarchyInfo.infos).c_str());
    FillOmniPipeSubCommRanks(algHierarchyInfo, myRank_, subCommRanks0, subCommRanks1, subCommRanks2);
    CHK_PRT_RET(
        subCommRanks0.empty() || subCommRanks1.empty() || subCommRanks2.empty() || subCommRanks0[0].empty()
            || subCommRanks1[0].empty() || subCommRanks2[0].empty(),
        HCCL_ERROR(
            "[InsV2AllGatherVOmniPipeExecutor][BuildSubCommAndTempMap] sub-communicator is empty, rank[%u].", myRank_),
        HCCL_E_INTERNAL);
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        HCCL_INFO("[InsV2AllGatherVOmniPipeExecutor][BuildSubCommAndTempMap] UBX specific optimization flags.");
        omniNeedSetStepNum_ = (subCommRanks1[0].size() == RANK_LEVEL_4) ? OmniNeedSetStepNum::OMNIPIPE_UBX_16P :
                                                                          OmniNeedSetStepNum::OMNIPIPE_DEFAULT;
    }

    // 两级分解：subCommRanks2在TopoMatchTwoLevel下恒回填{myRank}，无第三级模板参与
    rankSizeLevel_[OMNIPIPE_LEVEL0] = subCommRanks0[0].size();
    rankSizeLevel_[OMNIPIPE_LEVEL1] = subCommRanks1[0].size();
    rankSizeLevel_[OMNIPIPE_LEVEL2] = subCommRanks2[0].size(); // 恒为1，填入以匹配共享结构体维度
    tempMap.clear();
    if (rankSizeLevel_[OMNIPIPE_LEVEL0] > 1) {
        tempMap[OMNIPIPE_LEVEL0] = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
        tempMap[OMNIPIPE_LEVEL1] = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1);
    }
    return HCCL_SUCCESS;
}
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherVOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherVOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, topoInfo, algHierarchyInfo));
    // 模板的CalcRes（如NHR建链）读取DataDes，必须传均匀视图的param
    const OpParam uniformParam = MakeUniformParam(param);
    // 计算subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;
    rankSizeLevel_.resize(OMNIPIPE_LEVEL_NUM);
    rankIdxLevel_.resize(OMNIPIPE_LEVEL_NUM);

    CHK_RET(BuildSubCommAndTempMap(
        uniformParam, algHierarchyInfo, subCommRanks0, subCommRanks1, subCommRanks2, tempMap, topoInfo));

    rankIdxLevel_[OMNIPIPE_LEVEL0] = myRank_ % rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL1] = myRank_ % (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1])
                                     / rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL2] = 0; // 两级分解下z轴索引恒为0，填入以匹配共享结构体维度

    for (auto& temp : tempMap) {
        CHK_RET(CalcResLevel(comm, uniformParam, topoInfo, temp.second, resourceRequest));
    }
    HCCL_INFO(
        "[InsV2AllGatherVOmniPipeExecutor][CalcRes] finish calculating template resources, "
        "templateCount[%zu].",
        tempMap.size());

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherVOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcResLevel(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    std::shared_ptr<InsAlgTemplateBase> tempAlg, AlgResourceRequest& resourceRequest) const
{
    AlgResourceRequest resReqlevel;
    CHK_RET(tempAlg->CalcRes(comm, param, topoInfo, resReqlevel));
    resourceRequest.slaveThreadNum += resReqlevel.slaveThreadNum + 1;
    resourceRequest.notifyNumOnMainThread += 1;
    resourceRequest.notifyNumPerThread.emplace_back(resReqlevel.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(
        resourceRequest.notifyNumPerThread.end(), resReqlevel.notifyNumPerThread.begin(),
        resReqlevel.notifyNumPerThread.end());
    if (!resReqlevel.channels.empty()) {
        resourceRequest.channels.emplace_back(resReqlevel.channels[0]);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherVOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplateLevel(
    u32 level, std::shared_ptr<InsAlgTemplateBase>& tempBase)
{
    u32 levelThreadNum = tempBase->GetThreadNum();
    if (level == OMNIPIPE_LEVEL0) {
        levelThreads_[OMNIPIPE_LEVEL0].assign(threads_.begin() + 1, threads_.begin() + 1 + levelThreadNum);
        tempMainThreadsXY_.push_back(levelThreads_[OMNIPIPE_LEVEL0].at(0));
    } else {
        levelThreads_[OMNIPIPE_LEVEL1].assign(
            threads_.begin() + 1 + levelThreads_[OMNIPIPE_LEVEL0].size(),
            threads_.begin() + 1 + levelThreads_[OMNIPIPE_LEVEL0].size() + levelThreadNum);
        tempMainThreadsXY_.push_back(levelThreads_[OMNIPIPE_LEVEL1].at(0));
    }

    // 获取当前template各自的主thread上有多少notify
    AlgResourceRequest levelTempRequest;
    CHK_RET(tempBase->GetRes(levelTempRequest));
    ntfIdxCtrlToTempXY_.push_back(levelTempRequest.notifyNumOnMainThread);
    ntfIdxTempToCtrlXY_.push_back(tempMainThreadsXY_.size() - 1);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherVOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataType_ = param.vDataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.vDataDes.dataType];
    CHK_RET(ParseCountsAndDispls(param));
    reduceOp_ = param.reduceType;
    opMode_ = param.opMode;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    maxTmpMemSize_ = resCtx.cclMem.size;
    // 模板的KernelRun读取DataDes.dataType，后续统一使用均匀视图的param
    const OpParam uniformParam = MakeUniformParam(param);

    // 计算subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;

    rankSizeLevel_.resize(OMNIPIPE_LEVEL_NUM);
    rankIdxLevel_.resize(OMNIPIPE_LEVEL_NUM);

    CHK_RET(BuildSubCommAndTempMap(
        uniformParam, algHierarchyInfo_, subCommRanks0, subCommRanks1, subCommRanks2, tempMap, &resCtx.topoInfo));

    rankIdxLevel_[OMNIPIPE_LEVEL0] = myRank_ % rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL1] = myRank_ % (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1])
                                     / rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL2] = 0; // 两级分解下z轴索引恒为0，填入以匹配共享结构体维度

    // 为temp分配thread；清空上次Orchestrate残留的线程/notify索引容器，
    threads_ = resCtx.threads;
    controlThread_ = threads_.at(0);
    levelThreads_.assign(OMNIPIPE_LEVEL_NUM, {});
    tempMainThreadsXY_.clear();
    ntfIdxCtrlToTempXY_.clear();
    ntfIdxTempToCtrlXY_.clear();

    const std::vector<const std::vector<std::vector<u32>>*> subCommsByLevel
        = {&subCommRanks0, &subCommRanks1, &subCommRanks2};
    CHK_RET(ClassifyOmniPipeChannelsByLevel(
        myRank_, resCtx.channels, subCommsByLevel, rankSizeLevel_, remoteRankToChannelInfo_));
    if ((resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix)) {
        if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
            CHK_RET(tempMap[OMNIPIPE_LEVEL1]->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
        }
    }
    for (auto& temp : tempMap) {
        CHK_RET(PrepareResForTemplateLevel(temp.first, temp.second));
    }
    // 算法展开
    HcclResult ret = OrchestrateLoop(uniformParam, resCtx, tempMap);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2AllGatherVOmniPipeExecutor][Orchestrate] all-gather-v execution failed, "
            "rank[%u], errorCode[0x%016llx].",
            myRank_, HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult
InsV2AllGatherVOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsByDimData(
    TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo) const
{
    CHK_RET(FillOmniPipeTemplateAlgParams(tempAlgParams, stepSliceInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherVOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx,
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap)
{
    HCCL_INFO(
        "[InsV2AllGatherVOmniPipeExecutor][OrchestrateLoop] start all-gather-v pipeline loops, rank[%u], "
        "uniformCount[%llu].",
        myRank_, dataCount_);

    // 带宽赋值
    double bw_ag_l0 = BW_OMNI_DEFAULT;
    double bw_ag_l1 = BW_OMNI_DEFAULT;
    double bw_ag_l2 = BW_OMNI_UBX_ROCE;

    if (resCtx.topoInfo.level0PcieMix) { // PCIE
        if (rankSizeLevel_[OMNIPIPE_LEVEL1] == RANK_LEVEL_2) {
            bw_ag_l1 = BW_OMNI_PCIE_EIGHT_CLOS;
        } else if (rankSizeLevel_[OMNIPIPE_LEVEL1] == RANK_LEVEL_4) {
            bw_ag_l1 = BW_OMNI_PCIE_SIXTEEN_CLOS;
        }
        // UBX
    } else if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS) {
        bw_ag_l1 = BW_OMNI_UBX_AG_CLOS;
    }

    std::vector<double> endpointAttrBw{bw_ag_l0, bw_ag_l1, bw_ag_l2};

    // 计算等价带宽
    double eqBw0 = endpointAttrBw[0]; // L0 mesh
    double eqBw1 = endpointAttrBw[1]; // L1 NHR
    double eqBw2 = endpointAttrBw[2]; // L2 NHR（两级分解下rankSize恒为1，等价带宽即原值）

    HCCL_DEBUG(
        "[InsV2AllGatherVOmniPipeExecutor][OrchestrateLoop] initialize per-level equivalent "
        "bandwidth, level0[%f], level1[%f], level2[%f].",
        eqBw0, eqBw1, eqBw2);

    // level0为mesh,等价mesh为其本身；level1为nhr；level2当前无第三级参与
    eqBw1 = rankSizeLevel_[OMNIPIPE_LEVEL1] > 1 ? eqBw1 / (rankSizeLevel_[OMNIPIPE_LEVEL1] - 1) : eqBw1;

    std::vector<double> endpointAttrBwNew{eqBw0, eqBw1, eqBw2};
    CHK_PRT_RET(dataTypeSize_ == 0, HCCL_ERROR("[%s] dataTypeSize_ is 0", __func__), HCCL_E_INTERNAL);
    u64 scratchBoundDataSize = maxTmpMemSize_ / rankSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 maxCountPerLoop = std::min(scratchBoundDataSize, transportBoundDataSize);
    CHK_PRT_RET(
        maxCountPerLoop == 0,
        HCCL_ERROR(
            "[%s] maxCountPerLoop is 0, maxTmpMemSize_[%llu], rankSize_[%u], dataTypeSize_[%llu]", __func__,
            maxTmpMemSize_, rankSize_, dataTypeSize_),
        HCCL_E_INTERNAL);
    // 均匀槽位按dataCount_（各rank counts最大值）切loop，模板无变长感知
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);

    u64 perLoopSize = maxCountPerLoop * dataTypeSize_;
    std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
    std::vector<u64> dataWholeSize(rankSize_, perLoopSize);

    OmniPipeSliceParam omniPipeSliceParam;
    omniPipeSliceParam.levelRankSize
        = {rankSizeLevel_[OMNIPIPE_LEVEL0], rankSizeLevel_[OMNIPIPE_LEVEL1], rankSizeLevel_[OMNIPIPE_LEVEL2]};
    omniPipeSliceParam.endpointAttrBw = endpointAttrBwNew;
    omniPipeSliceParam.dataSizePerLoop = dataSizePerLoop;
    omniPipeSliceParam.dataTypeSize = dataTypeSize_;
    omniPipeSliceParam.levelRankId
        = {rankIdxLevel_[OMNIPIPE_LEVEL0], rankIdxLevel_[OMNIPIPE_LEVEL1], rankIdxLevel_[OMNIPIPE_LEVEL2]};
    omniPipeSliceParam.opMode = opMode_;
    omniPipeSliceParam.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    omniPipeSliceParam.dataWholeSize = dataWholeSize;
    omniPipeSliceParam.needSetStepNum = omniNeedSetStepNum_;
    if (resCtx.topoInfo.level0PcieMix
        && param.opConfig.multipleDimensionSplitRatioSource != MultipleDimensionSplitRatioSource::BUILTIN_FORMULA) {
        omniPipeSliceParam.multipleDimensionSplitRatio = param.opConfig.multipleDimensionSplitRatio;
    }

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
        tempResMap[temp.first].npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
        tempResMap[temp.first].dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
        // 下发用户输入输出地址，数据面统一走ccl scratch。
        tempAlgParamMap[temp.first].buffInfo.inputPtr = param.inputPtr;
        tempAlgParamMap[temp.first].buffInfo.outputPtr = param.outputPtr;
    }
    HCCL_DEBUG(
        "[InsV2AllGatherVOmniPipeExecutor][OrchestrateLoop] split operation into executor loops, "
        "loopCount[%llu], maxCountPerLoop[%llu].",
        loopTimes, maxCountPerLoop);
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        u64 myValidCount = (counts_[myRank_] > processedDataCount) ?
                               std::min(counts_[myRank_] - processedDataCount, currDataCount) :
                               0;
        if (myValidCount > 0) {
            DataSlice src(
                param.inputPtr, processedDataCount * dataTypeSize_, myValidCount * dataTypeSize_, myValidCount);
            DataSlice dst(
                resCtx.cclMem.addr, myRank_ * currDataCount * dataTypeSize_, myValidCount * dataTypeSize_,
                myValidCount);
            CHK_RET(LocalCopy(controlThread_, src, dst));
        }

        if (loop == loopTimes - 1 && dataCount_ % maxCountPerLoop != 0) {
            omniPipeSliceInfo = tailSliceInfo;
        } else {
            omniPipeSliceInfo = alignSliceInfo;
        }

        CHK_PRT_RET(
            omniPipeSliceInfo.dataSliceLevel0.size() == 0 || omniPipeSliceInfo.dataSliceLevel1.size() == 0
                || omniPipeSliceInfo.dataSliceLevel2.size() == 0,
            HCCL_ERROR(
                "[InsV2AllGatherVOmniPipeExecutor][OrchestrateLoop] slice plan is empty, "
                "level0[%zu], level1[%zu], level2[%zu], rank[%u], loop[%llu].",
                omniPipeSliceInfo.dataSliceLevel0.size(), omniPipeSliceInfo.dataSliceLevel1.size(),
                omniPipeSliceInfo.dataSliceLevel2.size(), myRank_, loop),
            HCCL_E_PARA);

        // 两级分解：level2StepCount在z轴退化(rankSize=1)下恒为1，i循环单次，j循环为XY两轴步进
        u32 level2StepCount = omniPipeSliceInfo.dataSliceLevel2.size();
        u32 level0StepCount = omniPipeSliceInfo.dataSliceLevel0.size() / omniPipeSliceInfo.dataSliceLevel2.size();

        for (int i = 0; i < level2StepCount; i++) {
            bool hasXYThreads = !tempMainThreadsXY_.empty();
            for (int j = 0; j < level0StepCount; j++) {
                if (hasXYThreads) {
                    CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsXY_, ntfIdxCtrlToTempXY_));
                }
                tempAlgParamMap[OMNIPIPE_LEVEL0].omniLastStepRead_ = false;
                tempAlgParamMap[OMNIPIPE_LEVEL1].omniLastStepRead_ = false;
                if (rankSizeLevel_[OMNIPIPE_LEVEL0] > 1) {
                    CHK_RET(GenTemplateAlgParamsByDimData(
                        tempAlgParamMap[OMNIPIPE_LEVEL0], omniPipeSliceInfo.dataSliceLevel0[i * level0StepCount + j]));
                    CHK_RET(tempMap[OMNIPIPE_LEVEL0]->KernelRun(
                        param, tempAlgParamMap[OMNIPIPE_LEVEL0], tempResMap[OMNIPIPE_LEVEL0]));
                }
                if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
                    CHK_RET(GenTemplateAlgParamsByDimData(
                        tempAlgParamMap[OMNIPIPE_LEVEL1], omniPipeSliceInfo.dataSliceLevel1[i * level0StepCount + j]));
                    CHK_RET(tempMap[OMNIPIPE_LEVEL1]->KernelRun(
                        param, tempAlgParamMap[OMNIPIPE_LEVEL1], tempResMap[OMNIPIPE_LEVEL1]));
                }
                if (hasXYThreads) {
                    CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsXY_, ntfIdxTempToCtrlXY_));
                }
            }
        }

        HCCL_INFO("ccl->out_localcopy allgatherv trim");
        for (u32 rank = 0; rank < rankSize_; rank++) {
            u64 rankValidCount = (counts_[rank] > processedDataCount) ?
                                     std::min(counts_[rank] - processedDataCount, currDataCount) :
                                     0;
            if (rankValidCount == 0) {
                continue;
            }
            DataSlice dst(
                param.outputPtr, (displs_[rank] + processedDataCount) * dataTypeSize_, rankValidCount * dataTypeSize_,
                rankValidCount);
            DataSlice src(
                resCtx.cclMem.addr, rank * currDataCount * dataTypeSize_, rankValidCount * dataTypeSize_,
                rankValidCount);
            CHK_RET(LocalCopy(controlThread_, src, dst));
        }
        processedDataCount += currDataCount;
    }
    HCCL_INFO(
        "[InsV2AllGatherVOmniPipeExecutor][OrchestrateLoop] finish all-gather-v pipeline loops, rank[%u].", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherVOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::RestoreChannelMap(
    const AlgResourceCtxSerializable& resCtx,
    std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo) const
{
    HCCL_WARNING(
        "[InsV2AllGatherVOmniPipeExecutor][RestoreChannelMap] unexpected call: channel classification is "
        "already done in Orchestrate, rank[%u].",
        myRank_);
    (void)resCtx;
    rankIdToChannelInfo.resize(OMNIPIPE_LEVEL_NUM);
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_ALLGATHER_V, AicpuAllGatherVPipeLineMeshNHR, InsV2AllGatherVOmniPipeExecutor,
    TopoMatchTwoLevel, InsTempAllGatherOmniPipeMesh1D, InsTempAllGatherOmniPipeNHR);
REGISTER_ALG_ATTRS(
    AicpuAllGatherVPipeLineMeshNHR, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS;
    topo.maxTopoLevelNum = TOPO_LEVEL_NUM_1; topo.isSupportLevel0PcieMix = false; op.isSupportInplace = false;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        bool isEqual = false;
        bool isMultiple = false;
        AutoSelectorBase::CheckMeshNumEqualToClosNum(topo, isEqual);
        AutoSelectorBase::CheckClosNumMultipleOfMeshNum(topo, isMultiple);
        return !topo->level0PcieMix && (topo->topoLevelNums == TOPO_LEVEL_NUM_1) && (!(isEqual) && isMultiple);
    });
} // namespace ops_hccl
