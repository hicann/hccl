/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_sequence_executor_3level.h"
#include <cmath>
#include "alg_data_trans_wrapper.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "ins_temp_all_gather_nhr.h"

#include "alg_attrs_registry.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2AllGatherSequenceExecutor3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InsV2AllGatherSequenceExecutor3Level()
{}
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
std::vector<CostModelParam>
InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)param;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2AllGatherSequenceExecutor3Level][CalcCostCoeff]",
            TopoProbeScene::PROBE_CALC_COST_COEFF)) {
        return {};
    }
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = topoInfo->isPod;
    u32 rankSizeLevel0 = algHierarchyInfo.infos[0][0].size();
    u32 rankSizeLevel1 = (algHierarchyInfo.infos.size() > 1) ? algHierarchyInfo.infos[1][0].size() : 1;
    u32 rankSizeLevel2 = (algHierarchyInfo.infos.size() > 2) ? algHierarchyInfo.infos[2][0].size() : 1;
    // algo level 对应物理层(MeshConcur 为 2元组)逐层下传, 供跨物理层模板用; 旧字段取各层首元素
    CHK_PRT_RET(
        algHierarchyInfo.physicalIdxForAlgoLevels.size() != TOPO_LEVEL_NUM_3,
        HCCL_WARNING(
            "[InsV2AllGatherSequenceExecutor3Level][CalcCostCoeff] physicalIdxForAlgoLevels size[%zu] != 3.",
            algHierarchyInfo.physicalIdxForAlgoLevels.size()),
        {});
    const std::vector<std::vector<PhysicalLevelIndex>>& phyIdxForAlgoLevels = algHierarchyInfo.physicalIdxForAlgoLevels;
    std::vector<std::vector<CommTopo>> phyLevelNetTypes(phyIdxForAlgoLevels.size());
    std::vector<std::vector<std::vector<u32>>> phyLevelPortNums(phyIdxForAlgoLevels.size());
    AppendPhysicalLevelInfo(topoInfo, phyIdxForAlgoLevels, phyLevelNetTypes, phyLevelPortNums);
    // 先校验各层物理信息非空再取首元素, 避免 topo match 异常产出空层时下方 [0] 访问越界
    for (u32 lvl = 0; lvl < TOPO_LEVEL_NUM_3; lvl++) {
        if (phyLevelNetTypes[lvl].empty() || phyLevelPortNums[lvl].empty()) {
            HCCL_WARNING(
                "[InsV2AllGatherSequenceExecutor3Level][CalcCostCoeff] phyLevel[%u] netTypes or portNums is empty.",
                lvl);
            return {};
        }
    }
    CommTopo netTypeLevel0 = phyLevelNetTypes[0][0];
    CommTopo netTypeLevel1 = phyLevelNetTypes[1][0];
    CommTopo netTypeLevel2 = phyLevelNetTypes[2][0];
    const std::vector<u32>& portNumLevel0 = phyLevelPortNums[0][0];
    const std::vector<u32>& portNumLevel1 = phyLevelPortNums[1][0];
    const std::vector<u32>& portNumLevel2 = phyLevelPortNums[2][0];
    if (portNumLevel0.empty() || portNumLevel1.empty() || portNumLevel2.empty()) {
        HCCL_WARNING("[InsV2AllGatherSequenceExecutor3Level][CalcCostCoeff] portNum is empty");
        return {};
    }
    HCCL_INFO(
        "[InsV2AllGatherSequenceExecutor3Level][CalcCostCoeff] rankSize=%d, rankSizeLevel0=%d, rankSizeLevel1=%d, "
        "rankSizeLevel2=%d, portNumLevel0=%zu, portNumLevel1=%zu, portNumLevel2=%zu, netTypeLevel0=%d, "
        "netTypeLevel1=%d, netTypeLevel2=%d",
        rankSize, rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, portNumLevel0.size(), portNumLevel1.size(),
        portNumLevel2.size(), static_cast<int>(netTypeLevel0), static_cast<int>(netTypeLevel1),
        static_cast<int>(netTypeLevel2));
    std::vector<CostModelParam> params
        = [rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, portNumLevel0, portNumLevel1, portNumLevel2, netTypeLevel0,
           netTypeLevel1, netTypeLevel2, isPod, phyIdxForAlgoLevels, phyLevelNetTypes, phyLevelPortNums] {
              std::vector<CostModelParam> v;
              auto p0 = InsAlgTemplate0::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel0, 1.0f * rankSizeLevel2 * rankSizeLevel1, netTypeLevel0, BufferType::HCCL_BUFFER,
                  BufferType::OUTPUT, BufferType::HCCL_BUFFER, portNumLevel0, isPod, nullptr, nullptr, nullptr, 1u,
                  phyIdxForAlgoLevels[0], phyLevelNetTypes[0], phyLevelPortNums[0]});
              v.insert(v.end(), p0.begin(), p0.end());
              auto p1 = InsAlgTemplate1::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel1, 1.0f * rankSizeLevel2, netTypeLevel1, BufferType::HCCL_BUFFER,
                  BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, portNumLevel1, isPod, nullptr, nullptr, nullptr, 1u,
                  phyIdxForAlgoLevels[1], phyLevelNetTypes[1], phyLevelPortNums[1]});
              v.insert(v.end(), p1.begin(), p1.end());
              auto p2 = InsAlgTemplate2::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel2, 1.0f, netTypeLevel2, BufferType::INPUT, BufferType::HCCL_BUFFER,
                  BufferType::HCCL_BUFFER, portNumLevel2, isPod, nullptr, nullptr, nullptr, 1u, phyIdxForAlgoLevels[2],
                  phyLevelNetTypes[2], phyLevelPortNums[2]});
              v.insert(v.end(), p2.begin(), p2.end());
              return v;
          }();
    return params;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
AlgNetMeta
InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    return this->template AgSeq3LevelGetAlgNetMeta<AlgTopoMatch>(
        topoInfo, param, algName, "InsV2AllGatherSequenceExecutor3Level");
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    return AgSeq3LevelCalcRes<InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>(
        comm, param, topoInfo, algHierarchyInfo, resourceRequest, "InsV2AllGatherSequenceExecutor3Level");
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllGatherSequenceExecutor3Level][Orchestrate] Orchestrate Start");
    CHK_RET(InitOrchestrateState(param, resCtx, "InsV2AllGatherSequenceExecutor3Level"));

    // 构建template
    InsAlgTemplate0 Level0TempAlg(param, resCtx.topoInfo.userRank, levels_[0].hierarchyInfo);
    InsAlgTemplate1 Level1TempAlg(param, resCtx.topoInfo.userRank, levels_[1].hierarchyInfo);
    InsAlgTemplate2 Level2TempAlg(param, resCtx.topoInfo.userRank, levels_[2].hierarchyInfo);

    rankIdxLevel0_ = myRank_ % levels_[0].rankSize; // level0 组内偏移
    rankIdxLevel1_ = (myRank_ % (levels_[0].rankSize * levels_[1].rankSize)) / levels_[0].rankSize; // level1 组编号
    skipLevel1_ = (levels_[1].rankSize == 1);
    if (skipLevel1_) {
        HCCL_INFO("[InsV2AllGatherSequenceExecutor3Level][Orchestrate] level1 rankSize is 1, skip level1");
    }

    CHK_RET(Level0TempAlg.SetchannelsPerRank(levels_[0].channels));
    if (!skipLevel1_) {
        CHK_RET(Level1TempAlg.SetchannelsPerRank(levels_[1].channels));
    }
    CHK_RET(Level2TempAlg.SetchannelsPerRank(levels_[2].channels));
    // 将计算资源分配个每个算法
    CHK_RET(AgSeq3LevelPrepareResForTemplate(levels_, threads_, Level0TempAlg, Level1TempAlg, Level2TempAlg));

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx, Level0TempAlg, Level1TempAlg, Level2TempAlg);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2AllGatherSequenceExecutor3Level][Orchestrate]errNo[0x%016llx] All Gather executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, InsAlgTemplate0& tempAlgLevel0,
    InsAlgTemplate1& tempAlgLevel1, InsAlgTemplate2& tempAlgLevel2)
{
    HCCL_INFO(
        "[InsV2AllGatherSequenceExecutor3Level] AlgTemplate Level0 server is [%s]", tempAlgLevel0.Describe().c_str());
    HCCL_INFO(
        "[InsV2AllGatherSequenceExecutor3Level] AlgTemplate Level1 server is [%s]", tempAlgLevel1.Describe().c_str());
    HCCL_INFO(
        "[InsV2AllGatherSequenceExecutor3Level] AlgTemplate Level2 server is [%s]", tempAlgLevel2.Describe().c_str());

    u32 templateScratchMultiplierLevel1
        = tempAlgLevel1.CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplierLevel2 = tempAlgLevel2.CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);
    u32 totalScratchMultiple = templateScratchMultiplierLevel1 * templateScratchMultiplierLevel2;

    u64 scratchMemBlockSize = maxTmpMemSize_;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    if (totalScratchMultiple > 0) {
        scratchMemBlockSize = (maxTmpMemSize_ / AICPU_ALIGN_SIZE / totalScratchMultiple) * AICPU_ALIGN_SIZE;
        scratchMemBlockSize = std::min(scratchMemBlockSize, transportBoundDataSize);
    }
    u64 maxCountPerLoop = std::min(static_cast<u64>(scratchMemBlockSize), static_cast<u64>(UB_MAX_DATA_SIZE))
                          / AICPU_ALIGN_SIZE * AICPU_ALIGN_SIZE / dataTypeSize_;

    if (param.supportSymmetricMemory) {
        maxCountPerLoop = dataCount_;
        HCCL_INFO(
            "[InsV2AllGatherSequenceExecutor3Level][OrchestrateLoop] %s: symmetric memory enabled", param.algName);
    }

    if (maxCountPerLoop == 0) {
        HCCL_ERROR(
            "[InsV2AllGatherSequenceExecutor3Level] myRank[%u] maxCountPerLoop is 0, "
            "scratchMultiplier[%u] too large for cclBuffSize[%llu]",
            myRank_, totalScratchMultiple, scratchMemBlockSize);
        return HCCL_E_INTERNAL;
    }
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    const AgSeq3LevelCtx ctx = MakeCtx();

    TemplateResource Level2TempAlgRes, Level1TempAlgRes, Level0TempAlgRes;
    TemplateDataParams tempAlgParamsLevel2, tempAlgParamsLevel1, tempAlgParamsLevel0;
    CHK_RET(AgSeq3LevelGenTempResource(levels_, 2, resCtx, Level2TempAlgRes));
    CHK_RET(AgSeq3LevelGenTempResource(levels_, 1, resCtx, Level1TempAlgRes));
    CHK_RET(AgSeq3LevelGenTempResource(levels_, 0, resCtx, Level0TempAlgRes));

    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCount = (loopIndex == loopTimes - 1) ? (dataCount_ - loopIndex * maxCountPerLoop) : maxCountPerLoop;
        u64 dataOffset = loopIndex * maxCountPerLoop * dataTypeSize_;
        // 数据0的server内的mesh算法
        GenTemplateAlgParamsLevel2(param, resCtx, currCount, dataOffset, tempAlgParamsLevel2);
        CHK_RET(tempAlgLevel2.KernelRun(param, tempAlgParamsLevel2, Level2TempAlgRes));

        // 数据1的server间的nhr算法
        if (!skipLevel1_) {
            AgSeq3LevelGenLevel1Params(
                ctx, param, resCtx, currCount, dataOffset, tempAlgParamsLevel1, "InsV2AllGatherSequenceExecutor3Level");
            CHK_RET(tempAlgLevel1.KernelRun(param, tempAlgParamsLevel1, Level1TempAlgRes));
        }

        AgSeq3LevelGenLevel0Params(
            ctx, param, resCtx, currCount, dataOffset, tempAlgParamsLevel0, "InsV2AllGatherSequenceExecutor3Level");
        CHK_RET(tempAlgLevel0.KernelRun(param, tempAlgParamsLevel0, Level0TempAlgRes));
        // 尾同步
    }

    HCCL_INFO("[InsV2AllGatherSequenceExecutor3Level][OrchestrateLoop] End.");
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenTemplateAlgParamsLevel2(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, const u64 curCount, const u64 dataOffset,
        TemplateDataParams& tempAlgParamsLevel2) const
{
    tempAlgParamsLevel2.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsLevel2.enableRemoteMemAccess = param.supportSymmetricMemory;
    if (tempAlgParamsLevel2.enableRemoteMemAccess) {
        tempAlgParamsLevel2.buffInfo.outputPtr = param.outputPtr;
        tempAlgParamsLevel2.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParamsLevel2.buffInfo.outBuffBaseOff
            = (rankIdxLevel1_ * levels_[0].rankSize + rankIdxLevel0_) * dataSize_;
        tempAlgParamsLevel2.outputSliceStride = levels_[0].rankSize * levels_[1].rankSize * dataSize_;
    } else {
        tempAlgParamsLevel2.buffInfo.outputPtr = resCtx.cclMem.addr;
        tempAlgParamsLevel2.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsLevel2.buffInfo.outBuffBaseOff = rankIdxLevel1_ * curCount * dataTypeSize_;
        tempAlgParamsLevel2.outputSliceStride = curCount * dataTypeSize_ * levels_[1].rankSize;
    }
    tempAlgParamsLevel2.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsLevel2.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsLevel2.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.inputSize = param.inputSize;
    tempAlgParamsLevel2.buffInfo.outputSize = param.outputSize;
    tempAlgParamsLevel2.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsLevel2.buffInfo.hcclBuffBaseOff = rankIdxLevel1_ * curCount * dataTypeSize_;

    tempAlgParamsLevel2.sliceSize = curCount * dataTypeSize_;
    tempAlgParamsLevel2.count = curCount;
    tempAlgParamsLevel2.tailSize = tempAlgParamsLevel2.sliceSize;
    tempAlgParamsLevel2.inputSliceStride = 0;
    tempAlgParamsLevel2.repeatNum = 1;
    tempAlgParamsLevel2.inputRepeatStride = 0;
    tempAlgParamsLevel2.outputRepeatStride = 0;

    HCCL_DEBUG(
        "[InsV2AllGatherSequenceExecutor3Level][GenTemplateAlgParamsLevel2] rank[%u] inBuffBaseOff[%llu] "
        "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu]",
        myRank_, tempAlgParamsLevel2.buffInfo.inBuffBaseOff, tempAlgParamsLevel2.buffInfo.outBuffBaseOff,
        tempAlgParamsLevel2.buffInfo.hcclBuffBaseOff, tempAlgParamsLevel2.sliceSize,
        tempAlgParamsLevel2.outputSliceStride);
}

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_ALLGATHER, AicpuAllGatherSequenceMeshConcurNHRNHR, InsV2AllGatherSequenceExecutor3Level,
    TopoMatchThreeLevel, InsTempAllGatherMesh1D1DZAxisDetour, InsTempAllGatherNHR, InsTempAllGatherNHR);
REGISTER_ALG_ATTRS(AicpuAllGatherSequenceMeshConcurNHRNHR, topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3;
                   topo.minTopoLevelNum = TOPO_LEVEL_NUM_3; topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D);
} // namespace ops_hccl
// 算法注册
