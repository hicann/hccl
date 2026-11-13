/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_sequence_executor_3level_dpu.h"
#include <cmath>
#include "alg_data_trans_wrapper.h"
#include "coll_alg_v2_exec_registry.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_all_gather_nhr_dpu.h"
#include "ins_temp_all_gather_rhd.h"
#include "ins_temp_all_gather_rhd_dpu.h"
#include "alg_attrs_registry.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2AllGatherSequenceExecutor3LevelDpu<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InsV2AllGatherSequenceExecutor3LevelDpu()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
std::vector<CostModelParam>
InsV2AllGatherSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)param;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2AllGatherSequenceExecutor3LevelDpu][CalcCostCoeff]",
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
            "[InsV2AllGatherSequenceExecutor3LevelDpu][CalcCostCoeff] physicalIdxForAlgoLevels size[%zu] != 3.",
            algHierarchyInfo.physicalIdxForAlgoLevels.size()),
        {});
    const std::vector<std::vector<PhysicalLevelIndex>>& phyIdxForAlgoLevels = algHierarchyInfo.physicalIdxForAlgoLevels;
    std::vector<std::vector<CommTopo>> phyLevelNetTypes(phyIdxForAlgoLevels.size());
    std::vector<std::vector<std::vector<u32>>> phyLevelPortNums(phyIdxForAlgoLevels.size());
    AppendPhysicalLevelInfo(topoInfo, phyIdxForAlgoLevels, phyLevelNetTypes, phyLevelPortNums);
    // 各层落在 DEVICE 还是 HOST 由本算法单独收集（准入 ③ 用）：基类 helper 只产出网型与端口数，
    // 没有 locType 出口。直接读 physicalLevels[].locType（不改基类）：区分同形态(CLOS)落在
    // DEVICE(设备/UB) 还是 HOST(RoCE)，故须单独收集。
    const auto getLocType = [topoInfo](u32 levelIdx) -> EndpointLocType {
        if (topoInfo == nullptr || levelIdx >= topoInfo->physicalLevels.size()) {
            return EndpointLocType::ENDPOINT_LOC_TYPE_RESERVED;
        }
        return topoInfo->physicalLevels[levelIdx].locType;
    };
    std::vector<std::vector<EndpointLocType>> phyLevelLocTypes(phyIdxForAlgoLevels.size());
    for (u32 lvl = 0; lvl < phyIdxForAlgoLevels.size(); lvl++) {
        for (PhysicalLevelIndex phyIdx : phyIdxForAlgoLevels[lvl]) {
            phyLevelLocTypes[lvl].push_back(getLocType(static_cast<u32>(phyIdx)));
        }
    }
    // 先校验各层物理信息非空再取首元素, 避免 topo match 异常产出空层时下方 [0] 访问越界
    for (u32 lvl = 0; lvl < TOPO_LEVEL_NUM_3; lvl++) {
        if (phyLevelNetTypes[lvl].empty() || phyLevelPortNums[lvl].empty()) {
            HCCL_WARNING(
                "[InsV2AllGatherSequenceExecutor3LevelDpu][CalcCostCoeff] phyLevel[%u] netTypes or portNums is empty.",
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
        HCCL_WARNING("[InsV2AllGatherSequenceExecutor3LevelDpu][CalcCostCoeff] portNum is empty");
        return {};
    }

    // ① d2 >= 2：最外层组数不足 2 时该级退化，三级不存在 host-DPU 通信，交回二级 Sequence 竞争。
    CHK_PRT_RET(
        rankSizeLevel2 < 2,
        HCCL_INFO(
            "[InsV2AllGatherSequenceExecutor3LevelDpu][CalcCostCoeff] hostdpu admit: rankSizeLevel2[%u] < 2, skip.",
            rankSizeLevel2),
        {});
    // ② 中间层须为 CLOS：非 CLOS 时设备侧 NHR 的可行性未经验证，本算法不参与竞争。
    CHK_PRT_RET(
        netTypeLevel1 != CommTopo::COMM_TOPO_CLOS,
        HCCL_INFO(
            "[InsV2AllGatherSequenceExecutor3LevelDpu][CalcCostCoeff] hostdpu admit: level1 netType[%d] != CLOS, skip.",
            static_cast<int>(netTypeLevel1)),
        {});
    // ③ 中间层须落在设备侧：仅判 CLOS 不够 —— host RoCE 的 CLOS 也会被 topomatch 当作中间层，
    // 而设备 NHR 只消费非 ROCE 通道（ins_temp_all_gather_nhr.cc:71-75），host CLOS 下通道为空，
    // 建链阶段必然失败。故须拦截，回退二级。
    CHK_PRT_RET(
        phyLevelLocTypes[1][0] != EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE,
        HCCL_INFO(
            "[InsV2AllGatherSequenceExecutor3LevelDpu][CalcCostCoeff] hostdpu admit: level1 locType[%d] != DEVICE, "
            "skip.",
            static_cast<int>(phyLevelLocTypes[1][0])),
        {});

    HCCL_INFO(
        "[InsV2AllGatherSequenceExecutor3LevelDpu][CalcCostCoeff] rankSize=%d, rankSizeLevel0=%d, rankSizeLevel1=%d, "
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
InsV2AllGatherSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    return this->template AgSeq3LevelGetAlgNetMeta<AlgTopoMatch>(
        topoInfo, param, algName, "InsV2AllGatherSequenceExecutor3LevelDpu");
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    return AgSeq3LevelCalcRes<InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>(
        comm, param, topoInfo, algHierarchyInfo, resourceRequest, "InsV2AllGatherSequenceExecutor3LevelDpu");
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllGatherSequenceExecutor3LevelDpu][Orchestrate] Orchestrate Start");
    // 对称内存路径当前未支持: 入口侧对称窗口检查仅覆盖 AICPU_TS 引擎的白名单算法(all_gather.cc),
    // Dpu 前缀算法的引擎映射为 HOSTCPU(alg_parse.cc), 对称标志不会被置位。此处显式拒绝,
    // 防止后续入口白名单放开后静默走入未验证的对称布局。
    CHK_PRT_RET(
        param.supportSymmetricMemory,
        HCCL_ERROR(
            "[InsV2AllGatherSequenceExecutor3LevelDpu][Orchestrate] symmetric memory is not supported by DPU 3level "
            "sequence executor."),
        HCCL_E_NOT_SUPPORT);
    CHK_RET(InitOrchestrateState(param, resCtx, "InsV2AllGatherSequenceExecutor3LevelDpu"));

    // 构建template
    InsAlgTemplate0 Level0TempAlg(param, resCtx.topoInfo.userRank, levels_[0].hierarchyInfo);
    InsAlgTemplate1 Level1TempAlg(param, resCtx.topoInfo.userRank, levels_[1].hierarchyInfo);
    InsAlgTemplate2 Level2TempAlg(param, resCtx.topoInfo.userRank, levels_[2].hierarchyInfo);

    rankIdxLevel0_ = myRank_ % levels_[0].rankSize; // level0 组内偏移
    rankIdxLevel1_ = (myRank_ % (levels_[0].rankSize * levels_[1].rankSize)) / levels_[0].rankSize; // level1 组编号
    skipLevel1_ = (levels_[1].rankSize == 1);
    if (skipLevel1_) {
        HCCL_INFO("[InsV2AllGatherSequenceExecutor3LevelDpu][Orchestrate] level1 rankSize is 1, skip level1");
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
            "[InsV2AllGatherSequenceExecutor3LevelDpu][Orchestrate]errNo[0x%016llx] All Gather executor kernel run "
            "failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    OrchestrateLoop(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, InsAlgTemplate0& tempAlgLevel0,
        InsAlgTemplate1& tempAlgLevel1, InsAlgTemplate2& tempAlgLevel2)
{
    HCCL_INFO(
        "[InsV2AllGatherSequenceExecutor3LevelDpu] AlgTemplate Level0 server is [%s]",
        tempAlgLevel0.Describe().c_str());
    HCCL_INFO(
        "[InsV2AllGatherSequenceExecutor3LevelDpu] AlgTemplate Level1 server is [%s]",
        tempAlgLevel1.Describe().c_str());
    HCCL_INFO(
        "[InsV2AllGatherSequenceExecutor3LevelDpu] AlgTemplate Level2 server is [%s]",
        tempAlgLevel2.Describe().c_str());

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

    if (maxCountPerLoop == 0) {
        HCCL_ERROR(
            "[InsV2AllGatherSequenceExecutor3LevelDpu] myRank[%u] maxCountPerLoop is 0, "
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

        // L2：最外层 host-DPU NHR（反序，先收最外层）
        GenTemplateAlgParamsLevel2(param, resCtx, currCount, dataOffset, tempAlgParamsLevel2);
        CHK_RET(tempAlgLevel2.KernelRun(param, tempAlgParamsLevel2, Level2TempAlgRes));

        // L1：中间 CLOS 的设备侧 NHR
        if (!skipLevel1_) {
            AgSeq3LevelGenLevel1Params(
                ctx, param, resCtx, currCount, dataOffset, tempAlgParamsLevel1,
                "InsV2AllGatherSequenceExecutor3LevelDpu");
            CHK_RET(tempAlgLevel1.KernelRun(param, tempAlgParamsLevel1, Level1TempAlgRes));
        }

        // L0：框内设备侧 Mesh1D 绕路
        AgSeq3LevelGenLevel0Params(
            ctx, param, resCtx, currCount, dataOffset, tempAlgParamsLevel0, "InsV2AllGatherSequenceExecutor3LevelDpu");
        CHK_RET(tempAlgLevel0.KernelRun(param, tempAlgParamsLevel0, Level0TempAlgRes));
        // 尾同步
    }

    HCCL_INFO("[InsV2AllGatherSequenceExecutor3LevelDpu][OrchestrateLoop] End.");
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void InsV2AllGatherSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenTemplateAlgParamsLevel2(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, const u64 curCount, const u64 dataOffset,
        TemplateDataParams& tempAlgParamsLevel2) const
{
    tempAlgParamsLevel2.buffInfo.inputPtr = param.inputPtr;
    // 对称内存不支持(Orchestrate 入口已显式拒绝), 输出恒为 CCL scratch 跨步布局
    tempAlgParamsLevel2.enableRemoteMemAccess = false;
    tempAlgParamsLevel2.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel2.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.outBuffBaseOff = rankIdxLevel1_ * curCount * dataTypeSize_;
    tempAlgParamsLevel2.outputSliceStride = curCount * dataTypeSize_ * levels_[1].rankSize;
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

    // allRank* 契约：L2 的 DPU 模板 InsTempAllGatherNHRDPU -> BatchTransferNHR 以 .at() 读这三个数组，
    // 为空会抛 std::out_of_range。三级 AG 的 L2 每片恒等长（尾块由外层 loop 消化）。
    // 输出即 CCL scratch（HCCL_BUFFER），按 outputSliceStride 跨步填 displs：BatchTransferNHR 的
    // srcDstOffset = hcclBuffBaseOff + allRankDispls[id]，NHR 环的 tx/rx 对称同偏移，各成员分片直接落
    // 到最终槽位 (rankIdxLevel1_ + d1*id)*S 上（对齐设备侧 InsTempAllGatherNHR 的 isOutputCclBuff_
    // 设计），PostLocalCopy 整体跳过，消除紧凑->跨步的原地重排拷贝。此处 repeatNum 恒为 1（上方
    // 硬编码），BatchTransferNHR 偏移中的 repeat*rankSize*sliceSize 项为 0，跨步 displs 安全。
    const u32 sliceNum = levels_[2].rankSize;
    const u64 sliceSize = curCount * dataTypeSize_;
    tempAlgParamsLevel2.allRankSliceSize.clear();
    tempAlgParamsLevel2.allRankDispls.clear();
    tempAlgParamsLevel2.allRankProcessedDataCount.clear();
    tempAlgParamsLevel2.allRankSliceSize.reserve(sliceNum);
    tempAlgParamsLevel2.allRankDispls.reserve(sliceNum);
    tempAlgParamsLevel2.allRankProcessedDataCount.reserve(sliceNum);
    for (u32 i = 0; i < sliceNum; i++) {
        tempAlgParamsLevel2.allRankDispls.emplace_back(i * tempAlgParamsLevel2.outputSliceStride);
        tempAlgParamsLevel2.allRankSliceSize.emplace_back(sliceSize);
        tempAlgParamsLevel2.allRankProcessedDataCount.emplace_back(curCount);
    }
    HCCL_DEBUG(
        "[InsV2AllGatherSequenceExecutor3LevelDpu][GenTemplateAlgParamsLevel2] rank[%u] allRank num[%zu] "
        "sliceBytes[%llu] lastDispl[%llu]",
        myRank_, tempAlgParamsLevel2.allRankSliceSize.size(), curCount * dataTypeSize_,
        tempAlgParamsLevel2.allRankDispls.empty() ? 0ULL : tempAlgParamsLevel2.allRankDispls.back());

    HCCL_DEBUG(
        "[InsV2AllGatherSequenceExecutor3LevelDpu][GenTemplateAlgParamsLevel2] rank[%u] inBuffBaseOff[%llu] "
        "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu]",
        myRank_, tempAlgParamsLevel2.buffInfo.inBuffBaseOff, tempAlgParamsLevel2.buffInfo.outBuffBaseOff,
        tempAlgParamsLevel2.buffInfo.hcclBuffBaseOff, tempAlgParamsLevel2.sliceSize,
        tempAlgParamsLevel2.outputSliceStride);
}

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_ALLGATHER, DpuAllGatherSequenceMeshConcurNHRNHR, InsV2AllGatherSequenceExecutor3LevelDpu,
    TopoMatchThreeLevel, InsTempAllGatherMesh1D1DZAxisDetour, InsTempAllGatherNHR, InsTempAllGatherNHRDPU);
REGISTER_ALG_ATTRS(DpuAllGatherSequenceMeshConcurNHRNHR, topo.isHostDpuOnly = true;
                   topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; topo.minTopoLevelNum = TOPO_LEVEL_NUM_3;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D);

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_ALLGATHER, DpuAllGatherSequenceMeshConcurRHDRHD, InsV2AllGatherSequenceExecutor3LevelDpu,
    TopoMatchThreeLevel, InsTempAllGatherMesh1D1DZAxisDetour, InsTempAllGatherRHD, InsTempAllGatherRHDDPU);
REGISTER_ALG_ATTRS(
    DpuAllGatherSequenceMeshConcurRHDRHD, topo.isHostDpuOnly = true; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3;
    topo.minTopoLevelNum = TOPO_LEVEL_NUM_3; topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails*) -> bool {
        return false;
    };);
} // namespace ops_hccl
// 算法注册
