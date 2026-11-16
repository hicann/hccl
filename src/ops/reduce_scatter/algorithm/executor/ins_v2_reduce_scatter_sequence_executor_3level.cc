/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_scatter_sequence_executor_3level.h"
#include "cost_model.h"
#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "ins_temp_reduce_scatter_rhd.h"
#include "alg_data_trans_wrapper.h"
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ins_temp_reduce_scatter_rhd_dpu_inter.h"
#endif

#include "alg_attrs_registry.h"
#include "auto_selector_base.h"
namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2ReduceScatterSequenceExecutor3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InsV2ReduceScatterSequenceExecutor3Level()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    return RsSeq3LevelCalcRes<InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>(
        comm, param, topoInfo, algHierarchyInfo, resourceRequest, *this);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    CHK_RET(RsSeq3LevelRestoreOrchestrateCtx(param, resCtx, *this));
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2ReduceScatterSequenceExecutor3Level][Orchestrate] myRank[%u] errNo[0x%016llx] "
            "Reduce scatter executor kernel run failed",
            myRank_, HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenInterTemplateParams2(
        TemplateDataParams& tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount,
        const u64 loop) const
{
    tempAlgParamsInter.count = currDataCount;
    u64 symMemBaseOff2 = supportSymmetricMemory_ ? processedDataCount * dataTypeSize_ : 0;
    tempAlgParamsInter.buffInfo.inBuffBaseOff
        = symMemBaseOff2 + (rankIdxLevel0_ + (rankIdxLevel1_ * rankSizeLevel0_)) * currDataCount * dataTypeSize_;
    tempAlgParamsInter.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    if constexpr (InsAlgTemplate2::kNeedsAccRecvScratchIsolation) {
        // L2 DPU inter（RhdDpuInter）scratch 为 acc+recv 双区（2·d2·C），须与 L0/L1 原地规约区
        // [0, R·C) 隔离、排在其后（R = d0·d1·d2）；基址全 rank 同值——DPU 对端 remoteCclMem 同偏移即
        // 对端槽位（rankIdx 错位口径会破坏该同基址契约）。对称内存路径 L0/L1 在 input 对称窗口原地
        // 规约、不占 ccl，ccl 仅作 L2 scratch，基址取 0。
        // 公式单一事实来源：RhdIsolatedScratchBaseOff（与本文件预算分支 RhdIsolatedScratchBudgetMul 配套）。
        tempAlgParamsInter.buffInfo.hcclBuffBaseOff
            = supportSymmetricMemory_ ?
                  0 :
                  RhdIsolatedScratchBaseOff(
                      rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_, currDataCount, dataTypeSize_);
    } else {
        tempAlgParamsInter.buffInfo.hcclBuffBaseOff
            = (rankIdxLevel0_ + (rankIdxLevel1_ * rankSizeLevel0_)) * currDataCount * dataTypeSize_;
    }

    tempAlgParamsInter.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsInter.tailSize = tempAlgParamsInter.sliceSize;

    tempAlgParamsInter.inputSliceStride = rankSizeLevel0_ * rankSizeLevel1_ * currDataCount * dataTypeSize_;
    tempAlgParamsInter.outputSliceStride = 0;
    tempAlgParamsInter.repeatNum = 1;
    tempAlgParamsInter.inputRepeatStride = 0;
    tempAlgParamsInter.outputRepeatStride = 0;

    // allRank* 变长分片契约（需求 5.2，DPU inter 模板消费）：L2 组内逐超节点份额表——均匀退化
    // fill（allRankSliceSize[j] ≡ sliceSize、allRankDispls[j] = j·inputSliceStride，与 stride 契约
    // 等价）。allRankSliceSize 须全 rank 一致（全局表）；allRankDispls 为本 rank 本地输入布局。
    // stride 契约字段保留供 device RHD L2 消费，两套并存
    tempAlgParamsInter.allRankSliceSize.clear();
    tempAlgParamsInter.allRankDispls.clear();
    tempAlgParamsInter.allRankProcessedDataCount.clear();
    for (u32 j = 0; j < rankSizeLevel2_; ++j) {
        tempAlgParamsInter.allRankSliceSize.push_back(tempAlgParamsInter.sliceSize);
        tempAlgParamsInter.allRankDispls.push_back(j * tempAlgParamsInter.inputSliceStride);
        tempAlgParamsInter.allRankProcessedDataCount.push_back(currDataCount);
    }

    HCCL_INFO(
        "[InsV2ReduceScatterSequenceExecutor3Level] myRank[%u] loop[%llu] Inter2 inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] "
        "repeatNum[%llu] inputRepeatStride[%llu] outputRepeatStride[%llu]",
        myRank_, loop, tempAlgParamsInter.inputSliceStride, tempAlgParamsInter.outputSliceStride,
        tempAlgParamsInter.sliceSize, tempAlgParamsInter.buffInfo.inBuffBaseOff,
        tempAlgParamsInter.buffInfo.outBuffBaseOff, tempAlgParamsInter.repeatNum, tempAlgParamsInter.inputRepeatStride,
        tempAlgParamsInter.outputRepeatStride);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    RsSeq3LevelCtx rsCtx = RsSeq3LevelMakeCtx(*this);

    TemplateDataParams tempAlgParamsLevel0;
    SetTemplateBuffInfo(
        tempAlgParamsLevel0, BufferType::INPUT, BufferType::HCCL_BUFFER, param.inputPtr, resCtx.cclMem.addr,
        resCtx.cclMem);

    std::shared_ptr<InsAlgTemplate0> algTemplateLevel0
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);
    if (rankSizeLevel0_ > 1) {
        CHK_RET(algTemplateLevel0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));
    }

    TemplateDataParams tempAlgParamsLevel1;
    SetTemplateBuffInfo(
        tempAlgParamsLevel1, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, resCtx.cclMem.addr, resCtx.cclMem.addr,
        resCtx.cclMem);

    std::shared_ptr<InsAlgTemplate1> algTemplateLevel1
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);
    if (!skipLevel1_) {
        CHK_RET(algTemplateLevel1->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }

    TemplateDataParams tempAlgParamsLevel2;
    SetTemplateBuffInfo(
        tempAlgParamsLevel2, BufferType::HCCL_BUFFER, BufferType::OUTPUT, resCtx.cclMem.addr, param.outputPtr,
        resCtx.cclMem);

    std::shared_ptr<InsAlgTemplate2> algTemplateLevel2
        = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[2]);
    CHK_RET(algTemplateLevel2->SetchannelsPerRank(remoteRankToChannelInfo_[2]));

    u32 templateScratchMultiplier0 = algTemplateLevel0->CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplier1
        = skipLevel1_ ? 1 : algTemplateLevel1->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplier2
        = algTemplateLevel2->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::OUTPUT);
    // scratch 预算分区重算（L2 DPU inter，参考 3level_dpu.cc:167-172）：device 模板 L2 原地归约、
    // 三层共用 [0, m0·m1·m2·C) 连乘预算即可；RhdDpuInter 的 L2 scratch 为 acc+recv 双区（m2=2·d2），
    // 须与 L0/L1 原地规约区 [0, R·C)（R = d0·d1·d2）隔离、排在其后，总预算 = L0/L1 区 + L2 scratch
    // = R + m2，与 GenInterTemplateParams2 的 hcclBuffBaseOff（R·C）配套。不可照抄连乘 m0·m1·m2
    // （m2=2·d2 会把 L0/L1 区虚增至 2 倍）；R 用 rankSizeLevel 连乘表达（与 hcclBuffBaseOff 同口径）。
    u32 templateScratchMultiplier;
    if constexpr (InsAlgTemplate2::kNeedsAccRecvScratchIsolation) {
        // 公式单一事实来源：RhdIsolatedScratchBudgetMul（与 GenInterTemplateParams2 的
        // hcclBuffBaseOff=RhdIsolatedScratchBaseOff 配套，UT 钉死两侧口径一致性）
        templateScratchMultiplier = RhdIsolatedScratchBudgetMul(
            rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_, templateScratchMultiplier2);
    } else {
        templateScratchMultiplier
            = templateScratchMultiplier0 * templateScratchMultiplier1 * templateScratchMultiplier2;
    }

    TemplateResource templateResource0;
    CHK_RET(RsSeq3LevelGenTempResource(resCtx, 0, remoteRankToChannelInfo_, algTemplateLevel0, templateResource0));
    TemplateResource templateResource1;
    if (!skipLevel1_) {
        CHK_RET(RsSeq3LevelGenTempResource(resCtx, 1, remoteRankToChannelInfo_, algTemplateLevel1, templateResource1));
    }
    TemplateResource templateResource2;
    CHK_RET(RsSeq3LevelGenTempResource(resCtx, 2, remoteRankToChannelInfo_, algTemplateLevel2, templateResource2));

    if (templateScratchMultiplier == 0) {
        HCCL_ERROR("[%s] templateScratchMultiplier is 0, division by zero.", __func__);
        return HCCL_E_INTERNAL;
    }
    u64 maxCountPerLoop = tempAlgParamsLevel2.buffInfo.hcclBuff.size / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN
                          * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    if (maxCountPerLoop == 0) {
        HCCL_ERROR(
            "[InsV2ReduceScatterSequenceExecutor3Level] myRank[%u] maxCountPerLoop is 0, "
            "scratchMultiplier[%u] too large for cclBuffSize[%llu]",
            myRank_, templateScratchMultiplier, tempAlgParamsLevel2.buffInfo.hcclBuff.size);
        return HCCL_E_INTERNAL;
    }
    if constexpr (InsAlgTemplate2::kNeedsAccRecvScratchIsolation) {
        // 隔离双公式独立互验（预算侧 RhdIsolatedScratchBudgetMul 与基址侧 RhdIsolatedScratchBaseOff
        // 各自独立计算，配对恒等式 budget·C·dts == baseOff + m2·C·dts 仅在两侧同口径时成立）：
        // 任一侧被单独改动即在此显式报错，而非静默越界写 ccl；UT（rhd_step_test 配套不变量）
        // 与本运行期互验双层兜底。不用「乘数×maxCount≤ccl」式——maxCount 由乘数除法反推，
        // 该式恒真无独立信息量。
        const u64 isolatedBudgetTotal = static_cast<u64>(templateScratchMultiplier) * maxCountPerLoop * dataTypeSize_;
        const u64 isolatedLayoutTotal
            = RhdIsolatedScratchBaseOff(
                  rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_, maxCountPerLoop, dataTypeSize_)
              + static_cast<u64>(templateScratchMultiplier2) * maxCountPerLoop * dataTypeSize_;
        CHK_PRT_RET(
            isolatedBudgetTotal != isolatedLayoutTotal,
            HCCL_ERROR(
                "[InsV2ReduceScatterSequenceExecutor3Level] myRank[%u] isolated scratch budget[%llu] != layout "
                "baseOff[%llu] + L2 scratch[%llu], formula pair may have drifted.",
                myRank_, isolatedBudgetTotal,
                RhdIsolatedScratchBaseOff(
                    rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_, maxCountPerLoop, dataTypeSize_),
                static_cast<u64>(templateScratchMultiplier2) * maxCountPerLoop * dataTypeSize_),
            HcclResult::HCCL_E_INTERNAL);
    }
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    if (supportSymmetricMemory_) {
        loopTimes = 1;
        tempAlgParamsLevel0.buffInfo.outputPtr = param.inputPtr;
        tempAlgParamsLevel0.buffInfo.outBuffType = BufferType::INPUT;
        tempAlgParamsLevel1.buffInfo.inputPtr = param.inputPtr;
        tempAlgParamsLevel1.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParamsLevel1.buffInfo.outputPtr = param.inputPtr;
        tempAlgParamsLevel1.buffInfo.outBuffType = BufferType::INPUT;
        tempAlgParamsLevel2.buffInfo.inputPtr = param.inputPtr;
        tempAlgParamsLevel2.buffInfo.inBuffType = BufferType::INPUT;
        HCCL_INFO(
            "[InsV2ReduceScatterSequenceExecutor3Level][OrchestrateLoop] %s: symmetric memory enabled", param.algName);
    }
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        RsSeq3LevelGenIntraParams(rsCtx, tempAlgParamsLevel0, processedDataCount, currDataCount, loop);
        CHK_RET(algTemplateLevel0->KernelRun(param, tempAlgParamsLevel0, templateResource0));

        if (!skipLevel1_) {
            RsSeq3LevelGenInter1Params(rsCtx, tempAlgParamsLevel1, processedDataCount, currDataCount, loop);
            CHK_RET(algTemplateLevel1->KernelRun(param, tempAlgParamsLevel1, templateResource1));
        }

        GenInterTemplateParams2(tempAlgParamsLevel2, processedDataCount, currDataCount, loop);
        CHK_RET(algTemplateLevel2->KernelRun(param, tempAlgParamsLevel2, templateResource2));
        processedDataCount += currDataCount;
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
std::vector<CostModelParam>
InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    CalcCostCoeff(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2ReduceScatterSequenceExecutor3Level][CalcCostCoeff]",
            TopoProbeScene::PROBE_CALC_COST_COEFF)) {
        return {};
    }
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = topoInfo->isPod;
    u32 rankSizeLevel0 = algHierarchyInfo.infos[0][0].size();
    u32 rankSizeLevel1 = (algHierarchyInfo.infos.size() > 1) ? algHierarchyInfo.infos[1][0].size() : 1;
    u32 rankSizeLevel2 = (algHierarchyInfo.infos.size() > 2) ? algHierarchyInfo.infos[2][0].size() : 1;
    // algo level 对应物理层(MeshConcur 为 2元组)逐层下传, 供跨物理层模板用; 层数不足时高层复用低层
    const auto& physIdx = algHierarchyInfo.physicalIdxForAlgoLevels;
    CHK_PRT_RET(
        physIdx.size() != TOPO_LEVEL_NUM_3,
        HCCL_WARNING(
            "[InsV2ReduceScatterSequenceExecutor3Level][CalcCostCoeff] physicalIdxForAlgoLevels size[%zu] != 3.",
            physIdx.size()),
        {});
    std::vector<std::vector<CommTopo>> phyLevelNetTypes(physIdx.size());
    std::vector<std::vector<std::vector<u32>>> phyLevelPortNums(physIdx.size());
    AppendPhysicalLevelInfo(topoInfo, physIdx, phyLevelNetTypes, phyLevelPortNums);
    CommTopo netTypeLevel0 = phyLevelNetTypes[0][0];
    CommTopo netTypeLevel1 = phyLevelNetTypes[1][0];
    CommTopo netTypeLevel2 = phyLevelNetTypes[2][0];
    const std::vector<u32>& portNumLevel0 = phyLevelPortNums[0][0];
    const std::vector<u32>& portNumLevel1 = phyLevelPortNums[1][0];
    const std::vector<u32>& portNumLevel2 = phyLevelPortNums[2][0];
    if (portNumLevel0.empty() || portNumLevel1.empty() || portNumLevel2.empty()) {
        HCCL_WARNING("[InsV2ReduceScatterSequenceExecutor3Level][CalcCostCoeff] portNum is empty");
        return {};
    }
    HCCL_INFO(
        "[InsV2ReduceScatterSequenceExecutor3Level][CalcCostCoeff] rankSize=%d, rankSizeLevel0=%d, rankSizeLevel1=%d, "
        "rankSizeLevel2=%d, "
        "portNumLevel0=%d, portNumLevel1=%d, portNumLevel2=%d, netTypeLevel0=%d, netTypeLevel1=%d, netTypeLevel2=%d",
        rankSize, rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, portNumLevel0.empty() ? 0 : portNumLevel0[0],
        portNumLevel1.empty() ? 0 : portNumLevel1[0], portNumLevel2.empty() ? 0 : portNumLevel2[0],
        static_cast<int>(netTypeLevel0), static_cast<int>(netTypeLevel1), static_cast<int>(netTypeLevel2));
    std::vector<CostModelParam> params = [rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, rankSize, portNumLevel0,
                                          portNumLevel1, portNumLevel2, netTypeLevel0, netTypeLevel1, netTypeLevel2,
                                          isPod, algName, comm, topoInfo, physIdx, phyLevelNetTypes, phyLevelPortNums] {
        std::vector<CostModelParam> v;
        // Step1: 框内 RS（level0 mesh，全量输入的 1/rankSizeLevel0）
        auto p0 = InsAlgTemplate0::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel0, 1.0f * rankSizeLevel1 * rankSizeLevel2 / rankSize, netTypeLevel0, BufferType::INPUT,
            BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, portNumLevel0, isPod, algName, comm, topoInfo,
            rankSizeLevel1 * rankSizeLevel2, physIdx[0], phyLevelNetTypes[0], phyLevelPortNums[0]});
        // Step2: 框间 RS（level1 NHR，level0 归约后的 1/level0 份）
        auto p1 = InsAlgTemplate1::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel1, 1.0f * rankSizeLevel2 / rankSize, netTypeLevel1, BufferType::HCCL_BUFFER,
            BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, portNumLevel1, isPod, algName, comm, topoInfo,
            rankSizeLevel0 * rankSizeLevel2, physIdx[1], phyLevelNetTypes[1], phyLevelPortNums[1]});
        // Step3: 跨 super-pod RS（level2 NHR，level1 归约后的 1/(level0*level1) 份）
        auto p2 = InsAlgTemplate2::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel2, 1.0f / rankSize, netTypeLevel2, BufferType::HCCL_BUFFER, BufferType::OUTPUT,
            BufferType::HCCL_BUFFER, portNumLevel2, isPod, algName, comm, topoInfo, rankSizeLevel0 * rankSizeLevel1,
            physIdx[2], phyLevelNetTypes[2], phyLevelPortNums[2]});
        // 任一 template 未实现 CalcCostCoeff（返回空）则整个算法不参与 CostModel
        if (p0.empty() || p1.empty() || p2.empty()) {
            HCCL_WARNING(
                "[InsV2ReduceScatterSequenceExecutor3Level] CalcCostCoeff incomplete, skip "
                "(p0=%zu p1=%zu p2=%zu).",
                p0.size(), p1.size(), p2.size());
            return v;
        }
        v.insert(v.end(), p0.begin(), p0.end());
        v.insert(v.end(), p1.begin(), p1.end());
        v.insert(v.end(), p2.begin(), p2.end());
        return v;
    }();
    return params;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
AlgNetMeta InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)param;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2ReduceScatterSequenceExecutor3Level][GetAlgNetMeta]",
            TopoProbeScene::PROBE_GET_ALG_NET_META)) {
        return {};
    }
    // 三段 dataRatio 为 ÷rankSize 口径 {d1*d2/R, d2/R, 1/R}(与 CalcCostCoeff 一致), utils 除数取 userRankSize
    return RsSeq3LevelBuildNetMeta(topoInfo, algHierarchyInfo, topoInfo->userRankSize, *this);
}

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AicpuReduceScatterSequenceMeshConcurNHRNHR,
    InsV2ReduceScatterSequenceExecutor3Level, TopoMatchThreeLevel, InsTempReduceScatterMesh1DZAxisDetour,
    InsTempReduceScatterNHR, InsTempReduceScatterNHR);
REGISTER_ALG_ATTRS(AicpuReduceScatterSequenceMeshConcurNHRNHR, topo.minTopoLevelNum = TOPO_LEVEL_NUM_3;
                   topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; op.isSupportProd = false;
                   op.unsupportedDataTypes = UNSUPPORTED_64BIT);

// RHD 三级（Aicpu）：level0 z 轴绕路 Mesh + level1/level2 均 device RHD（同一模板实例）。
// preferred-only + 占位 cost，口径同二级注册。
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AicpuReduceScatterSequenceMeshConcurRHDRHD,
    InsV2ReduceScatterSequenceExecutor3Level, TopoMatchThreeLevel, InsTempReduceScatterMesh1DZAxisDetour,
    InsTempReduceScatterRHD, InsTempReduceScatterRHD);
REGISTER_ALG_ATTRS(
    AicpuReduceScatterSequenceMeshConcurRHDRHD, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
    topo.minTopoLevelNum = TOPO_LEVEL_NUM_3; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; op.isSupportProd = false;
    // Level1Nhr 硬过滤位（同 Dpu 三级注册 :354 口径）：selector 在软策略
    // （preferred）之前按 `Level1Nhr && !isSupportLevel1Nhr` 过滤，缺位时 Level0 GCD=1（level0 mesh
    // 退化）组网被静默回退。两注册共用同一 L0 模板 InsTempReduceScatterMesh1DZAxisDetour，Level0
    // GCD=1 时不受 level0 mesh 规则性约束，显式放行；物理前提由下方 topoPrecondCheck 保证，
    // 无额外排除理由。
    topo.isSupportLevel1Nhr = true;
    op.unsupportedDataTypes = UNSUPPORTED_64BIT; topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails*) -> bool {
        return false;
    };
    // 物理硬前提（topoPrecondCheck，preferred 不可绕过）：L1/L2 均须
    // 设备侧 CLOS——两层级均绑定 device 版 RHD 模板，其 CalcRes 过滤全部 ROCE 通道
    // （ins_temp_reduce_scatter_rhd.cc:45-52），任一层为 HOST 链路时通道必空、运行期以
    // HCCL_E_INTERNAL 失败而非选择期过滤。对齐同文件 Dpu 三级注册的同款前提。
    topo.topoPrecondCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->physicalLevels.size() == TOPO_LEVEL_NUM_3
               && topo->physicalLevels[1].topoType == CommTopo::COMM_TOPO_CLOS
               && topo->physicalLevels[1].locType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE
               && topo->physicalLevels[2].locType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    };);

#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
// RHD（递归折半倍增，支持层内非 2 幂）三级 hostdpu：level0 z 轴绕路 Mesh + level1 device RHD +
// level2 RHD DPU inter（host RoCE、DPU 只搬数、步间 AICPU LocalReduce 交替，老路径先行）。
// 绑 plain 版三级 executor（不用 3LevelDpu 版）：RhdDpuInter 消费五参数 stride 契约（KernelRun 入口
// 按 sliceSize/inputSliceStride 自推导 allRank* 表），而 3LevelDpu 版 L2 fill 为 allRank* 契约且把
// sliceSize/inputSliceStride 置 0（适配 Mesh DPU inter），换绑会致自推导零表、静默不规约。但 plain
// 版不能零改动直绑——RhdDpuInter 的 L2 scratch 为 acc+recv 双区（2·d2），须与 L0/L1 原地规约区
// [0, R·C) 隔离，故 plain 版经 kNeedsAccRecvScratchIsolation 分支注入 +m2 scratch 隔离（OrchestrateLoop
// 预算重算 + GenInterTemplateParams2 的 hcclBuffBaseOff = R·C）。
// preferred-only + 占位 cost（{{0,0,1,0}} 恒非空），attrs 对齐 Aicpu 三级版 + isHostDpuOnly
// （hostdpu 组网入候选）；版本守卫对齐模板编译段（RHD DPU inter 模板仅在 9.x 段编译）。
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, DpuReduceScatterSequenceMeshConcurRHDRHD,
    InsV2ReduceScatterSequenceExecutor3Level, TopoMatchThreeLevel, InsTempReduceScatterMesh1DZAxisDetour,
    InsTempReduceScatterRHD, InsTempReduceScatterRhdDpuInter);
REGISTER_ALG_ATTRS(
    DpuReduceScatterSequenceMeshConcurRHDRHD, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
    topo.minTopoLevelNum = TOPO_LEVEL_NUM_3; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; topo.isHostDpuOnly = true;
    // Level1Nhr 硬过滤位（同二级注册，见 executor.cc 对应注释；
    // Level0 GCD=1 的组网 RHD 三级不受 level0 mesh 规则性约束，显式放行）
    topo.isSupportLevel1Nhr = true;
    op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_64BIT;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails*) -> bool {
        return false;
    };
    // 物理硬前提（topoPrecondCheck：无条件执行，preferred/tuner均不可绕过）：L1框间层须为设备侧CLOS。
    // L1绑定device版RHD模板，其CalcRes经CalcChannelRequestNhr建链后过滤全部ROCE通道
    // （ins_temp_reduce_scatter_rhd.cc:45-52，device链路口径），L1为HOST链路时通道必空，
    // 运行期SetchannelsPerRank(空)以HCCL_E_INTERNAL失败（rD5）。违反前提的组网（如hostdpu
    // 全HOST标准三层布局）在选择期即被过滤，preferred配置由cost_model以WARNING静默回退自动选路
    // （回退默认二级DpuReduceScatterSequenceMeshMesh）；topoCustomCheck保持恒false（preferred-only）。
    // 谓词对齐NHRMesh本体（executor_3level_dpu.cc，L1=physicalLevels[1]由FindAnchors三层锚定保证）。
    topo.topoPrecondCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->physicalLevels.size() == TOPO_LEVEL_NUM_3
               && topo->physicalLevels[1].topoType == CommTopo::COMM_TOPO_CLOS
               && topo->physicalLevels[1].locType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    };);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)

} // namespace ops_hccl
