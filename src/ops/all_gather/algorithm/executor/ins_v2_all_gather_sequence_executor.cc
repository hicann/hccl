/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_sequence_executor.h"
#include "ins_temp_all_gather_mesh_1D.h"
#include "ins_temp_all_gather_nhr_dpu.h"
#include "coll_alg_v2_exec_registry.h"
#include "topo_match_two_level.h"
#include "alg_attrs_registry.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitCommInfo(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void)comm;
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    algHierarchyInfo_ = algHierarchyInfo;

    HCCL_INFO(
        "[InsV2AllGatherSequenceExecutor][InitCommInfo] myRank[%u], rankSize[%u], dataType[%u], dataTypeSize[%u]",
        myRank_, rankSize_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void)comm;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, AlgAttrs{}));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
std::vector<CostModelParam>
InsV2AllGatherSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)algName;
    (void)comm;
    AlgHierarchyInfoForAllLevel algHierarchyInfo; // TODO: unused for now, costmodel fallback
    (void)algHierarchyInfo;
    // TODO: CalcAlgHierarchyInfo(comm, topoInfo, algHierarchyInfo);
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = false;
    auto rs = CostModelManager::Global()->CalcRankSizeByTopo(topoInfo);
    u32 rankSizeLevel0 = rs.level0;
    u32 rankSizeLevel1 = rs.level1;
    // TODO: CommTopo netTypeLevel0 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[0]);
    CommTopo netTypeLevel0 = CommTopo::COMM_TOPO_1DMESH;
    // TODO: CommTopo netTypeLevel1 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[1]);
    CommTopo netTypeLevel1 = CommTopo::COMM_TOPO_CLOS;
    // TODO: std::vector<u32> portNumLevel0 = GetPortNumLevel(topoInfo, algHierarchyInfo.index[0]);
    std::vector<u32> portNumLevel0 = {1};
    // TODO: std::vector<u32> portNumLevel1 = GetPortNumLevel(topoInfo, algHierarchyInfo.index[1]);
    std::vector<u32> portNumLevel1 = {1};
    HCCL_INFO(
        "[CalcCostCoeff] rankSize=%d, rankSizeLevel0=%d, rankSizeLevel1=%d, portNumLevel0=%d, portNumLevel1=%d, "
        "netTypeLevel0=%d, netTypeLevel1=%d",
        rankSize, rankSizeLevel0, rankSizeLevel1, portNumLevel0, portNumLevel1, static_cast<int>(netTypeLevel0),
        static_cast<int>(netTypeLevel1));
    std::vector<CostModelParam> params
        = [rankSizeLevel0, rankSizeLevel1, portNumLevel0, portNumLevel1, netTypeLevel0, netTypeLevel1, isPod] {
              std::vector<CostModelParam> v;
              auto p0 = InsAlgTemplate0::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel0, 1.0f * rankSizeLevel1, netTypeLevel0, BufferType::OUTPUT, BufferType::OUTPUT,
                  BufferType::HCCL_BUFFER, portNumLevel0, isPod});
              v.insert(v.end(), p0.begin(), p0.end());
              auto p1 = InsAlgTemplate1::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel1, 1.0f, netTypeLevel1, BufferType::INPUT, BufferType::OUTPUT, BufferType::HCCL_BUFFER,
                  portNumLevel1, isPod});
              v.insert(v.end(), p1.begin(), p1.end());
              return v;
          }();
    return params;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
AlgNetMeta InsV2AllGatherSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const
{
    (void)param;
    auto rs = CostModelManager::Global()->CalcRankSizeByTopo(topoInfo);
    u32 rankSizeLevel0 = rs.level0;
    u32 rankSizeLevel1 = rs.level1;
    // TODO: CommTopo netTypeLevel0 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[0]);
    CommTopo netTypeLevel0 = CommTopo::COMM_TOPO_1DMESH;
    // TODO: CommTopo netTypeLevel1 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[1]);
    CommTopo netTypeLevel1 = CommTopo::COMM_TOPO_CLOS;
    AlgNetMeta meta;
    meta.netTypes.push_back(netTypeLevel0);
    meta.netTypes.push_back(netTypeLevel1);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1, 1};
    meta.dataRatios = {1.0f * rankSizeLevel1, 1.0f};
    meta.rankSizes = {rankSizeLevel0, rankSizeLevel1};
    return meta;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    InitCommInfo(comm, param, topoInfo, algHierarchyInfo);
    if (algHierarchyInfo.infos.size() < TOPO_LEVEL_NUM_2 || algHierarchyInfo.infos[0].empty()
        || algHierarchyInfo.infos[1].empty() || algHierarchyInfo.infos[0][0].empty()
        || algHierarchyInfo.infos[1][0].empty()) {
        HCCL_ERROR("[%s] invalid algHierarchyInfo infos.", __func__);
        return HCCL_E_PARA;
    }

    InsAlgTemplate0 intraTempAlg(param, myRank_, algHierarchyInfo.infos[0]);
    InsAlgTemplate1 interTempAlg(param, myRank_, algHierarchyInfo.infos[1]);

    AlgResourceRequest resReqIntra;
    CHK_RET(intraTempAlg.CalcRes(comm, param, topoInfo, resReqIntra));
    AlgResourceRequest resReqInter;
    CHK_RET(interTempAlg.CalcRes(comm, param, topoInfo, resReqInter));

    // 分级算法，slaveThread和对应notify可以复用
    resourceRequest.slaveThreadNum = std::max(resReqIntra.slaveThreadNum, resReqInter.slaveThreadNum);
    resourceRequest.notifyNumPerThread = resReqIntra.notifyNumPerThread; // dpu目前没有notify
    resourceRequest.notifyNumOnMainThread
        = std::max(resReqIntra.notifyNumOnMainThread, resReqInter.notifyNumOnMainThread);

    resourceRequest.channels = {resReqIntra.channels[0], resReqInter.channels[0]};
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllGatherSequenceExecutor][Orchestrate] Orchestrate Start");
    // 参数填充
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;
    if (algHierarchyInfo_.infos.size() < TOPO_LEVEL_NUM_2 || algHierarchyInfo_.infos[0].empty()
        || algHierarchyInfo_.infos[1].empty() || algHierarchyInfo_.infos[0][0].empty()
        || algHierarchyInfo_.infos[1][0].empty()) {
        HCCL_ERROR("[%s] invalid algHierarchyInfo infos.", __func__);
        return HCCL_E_PARA;
    }
    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2AllGatherSequenceExecutor][Orchestrate]errNo[0x%016llx] Orchestrate failed", HCCL_ERROR_CODE(ret)),
        ret);
    HCCL_INFO("[InsV2AllGatherSequenceExecutor][Orchestrate] Orchestrate End");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllGatherSequenceExecutor][OrchestrateLoop] Start");

    // 框间template
    TemplateDataParams interTempDataParams;
    interTempDataParams.buffInfo.inputPtr = param.inputPtr;
    interTempDataParams.buffInfo.outputPtr = param.outputPtr;
    interTempDataParams.buffInfo.hcclBuff = resCtx.cclMem;
    interTempDataParams.buffInfo.inBuffType = BufferType::INPUT;
    interTempDataParams.buffInfo.outBuffType = BufferType::OUTPUT;
    interTempDataParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建框间template
    InsAlgTemplate1 interTempAlg(param, myRank_, algHierarchyInfo_.infos[1]);

    // 框内template
    TemplateDataParams intraTempDataParams;
    intraTempDataParams.buffInfo.inputPtr = param.outputPtr;
    intraTempDataParams.buffInfo.outputPtr = param.outputPtr;
    intraTempDataParams.buffInfo.hcclBuff = resCtx.cclMem;
    intraTempDataParams.buffInfo.inBuffType = BufferType::OUTPUT;
    intraTempDataParams.buffInfo.outBuffType = BufferType::OUTPUT;
    intraTempDataParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    intraTempDataParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    // 构建框内template
    InsAlgTemplate0 intraTempAlg(param, myRank_, algHierarchyInfo_.infos[0]);

    u32 intraTemplateScratchMultiplier = intraTempAlg.CalcScratchMultiple(BufferType::OUTPUT, BufferType::OUTPUT);
    u32 interTemplateScratchMultiplier = interTempAlg.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 templateScratchMultiplier
        = std::max(interTemplateScratchMultiplier, intraTemplateScratchMultiplier * rankSizeLevel1_);

    // 构造框间template资源
    TemplateResource templateResourceInter;
    templateResourceInter.channels = remoteRankToChannelInfo_[1];
    templateResourceInter.threads = resCtx.threads;
    templateResourceInter.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceInter.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    // 构造框内template资源
    TemplateResource templateResourceIntra;
    templateResourceIntra.channels = remoteRankToChannelInfo_[0];
    templateResourceIntra.threads = resCtx.threads;
    templateResourceIntra.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceIntra.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;

    if (templateScratchMultiplier == 0) {
        HCCL_ERROR("[%s] templateScratchMultiplier is 0, division by zero.", __func__);
        return HCCL_E_INTERNAL;
    }
    u64 maxCountPerLoop = interTempDataParams.buffInfo.hcclBuff.size / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN
                          * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        // 框间的数据偏移和搬运计算
        interTempDataParams.count = currDataCount;
        interTempDataParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        u64 rankIdxInLevel0 = myRank_ % rankSizeLevel0_;
        interTempDataParams.buffInfo.outBuffBaseOff = rankIdxInLevel0 * dataSize_ + processedDataCount * dataTypeSize_;
        interTempDataParams.buffInfo.hcclBuffBaseOff = 0;

        interTempDataParams.sliceSize = currDataCount * dataTypeSize_;
        interTempDataParams.tailSize = interTempDataParams.sliceSize;
        // 这里的stride当成传统意义上的stride间隔
        interTempDataParams.inputSliceStride = 0;
        interTempDataParams.outputSliceStride = dataSize_ * rankSizeLevel0_;

        interTempDataParams.repeatNum = 1;
        interTempDataParams.inputRepeatStride = 0;
        interTempDataParams.outputRepeatStride = 0;

        HCCL_INFO(
            "[InsV2AllGatherSequenceExecutor] loop[%llu] interTempDataParams.inputSliceStride[%llu] "
            "interTempDataParams.outputSliceStride[%llu] interTempDataParams.sliceSize[%llu] "
            "interTempDataParams.buffInfo.inBuffBaseOff[%llu] interTempDataParams.buffInfo.outBuffBaseOff[%llu] "
            "interTempDataParams.repeatNum[%llu] interTempDataParams.inputRepeatStride[%llu] "
            "interTempDataParams.outputRepeatStride[%llu]",
            loop, interTempDataParams.inputSliceStride, interTempDataParams.outputSliceStride,
            interTempDataParams.sliceSize, interTempDataParams.buffInfo.inBuffBaseOff,
            interTempDataParams.buffInfo.outBuffBaseOff, interTempDataParams.repeatNum,
            interTempDataParams.inputRepeatStride, interTempDataParams.outputRepeatStride);

        CHK_RET(SplitData(currDataCount, rankSizeLevel1_, interTempDataParams));
        CHK_RET(interTempAlg.KernelRun(param, interTempDataParams, templateResourceInter));

        // 框内的数据偏移和搬运量计算
        intraTempDataParams.count = currDataCount;
        intraTempDataParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        intraTempDataParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        intraTempDataParams.buffInfo.hcclBuffBaseOff = 0;

        intraTempDataParams.sliceSize = currDataCount * dataTypeSize_;
        intraTempDataParams.tailSize = intraTempDataParams.sliceSize;
        // 这里的stride当成传统意义上的stride间隔
        intraTempDataParams.inputSliceStride = dataSize_;
        intraTempDataParams.outputSliceStride = dataSize_;

        intraTempDataParams.repeatNum = rankSizeLevel1_;
        intraTempDataParams.inputRepeatStride = dataSize_ * rankSizeLevel0_;
        intraTempDataParams.outputRepeatStride = dataSize_ * rankSizeLevel0_;

        HCCL_INFO(
            "[InsV2AllGatherSequenceExecutor] loop[%llu] intraTempDataParams.inputSliceStride[%llu] "
            "intraTempDataParams.outputSliceStride[%llu] intraTempDataParams.sliceSize[%llu] "
            "intraTempDataParams.buffInfo.inBuffBaseOff[%llu] intraTempDataParams.buffInfo.outBuffBaseOff[%llu] "
            "intraTempDataParams.repeatNum[%llu] intraTempDataParams.inputRepeatStride[%llu] "
            "intraTempDataParams.outputRepeatStride[%llu]",
            loop, intraTempDataParams.inputSliceStride, intraTempDataParams.outputSliceStride,
            intraTempDataParams.sliceSize, intraTempDataParams.buffInfo.inBuffBaseOff,
            intraTempDataParams.buffInfo.outBuffBaseOff, intraTempDataParams.repeatNum,
            intraTempDataParams.inputRepeatStride, intraTempDataParams.outputRepeatStride);

        CHK_RET(intraTempAlg.KernelRun(param, intraTempDataParams, templateResourceIntra));

        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2AllGatherSequenceExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::SplitData(
    const u64 dataCount, const u64 rankSize, TemplateDataParams& tempAlgParams)
{
    u32 sliceNum = rankSize;
    tempAlgParams.allRankSliceSize.clear();
    tempAlgParams.allRankDispls.clear();
    tempAlgParams.allRankProcessedDataCount.clear();
    tempAlgParams.allRankSliceSize.reserve(sliceNum);
    tempAlgParams.allRankDispls.reserve(sliceNum);
    tempAlgParams.allRankProcessedDataCount.reserve(sliceNum);

    u64 sliceSize = dataCount * dataTypeSize_;
    for (u32 i = 0; i < sliceNum; i++) {
        tempAlgParams.allRankDispls.emplace_back(i * sliceSize);
        tempAlgParams.allRankSliceSize.emplace_back(sliceSize);
        tempAlgParams.allRankProcessedDataCount.emplace_back(dataCount);
    }
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_ALLGATHER, DpuAllGatherSequenceMeshNHR, InsV2AllGatherSequenceExecutor, TopoMatchTwoLevel,
    InsTempAllGatherMesh1D, InsTempAllGatherNHRDPU);
REGISTER_ALG_ATTRS(
    DpuAllGatherSequenceMeshNHR, topo.isSupportLevel0PcieMix = true; topo.minTopoLevelNum = 2; topo.maxTopoLevelNum = 3;
    topo.isHostDpuOnly = true;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS | LEVEL0_TOPO_CLOS;
    // MESH_1D_CLOS 非pcieMix 且每框多卡时走 PipeLineUBX，其余场景走本算法，通信域初始化时过滤
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        if (topo->level0Topo != Level0Shape::MESH_1D_CLOS) {
            return true;
        }
        return topo->level0PcieMix || topo->netLayerDetails.localNetInsSizeOfLayer.empty()
               || topo->netLayerDetails.localNetInsSizeOfLayer[0] == 1;
    };);

} // namespace ops_hccl
