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

#include "topo_match_multilevel.h"
#include "topo_match_ubx.h"
#include "topo_match_pcie_mix.h"

namespace ops_hccl {
constexpr u32 OMNIPIPE_LEVEL2_IDX = 2;

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InsV2AllGatherSequenceExecutor3Level()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
    AlgResourceRequest &resourceRequest)
{
    if (algHierarchyInfo.infos.size() != SEQUENCE_EXECUTOR_3_LEVEL_NUM) {
        HCCL_ERROR("[InsV2AllGatherSequenceExecutor3Level] algHierarchyInfo size %u should be %u", algHierarchyInfo.infos.size(), SEQUENCE_EXECUTOR_3_LEVEL_NUM);
        return HCCL_E_INTERNAL;
    }
    // 构建template
    InsAlgTemplate0 Level0TempAlg(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    InsAlgTemplate1 Level1TempAlg(param, topoInfo->userRank, algHierarchyInfo.infos[1]);
    InsAlgTemplate2 Level2TempAlg(param, topoInfo->userRank, algHierarchyInfo.infos[2]);
    
    // 调用计算资源的函数
    AlgResourceRequest Level0TempRequest, Level1TempRequest, Level2TempRequest;
    CHK_RET(Level0TempAlg.CalcRes(comm, param, topoInfo, Level0TempRequest));
    CHK_RET(Level1TempAlg.CalcRes(comm, param, topoInfo, Level1TempRequest));
    CHK_RET(Level2TempAlg.CalcRes(comm, param, topoInfo, Level2TempRequest));

    resourceRequest.notifyNumOnMainThread = std::max({Level0TempRequest.notifyNumOnMainThread, Level1TempRequest.notifyNumOnMainThread, Level2TempRequest.notifyNumOnMainThread});
    resourceRequest.slaveThreadNum = std::max({Level0TempRequest.slaveThreadNum, Level1TempRequest.slaveThreadNum, Level2TempRequest.slaveThreadNum});
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              Level0TempRequest.notifyNumPerThread.begin(),
                                              Level0TempRequest.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              Level1TempRequest.notifyNumPerThread.begin(),
                                              Level1TempRequest.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              Level2TempRequest.notifyNumPerThread.begin(),
                                              Level2TempRequest.notifyNumPerThread.end());
    CHK_PRT_RET(Level0TempRequest.channels.empty() || Level1TempRequest.channels.empty() || Level2TempRequest.channels.empty(),
                     HCCL_ERROR("[InsV2AllGatherSequenceExecutor3Level][CalcRes] Level0Template, Level1Template or Level2TempRequest has empty channels."),
                     HcclResult::HCCL_E_INTERNAL);
    resourceRequest.channels.emplace_back(Level0TempRequest.channels[0]);
    resourceRequest.channels.emplace_back(Level1TempRequest.channels[0]);
    resourceRequest.channels.emplace_back(Level2TempRequest.channels[0]);
    HCCL_DEBUG("[InsV2AllGatherSequenceExecutor3Level][CalcRes] notifyNumOnMainThread[%u], slaveThreadNum[%u], "
               "channels[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
               resourceRequest.channels.size());
    for (auto i = 0; i < resourceRequest.notifyNumPerThread.size(); i++) {
        HCCL_DEBUG("[InsV2AllGatherSequenceExecutor3Level][CalcRes] myRank[%u], notifyNumPerThread[%u]=[%u]", i,
                   resourceRequest.notifyNumPerThread[i]);
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherSequenceExecutor3Level][Orchestrate] Orchestrate Start");
    maxTmpMemSize_ = resCtx.cclMem.size;  // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    myRank_ = resCtx.topoInfo.userRank;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    levels_.emplace_back(remoteRankToChannelInfo_[0], resCtx.algHierarchyInfo.infos[0]);
    levels_.emplace_back(remoteRankToChannelInfo_[1], resCtx.algHierarchyInfo.infos[1]);
    levels_.emplace_back(remoteRankToChannelInfo_[2], resCtx.algHierarchyInfo.infos[2]);
    // 实例化算法模板类
    // 构建template
    InsAlgTemplate0 Level0TempAlg(param, resCtx.topoInfo.userRank, levels_[0].hierarchyInfo);
    InsAlgTemplate1 Level1TempAlg(param, resCtx.topoInfo.userRank, levels_[1].hierarchyInfo);
    InsAlgTemplate2 Level2TempAlg(param, resCtx.topoInfo.userRank, levels_[2].hierarchyInfo);

    rankIdxLevel0_ = myRank_ % levels_[0].rankSize;                                    // level0 组内偏移
    rankIdxLevel1_ = myRank_ % (levels_[0].rankSize * levels_[1].rankSize);            // level1 组编号

    CHK_RET(Level0TempAlg.SetchannelsPerRank(levels_[0].channels));
    // 将计算资源分配个每个算法
    CHK_RET(PrepareResForTemplate(Level0TempAlg, Level1TempAlg, Level2TempAlg));
    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx, Level0TempAlg, Level1TempAlg, Level2TempAlg);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllGatherSequenceExecutor3Level][Orchestrate]errNo[0x%016llx] All Gather excutor kernel run failed",
                   HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
uint64_t InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::GetRankSize(const std::vector<std::vector<u32>> &vTopo)
{
    uint64_t count = 1;
    for (const auto &i : vTopo) {
        count *= i.size();
    }
    return count;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::GenTempResource(int idx, TemplateResource &res) const
{
    res.channels = levels_[idx].channels;
    res.threads = levels_[idx].threads;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::PrepareResForTemplate(
    InsAlgTemplate0 &tempAlgLevel0, InsAlgTemplate1 &tempAlgLevel1, InsAlgTemplate2 &tempAlgLevel2)
{
    AlgResourceRequest Level0TempRequest;
    AlgResourceRequest Level1TempRequest;
    AlgResourceRequest Level2TempRequest;
    tempAlgLevel0.GetRes(Level0TempRequest);
    tempAlgLevel1.GetRes(Level1TempRequest);
    tempAlgLevel2.GetRes(Level2TempRequest);

    levels_[0].threads.assign(threads_.begin(), threads_.begin() + Level0TempRequest.slaveThreadNum + 1);
    levels_[1].threads.assign(threads_.begin(), threads_.begin() + Level1TempRequest.slaveThreadNum + 1);
    levels_[2].threads.assign(threads_.begin(), threads_.begin() + Level2TempRequest.slaveThreadNum + 1);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgLevel0,
    InsAlgTemplate1 &tempAlgLevel1, InsAlgTemplate2 &tempAlgLevel2)
{
    HCCL_INFO("[InsV2AllGatherParallelExecutor] AlgTemplate Level0 server is [%s]", tempAlgLevel0.Describe().c_str());
    HCCL_INFO("[InsV2AllGatherParallelExecutor] AlgTemplate Level1 server is [%s]", tempAlgLevel1.Describe().c_str());
    HCCL_INFO("[InsV2AllGatherParallelExecutor] AlgTemplate Level2 server is [%s]", tempAlgLevel2.Describe().c_str());

    u32 templateScratchMultiplierLevel0 = tempAlgLevel0.CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::OUTPUT);
 	u32 templateScratchMultiplierLevel1 = tempAlgLevel1.CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER);
 	u32 templateScratchMultiplierLevel2 = tempAlgLevel2.CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);
 	u32 totalScratchMultiple = templateScratchMultiplierLevel0 * templateScratchMultiplierLevel1 * templateScratchMultiplierLevel2;

    u64 scratchMemBlockSize = maxTmpMemSize_;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    if (totalScratchMultiple > 0) {
        scratchMemBlockSize = (maxTmpMemSize_ / HCCL_MIN_SLICE_ALIGN / totalScratchMultiple) * HCCL_MIN_SLICE_ALIGN;
        scratchMemBlockSize = std::min(scratchMemBlockSize, transportBoundDataSize);
    }

    u64 maxCountPerLoop =
        (std::min(static_cast<u64>(scratchMemBlockSize), static_cast<u64>(UB_MAX_DATA_SIZE)) / dataTypeSize_ / 10) * 10;
    if (maxCountPerLoop == 0) {
        HCCL_ERROR("[InsV2AllGatherParallelExecutor] myRank[%u] maxCountPerLoop is 0, "
            "scratchMultiplier[%u] too large for cclBuffSize[%llu]",
            totalScratchMultiple, scratchMemBlockSize);
        return HCCL_E_INTERNAL;
    }
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    TemplateResource Level2TempAlgRes, Level1TempAlgRes, Level0TempAlgRes;
    TemplateDataParams tempAlgParamsLevel2, tempAlgParamsLevel1, tempAlgParamsLevel0;
    CHK_RET(GenTempResource(OMNIPIPE_LEVEL2_IDX, Level2TempAlgRes));
    CHK_RET(GenTempResource(1, Level1TempAlgRes));
    CHK_RET(GenTempResource(0, Level0TempAlgRes));

    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCount = (loopIndex == loopTimes - 1) ? (dataCount_ - loopIndex * maxCountPerLoop) : maxCountPerLoop;
        u64 dataOffset = loopIndex * maxCountPerLoop * dataTypeSize_;
        // 数据0的server内的mesh算法
        GenTemplateAlgParamsLevel2(param, resCtx, currCount, dataOffset, tempAlgParamsLevel2);
        CHK_RET(tempAlgLevel2.KernelRun(param, tempAlgParamsLevel2, Level2TempAlgRes));

        // 数据1的server间的nhr算法
        GenTemplateAlgParamsLevel1(param, resCtx, currCount, dataOffset, tempAlgParamsLevel1);
        CHK_RET(tempAlgLevel1.KernelRun(param, tempAlgParamsLevel1, Level1TempAlgRes));

        GenTemplateAlgParamsLevel0(param, resCtx, currCount, dataOffset, tempAlgParamsLevel0);
        CHK_RET(tempAlgLevel0.KernelRun(param, tempAlgParamsLevel0, Level0TempAlgRes));
        // 尾同步
    }

    HCCL_INFO("[InsV2AllGatherParallelExecutor][OrchestrateLoop] End.");
    return HcclResult::HCCL_SUCCESS;
};

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::GenTemplateAlgParamsLevel2(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 curCount, const u64 dataOffset, TemplateDataParams &tempAlgParamsLevel2) const
{
    tempAlgParamsLevel2.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsLevel2.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel2.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsLevel2.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsLevel2.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.inputSize = param.inputSize;
    tempAlgParamsLevel2.buffInfo.outputSize = param.outputSize;

    tempAlgParamsLevel2.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsLevel2.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsLevel2.buffInfo.hcclBuffBaseOff = levels_[2].rankSize * levels_[1].rankSize * curCount * dataTypeSize_;
    tempAlgParamsLevel2.sliceSize = curCount * dataTypeSize_;
    tempAlgParamsLevel2.count = curCount;
    tempAlgParamsLevel2.tailSize = tempAlgParamsLevel2.sliceSize;

    tempAlgParamsLevel2.inputSliceStride = 0;
    tempAlgParamsLevel2.outputSliceStride = 0;
    tempAlgParamsLevel2.repeatNum = 1;
    tempAlgParamsLevel2.inputRepeatStride = 0;
    tempAlgParamsLevel2.outputRepeatStride = 0;
    tempAlgParamsLevel2.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
    HCCL_DEBUG("[InsV2AllGatherSequenceExecutor3Level][GenTemplateAlgParamsLevel2] rank[%u] inBuffBaseOff[%llu] "
               "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu]",
               myRank_, tempAlgParamsLevel2.buffInfo.inBuffBaseOff, tempAlgParamsLevel2.buffInfo.outBuffBaseOff,
               tempAlgParamsLevel2.buffInfo.hcclBuffBaseOff, tempAlgParamsLevel2.sliceSize,
               tempAlgParamsLevel2.outputSliceStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::GenTemplateAlgParamsLevel1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 curCount, const u64 dataOffset, TemplateDataParams &tempAlgParamsLevel1) const
{
    tempAlgParamsLevel1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsLevel1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.inputSize = param.inputSize;
    tempAlgParamsLevel1.buffInfo.outputSize = param.outputSize;

    tempAlgParamsLevel1.buffInfo.inBuffBaseOff = levels_[2].rankSize * levels_[1].rankSize * curCount * dataTypeSize_;
    tempAlgParamsLevel1.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsLevel1.buffInfo.hcclBuffBaseOff = 0;
    tempAlgParamsLevel1.sliceSize = curCount * dataTypeSize_;
    tempAlgParamsLevel1.count = curCount;
    tempAlgParamsLevel1.tailSize = tempAlgParamsLevel1.sliceSize;

    tempAlgParamsLevel1.inputSliceStride = 0;
    tempAlgParamsLevel1.outputSliceStride = 0;
    tempAlgParamsLevel1.repeatNum = levels_[2].rankSize;
    tempAlgParamsLevel1.inputRepeatStride = curCount * dataTypeSize_;
    tempAlgParamsLevel1.outputRepeatStride = 0;
    tempAlgParamsLevel1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
    HCCL_DEBUG("[InsV2AllGatherSequenceExecutor3Level][GenTemplateAlgParamsLevel10] rank[%u] inBuffBaseOff[%llu] "
               "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu] "
               "outputRepeatStride[%llu]",
               myRank_, tempAlgParamsLevel1.buffInfo.inBuffBaseOff, tempAlgParamsLevel1.buffInfo.outBuffBaseOff,
               tempAlgParamsLevel1.buffInfo.hcclBuffBaseOff, tempAlgParamsLevel1.sliceSize,
               tempAlgParamsLevel1.outputSliceStride, tempAlgParamsLevel1.outputRepeatStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void InsV2AllGatherSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::GenTemplateAlgParamsLevel0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 curCount, const u64 dataOffset, TemplateDataParams &tempAlgParamsLevel0) const
{
    tempAlgParamsLevel0.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsLevel0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsLevel0.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsLevel0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel0.buffInfo.inputSize = param.inputSize;
    tempAlgParamsLevel0.buffInfo.outputSize = param.outputSize;

    tempAlgParamsLevel0.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsLevel0.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsLevel0.buffInfo.hcclBuffBaseOff = 0;
    tempAlgParamsLevel0.sliceSize = curCount * dataTypeSize_;
    tempAlgParamsLevel0.count = curCount;
    tempAlgParamsLevel0.tailSize = tempAlgParamsLevel0.sliceSize;

    tempAlgParamsLevel0.inputSliceStride = 0;
    tempAlgParamsLevel0.outputSliceStride = dataSize_;
    tempAlgParamsLevel0.repeatNum = levels_[1].rankSize * levels_[2].rankSize;
    tempAlgParamsLevel0.inputRepeatStride = curCount * dataTypeSize_;
    tempAlgParamsLevel0.outputRepeatStride = levels_[0].rankSize * dataSize_;
    tempAlgParamsLevel0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    HCCL_DEBUG(
        "[InsV2AllGatherSequenceExecutor3Level][GenTemplateAlgParamsLevel0] rank[%d] inBuffBaseOff[%llu] "
        "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu] levels_[0].rankSize[%u] "
        "levels_[1].rankSize[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]",
        myRank_, tempAlgParamsLevel0.buffInfo.inBuffBaseOff, tempAlgParamsLevel0.buffInfo.outBuffBaseOff,
        tempAlgParamsLevel0.buffInfo.hcclBuffBaseOff, tempAlgParamsLevel0.sliceSize,
        tempAlgParamsLevel0.outputSliceStride, levels_[0].rankSize, levels_[1].rankSize, rankIdxLevel0_, rankIdxLevel1_);
    return;
}

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER, 
    InsAllGatherSequenceNHRNHRMesh1D,
    InsV2AllGatherSequenceExecutor3Level, 
    TopoMatchMultilevel,
    InsTempAllGatherMesh1D1DZAxisDetour,
    InsTempAllGatherNHR, 
    InsTempAllGatherNHR);
}
// 算法注册