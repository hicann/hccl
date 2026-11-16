/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_scatter_sequence_executor_3level_dpu.h"
#include "cost_model.h"
#include "aicpu_temp_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_scatter_rhd.h"
#include "ins_temp_scatter_rhd_dpu_inter_node.h"
#include "alg_attrs_registry.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2ScatterSequenceExecutor3LevelDpu<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InsV2ScatterSequenceExecutor3LevelDpu()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataCount_ = param.DataDes.count;
    dataSize_ = dataCount_ * dataTypeSize_;

    if (algHierarchyInfo.infos.empty() || algHierarchyInfo.infos.size() < TOPO_LEVEL_NUM_2
        || algHierarchyInfo.infos[0].empty() || algHierarchyInfo.infos[1].empty()
        || algHierarchyInfo.infos[0][0].empty() || algHierarchyInfo.infos[1][0].empty()) {
        HCCL_ERROR("[InsV2ScatterSequenceExecutor3LevelDpu][CalcRes] algHierarchyInfo.infos is invalid.");
        return HCCL_E_PARA;
    }
    skipLevel1_ = (algHierarchyInfo.infos[1][0].size() == 1);
    skipLevel2_ = (algHierarchyInfo.infos.size() == TOPO_LEVEL_NUM_2);
    // 三级形态下 infos[2] 同样防御（对齐 Orchestrate 侧与 aicpu 兄弟对 infos[0]/[1] 的口径）
    if (!skipLevel2_ && (algHierarchyInfo.infos[2].empty() || algHierarchyInfo.infos[2][0].empty())) {
        HCCL_ERROR("[InsV2ScatterSequenceExecutor3LevelDpu][CalcRes] algHierarchyInfo.infos[2] is invalid.");
        return HCCL_E_PARA;
    }
    std::shared_ptr<InsAlgTemplate0> tempAlgLevel0
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> tempAlgLevel1
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate2> tempAlgLevel2;
    if (!skipLevel2_) {
        tempAlgLevel2 = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo.infos[2]);
    }

    AlgResourceRequest resReq0;
    AlgResourceRequest resReq1;
    AlgResourceRequest resReq2;
    CHK_RET(tempAlgLevel0->CalcRes(comm, param, topoInfo, resReq0));
    if (skipLevel1_) {
        HCCL_INFO(
            "[InsV2ScatterSequenceExecutor3LevelDpu][CalcRes] myRank[%u] level1 rankSize is 1, skip level1 CalcRes",
            myRank_);
    } else {
        CHK_RET(tempAlgLevel1->CalcRes(comm, param, topoInfo, resReq1));
    }
    if (skipLevel2_) {
        HCCL_INFO(
            "[InsV2ScatterSequenceExecutor3LevelDpu][CalcRes] myRank[%u] level2 rankSize is 1, skip level2 CalcRes",
            myRank_);
    } else {
        CHK_RET(tempAlgLevel2->CalcRes(comm, param, topoInfo, resReq2));
    }

    auto maxThreadNum = resReq0.slaveThreadNum;
    if (!skipLevel1_) {
        maxThreadNum = std::max(maxThreadNum, resReq1.slaveThreadNum);
    }
    if (!skipLevel2_) {
        maxThreadNum = std::max(maxThreadNum, resReq2.slaveThreadNum);
    }
    resourceRequest.slaveThreadNum = maxThreadNum;
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.resize(resourceRequest.slaveThreadNum);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        if (i < resReq0.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReq0.notifyNumPerThread[i]);
        }
        if (!skipLevel1_ && i < resReq1.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReq1.notifyNumPerThread[i]);
        }
        if (!skipLevel2_ && i < resReq2.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReq2.notifyNumPerThread[i]);
        }
    }
    auto maxNotifyNum = resReq0.notifyNumOnMainThread;
    if (!skipLevel1_) {
        maxNotifyNum = std::max(maxNotifyNum, resReq1.notifyNumOnMainThread);
    }
    if (!skipLevel2_) {
        maxNotifyNum = std::max(maxNotifyNum, resReq2.notifyNumOnMainThread);
    }
    resourceRequest.notifyNumOnMainThread = maxNotifyNum;
    resourceRequest.channels.resize(TOPO_LEVEL_NUM_3);
    if (resReq0.channels.empty() || (!skipLevel1_ && resReq1.channels.empty())
        || (!skipLevel2_ && resReq2.channels.empty())) {
        HCCL_ERROR(
            "[InsV2ScatterSequenceExecutor3LevelDpu][CalcRes] myRank[%u] channels empty, level0[%zu] level1[%zu] "
            "level2[%zu]",
            myRank_, resReq0.channels.size(), resReq1.channels.size(), resReq2.channels.size());
        return HCCL_E_INTERNAL;
    }
    resourceRequest.channels[0] = resReq0.channels[0];
    if (!skipLevel1_) {
        resourceRequest.channels[1] = resReq1.channels[0];
    }
    if (!skipLevel2_) {
        resourceRequest.channels[2] = resReq2.channels[0];
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    if (algHierarchyInfo_.infos.size() < TOPO_LEVEL_NUM_2 || algHierarchyInfo_.infos[0].empty()
        || algHierarchyInfo_.infos[1].empty() || algHierarchyInfo_.infos[0][0].empty()
        || algHierarchyInfo_.infos[1][0].empty()) {
        HCCL_ERROR("[InsV2ScatterSequenceExecutor3LevelDpu][Orchestrate] invalid algHierarchyInfo infos.");
        return HCCL_E_PARA;
    }
    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();

    skipLevel1_ = (rankSizeLevel1_ == 1);
    skipLevel2_ = (algHierarchyInfo_.infos.size() == TOPO_LEVEL_NUM_2);
    if (skipLevel2_) {
        rankSizeLevel2_ = 1;
    } else {
        rankSizeLevel2_ = algHierarchyInfo_.infos[2][0].size();
    }
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2ScatterSequenceExecutor3LevelDpu][Orchestrate] myRank[%u] errNo[0x%016llx] scatter executor "
            "kernel run failed",
            myRank_, HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void InsV2ScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenIntraTemplateParams(
        TemplateDataParams& tempAlgParamsIntra, const u64 processedDataCount, const u64 currDataCount,
        const u64 loop) const
{
    tempAlgParamsIntra.count = currDataCount;
    tempAlgParamsIntra.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsIntra.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsIntra.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsIntra.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsIntra.tailSize = tempAlgParamsIntra.sliceSize;

    tempAlgParamsIntra.inputSliceStride = dataSize_;
    tempAlgParamsIntra.outputSliceStride = 0;
    tempAlgParamsIntra.repeatNum = rankSizeLevel1_ * rankSizeLevel2_;
    tempAlgParamsIntra.inputRepeatStride = rankSizeLevel0_ * dataSize_;
    tempAlgParamsIntra.outputRepeatStride = currDataCount * dataTypeSize_;

    HCCL_INFO(
        "[InsV2ScatterSequenceExecutor3LevelDpu] myRank[%u] loop[%llu] Intra inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] repeatNum[%llu] "
        "inputRepeatStride[%llu] outputRepeatStride[%llu]",
        myRank_, loop, tempAlgParamsIntra.inputSliceStride, tempAlgParamsIntra.outputSliceStride,
        tempAlgParamsIntra.sliceSize, tempAlgParamsIntra.buffInfo.inBuffBaseOff,
        tempAlgParamsIntra.buffInfo.outBuffBaseOff, tempAlgParamsIntra.repeatNum, tempAlgParamsIntra.inputRepeatStride,
        tempAlgParamsIntra.outputRepeatStride);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void InsV2ScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenInterTemplateParams1(
        TemplateDataParams& tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount,
        const u64 loop) const
{
    tempAlgParamsInter.count = currDataCount;
    tempAlgParamsInter.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsInter.buffInfo.outBuffBaseOff = skipLevel2_ ? processedDataCount * dataTypeSize_ : 0;
    tempAlgParamsInter.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsInter.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsInter.tailSize = tempAlgParamsInter.sliceSize;

    tempAlgParamsInter.inputSliceStride = 0;
    tempAlgParamsInter.outputSliceStride = 0;
    tempAlgParamsInter.repeatNum = rankSizeLevel2_;
    tempAlgParamsInter.inputRepeatStride = 0;

    HCCL_INFO(
        "[InsV2ScatterSequenceExecutor3LevelDpu] myRank[%u] loop[%llu] Inter1 inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] repeatNum[%llu] "
        "inputRepeatStride[%llu]",
        myRank_, loop, tempAlgParamsInter.inputSliceStride, tempAlgParamsInter.outputSliceStride,
        tempAlgParamsInter.sliceSize, tempAlgParamsInter.buffInfo.inBuffBaseOff,
        tempAlgParamsInter.buffInfo.outBuffBaseOff, tempAlgParamsInter.repeatNum, tempAlgParamsInter.inputRepeatStride);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void InsV2ScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenInterTemplateParams2(
        TemplateDataParams& tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount,
        const u64 loop) const
{
    tempAlgParamsInter.count = currDataCount;
    tempAlgParamsInter.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsInter.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsInter.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsInter.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsInter.tailSize = tempAlgParamsInter.sliceSize;

    tempAlgParamsInter.inputSliceStride = 0;
    tempAlgParamsInter.outputSliceStride = 0;
    tempAlgParamsInter.repeatNum = 1;
    tempAlgParamsInter.inputRepeatStride = 0;

    // allRank* 变长契约（对齐二级 executor 的 L1 填法，均匀退化；count==0 的对端在模板侧跳过通信）
    tempAlgParamsInter.allRankSliceSize.clear();
    tempAlgParamsInter.allRankDispls.clear();
    tempAlgParamsInter.allRankProcessedDataCount.clear();
    for (u32 j = 0; j < rankSizeLevel2_; j++) {
        tempAlgParamsInter.allRankSliceSize.push_back(currDataCount * dataTypeSize_);
        tempAlgParamsInter.allRankDispls.push_back(j * currDataCount * dataTypeSize_);
        tempAlgParamsInter.allRankProcessedDataCount.push_back(currDataCount);
    }

    HCCL_INFO(
        "[InsV2ScatterSequenceExecutor3LevelDpu] myRank[%u] loop[%llu] Inter2 inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] repeatNum[%llu] "
        "inputRepeatStride[%llu]",
        myRank_, loop, tempAlgParamsInter.inputSliceStride, tempAlgParamsInter.outputSliceStride,
        tempAlgParamsInter.sliceSize, tempAlgParamsInter.buffInfo.inBuffBaseOff,
        tempAlgParamsInter.buffInfo.outBuffBaseOff, tempAlgParamsInter.repeatNum, tempAlgParamsInter.inputRepeatStride);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
template <typename InsAlgTemplate>
HcclResult
InsV2ScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::GenTempResource(
    const AlgResourceCtxSerializable& resCtx, const u32 channelLevelIdx,
    const std::shared_ptr<InsAlgTemplate>& algTemplate, TemplateResource& tempResource) const
{
    AlgResourceRequest req;
    algTemplate->GetRes(req);
    if (channelLevelIdx >= remoteRankToChannelInfo_.size()) {
        HCCL_ERROR(
            "[InsV2ScatterSequenceExecutor3LevelDpu][GenTempResource] myRank[%u] channelLevelIdx[%u] should be "
            "lower than remoteRankToChannelInfo_.size()[%zu]",
            myRank_, channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    tempResource.channels = remoteRankToChannelInfo_[channelLevelIdx];
    if (resCtx.threads.size() < static_cast<size_t>(1) + req.slaveThreadNum) {
        HCCL_ERROR(
            "[InsV2ScatterSequenceExecutor3LevelDpu][GenTempResource] myRank[%u] threads size[%zu] less than "
            "required[%llu]",
            myRank_, resCtx.threads.size(), static_cast<unsigned long long>(1) + req.slaveThreadNum);
        return HCCL_E_INTERNAL;
    }
    tempResource.threads.assign(
        resCtx.threads.begin(), resCtx.threads.begin() + static_cast<size_t>(1) + req.slaveThreadNum);
    // shmem 指针透传（照 AG/RS 三级 DPU 先例）：二字段为裸 void* 无默认初始化，不赋值是野值；
    // 仅 L2 DPU 模板消费，L0/L1 设备模板不读该字段，无害
    tempResource.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    tempResource.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    u32 podSize = rankSizeLevel0_ * rankSizeLevel1_;
    u32 rootPodStartRank = param.root / podSize * podSize;
    u32 rootPodEndRank = rootPodStartRank + podSize - 1;
    TemplateDataParams tempAlgParamsLevel0;
    tempAlgParamsLevel0.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsLevel0.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsLevel0.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel0.buffInfo.hcclBuff = resCtx.cclMem;
    if (skipLevel1_ && skipLevel2_) {
        tempAlgParamsLevel0.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParamsLevel0.buffInfo.outputPtr = param.outputPtr;
    }

    std::shared_ptr<InsAlgTemplate0> algTemplateLevel0
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);
    if (rankSizeLevel0_ > 1) {
        CHK_RET(algTemplateLevel0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));
    }

    TemplateDataParams tempAlgParamsLevel1;
    tempAlgParamsLevel1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel1.buffInfo.hcclBuff = resCtx.cclMem;
    if (skipLevel2_) {
        tempAlgParamsLevel1.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParamsLevel1.buffInfo.outputPtr = param.outputPtr;
    } else {
        tempAlgParamsLevel1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsLevel1.buffInfo.outputPtr = resCtx.cclMem.addr;
    }

    std::shared_ptr<InsAlgTemplate1> algTemplateLevel1 = nullptr;
    if (!skipLevel1_) {
        algTemplateLevel1 = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);
        // 层内 root：root 所在框的同号卡（发起框间 scatter）
        algTemplateLevel1->SetRoot((param.root / rankSizeLevel0_) * rankSizeLevel0_ + (myRank_ % rankSizeLevel0_));
        CHK_RET(algTemplateLevel1->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }

    TemplateDataParams tempAlgParamsLevel2;
    tempAlgParamsLevel2.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsLevel2.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel2.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsLevel2.buffInfo.hcclBuff = resCtx.cclMem;

    std::shared_ptr<InsAlgTemplate2> algTemplateLevel2 = nullptr;
    if (!skipLevel2_) {
        algTemplateLevel2 = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[2]);
        // 层内 root：root 所在 pod 的同位卡（发起跨超节点 scatter）
        algTemplateLevel2->SetRoot(param.root / podSize * podSize + myRank_ % podSize);
        CHK_RET(algTemplateLevel2->SetchannelsPerRank(remoteRankToChannelInfo_[2]));
    }

    u32 templateScratchMultiplier0 = algTemplateLevel0->CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplier1
        = skipLevel1_ ? 1 : algTemplateLevel1->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplier2
        = skipLevel2_ ? 1 : algTemplateLevel2->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::OUTPUT);
    u32 templateScratchMultiplier
        = templateScratchMultiplier0 * templateScratchMultiplier1 * templateScratchMultiplier2;

    TemplateResource templateResource0;
    CHK_RET(GenTempResource(resCtx, 0, algTemplateLevel0, templateResource0));
    TemplateResource templateResource1;
    if (!skipLevel1_) {
        CHK_RET(GenTempResource(resCtx, 1, algTemplateLevel1, templateResource1));
    }
    TemplateResource templateResource2;
    if (!skipLevel2_) {
        CHK_RET(GenTempResource(resCtx, 2, algTemplateLevel2, templateResource2));
    }

    if (templateScratchMultiplier == 0) {
        HCCL_ERROR(
            "[InsV2ScatterSequenceExecutor3LevelDpu] myRank[%u] templateScratchMultiplier is 0, cannot compute "
            "maxCountPerLoop.",
            myRank_);
        return HCCL_E_INTERNAL;
    }
    u64 maxCountPerLoop = tempAlgParamsLevel0.buffInfo.hcclBuff.size / templateScratchMultiplier / AICPU_ALIGN_SIZE
                          * AICPU_ALIGN_SIZE / dataTypeSize_;
    if (maxCountPerLoop == 0) {
        HCCL_ERROR(
            "[InsV2ScatterSequenceExecutor3LevelDpu] myRank[%u] maxCountPerLoop is 0, scratchMultiplier[%u] too "
            "large for cclBuffSize[%llu]",
            myRank_, templateScratchMultiplier, tempAlgParamsLevel0.buffInfo.hcclBuff.size);
        return HCCL_E_INTERNAL;
    }
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        GenIntraTemplateParams(tempAlgParamsLevel0, processedDataCount, currDataCount, loop);
        CHK_RET(algTemplateLevel0->KernelRun(param, tempAlgParamsLevel0, templateResource0));

        // 仅 root 所在 pod 的 rank 执行框间 scatter（root-pod 门控，对齐 AICPU 三级）
        if (!skipLevel1_ && myRank_ >= rootPodStartRank && myRank_ <= rootPodEndRank) {
            GenInterTemplateParams1(tempAlgParamsLevel1, processedDataCount, currDataCount, loop);
            // L1（InsTempScatterRHD）宿主机进程内直接执行，executor 侧 SetRoot 有效，
            // 不经 DPURunInfo 序列化——无需 tempAlgParams.root
            CHK_RET(algTemplateLevel1->KernelRun(param, tempAlgParamsLevel1, templateResource1));
        }
        if (!skipLevel2_) {
            GenInterTemplateParams2(tempAlgParamsLevel2, processedDataCount, currDataCount, loop);
            // 仅 L2 DPU 模板经 DPURunInfo 序列化需要 tempAlgParams.root（device 侧 KernelRun 内
            // SetRoot(tempAlgParams.root) 消费；漏填时 device 侧恒用默认 0，跨 pod 的 rootAlgRank
            // 与 σ 置换全部错位）：L2 = root 所在 pod 的同位卡
            tempAlgParamsLevel2.root = param.root / podSize * podSize + myRank_ % podSize;
            CHK_RET(algTemplateLevel2->KernelRun(param, tempAlgParamsLevel2, templateResource2));
        }
        processedDataCount += currDataCount;
    }
    return HCCL_SUCCESS;
}

// preferred-only：固定占位系数（非 probe 版；空系数会使算法在 cost 表注册阶段被跳过，
// 显式配置也无法选中——RHD 不参与默认比价，占位不实际参与选路竞争）
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
std::vector<CostModelParam>
InsV2ScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)topoInfo;
    (void)algName;
    (void)param;
    std::vector<CostModelParam> params;
    params.push_back({0.0f, 0.0f, 1.0f, 0.0f});
    return params;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
AlgNetMeta
InsV2ScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)param;
    (void)algName;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2ScatterSequenceExecutor3LevelDpu][GetAlgNetMeta]",
            TopoProbeScene::PROBE_GET_ALG_NET_META)) {
        return {};
    }
    AlgNetMeta meta;
    u32 rankSizeLevel0 = algHierarchyInfo.infos[0][0].size();
    u32 rankSizeLevel1 = (algHierarchyInfo.infos.size() > 1) ? algHierarchyInfo.infos[1][0].size() : 1;
    u32 rankSizeLevel2 = (algHierarchyInfo.infos.size() > 2) ? algHierarchyInfo.infos[2][0].size() : 1;
    const auto& physIdx = algHierarchyInfo.physicalIdxForAlgoLevels;
    u32 physIdxLevel0 = static_cast<u32>(physIdx[0][0]);
    u32 physIdxLevel1 = (physIdx.size() > 1) ? static_cast<u32>(physIdx[1][0]) : physIdxLevel0;
    u32 physIdxLevel2 = (physIdx.size() > 2) ? static_cast<u32>(physIdx[2][0]) : physIdxLevel0;
    meta.netTypes.push_back(GetPhysicalLevelTopoType(topoInfo, physIdxLevel0));
    meta.netTypes.push_back(GetPhysicalLevelTopoType(topoInfo, physIdxLevel1));
    meta.netTypes.push_back(GetPhysicalLevelTopoType(topoInfo, physIdxLevel2));
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1, 1, 1};
    // 三段 dataRatio 对齐 AICPU 三级口径 {d1*d2, d2, 1}（逐级扇出量纲）
    meta.dataRatios = {static_cast<float>(rankSizeLevel1 * rankSizeLevel2), static_cast<float>(rankSizeLevel2), 1.0f};
    meta.rankSizes = {rankSizeLevel0, rankSizeLevel1, rankSizeLevel2};
    // 显式逐段 algoTypes（照 RS 三级 DPU 先例：该函数正确性影响 util 查表口径，须显式设置）
    meta.algoTypes = {AlgoType::MESH_CONCUR, AlgoType::RHD, AlgoType::RHD};
    return meta;
}

// Scatter DPU 三级 RHD（preferred-only）：L0 绕路（框内，root 所在框）→ L1 DevRHD（仅 root 所在
// pod 执行）→ L2 DpuRHD（全员）。命名硬约束：L0 段写 MeshConcur（触发 FillPhysicalIdxForAlgoLevels
// 双物理层配对；AnchorHostDpu 对最高算法层 AlgoType 不敏感，登记名字即可）
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_SCATTER, DpuScatterSequenceMeshConcurRHDRHD, InsV2ScatterSequenceExecutor3LevelDpu,
    TopoMatchThreeLevel, AicpuTempScatterMesh1DZAxisDetour, InsTempScatterRHD, InsTempScatterRhdDpuInterNode);
REGISTER_ALG_ATTRS(
    DpuScatterSequenceMeshConcurRHDRHD, topo.minTopoLevelNum = TOPO_LEVEL_NUM_3;
    topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; topo.isHostDpuOnly = true; topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails*) -> bool {
        return false;
    };);

} // namespace ops_hccl
