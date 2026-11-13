/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_nhr_dpu.h"
#include "dpu_alg_nhr_opt_wrapper.h"
#include "alg_data_trans_wrapper.h"
#include "channel.h"
#include "alg_v2_template_register.h"
#include "exec_timeout_manager.h"

namespace ops_hccl {

InsTempReduceScatterNhrDpu::InsTempReduceScatterNhrDpu(
    const OpParam& param, const uint32_t rankId, const std::vector<std::vector<uint32_t>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

std::vector<CostModelParam> InsTempReduceScatterNhrDpu::CalcCostCoeff(CalcCostCoeffParam param)
{
    int portNum = (param.isPod && param.netType == CommTopo::COMM_TOPO_CLOS && param.portNum.size() >= 2) ?
                      (param.portNum[0] + param.portNum[1]) :
                      param.portNum[0];

    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    CostModelManager::Global()->CalcMeshParam(
        param.dataRatio, param.netType, portNum, param.rankSize, A, false, EngineType::CPU);
    CostModelManager::Global()->CalcLocalCopyParams(
        param.dataRatio * (1.0f + 1.0f / static_cast<float>(param.rankSize)), EngineType::AICPU, B);
    CostModelManager::Global()->CalcDpuLatencyParams(
        static_cast<int>(GetNHRStepNum(param.rankSize)), 1, 2, static_cast<int>(param.rankSize - 1), C);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    HCCL_DEBUG("[%s] CalcCostCoeff A=%f B=%f C=%f D=%f.", __func__, A, B, C, D);
    return params;
}

HcclResult InsTempReduceScatterNhrDpu::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhrHighestHostRoce(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO(
        "[InsTempReduceScatterNhrDpu][CalcRes]slaveThreadNum[%u] notifyNumOnMainThread[%u] level1Channels[%zu].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, level1Channels.size());
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterNhrDpu::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void)inBufferType;
    (void)outBufferType;

    u64 scratchMultiple = templateRankSize_;
    HCCL_INFO("[InsTempReduceScatterNhrDpu][CalcScratchMultiple] templateScratchMultiplier[%llu]", scratchMultiple);
    return scratchMultiple;
}

HcclResult InsTempReduceScatterNhrDpu::NormalizeStrideLayout(
    const TemplateDataParams& rawParams, TemplateDataParams& workParams, const bool strideLayout)
{
    if (!strideLayout) {
        // 调用方已按 SplitData 约定给表，此处只做长度一致性校验，避免后续 .at() 越界
        CHK_PRT_RET(
            rawParams.allRankSliceSize.size() != templateRankSize_
                || rawParams.allRankDispls.size() != templateRankSize_
                || rawParams.allRankProcessedDataCount.size() != templateRankSize_,
            HCCL_ERROR(
                "[InsTempReduceScatterNhrDpu][NormalizeStrideLayout] inconsistent slice metadata, rankSize[%u], "
                "sizes[%zu], displs[%zu], counts[%zu].",
                templateRankSize_, rawParams.allRankSliceSize.size(), rawParams.allRankDispls.size(),
                rawParams.allRankProcessedDataCount.size()),
            HCCL_E_PARA);
        return HCCL_SUCCESS;
    }
    // stride 约定归一化为 N 个等大片，槽位 j 偏移 = j * outputSliceStride
    workParams.allRankSliceSize.assign(templateRankSize_, rawParams.sliceSize);
    workParams.allRankProcessedDataCount.assign(templateRankSize_, rawParams.count);
    workParams.allRankDispls.resize(templateRankSize_);
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; ++rankIdx) {
        workParams.allRankDispls[rankIdx] = static_cast<u64>(rankIdx) * rawParams.outputSliceStride;
    }
    HCCL_INFO(
        "[InsTempReduceScatterNhrDpu][NormalizeStrideLayout] rank[%u] normalize stride layout to uniform slices, "
        "sliceSize[%llu] outputSliceStride[%llu]",
        myRank_, static_cast<unsigned long long>(rawParams.sliceSize),
        static_cast<unsigned long long>(rawParams.outputSliceStride));
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterNhrDpu::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempReduceScatterNhrDpu] Run Start");
    CHK_PRT_RET(
        subCommRanks_.empty() || subCommRanks_[0].empty(),
        HCCL_ERROR("[InsTempReduceScatterNhrDpu][KernelRun] invalid empty subCommRanks, myRank[%u]", myRank_),
        HCCL_E_PARA);
    if (templateResource.threads.size() < 1) {
        HCCL_ERROR("[InsTempReduceScatterNhrDpu] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    // stride 约定先归一化为 SplitData 表，后续预拷贝/序列化/收尾统一按表寻址
    TemplateDataParams workParams = tempAlgParams;
    // 是否 stride 布局在此判定一次并显式下传，避免预拷贝再按 rawParams 重算一遍
    const bool strideLayout = tempAlgParams.allRankSliceSize.empty();
    CHK_RET(NormalizeStrideLayout(tempAlgParams, workParams, strideLayout));

    TemplateDataParams dpuAlgParams = workParams;
    dpuAlgParams.dataType = param.DataDes.dataType;
    dpuAlgParams.reduceOp = param.reduceType;

    if (templateResource.isDpuThreadEnabled) {
        // 新路径：预拷贝结果作为 DPU write-reduce 的累加初值，须先于 DPU 开工完成；
        // 随后主流通知DPU线程开工，把write-reduce任务下发到DPU线程，最后等其完工通知
        CHK_RET(LocalDataCopy(tempAlgParams, workParams, strideLayout, templateResource));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            templateResource.threads[0], templateResource.dpuThread, HOST_DPU_THREAD_NOTIFY_IDX)));
        // 获取执行超时时间
        u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(templateResource.dpuThread, HOST_DPU_THREAD_NOTIFY_IDX, execTimeout)));
        templateRankSize_ = subCommRanks_[0].size();
        CHK_RET(RunNHR(
            dpuAlgParams, templateResource.channels, templateResource.dpuThread,
            templateResource.isHcommChannelDrainSupported));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            templateResource.dpuThread, templateResource.threads[0], templateResource.dpuThreadNotifyIdxOnMain)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            templateResource.threads[0], templateResource.dpuThreadNotifyIdxOnMain, execTimeout)));
    } else {
        // 老路径：保持原共享内存请求/响应流程
        if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempReduceScatterNhrDpu] failed set batch mode, tag is %s.", param.algTag);
            return HCCL_E_INTERNAL;
        }
        // 预拷贝结果作为 DPU write-reduce 的累加初值，须先于 DPU 开工完成
        CHK_RET(LocalDataCopy(tempAlgParams, workParams, strideLayout, templateResource));

        if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempReduceScatterNhrDpu] failed set eager mode, tag is %s.", param.algTag);
            return HCCL_E_INTERNAL;
        }
        if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempReduceScatterNhrDpu] failed set batch mode, tag is %s.", param.algTag);
            return HCCL_E_INTERNAL;
        }

        HcclResult result = static_cast<HcclResult>(HcommThreadSynchronize(templateResource.threads[0]));
        if (result != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempReduceScatterNhrDpu] HcommThreadSynchronize failed");
            return result;
        }

        u32 sendMsgId = 0;
        DPURunInfo dpuRunInfo;
        dpuRunInfo.execTimeout = param.opConfig.execTimeout;
        dpuRunInfo.templateName = "InsTempReduceScatterNhrDpu";
        dpuRunInfo.tempAlgParams = dpuAlgParams;
        dpuRunInfo.channels = templateResource.channels;
        dpuRunInfo.myRank = myRank_;
        dpuRunInfo.subCommRanks = subCommRanks_;
        auto dpuRunInfoSeqData = dpuRunInfo.Serialize();
        result = static_cast<HcclResult>(HcommSendRequest(
            reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr), param.algTag,
            static_cast<void*>(dpuRunInfoSeqData.data()), dpuRunInfoSeqData.size(), &sendMsgId));
        if (result != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempReduceScatterNhrDpu] HcommSendRequest failed");
            return result;
        }
        HCCL_INFO("[InsTempReduceScatterNhrDpu] HcommSendRequest run over, sendMsgId[%u]", sendMsgId);

        // 等待DPU write-reduce传输（即规约）完成
        void* recvData = nullptr;
        u32 recvMsgId = 0;
        result = static_cast<HcclResult>(
            HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId));
        if (result == HCCL_SUCCESS && recvMsgId != sendMsgId) {
            HCCL_ERROR("[InsTempReduceScatterNhrDpu] recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
            result = HCCL_E_INTERNAL;
        }
        HCCL_INFO("[InsTempReduceScatterNhrDpu] HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);
        CHK_RET(result);
    }

    CHK_RET(PostLocalCopy(tempAlgParams, workParams, templateResource));

    HCCL_INFO("[InsTempReduceScatterNhrDpu] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterNhrDpu::DPUKernelRun(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>>& subCommRanks)
{
    CHK_PRT_RET(
        subCommRanks.empty() || subCommRanks[0].empty(),
        HCCL_ERROR("[InsTempReduceScatterNhrDpu][DPUKernelRun] invalid empty subCommRanks, myRank[%u]", myRank),
        HCCL_E_PARA);
    myRank_ = myRank;
    templateRankSize_ = subCommRanks[0].size();
    subCommRanks_ = subCommRanks;
    CHK_RET(RunNHR(tempAlgParams, channels));

    return HcclResult::HCCL_SUCCESS;
}

/**
 * @brief RS NHR step 几何：deltaRank 随步递增（与 AG 的 NHR 方向相反），
 *        发对端拥有的分片、收自己拥有的分片；tx/rx 分片集合按 deltaSliceIndex 递减步进。
 */
HcclResult InsTempReduceScatterNhrDpu::GetStepInfo(uint32_t step, uint32_t nSteps, AicpuNHRStepInfo& stepInfo)
{
    (void)nSteps;
    CHK_PRT_RET(
        templateRankSize_ == 0, HCCL_ERROR("[InsTempReduceScatterNhrDpu][GetStepInfo] templateRankSize_ is zero."),
        HCCL_E_PARA);
    uint32_t rankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], rankIdx));
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;

    uint32_t deltaRank = 1 << step;
    uint32_t sendTo = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    uint32_t recvFrom = (rankIdx + deltaRank) % templateRankSize_;

    uint32_t nSlices = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
    uint32_t deltaSliceIndex = 1 << (step + 1);
    uint32_t txSliceIdx = sendTo;
    uint32_t rxSliceIdx = rankIdx;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    for (uint32_t i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG(
            "[InsTempReduceScatterNhrDpu][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempReduceScatterNhrDpu::GetRankFromMap(const uint32_t rankIdx) { return subCommRanks_[0].at(rankIdx); }

HcclResult InsTempReduceScatterNhrDpu::LocalDataCopy(
    const TemplateDataParams& rawParams, const TemplateDataParams& workParams, const bool strideLayout,
    const TemplateResource& templateResource)
{
    for (uint32_t algRankIdx = 0; algRankIdx < templateRankSize_; ++algRankIdx) {
        u64 sliceSize = workParams.allRankSliceSize.at(algRankIdx);
        u64 sliceCount = workParams.allRankProcessedDataCount.at(algRankIdx);
        u64 sliceOffset = workParams.allRankDispls.at(algRankIdx);
        u64 srcSliceOffset = strideLayout ? static_cast<u64>(algRankIdx) * rawParams.inputSliceStride : sliceOffset;

        if (sliceSize == 0) {
            HCCL_INFO(
                "[InsTempReduceScatterNhrDpu][LocalDataCopy] slice[%u] of rank[%u] is empty, skip.", algRankIdx,
                myRank_);
            continue;
        }

        for (uint64_t rpt = 0; rpt < rawParams.repeatNum; ++rpt) {
            const u64 inBaseOff = rawParams.buffInfo.inBuffBaseOff + rpt * rawParams.inputRepeatStride;
            const u64 scratchBaseoff
                = rawParams.buffInfo.hcclBuffBaseOff + rpt * templateRankSize_ * sliceSize + sliceOffset;

            DataSlice srcSlices(rawParams.buffInfo.inputPtr, inBaseOff + srcSliceOffset, sliceSize, sliceCount);
            DataSlice dstSlice(rawParams.buffInfo.hcclBuff.addr, scratchBaseoff, sliceSize, sliceCount);
            CHK_RET(LocalCopy(templateResource.threads[0], srcSlices, dstSlice));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterNhrDpu::RunNHR(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
    ThreadHandle thread, bool isDrainSupported)
{
    const uint32_t nSteps = GetNHRStepNum(templateRankSize_);

    // 几何与 rpt 无关，预计算一次供各重复块复用
    std::vector<AicpuNHRStepInfo> stepInfos(nSteps);
    for (uint32_t step = 0; step < nSteps; ++step) {
        CHK_RET(GetStepInfo(step, nSteps, stepInfos[step]));
        stepInfos[step].toRank = GetRankFromMap(stepInfos[step].toRank);
        stepInfos[step].fromRank = GetRankFromMap(stepInfos[step].fromRank);
        HCCL_DEBUG(
            "[InsTempReduceScatterNhrDpu] rank[%u] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] "
            "nSteps[%u] nSlices[%u]",
            myRank_, templateRankSize_, stepInfos[step].fromRank, stepInfos[step].toRank, step, nSteps,
            stepInfos[step].nSlices);
    }

    CHK_RET(ValidateReduceScatterNhrParams(tempAlgParams, templateRankSize_));
    for (uint32_t rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        for (uint32_t step = 0; step < nSteps; ++step) {
            CHK_RET(DpuBatchWriteReduceTransferNHR(
                stepInfos[step], channels, tempAlgParams, rpt, templateRankSize_, thread, isDrainSupported));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterNhrDpu::PostLocalCopy(
    const TemplateDataParams& rawParams, const TemplateDataParams& workParams, const TemplateResource& templateResource)
{
    if (rawParams.buffInfo.outputPtr == nullptr) {
        return HcclResult::HCCL_SUCCESS;
    }

    uint32_t myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    u64 sliceSize = workParams.allRankSliceSize.at(myAlgRank);
    u64 sliceCount = workParams.allRankProcessedDataCount.at(myAlgRank);
    u64 sliceOffset = workParams.allRankDispls.at(myAlgRank);

    if (sliceSize == 0) {
        HCCL_INFO("[InsTempReduceScatterNhrDpu][PostLocalCopy] Rank %u has no data to process.", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    for (u32 rpt = 0; rpt < rawParams.repeatNum; ++rpt) {
        // 槽位经 write-reduce 后已是全量规约结果，拷回 output
        const u64 outBaseOff = rawParams.buffInfo.outBuffBaseOff + rpt * rawParams.outputRepeatStride;
        const u64 scratchBase = rawParams.buffInfo.hcclBuffBaseOff + rpt * templateRankSize_ * sliceSize + sliceOffset;

        DataSlice srcSlice(workParams.buffInfo.hcclBuff.addr, scratchBase, sliceSize, sliceCount);
        DataSlice dstSlice(rawParams.buffInfo.outputPtr, outBaseOff, sliceSize, sliceCount);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
    }
    return HcclResult::HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
REGISTER_TEMPLATE_V2("InsTempReduceScatterNhrDpu", InsTempReduceScatterNhrDpu);
#endif
} // namespace ops_hccl
