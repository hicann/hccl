/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_nhr_dpu.h"
#include "dpu_alg_nhr_opt_wrapper.h"
#include "alg_data_trans_wrapper.h"
#include "channel.h"
#include "alg_v2_template_register.h"
#include "exec_timeout_manager.h"

namespace ops_hccl {
InsTempAllGatherNHRDPU::InsTempAllGatherNHRDPU(
    const OpParam& param, const uint32_t rankId, const std::vector<std::vector<uint32_t>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

std::vector<CostModelParam> InsTempAllGatherNHRDPU::CalcCostCoeff(CalcCostCoeffParam param)
{
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    CostModelManager::Global()->CalcNHRParams(
        param.dataRatio, param.netType, param.portNum[0], param.rankSize, A, false, true, EngineType::CPU);
    if (param.inputBuffer != param.scratchBuffer) {
        if (param.outputBuffer == BufferType::HCCL_BUFFER) {
            // 输出即 scratch（三级 AG 的 L2）：NHR 环直接跨步落位、PostLocalCopy 跳过，仅剩 pre-copy 1 份
            CostModelManager::Global()->CalcLocalCopyParams(1, EngineType::CPU, B);
        } else {
            // 顶层路径（二级 Sequence 框间级）：pre-copy 1 份 + PostLocalCopy rankSize 份
            CostModelManager::Global()->CalcLocalCopyParams(param.rankSize + 1, EngineType::CPU, B);
        }
    } else {
        B = 0.0f;
    }
    CostModelManager::Global()->CalcDpuLatencyParams(GetNHRStepNum(param.rankSize), 1, 2, param.rankSize - 1, C);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    HCCL_DEBUG("[%s] CalcCostCoeff A=%f B=%f C=%f D=%f.", __func__, A, B, C, D);
    return params;
}

HcclResult InsTempAllGatherNHRDPU::CalcRes(
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
        "[InsTempAllGatherNHRDPU][CalcRes]slaveThreadNum[%u] notifyNumOnMainThread[%u] level1Channels[%zu].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, level1Channels.size());
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRDPU::GetRes(AlgResourceRequest& resourceRequest) const
{
    // 与 CalcRes 的线程/notify 需求保持一致: 仅主线程, 无从线程与 notify 需求。
    // PrepareResForTemplate 通过本接口获取各层线程数做线程分配, 缺省实现会打 ERROR 并返回错误码。
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherNHRDPU::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void)inBufferType;
    (void)outBufferType;
    u64 scratchMultiple = templateRankSize_;
    HCCL_INFO("[InsTempAllGatherNHRDPU][CalcScratchMultiple] templateScratchMultiplier[%llu]", scratchMultiple);
    return scratchMultiple;
}

HcclResult InsTempAllGatherNHRDPU::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllGatherNHRDPU] Run Start");

    if (templateResource.threads.size() < 1) {
        HCCL_ERROR("[InsTempAllGatherNHRDPU] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(LocalDataCopy(tempAlgParams, templateResource));

    if (templateResource.isDpuThreadEnabled) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            templateResource.threads[0], templateResource.dpuThread, HOST_DPU_THREAD_NOTIFY_IDX)));
        // 获取执行超时时间
        u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(templateResource.dpuThread, HOST_DPU_THREAD_NOTIFY_IDX, execTimeout)));
        templateRankSize_ = subCommRanks_[0].size();
        CHK_RET(RunNHR(
            tempAlgParams, templateResource.channels, templateResource.dpuThread,
            templateResource.isHcommChannelDrainSupported));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            templateResource.dpuThread, templateResource.threads[0], templateResource.dpuThreadNotifyIdxOnMain)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            templateResource.threads[0], templateResource.dpuThreadNotifyIdxOnMain, execTimeout)));
    } else {
        // 转换成eager-mode，保障AICPU指令下发执行完成
        if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempAllGatherNHRDPU] failed set eager mode, tag is %s.", param.algTag);
            return HCCL_E_INTERNAL;
        }

        if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
            HCCL_ERROR("[InsTempAllGatherNHRDPU] HcommThreadSynchronize failed");
            return HCCL_E_INTERNAL;
        }

        DPURunInfo dpuRunInfo;
        dpuRunInfo.execTimeout = param.opConfig.execTimeout;
        dpuRunInfo.templateName = "InsTempAllGatherNHRDPU";
        dpuRunInfo.tempAlgParams = tempAlgParams;
        dpuRunInfo.channels = templateResource.channels;
        dpuRunInfo.myRank = myRank_;
        dpuRunInfo.subCommRanks = subCommRanks_;
        auto dpuRunInfoSeqData = dpuRunInfo.Serialize();

        u32 sendMsgId = 0;
        if (HcommSendRequest(
                reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr), param.algTag,
                static_cast<void*>(dpuRunInfoSeqData.data()), dpuRunInfoSeqData.size(), &sendMsgId)
            != 0) {
            HCCL_ERROR("[InsTempAllGatherNHRDPU] HcommSendRequest failed");
            return HCCL_E_INTERNAL;
        }
        HCCL_INFO("[InsTempAllGatherNHRDPU] HcommSendRequest run over, sendMsgId[%u]", sendMsgId);

        // 等待DPU数据传输，然后回写结果回来
        void* recvData = nullptr;
        u32 recvMsgId = 0;
        if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId)
            != 0) {
            HCCL_ERROR("[InsTempAllGatherNHRDPU] HcommWaitResponse failed");
            return HCCL_E_INTERNAL;
        }
        HCCL_INFO("[InsTempAllGatherNHRDPU] HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

        if (recvMsgId != sendMsgId) {
            HCCL_ERROR("[InsTempAllGatherNHRDPU] recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
            return HCCL_E_INTERNAL;
        }

        // 将执行模式转换回到batch
        if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempAllGatherNHRDPU] failed set batch mode, tag is %s.", param.algTag);
            return HCCL_E_INTERNAL;
        }
    }

    CHK_RET(PostLocalCopy(tempAlgParams, templateResource));

    HCCL_INFO("[InsTempAllGatherNHRDPU] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRDPU::DPUKernelRun(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>>& subCommRanks)
{
    myRank_ = myRank;
    templateRankSize_ = subCommRanks[0].size();
    subCommRanks_ = subCommRanks;
    CHK_RET(RunNHR(tempAlgParams, channels));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRDPU::GetStepInfo(uint32_t step, uint32_t nSteps, AicpuNHRStepInfo& stepInfo) const
{
    uint32_t rankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], rankIdx));
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();

    // 计算通信对象
    uint32_t deltaRank = 1 << (nSteps - 1 - step);
    uint32_t recvFrom = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    uint32_t sendTo = (rankIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    uint32_t nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    uint32_t deltaSliceIndex = 1 << (nSteps - step);
    uint32_t txSliceIdx = rankIdx;
    uint32_t rxSliceIdx = (rankIdx - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.fromRank = recvFrom;
    stepInfo.toRank = sendTo;

    for (uint32_t i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG(
            "[InsTempAllGatherNHRDPU][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempAllGatherNHRDPU::GetRankFromMap(const uint32_t rankIdx) const { return subCommRanks_[0].at(rankIdx); }

HcclResult
InsTempAllGatherNHRDPU::LocalDataCopy(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    uint32_t algRankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], algRankIdx));
    // 输出即 CCL scratch（如三级 AG 的 L2）时，own slice 直接落最终跨步槽位，与 NHR 环
    // allRankDispls（=outputSliceStride 跨步）寻址一致，对齐设备侧 InsTempAllGatherNHR 的
    // isOutputCclBuff_ 分支；否则维持紧凑落位，供 PostLocalCopy 跨 buffer 重排（二级顶层路径）。
    const bool isOutputCclBuff = tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER;

    for (uint64_t rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams.buffInfo.inBuffBaseOff + rpt * tempAlgParams.inputRepeatStride;
        u64 scOff;
        if (isOutputCclBuff) {
            scOff = tempAlgParams.outputSliceStride * algRankIdx + tempAlgParams.buffInfo.hcclBuffBaseOff
                    + rpt * tempAlgParams.outputRepeatStride;
        } else {
            const u64 scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
            const u64 scratchBaseoff = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
            scOff = tempAlgParams.sliceSize * algRankIdx + scratchBaseoff;
        }

        const u64 inOff = tempAlgParams.inputSliceStride * algRankIdx + inBaseOff;

        DataSlice srcSlices(tempAlgParams.buffInfo.inputPtr, inOff, tempAlgParams.sliceSize, tempAlgParams.count);
        DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, scOff, tempAlgParams.sliceSize, tempAlgParams.count);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlices, dstSlice));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRDPU::RunNHR(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
    ThreadHandle thread, bool isDrainSupported) const
{
    const uint32_t nSteps = GetNHRStepNum(templateRankSize_);

    for (uint32_t rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        for (uint32_t step = 0; step < nSteps; ++step) {
            AicpuNHRStepInfo stepInfo;
            CHK_RET(GetStepInfo(step, nSteps, stepInfo));

            HCCL_DEBUG(
                "[InsTempAllGatherNHRDPU] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] "
                "nSlices[%u]",
                myRank_, templateRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

            stepInfo.toRank = GetRankFromMap(stepInfo.toRank);
            stepInfo.fromRank = GetRankFromMap(stepInfo.fromRank);
            HCCL_INFO(
                "[InsTempAllGatherNHRDPU][RunNHR] converted toRank=%u fromRank=%u channelsSize=%zu", stepInfo.toRank,
                stepInfo.fromRank, channels.size());
            CHK_RET(BatchTransferNHR(
                stepInfo, channels, tempAlgParams, rpt, myRank_, templateRankSize_, thread, isDrainSupported));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempAllGatherNHRDPU::PostLocalCopy(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    // 输出即 CCL scratch：NHR 环已按 allRankDispls（outputSliceStride 跨步）直接落位，无需重排，
    // 对齐设备侧 InsTempAllGatherNHR 在 outBuffType==HCCL_BUFFER 时的跳过逻辑；
    // 同时消除原地重叠拷贝缺陷（d1>=2 且 d2>=d1+1 时升序重排互相覆盖）。
    if (tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempAllGatherNHRDPU] PostLocalCopy skip because output is scratch");
        return HcclResult::HCCL_SUCCESS;
    }
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams.buffInfo.outBuffBaseOff + rpt * tempAlgParams.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (auto rank : subCommRanks_[0]) {
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
            u64 scratchOffset = tempAlgParams.sliceSize * algRank + scratchBase;
            u64 outOffset = tempAlgParams.outputSliceStride * algRank + outBaseOff;
            DataSlice srcSlice(
                tempAlgParams.buffInfo.hcclBuff.addr, scratchOffset, tempAlgParams.sliceSize, tempAlgParams.count);
            DataSlice dstSlice(
                tempAlgParams.buffInfo.outputPtr, outOffset, tempAlgParams.sliceSize, tempAlgParams.count);
            CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempAllGatherNHRDPU", InsTempAllGatherNHRDPU);
} // namespace ops_hccl
