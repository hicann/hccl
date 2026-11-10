/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_omnipipe_nhr_dpu.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "alg_v2_template_register.h"
#include "exec_timeout_manager.h"

namespace ops_hccl {
InsTempAllGatherOmniPipeNHRDPU::InsTempAllGatherOmniPipeNHRDPU(
    const OpParam& param, const uint32_t rankId, const std::vector<std::vector<uint32_t>>& subCommRanks)
    : InsTempAllGatherNHRDPU(param, rankId, subCommRanks)
{}

InsTempAllGatherOmniPipeNHRDPU::~InsTempAllGatherOmniPipeNHRDPU() {}
HcclResult InsTempAllGatherOmniPipeNHRDPU::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllGatherOmniPipeNHRDPU] Run Start");

    if (templateResource.threads.size() < 1) {
        HCCL_ERROR("[InsTempAllGatherOmniPipeNHRDPU] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    if (templateRankSize_ == 1) {
        HCCL_INFO("[InsTempAllGatherOmniPipeNHRDPU] Rank [%d], template ranksize is 1.", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    if (templateResource.isDpuThreadEnabled) {
        // 新路径：层线程通知DPU线程开工并下发搬运任务，最后在本层线程上等其完工通知
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            templateResource.threads[0], templateResource.dpuThread, HOST_DPU_THREAD_NOTIFY_IDX)));
        u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(templateResource.dpuThread, HOST_DPU_THREAD_NOTIFY_IDX, execTimeout)));
        // 照抄DPUKernelRun逻辑：入参就是本对象成员，只需对齐可能不同的templateRankSize_
        templateRankSize_ = subCommRanks_[0].size();
        CHK_RET(RunNHR(
            tempAlgParams, templateResource.channels, templateResource.dpuThread,
            templateResource.isHcommChannelDrainSupported));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            templateResource.dpuThread, templateResource.threads[0], AICPU_HOST_DPU_OMNIPIPE_THREAD_NOTIFY_IDX)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            templateResource.threads[0], AICPU_HOST_DPU_OMNIPIPE_THREAD_NOTIFY_IDX, execTimeout)));
    } else {
        // 转换成eager-mode，保障AICPU指令下发执行完成
        if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempAllGatherOmniPipeNHRDPU] failed set eager mode, tag is %s.", param.algTag);
            return HCCL_E_INTERNAL;
        }

        if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
            HCCL_ERROR("[InsTempAllGatherOmniPipeNHRDPU] HcommThreadSynchronize failed");
            return HCCL_E_INTERNAL;
        }

        DPURunInfo dpuRunInfo;
        dpuRunInfo.execTimeout = param.opConfig.execTimeout;
        dpuRunInfo.templateName = "InsTempAllGatherOmniPipeNHRDPU";
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
            HCCL_ERROR("[InsTempAllGatherOmniPipeNHRDPU] HcommSendRequest failed");
            return HCCL_E_INTERNAL;
        }
        HCCL_INFO("[InsTempAllGatherOmniPipeNHRDPU] HcommSendRequest run over, sendMsgId[%u]", sendMsgId);

        // 等待DPU数据传输，然后回写结果回来
        void* recvData = nullptr;
        u32 recvMsgId = 0;
        if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId)
            != 0) {
            HCCL_ERROR("[InsTempAllGatherOmniPipeNHRDPU] HcommWaitResponse failed");
            return HCCL_E_INTERNAL;
        }
        HCCL_INFO("[InsTempAllGatherOmniPipeNHRDPU] HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

        if (recvMsgId != sendMsgId) {
            HCCL_ERROR(
                "[InsTempAllGatherOmniPipeNHRDPU] recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
            return HCCL_E_INTERNAL;
        }

        // 将执行模式转换回到batch
        if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempAllGatherOmniPipeNHRDPU] failed set batch mode, tag is %s.", param.algTag);
            return HCCL_E_INTERNAL;
        }
    }

    HCCL_INFO("[InsTempAllGatherOmniPipeNHRDPU] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherOmniPipeNHRDPU::RunNHR(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
    ThreadHandle thread, bool isDrainSupported) const
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const uint32_t nSteps = GetNHRStepNum(templateRankSize_);

    for (uint32_t step = 0; step < nSteps; ++step) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));

        HCCL_DEBUG(
            "[InsTempAllGatherOmniPipeNHRDPU] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] "
            "nSteps[%u] nSlices[%u]",
            myRank_, templateRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

        auto rxChannel = channels.at(GetRankFromMap(stepInfo.fromRank));
        auto txChannel = channels.at(GetRankFromMap(stepInfo.toRank));

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        void* sendCclBuffAddr = txChannel[0].remoteCclMem.addr;
        void* recvCclBuffAddr = rxChannel[0].remoteCclMem.addr;

        for (u32 i = 0; i < stepInfo.nSlices; ++i) {
            const u32 txIdx = stepInfo.txSliceIdxs[i];
            const u32 rxIdx = stepInfo.rxSliceIdxs[i];
            for (uint32_t rpt = 0; rpt < tempAlgParams.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank].size();
                 ++rpt) {
                uint64_t txScratchBase = tempAlgParams.buffInfo.inBuffBaseOff
                                         + tempAlgParams.stepSliceInfo.inputOmniPipeSliceStride[txIdx][rpt];
                uint64_t rxScratchBase = tempAlgParams.buffInfo.outBuffBaseOff
                                         + tempAlgParams.stepSliceInfo.outputOmniPipeSliceStride[rxIdx][rpt];

                const u64 txScratchOff = txScratchBase + tempAlgParams.stepSliceInfo.stepInputSliceStride[txIdx];
                const u64 rxScratchOff = rxScratchBase + tempAlgParams.stepSliceInfo.stepOutputSliceStride[rxIdx];

                txSrcSlices.emplace_back(
                    tempAlgParams.buffInfo.hcclBuff.addr, txScratchOff,
                    tempAlgParams.stepSliceInfo.stepSliceSize[txIdx][rpt],
                    tempAlgParams.stepSliceInfo.stepCount[txIdx][rpt]);
                txDstSlices.emplace_back(
                    sendCclBuffAddr, txScratchOff, tempAlgParams.stepSliceInfo.stepSliceSize[txIdx][rpt],
                    tempAlgParams.stepSliceInfo.stepCount[txIdx][rpt]);
                rxSrcSlices.emplace_back(
                    recvCclBuffAddr, rxScratchOff, tempAlgParams.stepSliceInfo.stepSliceSize[rxIdx][rpt],
                    tempAlgParams.stepSliceInfo.stepCount[rxIdx][rpt]);
                rxDstSlices.emplace_back(
                    tempAlgParams.buffInfo.hcclBuff.addr, rxScratchOff,
                    tempAlgParams.stepSliceInfo.stepSliceSize[rxIdx][rpt],
                    tempAlgParams.stepSliceInfo.stepCount[rxIdx][rpt]);
            }
        }
        // write模式使用tx,rx地址不生效，仅使用对端link做Post/Wait
        // read 模式使用rx, tx地址不生效，仅使用对端link做Post/Wait
        TxRxChannels sendRecvChannels(txChannel[0], rxChannel[0]);
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        CHK_PRT_RET(
            DpuSendRecvWrite(sendRecvInfo, thread, isDrainSupported),
            HCCL_ERROR("[InsTempAllGatherOmniPipeNHRDPU] SendRecvWrite failed (step=%u)", step),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherOmniPipeNHRDPU::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_RET(InsTempAllGatherNHRDPU::CalcRes(comm, param, topoInfo, resourceRequest));
    resourceRequest.notifyNumOnMainThread = AICPU_HOST_DPU_OMNIPIPE_THREAD_NOTIFY_NUM;
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherOmniPipeNHRDPU::GetRes(AlgResourceRequest& resourceRequest) const
{
    // NHR算法只需要一条主流
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {}; // 没有从流
    resourceRequest.notifyNumOnMainThread = AICPU_HOST_DPU_OMNIPIPE_THREAD_NOTIFY_NUM;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherOmniPipeNHRDPU::GetThreadNum() const { return 1; }

REGISTER_TEMPLATE_V2("InsTempAllGatherOmniPipeNHRDPU", InsTempAllGatherOmniPipeNHRDPU);
} // namespace ops_hccl
