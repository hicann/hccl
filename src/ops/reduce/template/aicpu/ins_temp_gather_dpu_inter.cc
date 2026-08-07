/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_gather_dpu_inter.h"
#include "alg_data_trans_wrapper.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "channel.h"
#include "alg_v2_template_register.h"

namespace ops_hccl {
InsTempGatherDpuInter::InsTempGatherDpuInter(
    const OpParam& param, const uint32_t rankId, const std::vector<std::vector<uint32_t>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

HcclResult InsTempGatherDpuInter::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO(
        "[InsTempGatherDpuInter][CalcRes]slaveThreadNum[%u] notifyNumOnMainThread[%u] level1Channels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, level1Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempGatherDpuInter::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void)inBufferType;
    (void)outBufferType;
    u64 scratchMultiple = templateRankSize_;
    HCCL_INFO("[InsTempGatherDpuInter][CalcScratchMultiple] templateScratchMultiplier[%llu]", scratchMultiple);
    return scratchMultiple;
}

HcclResult InsTempGatherDpuInter::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempGatherDpuInter] Run Start");

    if (templateResource.threads.size() < 1) {
        HCCL_ERROR("[InsTempGatherDpuInter] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(LocalDataCopy(tempAlgParams, templateResource));

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempGatherDpuInter] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
        HCCL_ERROR("[InsTempGatherDpuInter] HcommThreadSynchronize failed");
        return HCCL_E_INTERNAL;
    }

    DPURunInfo dpuRunInfo;
    dpuRunInfo.templateName = "InsTempGatherDpuInter";
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
        HCCL_ERROR("[InsTempGatherDpuInter] HcommSendRequest failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempGatherDpuInter] HcommSendRequest run over, sendMsgId[%u]", sendMsgId);

    // 等待DPU数据传输，然后回写结果回来
    void* recvData = nullptr;
    u32 recvMsgId = 0;
    if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("[InsTempGatherDpuInter] HcommWaitResponse failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempGatherDpuInter] HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

    if (recvMsgId != sendMsgId) {
        HCCL_ERROR("[InsTempGatherDpuInter] recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
        return HCCL_E_INTERNAL;
    }

    // 将执行模式转换回到batch
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempGatherDpuInter] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(PostLocalCopy(tempAlgParams, templateResource));

    HCCL_INFO("[InsTempGatherDpuInter] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempGatherDpuInter::DPUKernelRun(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>>& subCommRanks)
{
    myRank_ = myRank;
    templateRankSize_ = subCommRanks[0].size();
    subCommRanks_ = subCommRanks;
    CHK_RET(RunNHR(tempAlgParams, channels));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempGatherDpuInter::GetStepInfo(uint32_t step, uint32_t nSteps, AicpuNHRStepInfo& stepInfo)
{
    uint32_t rankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], rankIdx));
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.myRank = rankIdx;
    stepInfo.step = step;

    // 计算通信对象
    uint32_t deltaRank = 1 << (nSteps - 1 - step);
    uint32_t recvFrom = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    uint32_t sendTo = (rankIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    uint32_t nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    uint32_t deltaSliceIndex = 1 << (nSteps - step);
    uint32_t txSliceIdx = rankIdx;
    uint32_t rxSliceIdx = (rankIdx - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;
    stepInfo.nSlices = nSlices;

    for (uint32_t i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG(
            "[InsTempGatherDpuInter][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempGatherDpuInter::GetRankFromMap(const uint32_t rankIdx) { return subCommRanks_[0].at(rankIdx); }

HcclResult
InsTempGatherDpuInter::LocalDataCopy(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    uint32_t algRankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], algRankIdx));

    u64 sliceSize = tempAlgParams.allRankSliceSize.at(algRankIdx);
    u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(algRankIdx);
    u64 sliceOffset = tempAlgParams.allRankDispls.at(algRankIdx);

    // 数据量为0的数据片无需Copy
    if (sliceSize == 0) {
        HCCL_INFO("[InsTempGatherDpuInter][LocalDataCopy] Rank %d has no data to process.", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    for (uint64_t rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams.buffInfo.inBuffBaseOff + rpt * tempAlgParams.inputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
        const u64 scratchBaseoff = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        const u64 inOff = inBaseOff;
        const u64 scOff = scratchBaseoff + sliceOffset;

        DataSlice srcSlices(tempAlgParams.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, scOff, sliceSize, sliceCount);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlices, dstSlice));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempGatherDpuInter::RunNHR(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels)
{
#ifndef AICPU_COMPILE
    const uint32_t nSteps = GetNHRStepNum(templateRankSize_);

    for (uint32_t rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const uint64_t scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
        const uint64_t scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (uint32_t step = 0; step < nSteps; ++step) {
            AicpuNHRStepInfo stepInfo;
            CHK_RET(GetStepInfo(step, nSteps, stepInfo));

            HCCL_DEBUG(
                "[InsTempGatherDpuInter] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] nSlices[%u]",
                myRank_, templateRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

            auto rxChannel = channels.at(GetRankFromMap(stepInfo.fromRank));
            auto txChannel = channels.at(GetRankFromMap(stepInfo.toRank));

            void* recvCclBuffAddr = rxChannel[0].remoteCclMem.addr;
            void* sendCclBuffAddr = txChannel[0].remoteCclMem.addr;

            for (u32 i = 0; i < stepInfo.nSlices; ++i) {
                const u32 txIdx = stepInfo.txSliceIdxs[i];
                const u32 rxIdx = stepInfo.rxSliceIdxs[i];

                u64 sendSize = tempAlgParams.allRankSliceSize.at(txIdx);
                u64 sendOffset = scratchBase + tempAlgParams.allRankDispls.at(txIdx);
                u64 sendCount = tempAlgParams.allRankProcessedDataCount.at(txIdx);

                u64 recvSize = tempAlgParams.allRankSliceSize.at(rxIdx);
                u64 recvOffset = scratchBase + tempAlgParams.allRankDispls.at(rxIdx);
                u64 recvCount = tempAlgParams.allRankProcessedDataCount.at(rxIdx);

                // 无需发送和接收数据
                if (sendSize == 0 && recvSize == 0) {
                    continue;
                }
                std::vector<DataSlice> txSrcSlices{
                    DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, sendOffset, sendSize, sendCount)};
                std::vector<DataSlice> txDstSlices{DataSlice(sendCclBuffAddr, sendOffset, sendSize, sendCount)};
                std::vector<DataSlice> rxSrcSlices{DataSlice(recvCclBuffAddr, recvOffset, recvSize, recvCount)};
                std::vector<DataSlice> rxDstSlices{
                    DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, recvOffset, recvSize, recvCount)};

                if (sendSize > 0 && recvSize > 0) {
                    if (txChannel[0].remoteRank == rxChannel[0].remoteRank) {
                        TxRxChannels sendRecvChannels(txChannel[0], rxChannel[0]);
                        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
                        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
                        CHK_PRT_RET(
                            SendRecvWrite(sendRecvInfo),
                            HCCL_ERROR("[InsTempGatherDpuInter] SendRecvWrite failed (step=%u, rpt=%u)", step, rpt),
                            HcclResult::HCCL_E_INTERNAL);
                        HCCL_INFO(
                            "[InsTempGatherDpuInter][RunNHR]SendRecvWrite on rank %u src offset %u, dst offset %u, "
                            "size %u",
                            myRank_, sendOffset, sendOffset, sendSize);
                        HCCL_INFO(
                            "[InsTempGatherDpuInter][RunNHR]SendRecvWrite src addr %p, dst addr %p",
                            tempAlgParams.buffInfo.hcclBuff.addr, sendCclBuffAddr);
                    } else if (txChannel[0].remoteRank < rxChannel[0].remoteRank) {
                        SlicesList sendSliceList(txSrcSlices, txDstSlices);
                        DataInfo sendInfo(txChannel[0], sendSliceList);
                        CHK_PRT_RET(
                            SendWrite(sendInfo),
                            HCCL_ERROR("[InsTempGatherDpuInter][RunNHR] Send failed (step=%u, rpt=%u)", step, rpt),
                            HcclResult::HCCL_E_INTERNAL);
                        HCCL_INFO(
                            "[InsTempGatherDpuInter][RunNHR]SendWrite src addr %p, dst addr %p",
                            tempAlgParams.buffInfo.hcclBuff.addr, sendCclBuffAddr);

                        SlicesList recvSliceList(rxSrcSlices, rxDstSlices);
                        DataInfo recvInfo(rxChannel[0], recvSliceList);
                        CHK_PRT_RET(
                            RecvWrite(recvInfo),
                            HCCL_ERROR("[InsTempGatherDpuInter][RunNHR] Recv failed (step=%u, rpt=%u)", step, rpt),
                            HcclResult::HCCL_E_INTERNAL);
                    } else {
                        SlicesList recvSliceList(rxSrcSlices, rxDstSlices);
                        DataInfo recvInfo(rxChannel[0], recvSliceList);
                        CHK_PRT_RET(
                            RecvWrite(recvInfo),
                            HCCL_ERROR("[InsTempGatherDpuInter][RunNHR] Recv failed (step=%u, rpt=%u)", step, rpt),
                            HcclResult::HCCL_E_INTERNAL);

                        SlicesList sendSliceList(txSrcSlices, txDstSlices);
                        DataInfo sendInfo(txChannel[0], sendSliceList);
                        CHK_PRT_RET(
                            SendWrite(sendInfo),
                            HCCL_ERROR("[InsTempGatherDpuInter][RunNHR] Send failed (step=%u, rpt=%u)", step, rpt),
                            HcclResult::HCCL_E_INTERNAL);
                        HCCL_INFO(
                            "[InsTempGatherDpuInter][RunNHR]SendWrite src addr %p, dst addr %p",
                            tempAlgParams.buffInfo.hcclBuff.addr, sendCclBuffAddr);
                    }
                } else if (sendSize > 0) {
                    SlicesList sendSliceList(txSrcSlices, txDstSlices);
                    DataInfo sendInfo(txChannel[0], sendSliceList);
                    CHK_PRT_RET(
                        SendWrite(sendInfo),
                        HCCL_ERROR("[InsTempGatherDpuInter][RunNHR] Send failed (step=%u, rpt=%u)", step, rpt),
                        HcclResult::HCCL_E_INTERNAL);
                    HCCL_INFO(
                        "[InsTempGatherDpuInter][RunNHR]SendWrite src addr %p, dst addr %p",
                        tempAlgParams.buffInfo.hcclBuff.addr, sendCclBuffAddr);
                } else if (recvSize > 0) {
                    SlicesList recvSliceList(rxSrcSlices, rxDstSlices);
                    DataInfo recvInfo(rxChannel[0], recvSliceList);
                    CHK_PRT_RET(
                        RecvWrite(recvInfo),
                        HCCL_ERROR("[InsTempGatherDpuInter][RunNHR] Recv failed (step=%u, rpt=%u)", step, rpt),
                        HcclResult::HCCL_E_INTERNAL);
                }
            }
        }
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempGatherDpuInter::PostLocalCopy(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    if (tempAlgParams.buffInfo.outputPtr == nullptr) {
        return HcclResult::HCCL_SUCCESS;
    }

    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams.buffInfo.outBuffBaseOff + rpt * tempAlgParams.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (auto rank : subCommRanks_[0]) {
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
            u64 scratchOffset = scratchBase + tempAlgParams.allRankDispls.at(algRank);
            u64 outOffset = outBaseOff + tempAlgParams.allRankDispls.at(algRank);

            u64 sliceSize = tempAlgParams.allRankSliceSize.at(algRank);
            u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(algRank);

            // 处理数据量为0情况
            if (sliceSize == 0) {
                continue;
            }

            DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams.buffInfo.outputPtr, outOffset, sliceSize, sliceCount);
            CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
            HCCL_INFO(
                "[InsTempGatherDpuInter][PostLocalCopy]LocalCopy on position %u src offset %u, dst offset %u, size %u",
                algRank, scratchOffset, outOffset, sliceSize);
            HCCL_INFO(
                "[InsTempGatherDpuInter][PostLocalCopy]LocalCopy src addr %p, dst addr %p",
                tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.outputPtr);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempGatherDpuInter", InsTempGatherDpuInter);
} // namespace ops_hccl
