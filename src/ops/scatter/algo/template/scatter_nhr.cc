/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "alg_template_register.h"
#include "scatter_nhr.h"

namespace ops_hccl {
ScatterNHR::ScatterNHR()
    : NHRBase(), interRank_(0), interRankSize_(0)
{
}

ScatterNHR::~ScatterNHR()
{
}

// scatter的入口函数
HcclResult ScatterNHR::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels)
{
    CHK_PTR_NULL(inputMem_.addr);
    CHK_PTR_NULL(outputMem_.addr);

    interRank_ = rank;
    interRankSize_ = rankSize;

    unitSize_ = DataUnitSize(dataType_);
    CHK_PRT_RET(unitSize_ == 0, HCCL_ERROR("[ScatterNHR][RunAsync] rank[%u] unit data size is zero", rank),
        HCCL_E_INTERNAL);

    // ranksize为1时，只有当input!=ouput时候进行拷贝
    if (interRankSize_ == 1) {
        if (inputMem_.addr != outputMem_.addr) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, outputMem_.addr, inputMem_.addr, inputMem_.size)));
        }
        return HCCL_SUCCESS;
    }

    // 带入vecotr为空，计算每个rank的结果偏移和大小
    if (slices_.size() == 0) {
        PrepareSlicesData(unitSize_, count_, interRankSize_);
    }

    CHK_PRT_RET(channels.size() < rankSize,
        HCCL_ERROR("[ScatterNHR][RunAsync] rank[%u] link size[%llu] is less than rank size", rank, channels.size()),
        HCCL_E_INTERNAL);

    if (sliceMap_.size() != rankSize) {
        GetRankMapping(rankSize, true); // 没有初始化过，说明不是由allreduce或者bcast调入，需要保序
    }

    // 需要判断input不等于outputmem，scatter输入只有一个input时不用拷贝
    if (inputMem_.addr != outputMem_.addr) {
        u32 targetIdx = sliceMap_[interRank_];
        void* src = static_cast<void *>(static_cast<u8 *>(inputMem_.addr) + slices_[targetIdx].offset);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, outputMem_.addr, src, slices_[targetIdx].size)));
    }

    // 运行scatter, NHR 算法
    CHK_RET(RunScatterNHR(channels));
    return HCCL_SUCCESS;
}

HcclResult ScatterNHR::SdmaRx(ChannelInfo &channelLeft, ChannelInfo &channelRight, InterServerAlgoStep &stepInfo)
{
    if (channelRight.isValid) {
        CHK_RET(static_cast<HcclResult>(HcommNotifyRecordOnThread(thread_, channelRight.handle, NOTIFY_IDX_ACK)));
    }
    if (channelLeft.isValid) {
        CHK_RET(static_cast<HcclResult>(HcommNotifyWaitOnThread(thread_, channelLeft.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));

        void* srcMemPtr = channelLeft.remoteOutput.addr;
        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            const Slice &rxSlice = slices_[stepInfo.rxSliceIdxs[i]];
            void* dst = static_cast<void *>(static_cast<u8 *>(outputMem_.addr) + rxSlice.offset);
            void* src = static_cast<void *>(static_cast<s8 *>(srcMemPtr) + baseOffset_ + rxSlice.offset);
            HCCL_DEBUG("[ScatterNHR][HcommReadOnThread] src[%p] dst[%p] size[%llu]", src, dst, rxSlice.size);
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread_, channelLeft.handle, dst, src, rxSlice.size)));
        }
        CHK_RET(static_cast<HcclResult>(HcommNotifyRecordOnThread(thread_, channelLeft.handle, NOTIFY_IDX_DATA_SIGNAL)));
    }
    if (channelRight.isValid) {
        CHK_RET(static_cast<HcclResult>(HcommNotifyWaitOnThread(thread_, channelRight.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterNHR::RdmaTxRx(ChannelInfo &channelLeft, ChannelInfo &channelRight, InterServerAlgoStep &stepInfo)
{
    HcclResult ret = HCCL_SUCCESS;

    if (channelLeft.isValid) {
        CHK_RET(static_cast<HcclResult>(HcommNotifyRecordOnThread(thread_, channelLeft.handle, NOTIFY_IDX_ACK)));
    }

    if (channelRight.isValid) {
        CHK_RET(static_cast<HcclResult>(HcommNotifyWaitOnThread(thread_, channelRight.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));

        void* dstMemPtr = channelRight.remoteOutput.addr;
        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            const Slice& txSlice = slices_[stepInfo.txSliceIdxs[i]];
            void* dst = static_cast<void *>(static_cast<u8 *>(dstMemPtr) + txSlice.offset + baseOffset_);
            void* src = static_cast<void *>(reinterpret_cast<u8 *>(outputMem_.addr) + txSlice.offset);
            HCCL_DEBUG("[ScatterNHR][HcommWriteOnThread] src[%p] dst[%p] size[%llu]", src, dst, txSlice.size);
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread_, channelRight.handle, dst, src, txSlice.size)));
        }
        CHK_RET(static_cast<HcclResult>(HcommNotifyRecordOnThread(thread_, channelRight.handle, NOTIFY_IDX_DATA_SIGNAL))); // rdma数据发完
        CHK_RET(static_cast<HcclResult>(HcommNotifyWaitOnThread(thread_, channelRight.handle, NOTIFY_IDX_FIN_ACK, CUSTOM_TIMEOUT)));
    }

    if (channelLeft.isValid) {
        CHK_RET(static_cast<HcclResult>(HcommNotifyWaitOnThread(thread_, channelLeft.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));  // rdma数据收完
        CHK_RET(static_cast<HcclResult>(HcommNotifyRecordOnThread(thread_, channelLeft.handle, NOTIFY_IDX_FIN_ACK)));
    }

    if (barrierSwitchOn_) {
        CHK_RET(ExecuteBarrier(channelLeft, channelRight));
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterNHR::RunScatterNHR(std::vector<ChannelInfo> &channels)
{
    // 计算通信步数
    u32 nSteps = GetStepNumInterServer(interRankSize_);

    // 逐步编排任务
    for (u32 step = 0; step < nSteps; step++) {
        InterServerAlgoStep stepInfo;
        GetStepInfo(step, nSteps, interRank_, interRankSize_, stepInfo);

        HCCL_DEBUG("[ScatterNHR][RunScatterNHR] rank[%u] recvFrom[%u] sendTo[%u] step[%u]",
            interRank_, stepInfo.fromRank, stepInfo.toRank, step);

        ChannelInfo channelLeft;
        ChannelInfo channelRight;
        if (stepInfo.txSliceIdxs.size() > 0) {
            channelRight = channels[stepInfo.toRank];
        }
        if (stepInfo.rxSliceIdxs.size() > 0) {
            channelLeft = channels[stepInfo.fromRank];
        }

        if ((channelRight.isValid && channelRight.protocol != COMM_PROTOCOL_ROCE) ||
            (channelLeft.isValid && channelLeft.protocol != COMM_PROTOCOL_ROCE)) {
            CHK_RET(SdmaRx(channelLeft, channelRight, stepInfo));
        } else {
            CHK_RET(RdmaTxRx(channelLeft, channelRight, stepInfo));
        }
    }
    return HCCL_SUCCESS;
}

void ScatterNHR::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const
{
    slices_.resize(rankSize);
    u64 sliceSize = (totalCount / rankSize) * unitSize;

    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
    }
    return;
}

// NHR每步的算法描述原理函数
HcclResult ScatterNHR::GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, InterServerAlgoStep &stepInfo)
{
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 0;
    stepInfo.toRank = rankSize;
    stepInfo.fromRank = rankSize;
    stepInfo.step = step;
    stepInfo.myRank = rank;

    u32 deltaRoot = (root_ + rankSize - rank) % rankSize;
    u32 deltaRankPair = 1 << step;

    // 数据份数和数据编号增量
    u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);

    // 判断是否是2的幂
    u32 nRanks = 0; // 本步需要进行收/发的rank数
    bool isPerfect = (rankSize & (rankSize - 1)) == 0;
    if (!isPerfect && step == nSteps - 1) {
        nRanks = rankSize - deltaRankPair;
    } else {
        nRanks = deltaRankPair;
    }

    if (deltaRoot < nRanks) { // 需要发
        u32 sendTo = (rank + rankSize - deltaRankPair) % rankSize;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetTxSliceIdx = sliceMap_[txSliceIdx];
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx);
            txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }

        stepInfo.toRank = sendTo;
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) { // 需要收
        u32 recvFrom = (rank + deltaRankPair) % rankSize;
        u32 rxSliceIdx = rank;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetRxSliceIdx = sliceMap_[rxSliceIdx];
            stepInfo.rxSliceIdxs.push_back(targetRxSliceIdx);
            rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }

        stepInfo.fromRank = recvFrom;
        stepInfo.nSlices = nSlices;
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterNHR::ExecuteBarrier(ChannelInfo &channelLeft, ChannelInfo &channelRight)
{
    if (channelLeft.isValid) {
        CHK_RET(static_cast<HcclResult>(HcommNotifyRecordOnThread(thread_, channelLeft.handle, NOTIFY_IDX_ACK)));
    }
    if (channelRight.isValid) {
        CHK_RET(static_cast<HcclResult>(HcommNotifyWaitOnThread(thread_, channelRight.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommNotifyRecordOnThread(thread_, channelRight.handle, NOTIFY_IDX_DATA_SIGNAL)));
    }
    if (channelLeft.isValid) {
        CHK_RET(static_cast<HcclResult>(HcommNotifyWaitOnThread(thread_, channelLeft.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommNotifyRecordOnThread(thread_, channelLeft.handle, NOTIFY_IDX_FIN_ACK)));
    }
    if (channelRight.isValid) {
        CHK_RET(static_cast<HcclResult>(HcommNotifyWaitOnThread(thread_, channelRight.handle, NOTIFY_IDX_FIN_ACK, CUSTOM_TIMEOUT)));
    }

    return HCCL_SUCCESS;
}


REGISTER_TEMPLATE(TemplateType::TEMPLATE_SCATTER_NHR, ScatterNHR);
}
