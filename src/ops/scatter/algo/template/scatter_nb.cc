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
#include "scatter_nb.h"

namespace ops_hccl {
ScatterNB::ScatterNB()
    : AlgTemplateBase(), interRank_(0), interRankSize_(0)
{
}

ScatterNB::~ScatterNB()
{
}

HcclResult ScatterNB::RunScatterNB(std::vector<ChannelInfo> &channels)
{
    HcclResult ret = HCCL_SUCCESS;
    // 需要判断input不等于outputmem，scatter 输入只有一个input时不用拷贝
    if (inputMem_.addr != outputMem_.addr) {
        CHK_RET(HcommLocalCopyOnThread(thread_, outputMem_.addr, inputMem_.addr, inputMem_.size));
    }

    // 计算通信步数：ceiling(log2(rankSize))
    u32 nSteps = CalcCeilLog2(interRankSize_);

    // 逐步编排任务
    u32 deltaRoot = (interRank_ + interRankSize_ - root_) % interRankSize_;
    for (u32 step = 0; step < nSteps; step++) {
        if (deltaRoot < u32(1<<step)) {
            if (step != nSteps - 1 || deltaRoot < (interRankSize_ - (1<<step))) {
                CHK_RET(RunScatterTx(step, channels));
            }
        } else if (deltaRoot < u32(1<<(step + 1)) && interRank_ != root_) {
            CHK_RET(RunScatterRx(step, channels));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterNB::RunScatterTx(const u32 step, std::vector<ChannelInfo> &channels)
{
    HcclResult ret = HCCL_SUCCESS;

    // 计算通信对象
    u32 deltaRank = 1 << step;
    u32 sendTo = (interRank_ + deltaRank) % interRankSize_;
    // 数据份数和数据编号增量
    u32 nSlices = (interRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);
    u32 sliceIdx = (interRank_ + (1<<step)) % interRankSize_;
    ChannelInfo &channelRight = channels[sendTo];

    std::vector<Slice> txSlices;
    for (u32 i = 0; i < nSlices; i++) {
        if (slicesFlag_[sliceIdx] == false) {
            continue;
        }
        txSlices.push_back(slices_[sliceIdx]);
        sliceIdx = (sliceIdx + deltaSliceIndex) % interRankSize_;
    }

    CHK_RET(HcommNotifyWaitOnThread(thread_, channelRight.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    if (channelRight.protocol == COMM_PROTOCOL_ROCE) {
        ret = RdmaTx(channelRight, txSlices);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][Scatter]rank[%u] step[%u] RightLink tx slices count [%u] Failed",
            interRank_, step, nSlices), ret);
    }

    CHK_RET(HcommNotifyRecordOnThread(thread_, channelRight.handle, NOTIFY_IDX_DATA_SIGNAL));

    // 为了避免在大数据量场景下触发网卡轮询机制，这里添加一组Data Notify，确保对端数据接收完成才进行下一次通信任务
    CHK_RET(HcommNotifyWaitOnThread(thread_, channelRight.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult ScatterNB::RunScatterRx(const u32 step, std::vector<ChannelInfo> &channels)
{
    HcclResult ret = HCCL_SUCCESS;

    // 计算通信对象
    u32 deltaRank = 1 << step;
    u32 recvFrom = (interRank_ + interRankSize_ - deltaRank) % interRankSize_;
    // 数据份数和数据编号增量
    u32 nSlices = (interRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);
    u32 sliceIdx = interRank_;
    ChannelInfo &channelLeft = channels[recvFrom];

    std::vector<Slice> rxSlices;
    for (u32 i = 0; i < nSlices; i++) {
        rxSlices.push_back(slices_[sliceIdx]);
        sliceIdx = (sliceIdx + deltaSliceIndex) % interRankSize_;
        slicesFlag_[sliceIdx] = true;
    }

    CHK_RET(HcommNotifyRecordOnThread(thread_, channelLeft.handle, NOTIFY_IDX_ACK));

    CHK_RET(HcommNotifyWaitOnThread(thread_, channelLeft.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    if (channelLeft.protocol != COMM_PROTOCOL_ROCE) {
        ret = SdmaRx(channelLeft, rxSlices);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][Scatter]rank[%u] step[%u] Right Link rx slices count [%u] "\
                "Failed", interRank_, step, nSlices), ret);
    }

    // 为了避免在大数据量场景下触发网卡轮询机制，这里添加一组Data Notify，确保对端数据接收完成才进行下一次通信任务
    CHK_RET(HcommNotifyRecordOnThread(thread_, channelLeft.handle, NOTIFY_IDX_DATA_SIGNAL));
    return HCCL_SUCCESS;
}

void ScatterNB::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const
{
    slices_.resize(rankSize);
    u64 sliceSize = (totalCount / rankSize) * unitSize;

    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
    }
}

// scatter的入口函数
HcclResult ScatterNB::RunAsync(const u32 rank, const u32 rankSize,
    std::vector<ChannelInfo> &channels)
{
    if (!outputMem_.addr || !inputMem_.addr) {
        HCCL_ERROR("[ScatterNB][RunAsync]run_async inputmem or outputmem is null");
        return HCCL_E_PTR;
    }

    interRank_ = rank;
    interRankSize_ = rankSize;
    if (interRank_ == root_) {
        slicesFlag_.resize(interRankSize_, true);
    } else {
        slicesFlag_.resize(interRankSize_, false);
    }

    // ranksize为1时，只有当input!=ouput 时候进行拷贝
    if (interRankSize_ == 1) {
        if (inputMem_.addr != outputMem_.addr) {
            CHK_RET(HcommLocalCopyOnThread(thread_, outputMem_.addr, inputMem_.addr, inputMem_.size));
        }
        return HCCL_SUCCESS;
    }

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(unitSize == 0, HCCL_ERROR("[ScatterNB][RunAsync]rank[%u] unit data size is zero", rank),
        HCCL_E_INTERNAL);

    // 带入vecotr为空，计算每个rank的结果偏移和大小
    if (slices_.size() == 0) {
        PrepareSlicesData(unitSize, count_, interRankSize_);
    }

    CHK_PRT_RET(channels.size() < rankSize,
        HCCL_ERROR("[ScatterNB][RunAsync]rank[%u] channel size[%llu] is less than rank size", rank, channels.size()),
        HCCL_E_INTERNAL);

    CHK_RET(RunScatterNB(channels));

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(channels[(interRank_ + interRankSize_ - 1) % interRankSize_],
            channels[(interRank_ + 1) % interRankSize_]));
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterNB::RdmaTx(ChannelInfo &channel, const std::vector<Slice> &txSlices)
{
    void *dstMemPtr = channel.remoteOutput.addr;
    for (const Slice& txSlice : txSlices) {
        void* dst = static_cast<void *>(static_cast<u8 *>(dstMemPtr) + txSlice.offset + baseOffset_);
        void* src = static_cast<void *>(reinterpret_cast<u8 *>(outputMem_.addr) + txSlice.offset);
        HCCL_DEBUG("[ScatterNB][HcommWriteOnThread] src[%p] dst[%p] size[%llu]", src, dst, txSlice.size);
        CHK_RET(HcommWriteOnThread(thread_, channel.handle, dst, src, txSlice.size));
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterNB::SdmaRx(ChannelInfo &channel, const std::vector<Slice> &rxSlices)
{
    void *srcMemPtr = channel.remoteOutput.addr;
    for (const Slice& rxSlice : rxSlices) {
        void* dst = static_cast<void *>(static_cast<u8 *>(outputMem_.addr) + rxSlice.offset);
        void* src = static_cast<void *>(static_cast<s8 *>(srcMemPtr) + rxSlice.offset + baseOffset_);
        HCCL_DEBUG("[ScatterNB][HcommReadOnThread] src[%p] dst[%p] size[%llu]", src, dst, rxSlice.size);
        CHK_RET(HcommReadOnThread(thread_, channel.handle, dst, src, rxSlice.size));
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_SCATTER_NB, ScatterNB);
}
