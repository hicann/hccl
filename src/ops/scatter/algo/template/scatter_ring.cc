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
#include "scatter_ring.h"

namespace ops_hccl {
ScatterRing::ScatterRing()
    : AlgTemplateBase(), interRank_(0), interRankSize_(0)
{
}

ScatterRing::~ScatterRing()
{
}

HcclResult ScatterRing::RunScatterOnRootRank()
{
    // rank存放scatter 结果的偏移
    u64 scatterOffset = slices_[interRank_].offset;
    u64 scatterResult = slices_[interRank_].size;

    HcclResult ret = HCCL_SUCCESS;
    // 需要判断input不等于outputmem，scatter 输入只有一个input时不用拷贝
    if (inputMem_.addr != outputMem_.addr) {
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, outputMem_.addr, inputMem_.addr, inputMem_.size)));
    }

    // 数据向下一个rank发送，依次发送后继所有rank的数据
    for (u32 i = 1; i < interRankSize_; i++) {
        u32 preRank = (interRank_ - i + interRankSize_) % interRankSize_;
        scatterOffset = slices_[preRank].offset;
        scatterResult = slices_[preRank].size;

        // 等待后一节点同步信号，进行下一轮操作
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread_, channelRight_.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));

        // 向root rank的后一rank发送
        HCCL_DEBUG(" root rank[%u] sendto dstrank[%u] from srcmem offset[%llu] size[%llu]",
            interRank_, preRank, scatterOffset, scatterResult);

        if (channelRight_.protocol == COMM_PROTOCOL_ROCE) {
            void* dst = static_cast<void *>(static_cast<u8 *>(channelRight_.remoteOutput.addr) + scatterOffset + baseOffset_);
            void* src = static_cast<void *>(static_cast<u8 *>(inputMem_.addr) + scatterOffset);
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread_, channelRight_.handle, dst, src, scatterResult)));
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread_, channelRight_.handle, NOTIFY_IDX_DATA_SIGNAL)));
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterRing::RunScatterOnEndRank()
{
    u64 scatterOffset = slices_[interRank_].offset;
    u64 scatterResult = slices_[interRank_].size;
    // 给前一节点发送同步，以便前一rank进行下一轮的操作
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread_, channelLeft_.handle, NOTIFY_IDX_ACK)));

    HCCL_DEBUG("last rank[%u] rx data ouputoffset[%llu] size[%llu]", interRank_, scatterOffset, scatterResult);

    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread_, channelLeft_.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    if (channelLeft_.protocol != COMM_PROTOCOL_ROCE) {
        void* src = static_cast<void *>(static_cast<s8 *>(channelLeft_.remoteOutput.addr) + scatterOffset + baseOffset_);
        void* dst = static_cast<void *>(static_cast<s8 *>(outputMem_.addr) + scatterOffset);
        HCCL_DEBUG("[ScatterRing][HcommReadOnThread] src[%p] dst[%p] size[%llu]", src, dst, scatterResult);
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread_, channelLeft_.handle, dst, src, scatterResult)));
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterRing::RunScatterOnMidRank()
{
    // 与root的rank号之差 + 接收的轮数 = rank_size,  每个rank 接收的次数为 root_+ranksize-rank%interRankSize_
    u32 round = (root_ + interRankSize_ - interRank_) % interRankSize_;
    HCCL_DEBUG("rank:[%u] will receive %u rounds data", interRank_, round);

    HcclResult ret = HCCL_SUCCESS;
    // 需要接收的和发送的轮数，包含接收自己的数据
    for (u32 i = 1; i <= round; i++) {
        u32 dataRank = (interRank_ + round - i) % interRankSize_; // 收到的数据应当是哪个rank的
        u64 scatterOffset = slices_[dataRank].offset;
        u64 scatterResult = slices_[dataRank].size;

        u32 lastDataRank = (interRank_ + round - i + 1) % interRankSize_; // 加1得到发送的数据应当是哪个rank的
        u64 scatterLastOffset = slices_[lastDataRank].offset;
        u64 scatterLastResult = slices_[lastDataRank].size;

        if (i != 1) {
            // 给前一节点发送同步，以便前一rank进行下一轮的操作
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread_, channelLeft_.handle, NOTIFY_IDX_ACK)));
            // 从后一rank接收同步信号
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread_, channelRight_.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            // 向后一rank发送数据
            HCCL_DEBUG("rank[%u] round[%u] tx async offset[%llu] size[%llu]", interRank_, \
                i, scatterLastOffset, scatterLastResult);

            if (channelRight_.protocol == COMM_PROTOCOL_ROCE) {
                void* dst = static_cast<void *>(static_cast<u8 *>(channelRight_.remoteOutput.addr) + scatterLastOffset + baseOffset_);
                void* src = static_cast<void *>(static_cast<u8 *>(outputMem_.addr) + scatterLastOffset);
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread_, channelRight_.handle, dst, src, scatterLastResult)));
            }
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread_, channelRight_.handle, NOTIFY_IDX_DATA_SIGNAL)));
        } else {
            // 给前一节点发送同步，以便前一rank进行下一轮的操作
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread_, channelLeft_.handle, NOTIFY_IDX_ACK)));
        }
        HCCL_DEBUG("rank[%u] round[%u] rcv with rank[%u]'s offset[%llu] size[%llu]", \
            interRank_, i, dataRank, scatterOffset, scatterResult);

        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread_, channelLeft_.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        if (channelLeft_.protocol != COMM_PROTOCOL_ROCE) {
            void* remoteAddr = (interRank_ == ((root_ + 1) % interRankSize_)) ?
                channelLeft_.remoteInput.addr : channelLeft_.remoteOutput.addr;

            void* src = static_cast<void *>(static_cast<s8 *>(remoteAddr) + scatterOffset + baseOffset_);
            void* dst = static_cast<void *>(static_cast<s8 *>(outputMem_.addr) + scatterOffset);
            HCCL_DEBUG("[ScatterRing][HcommReadOnThread] src[%p] dst[%p] size[%llu]", src, dst, scatterResult);
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread_, channelLeft_.handle, dst, src, scatterResult)));
        }
    }
    return HCCL_SUCCESS;
}

void ScatterRing::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const
{
    slices_.resize(rankSize);
    u64 sliceSize = (totalCount / rankSize) * unitSize;

    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
        HCCL_DEBUG("rank[%u] default slice[%u]: offset: [%llu] size[%llu]", interRank_, i, i * sliceSize, sliceSize);
    }
}

// scatter的入口函数
HcclResult ScatterRing::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels)
{
    if (!outputMem_.addr || !inputMem_.addr) {
        HCCL_ERROR("[ScatterRing][RunAsync]run_async inputmem or outputmem is null");
        return HCCL_E_PTR;
    }

    interRank_ = rank;
    interRankSize_ = rankSize;

    HCCL_INFO("ScatterRing run: rank[%u] totalrank[%u]  count[%llu] input[%p] output[%p]",
              interRank_, interRankSize_, count_, inputMem_.addr, outputMem_.addr);

    // ranksize为1时，只有当input!=ouput 时候进行拷贝
    if (interRankSize_ == 1) {
        if (inputMem_.addr != outputMem_.addr) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, outputMem_.addr, inputMem_.addr, inputMem_.size)));
        }
        return HCCL_SUCCESS;
    }

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(unitSize == 0, HCCL_ERROR("[ScatterRing][RunAsync]rank[%u] unit data size is zero", rank),
        HCCL_E_INTERNAL);

    // 带入vecotr为空，计算每个rank的结果偏移和大小
    if (slices_.size() == 0) {
        PrepareSlicesData(unitSize, count_, interRankSize_);
    }

    // 获取link的收、发缓存, 计算chunk_size
    u32 ringPrevRank = (rank + rankSize - 1) % rankSize;
    u32 ringNextRank = (rank + 1) % rankSize;

    if (channels.size() < rankSize) {
        HCCL_ERROR("[ScatterRing][RunAsync]rank[%u] link size[%llu] is less than rank size", rank, channels.size());
        return HCCL_E_INTERNAL;
    }

    channelLeft_ = channels[ringPrevRank];
    channelRight_ = channels[ringNextRank];

    // root rank向其他rank发送数据,
    if (interRank_ == root_) {
        CHK_RET(RunScatterOnRootRank());
    } else if (ringNextRank == root_) { // 最后一个节点只负责接收数据，拷贝至outputmem
        CHK_RET(RunScatterOnEndRank());
    } else {
        CHK_RET(RunScatterOnMidRank());
    }

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(channelLeft_, channelRight_));
    }
    HCCL_INFO("ScatterRing finished: rank:[%u] end", interRank_);

    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_SCATTER_RING, ScatterRing);
}
