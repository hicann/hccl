/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_URMA_ALL_GATHER_MESH_1D_H
#define AIV_URMA_ALL_GATHER_MESH_1D_H

#include "aiv_urma_comm_base.h"
#if (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)

template <typename T>
class AivUrmaAllGatherMesh1D : public AivUrmaCommBase {
public:
    __aicore__ inline void Process()
    {
        dataTypeSize_ = sizeof(T);

        if (blockIdx_ >= numBlocks_) {
            return;
        }

        CalcCoreAssignment();

        AIV_INFO(
            "[AivUrmaAllGatherMesh1D][Process] rank=%u level=%u blk=%u start=%u end=%u\n", rank_, subCommRankLevel_,
            blockIdx_, assignedRankStart_, assignedRankEnd_);

        SendRecvOutputAddr();

        SendData();

        WaitAllNotify();

        DrainAllChannels();

        SyncAll<true>();
    }

private:
    uint64_t dataTypeSize_{0};

    __aicore__ inline void CalcCoreAssignment()
    {
        assignedRankStart_ = 0;
        assignedRankEnd_ = 0;
        uint32_t activeCores = (numBlocks_ < subCommRankSize_) ? numBlocks_ : subCommRankSize_;
        if (activeCores == 0) {
            return;
        }
        uint32_t ranksPerCore = subCommRankSize_ / activeCores;
        uint32_t remainder = subCommRankSize_ % activeCores;

        uint32_t start = 0;
        for (uint32_t i = 0; i < blockIdx_ && i < activeCores; i++) {
            start += ranksPerCore + (i < remainder ? 1 : 0);
        }

        if (blockIdx_ < activeCores) {
            assignedRankStart_ = start;
            assignedRankEnd_ = start + ranksPerCore + (blockIdx_ < remainder ? 1 : 0);
        }
    }

    __aicore__ inline void SendData()
    {
        constexpr uint64_t dataNotifyBase = MAX_RANK_SIZE * sizeof(uint64_t);
        for (uint32_t i = assignedRankStart_; i < assignedRankEnd_; i++) {
            uint32_t peerRank = GetSubCommRank(i);
            if (peerRank == rank_ && input_ != output_) {
                LocalCopy(
                    output_ + mySubCommRankIdx_ * outputSliceStride_, input_ + mySubCommRankIdx_ * inputSliceStride_,
                    len_ * dataTypeSize_);
                continue;
            }
            if (peerRank == rank_) {
                continue;
            }

            // 等待该对端地址就绪
            uint64_t addrNotifySlot = notifyBase_ + peerRank * sizeof(uint64_t);
            WaitNotify(addrNotifySlot, 1, true);

            uint64_t peerOutputAddr = ReadPeerOutputAddr(peerRank);
            AIV_INFO(
                "[AivUrmaAllGatherMesh1D][SendData] rank=%u blk=%u peer=%u peerOut=0x%llx totalBytes=%llu chCount=%u\n",
                rank_, blockIdx_, peerRank, peerOutputAddr, len_ * dataTypeSize_, GetChannelsPerRank(peerRank));
            uint64_t localSrc = input_ + mySubCommRankIdx_ * inputSliceStride_;
            uint64_t remoteDst = peerOutputAddr + mySubCommRankIdx_ * outputSliceStride_;

            uint32_t chCount = GetChannelsPerRank(peerRank);

            // 计算各 channel 权重：
            // - level0ChNum == 0：权重 = portNum（纯 level1 或纯 mesh）
            // - level0ChNum > 0 且有 level1：level0 权重 = portNum * (subCommRankSize_ - 1)，level1 权重 = portNum（Z
            // 轴绕路）
            // - level0ChNum > 0 且无 level1：level0 权重 = portNum（纯 level0）
            uint32_t level0ChNum = level0ChannelNumPerRank_;
            uint64_t level0Multiplier = (level0ChNum > 0 && chCount > level0ChNum) ? (subCommRankSize_ - 1) : 1;

            uint64_t totalWeight = 0;
            for (uint32_t ch = 0; ch < chCount; ch++) {
                uint64_t weight = GetPortNum(peerRank, ch);
                if (ch < level0ChNum) {
                    weight *= level0Multiplier;
                }
                totalWeight += weight;
            }

            uint64_t accumCount = 0;
            for (uint32_t ch = 0; ch < chCount; ch++) {
                uint64_t weight = GetPortNum(peerRank, ch);
                if (ch < level0ChNum) {
                    weight *= level0Multiplier;
                }
                uint64_t chDataCount
                    = (ch == chCount - 1) ? (len_ - accumCount) : (totalWeight > 0 ? len_ * weight / totalWeight : 0);
                uint64_t chDataLen = chDataCount * dataTypeSize_;
                uint64_t channel = GetChannelHandle(peerRank, ch);
                AIV_INFO(
                    "[AivUrmaAllGatherMesh1D][SendData] rank=%u blk=%u peer=%u ch=%u/%u channel=0x%llx "
                    "chDataCount=%llu "
                    "chDataLen=%llu weight=%llu/%llu portNum=%u\n",
                    rank_, blockIdx_, peerRank, ch, chCount, channel, chDataCount, chDataLen, weight, totalWeight,
                    GetPortNum(peerRank, ch));
                uint64_t peerNotifyAddr = GetPeerCommInfoBase(peerRank) + FLAG1_OFFSET + dataNotifyBase
                                          + (rank_ * MAX_CHANNELS_PER_RANK + ch) * sizeof(uint64_t);

                if (chDataLen == 0) {
                    // 数据量为0时单独发notify，保证对端WaitAllNotify不挂死
                    hcomm_.WriteValueNbi<uint64_t>(static_cast<ChannelHandle>(channel), (GM_ADDR)peerNotifyAddr, 2ULL);
                } else {
                    for (uint64_t r = 0; r < repeatNum_; r++) {
                        uint64_t src = localSrc + r * inputRepeatStride_ + accumCount * dataTypeSize_;
                        uint64_t dst = remoteDst + r * outputRepeatStride_ + accumCount * dataTypeSize_;
                        bool isLastRepeat = (r == repeatNum_ - 1);
                        SubmitDataOnBatch(channel, dst, src, chDataLen, isLastRepeat, peerNotifyAddr, 2ULL);
                    }
                }

                accumCount += chDataCount;
                FlushBatch();
            }
        }
    }

    __aicore__ inline void WaitAllNotify()
    {
        constexpr uint64_t dataNotifyBase = MAX_RANK_SIZE * sizeof(uint64_t);
        for (uint32_t i = assignedRankStart_; i < assignedRankEnd_; i++) {
            uint32_t peerRank = GetSubCommRank(i);
            if (peerRank == rank_) {
                continue;
            }
            uint32_t chCount = GetChannelsPerRank(peerRank);
            for (uint32_t ch = 0; ch < chCount; ch++) {
                uint64_t notifySlot
                    = notifyBase_ + dataNotifyBase + (peerRank * MAX_CHANNELS_PER_RANK + ch) * sizeof(uint64_t);
                WaitNotify(notifySlot, 2, true);
                AIV_INFO(
                    "[AivUrmaAllGatherMesh1D][WaitAllNotify] rank=%u blk=%u peer=%u ch=%u/%u wait success\n", rank_,
                    blockIdx_, peerRank, ch, chCount);
            }
        }
    }
};

template <typename T>
__aicore__ inline void AivUrmaAllGatherV2Mesh1D(KERNEL_ARGS_DEF)
{
    AivUrmaAllGatherMesh1D<T> kernel;
    kernel.Init(KERNEL_CLASS_INIT, subCommRankLevel, level0ChannelNumPerRank);
    kernel.Process();
}
#endif // (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)

#endif // AIV_URMA_ALL_GATHER_MESH_1D_H
