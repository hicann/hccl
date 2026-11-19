/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_URMA_ALL_TO_ALL_V_MESH_1D_H
#define AIV_URMA_ALL_TO_ALL_V_MESH_1D_H

#include "aiv_urma_comm_base.h"
#if (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)

template <typename T>
class AivUrmaAllToAllVMesh1D : public AivUrmaCommBase {
public:
    __aicore__ inline void Process(ExtraArgs& extraArgs)
    {
        dataTypeSize_ = sizeof(T);
        extraArgs_ = &extraArgs;

        uint32_t roundCount = CalcRoundCount();
        AIV_INFO(
            "[AivUrmaAllToAllVMesh1D][Process] rank=%u level=%u blk=%u numBlocks=%u rankSize=%u rounds=%u\n", rank_,
            subCommRankLevel_, blockIdx_, numBlocks_, subCommRankSize_, roundCount);
        for (uint32_t round = 0; round < roundCount; round++) {
            CalcCommRankSetForOneRound(round);
            CalcRoundCoreAssignment();
            AIV_INFO(
                "[AivUrmaAllToAllVMesh1D][Process] rank=%u blk=%u round=%u/%u peerCnt=%u start=%u end=%u\n", rank_,
                blockIdx_, round, roundCount, roundPeerCount_, roundPeerStart_, roundPeerEnd_);
            // 每轮闭环: 地址交换 -> 等地址就绪 -> 发数据 -> 等数据到齐 -> 清空在途任务
            SendRecvOutputAddrForRound();
            WaitAddrReadyForRound();
            SendDataForRound();
            WaitAllNotifyForRound();
            DrainChannelsForRound();
        }

        LocalCopyOwnData();
        SyncAll<true>();
    }

private:
    uint64_t dataTypeSize_{0};
    ExtraArgs* extraArgs_{nullptr};
    // 本轮对端(global rank), 数量上限为 GROUP_SIZE
    uint32_t roundPeers_[AIV_URMA_ALLTOALLV_GROUP_SIZE];
    uint32_t roundPeerCount_{0};
    uint32_t roundPeerStart_{0};
    uint32_t roundPeerEnd_{0};

    __aicore__ inline uint32_t CalcRoundCount()
    {
        uint32_t totalPeers = subCommRankSize_ - 1;
        return (totalPeers + AIV_URMA_ALLTOALLV_GROUP_SIZE - 1) / AIV_URMA_ALLTOALLV_GROUP_SIZE;
    }

    // 左右对称分轮，对齐AICPU版CalcCommRankSetForOneLoop
    __aicore__ inline void CalcCommRankSetForOneRound(uint32_t round)
    {
        roundPeerCount_ = 0;
        uint32_t pairNumPerRound = AIV_URMA_ALLTOALLV_GROUP_SIZE / 2;
        uint32_t remainPeers = (subCommRankSize_ - 1) - round * AIV_URMA_ALLTOALLV_GROUP_SIZE;
        uint32_t pairSize = (remainPeers < AIV_URMA_ALLTOALLV_GROUP_SIZE) ? (remainPeers + 1) / 2 : pairNumPerRound;
        for (uint32_t i = round * pairNumPerRound + 1; i <= round * pairNumPerRound + pairSize; i++) {
            uint32_t leftRemoteRank = (rank_ + subCommRankSize_ - i) % subCommRankSize_;
            uint32_t rightRemoteRank = (rank_ + i) % subCommRankSize_;
            if (leftRemoteRank == rightRemoteRank) {
                roundPeers_[roundPeerCount_++] = leftRemoteRank;
                break;
            }
            roundPeers_[roundPeerCount_++] = leftRemoteRank;
            roundPeers_[roundPeerCount_++] = rightRemoteRank;
        }
    }

    // 本轮对端按核均分, 连续区间, 前remainder个核各多分1个对端
    __aicore__ inline void CalcRoundCoreAssignment()
    {
        roundPeerStart_ = 0;
        roundPeerEnd_ = 0;
        uint32_t activeCores = (numBlocks_ < roundPeerCount_) ? numBlocks_ : roundPeerCount_;
        if (activeCores == 0) {
            return;
        }
        uint32_t peersPerCore = roundPeerCount_ / activeCores;
        uint32_t remainder = roundPeerCount_ % activeCores;

        uint32_t start = 0;
        for (uint32_t i = 0; i < blockIdx_ && i < activeCores; i++) {
            start += peersPerCore + (i < remainder ? 1 : 0);
        }

        if (blockIdx_ < activeCores) {
            roundPeerStart_ = start;
            roundPeerEnd_ = start + peersPerCore + (blockIdx_ < remainder ? 1 : 0);
        }
    }

    // 双向均为0的对端两端一致跳过, 不产生任何notify写入与等待
    __aicore__ inline bool IsActivePeer(uint32_t peerRank)
    {
        return (extraArgs_->sendCounts[peerRank] != 0) || (extraArgs_->recvCounts[peerRank] != 0);
    }

    // 本端告知对端 output_+recvDispls[peer]*dataTypeSize_, 发送端直写该地址
    __aicore__ inline void SendRecvOutputAddrForRound()
    {
        for (uint32_t i = roundPeerStart_; i < roundPeerEnd_; i++) {
            uint32_t peerRank = roundPeers_[i]; // roundPeers_已存global rank
            if (!IsActivePeer(peerRank)) {
                continue;
            }
            uint64_t myOutputAddrForPeer = output_ + extraArgs_->recvDispls[peerRank] * dataTypeSize_;
            uint64_t channel = GetChannelHandle(peerRank, 0);
            uint64_t peerCommInfoBase = GetPeerCommInfoBase(peerRank);

            uint64_t peerOutputSlotAddr
                = peerCommInfoBase + AIV_URMA_OUTPUT_ADDR_SLOT_OFFSET + rank_ * sizeof(uint64_t);
            static constexpr struct UrmaWqeEntry ADDR_CFG = {.odr = 5, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 1};
            hcomm_.WriteValueNbi<uint64_t, true, PIPE_S, PIPE_MTE3, ADDR_CFG>(
                static_cast<ChannelHandle>(channel), (GM_ADDR)peerOutputSlotAddr, myOutputAddrForPeer);

            uint64_t peerNotifyAddr = peerCommInfoBase + FLAG1_OFFSET + rank_ * sizeof(uint64_t);
            static constexpr struct UrmaWqeEntry NOTIFY_CFG = {.odr = 6, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 1};
            hcomm_.WriteValueNbi<uint64_t, true, PIPE_S, PIPE_MTE3, NOTIFY_CFG>(
                static_cast<ChannelHandle>(channel), (GM_ADDR)peerNotifyAddr, 1ULL);
        }

        for (uint32_t i = roundPeerStart_; i < roundPeerEnd_; i++) {
            uint32_t peerRank = roundPeers_[i]; // roundPeers_已存global rank
            if (!IsActivePeer(peerRank)) {
                continue;
            }
            hcomm_.Drain(static_cast<ChannelHandle>(GetChannelHandle(peerRank, 0)));
        }
    }

    // 等待本轮全部活跃对端的地址就绪通知并消费清零
    __aicore__ inline void WaitAddrReadyForRound()
    {
        for (uint32_t i = roundPeerStart_; i < roundPeerEnd_; i++) {
            uint32_t peerRank = roundPeers_[i]; // roundPeers_已存global rank
            if (!IsActivePeer(peerRank)) {
                continue;
            }
            WaitNotify(notifyBase_ + peerRank * sizeof(uint64_t), 1, true);
        }
    }

    __aicore__ inline void SendDataForRound()
    {
        constexpr uint64_t dataNotifyBase = MAX_RANK_SIZE * sizeof(uint64_t);
        for (uint32_t i = roundPeerStart_; i < roundPeerEnd_; i++) {
            uint32_t peerRank = roundPeers_[i];
            uint64_t sendCount = extraArgs_->sendCounts[peerRank];
            if (sendCount == 0) {
                continue; // 空对端不发送, 对端recvCounts镜像为0不等待
            }

            uint64_t peerOutputAddr = ReadPeerOutputAddr(peerRank);
            uint64_t localSrc = input_ + extraArgs_->sendDispls[peerRank] * dataTypeSize_;
            uint64_t totalBytes = sendCount * dataTypeSize_;

            uint32_t chCount = GetChannelsPerRank(peerRank);
            AIV_INFO(
                "[AivUrmaAllToAllVMesh1D][SendData] rank=%u blk=%u peer=%u peerOut=0x%llx totalBytes=%llu chCount=%u\n",
                rank_, blockIdx_, peerRank, peerOutputAddr, totalBytes, chCount);
            uint32_t totalPorts = 0;
            for (uint32_t ch = 0; ch < chCount; ch++) {
                totalPorts += GetPortNum(peerRank, ch);
            }
            if (totalPorts == 0) {
                continue; // host建链已保证portSize>0, 此处仅防御
            }

            uint64_t accumCount = 0;
            for (uint32_t ch = 0; ch < chCount; ch++) {
                // 按元素个数切分再乘回dataTypeSize, 避免按字节整除把数据切碎
                uint64_t chDataCount = (ch == chCount - 1) ? (sendCount - accumCount) :
                                                             (sendCount * GetPortNum(peerRank, ch) / totalPorts);
                uint64_t chDataLen = chDataCount * dataTypeSize_;
                uint64_t accumLen = accumCount * dataTypeSize_;
                uint64_t channel = GetChannelHandle(peerRank, ch);
                uint64_t peerNotifyAddr = GetPeerCommInfoBase(peerRank) + FLAG1_OFFSET + dataNotifyBase
                                          + (rank_ * MAX_CHANNELS_PER_RANK + ch) * sizeof(uint64_t);
                AIV_INFO(
                    "[AivUrmaAllToAllVMesh1D][SendData] rank=%u blk=%u peer=%u ch=%u/%u channel=0x%llx "
                    "chDataCount=%llu "
                    "chDataLen=%llu portNum=%u/%u\n",
                    rank_, blockIdx_, peerRank, ch, chCount, channel, chDataCount, chDataLen, GetPortNum(peerRank, ch),
                    totalPorts);

                if (chDataLen == 0) {
                    // 分到0字节也补发notify, 防对端WaitAllNotifyForRound挂死
                    hcomm_.WriteValueNbi<uint64_t>(static_cast<ChannelHandle>(channel), (GM_ADDR)peerNotifyAddr, 2ULL);
                } else {
                    SubmitDataOnBatch(
                        channel, peerOutputAddr + accumLen, localSrc + accumLen, chDataLen, true, peerNotifyAddr, 2ULL);
                }
                accumCount += chDataCount;
                FlushBatch();
            }
        }
    }

    __aicore__ inline void WaitAllNotifyForRound()
    {
        constexpr uint64_t dataNotifyBase = MAX_RANK_SIZE * sizeof(uint64_t);
        for (uint32_t i = roundPeerStart_; i < roundPeerEnd_; i++) {
            uint32_t peerRank = roundPeers_[i]; // roundPeers_已存global rank
            if (extraArgs_->recvCounts[peerRank] == 0) {
                continue; // 空对端不等待
            }
            uint32_t chCount = GetChannelsPerRank(peerRank);
            for (uint32_t ch = 0; ch < chCount; ch++) {
                uint64_t notifySlot
                    = notifyBase_ + dataNotifyBase + (peerRank * MAX_CHANNELS_PER_RANK + ch) * sizeof(uint64_t);
                WaitNotify(notifySlot, 2, true);
                AIV_INFO(
                    "[AivUrmaAllToAllVMesh1D][WaitAllNotify] rank=%u blk=%u peer=%u ch=%u/%u wait success\n", rank_,
                    blockIdx_, peerRank, ch, chCount);
            }
        }
    }

    __aicore__ inline void DrainChannelsForRound()
    {
        for (uint32_t i = roundPeerStart_; i < roundPeerEnd_; i++) {
            uint32_t peerRank = roundPeers_[i]; // roundPeers_已存global rank
            if (!IsActivePeer(peerRank)) {
                continue;
            }
            uint32_t chCount = GetChannelsPerRank(peerRank);
            for (uint32_t ch = 0; ch < chCount; ch++) {
                hcomm_.Drain(static_cast<ChannelHandle>(GetChannelHandle(peerRank, ch)));
            }
        }
    }

    // 分轮对端不含本rank, 仅由0号核搬运防多核同写GM; 轮间有同步点(每轮Wait+Drain闭环), 无法与传输并行。
    __aicore__ inline void LocalCopyOwnData()
    {
        if (blockIdx_ != 0 || input_ == output_) {
            return;
        }
        // AICPU版LocalCopy对 src/dst size不等报HCCL_E_INTERNAL拒绝拷贝; device侧无法返回错误码,
        // 取min防御保证不越界写后续接收块(对齐alg_data_trans_wrapper.cc的LocalCopy安全语义)
        uint64_t sendBytes = extraArgs_->sendCounts[rank_] * dataTypeSize_;
        uint64_t recvBytes = extraArgs_->recvCounts[rank_] * dataTypeSize_;
        if (sendBytes != recvBytes) {
            AIV_INFO(
                "[AivUrmaAllToAllVMesh1D][LocalCopyOwnData] WARN rank[%u] sendCounts[%llu] != recvCounts[%llu]", rank_,
                extraArgs_->sendCounts[rank_], extraArgs_->recvCounts[rank_]);
        }
        uint64_t copyBytes = sendBytes < recvBytes ? sendBytes : recvBytes;
        if (copyBytes == 0) {
            return;
        }
        LocalCopy(
            output_ + extraArgs_->recvDispls[rank_] * dataTypeSize_,
            input_ + extraArgs_->sendDispls[rank_] * dataTypeSize_, copyBytes);
    }
};

template <typename T>
__aicore__ inline void AivUrmaAllToAllVV2Mesh1D(KERNEL_ARGS_DEF, ExtraArgs& extraArgs)
{
    AivUrmaAllToAllVMesh1D<T> kernel;
    kernel.Init(KERNEL_CLASS_INIT, subCommRankLevel, level0ChannelNumPerRank);
    kernel.Process(extraArgs);
}
#endif // (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)

#endif // AIV_URMA_ALL_TO_ALL_V_MESH_1D_H
