/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_URMA_COMM_BASE_H
#define AIV_URMA_COMM_BASE_H

#include "version/hcomm_version.h"
#include "version/asc_devkit_version.h"
#if (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)
#include "adv_api/hcomm/hcomm.h"
#include "aiv_defines.h"

struct AivUrmaChannelInfo {
    uint32_t rankSize;
    uint32_t channelsPerRank[MAX_RANK_SIZE];
    uint64_t channelHandle[MAX_RANK_SIZE * MAX_CHANNELS_PER_RANK];
    uint32_t portNum[MAX_RANK_SIZE * MAX_CHANNELS_PER_RANK];
};

class AivUrmaCommBase {
public:
    __aicore__ inline AivUrmaCommBase() = default;

    __aicore__ inline void Init(
        GM_ADDR buffIn, uint64_t input, uint64_t output, uint32_t rank, uint32_t sendRecvRemoteRank,
        uint32_t subCommRankSize, uint64_t len, uint32_t dataType, uint32_t reduceOp, uint32_t root,
        uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t repeatNum, uint64_t inputRepeatStride,
        uint64_t outputRepeatStride, GM_ADDR headCountMem, GM_ADDR tailCountMem, GM_ADDR addOneMem,
        uint32_t counterMemSize, bool isEnableCounter, uint32_t numBlocks, uint32_t subCommRankLevel,
        uint32_t level0ChannelNumPerRank)
    {
        buffIn_ = (uint64_t)buffIn;
        input_ = input;
        output_ = output;
        rank_ = rank;
        sendRecvRemoteRank_ = sendRecvRemoteRank;
        subCommRankSize_ = subCommRankSize;
        len_ = len;
        dataType_ = dataType;
        reduceOp_ = reduceOp;
        root_ = root;
        inputSliceStride_ = inputSliceStride;
        outputSliceStride_ = outputSliceStride;
        repeatNum_ = repeatNum;
        inputRepeatStride_ = inputRepeatStride;
        outputRepeatStride_ = outputRepeatStride;
        headCountMem_ = (uint64_t)headCountMem;
        tailCountMem_ = (uint64_t)tailCountMem;
        addOneMem_ = (uint64_t)addOneMem;
        counterMemSize_ = counterMemSize;
        isEnableCounter_ = isEnableCounter;
        numBlocks_ = numBlocks;
        subCommRankLevel_ = subCommRankLevel;
        level0ChannelNumPerRank_ = level0ChannelNumPerRank;
        blockIdx_ = GetBlockIdx();

        pipe_.InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
        pipe_.InitBuffer(dataBuf_, DATA_BUF_SIZE);
        pipe_.InitBuffer(hcommBatchBuf_, BATCH_BUFFER_BYTES);
        pipe_.InitBuffer(inOutQue_, 2, UB_DB_DATA_BATCH_SIZE);
        auto hcommTensor = hcommBuf_.Get<uint8_t>();
        hcomm_.Init(hcommTensor, HCOMM_INIT_SIZE);

        peerCommInfoBasePtr_ = (__gm__ uint64_t*)((__gm__ uint8_t*)buffIn_ + BUFFER_OUT_ADDR_OFFSET);
        outputAddrSlotPtr_ = (__gm__ uint64_t*)((__gm__ uint8_t*)buffIn_ + AIV_URMA_OUTPUT_ADDR_SLOT_OFFSET);
        notifyBase_ = buffIn_ + FLAG1_OFFSET;

        GetTag();
        LoadChannelInfo();
        LoadSubCommRanks();

        AIV_INFO(
            "[AivUrmaCommBase][Init] rank=%u level=%u tag=%u blk=%u subRankSize=%u numBlocks=%u mySubRankIdx=%u "
            "len=%llu\n",
            rank_, subCommRankLevel_, tag_, blockIdx_, subCommRankSize_, numBlocks_, mySubCommRankIdx_, len_);
    }

    __aicore__ inline uint64_t GetPeerCommInfoBase(uint32_t peerRank) { return peerCommInfoBasePtr_[peerRank]; }

    __aicore__ inline uint64_t ReadPeerOutputAddr(uint32_t peerRank)
    {
        __gm__ uint64_t* srcGM = outputAddrSlotPtr_ + peerRank;
        LocalTensor<uint64_t> tmp = dataBuf_.GetWithOffset<uint64_t>(1, 0);
        GlobalTensor<uint64_t> srcGT;
        srcGT.SetGlobalBuffer(srcGM, 1);
        DataCopyGM2UB(tmp, srcGT, 1);
        PipeBarrier<PIPE_ALL>();
        return tmp.GetValue(0);
    }

    __aicore__ inline void SubmitDataOnBatch(
        uint64_t channel, uint64_t remoteDst, uint64_t localSrc, uint64_t dataLen, bool needNotify, uint64_t notifyAddr,
        uint64_t notifyVal)
    {
        static constexpr struct UrmaWqeEntry DATA_CFG = {.odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
        static constexpr struct UrmaWqeEntry LAST_DATA_CFG = {.odr = 6, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};

        uint32_t numChunks = (dataLen + WQE_MAX_LEN - 1) / WQE_MAX_LEN;
        for (uint32_t c = 0; c < numChunks; c++) {
            uint64_t offset = (uint64_t)c * WQE_MAX_LEN;
            uint64_t thisLen = (c == numChunks - 1) ? (dataLen - offset) : WQE_MAX_LEN;
            bool isLast = (c == numChunks - 1);

            if (batchCount_ >= BATCH_CAPACITY) {
                hcomm_.BatchCommit(activeBatchHandle_);
                batchCount_ = 0;
            }

            if (batchCount_ == 0) {
                auto batchTensor = hcommBatchBuf_.Get<uint8_t>();
                activeBatchHandle_
                    = hcomm_.MakeBatchHandle(static_cast<ChannelHandle>(channel), batchTensor, BATCH_BUFFER_BYTES);
            }

            if (isLast && needNotify) {
                hcomm_.WriteWithNotifyNbi<LAST_DATA_CFG>(
                    activeBatchHandle_, (GM_ADDR)(remoteDst + offset), (GM_ADDR)(localSrc + offset), (uint32_t)thisLen,
                    (GM_ADDR)notifyAddr, notifyVal);
            } else {
                hcomm_.WriteNbi<DATA_CFG>(
                    activeBatchHandle_, (GM_ADDR)(remoteDst + offset), (GM_ADDR)(localSrc + offset), (uint32_t)thisLen);
            }
            batchCount_++;
        }
    }

    __aicore__ inline void FlushBatch()
    {
        if (batchCount_ > 0) {
            hcomm_.BatchCommit(activeBatchHandle_);
            batchCount_ = 0;
        }
    }

    __aicore__ inline void DrainAllChannels()
    {
        for (uint32_t i = assignedRankStart_; i < assignedRankEnd_; i++) {
            uint32_t peerRank = GetSubCommRank(i);
            if (peerRank == rank_) {
                continue;
            }
            uint32_t chCount = GetChannelsPerRank(peerRank);
            for (uint32_t ch = 0; ch < chCount; ch++) {
                int32_t ret = hcomm_.Drain(static_cast<ChannelHandle>(GetChannelHandle(peerRank, ch)));
                if (ret != 0) {
                    AIV_INFO(
                        "[AivUrmaCommBase][DrainAllChannels] Drain failed rank=%u blk=%u peer=%u ch=%u ret=%d\n", rank_,
                        blockIdx_, peerRank, ch, ret);
                }
            }
        }
    }

    __aicore__ inline void WaitNotify(uint64_t notifySlot, uint32_t expectedVal, bool clearAfter)
    {
        __gm__ uint64_t* notifySlotPtr = (__gm__ uint64_t*)notifySlot;
        while (true) {
            LocalTensor<uint64_t> tmp = dataBuf_.GetWithOffset<uint64_t>(1, 0);
            GlobalTensor<uint64_t> notifyGT;
            notifyGT.SetGlobalBuffer(notifySlotPtr, 1);
            DataCopyGM2UB(tmp, notifyGT, 1);
            PipeBarrier<PIPE_ALL>();
            if (tmp.GetValue(0) == expectedVal) {
                break;
            }
        }
        if (clearAfter) {
            LocalTensor<uint64_t> tmp = dataBuf_.GetWithOffset<uint64_t>(1, 0);
            tmp.SetValue(0, 0);
            PipeBarrier<PIPE_ALL>();
            GlobalTensor<uint64_t> notifyGT;
            notifyGT.SetGlobalBuffer(notifySlotPtr, 1);
            DataCopyUB2GM(notifyGT, tmp, 1);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void SendRecvOutputAddr()
    {
        uint64_t myOutputAddr = output_;

        for (uint32_t i = assignedRankStart_; i < assignedRankEnd_; i++) {
            uint32_t peerRank = GetSubCommRank(i);
            if (peerRank == rank_) {
                continue;
            }
            uint64_t channel = GetChannelHandle(peerRank, 0);
            uint64_t peerCommInfoBase = GetPeerCommInfoBase(peerRank);

            uint64_t peerOutputSlotAddr
                = peerCommInfoBase + AIV_URMA_OUTPUT_ADDR_SLOT_OFFSET + rank_ * sizeof(uint64_t);
            static constexpr struct UrmaWqeEntry ADDR_CFG = {.odr = 5, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 1};
            hcomm_.WriteValueNbi<uint64_t, true, PIPE_S, PIPE_MTE3, ADDR_CFG>(
                static_cast<ChannelHandle>(channel), (GM_ADDR)peerOutputSlotAddr, myOutputAddr);

            uint64_t peerNotifyAddr = peerCommInfoBase + FLAG1_OFFSET + rank_ * sizeof(uint64_t);
            static constexpr struct UrmaWqeEntry NOTIFY_CFG = {.odr = 6, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 1};
            hcomm_.WriteValueNbi<uint64_t, true, PIPE_S, PIPE_MTE3, NOTIFY_CFG>(
                static_cast<ChannelHandle>(channel), (GM_ADDR)peerNotifyAddr, 1ULL);
        }

        for (uint32_t i = assignedRankStart_; i < assignedRankEnd_; i++) {
            uint32_t peerRank = GetSubCommRank(i);
            if (peerRank == rank_) {
                continue;
            }
            int32_t ret = hcomm_.Drain(static_cast<ChannelHandle>(GetChannelHandle(peerRank, 0)));
            if (ret != 0) {
                AIV_INFO(
                    "[AivUrmaCommBase][SendRecvOutputAddr] Drain failed rank=%u blk=%u peer=%u ret=%d\n", rank_,
                    blockIdx_, peerRank, ret);
            }
        }
    }

    __aicore__ inline void LocalCopy(uint64_t dst, uint64_t src, uint64_t len)
    {
        uint64_t offset = 0;
        uint32_t maxCount = UB_DB_DATA_BATCH_SIZE / sizeof(uint8_t);
        while (offset < len) {
            uint32_t chunkLen
                = (len - offset > UB_DB_DATA_BATCH_SIZE) ? maxCount : (uint32_t)((len - offset) / sizeof(uint8_t));
            LocalTensor<uint8_t> ubBuf = inOutQue_.AllocTensor<uint8_t>();
            GlobalTensor<uint8_t> srcGT;
            srcGT.SetGlobalBuffer((__gm__ uint8_t*)(src + offset), chunkLen);
            DataCopyGM2UB(ubBuf, srcGT, chunkLen);
            inOutQue_.EnQue(ubBuf);
            LocalTensor<uint8_t> ubOut = inOutQue_.DeQue<uint8_t>();
            GlobalTensor<uint8_t> dstGT;
            dstGT.SetGlobalBuffer((__gm__ uint8_t*)(dst + offset), chunkLen);
            DataCopyUB2GM(dstGT, ubOut, chunkLen);
            inOutQue_.FreeTensor(ubOut);
            offset += chunkLen * sizeof(uint8_t);
        }
    }

    __aicore__ inline void ClearGM(uint64_t dst, uint64_t len)
    {
        uint64_t emptySrc = buffIn_ + AIV_FLAG_EMPTY_OFFSET;
        uint64_t offset = 0;
        uint32_t maxCount = UB_DB_DATA_BATCH_SIZE / sizeof(uint8_t);
        while (offset < len) {
            uint32_t chunkLen
                = (len - offset > UB_DB_DATA_BATCH_SIZE) ? maxCount : (uint32_t)((len - offset) / sizeof(uint8_t));
            LocalTensor<uint8_t> ubBuf = inOutQue_.AllocTensor<uint8_t>();
            GlobalTensor<uint8_t> srcGT;
            srcGT.SetGlobalBuffer((__gm__ uint8_t*)(emptySrc + offset), chunkLen);
            DataCopyGM2UB(ubBuf, srcGT, chunkLen);
            inOutQue_.EnQue(ubBuf);
            LocalTensor<uint8_t> ubOut = inOutQue_.DeQue<uint8_t>();
            GlobalTensor<uint8_t> dstGT;
            dstGT.SetGlobalBuffer((__gm__ uint8_t*)(dst + offset), chunkLen);
            DataCopyUB2GM(dstGT, ubOut, chunkLen);
            inOutQue_.FreeTensor(ubOut);
            offset += chunkLen * sizeof(uint8_t);
        }
    }

protected:
    AscendC::Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> hcomm_;
    TPipe pipe_;
    TBuf<> hcommBuf_;
    TBuf<> dataBuf_;
    TBuf<> hcommBatchBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inOutQue_;
    static constexpr uint32_t HCOMM_INIT_SIZE = 512U;
    static constexpr uint32_t DATA_BUF_SIZE = 512U;
    static constexpr uint32_t BATCH_CAPACITY = 128U;
    static constexpr uint32_t BATCH_BUFFER_BYTES = BATCH_CAPACITY * 64U;
    static constexpr uint64_t WQE_MAX_LEN = 256ULL * 1024 * 1024;

    uint64_t buffIn_{0};
    uint64_t input_{0};
    uint64_t output_{0};
    uint32_t rank_{0};
    uint32_t sendRecvRemoteRank_{0};
    uint32_t subCommRankSize_{0};
    uint64_t len_{0};
    uint32_t dataType_{0};
    uint32_t reduceOp_{0};
    uint32_t root_{0};
    uint64_t inputSliceStride_{0};
    uint64_t outputSliceStride_{0};
    uint64_t repeatNum_{0};
    uint64_t inputRepeatStride_{0};
    uint64_t outputRepeatStride_{0};
    uint64_t headCountMem_{0};
    uint64_t tailCountMem_{0};
    uint64_t addOneMem_{0};
    uint32_t counterMemSize_{0};
    bool isEnableCounter_{false};
    uint32_t numBlocks_{0};
    uint32_t blockIdx_{0};
    uint32_t subCommRankLevel_{0};
    uint32_t level0ChannelNumPerRank_{0};
    uint32_t tag_{0};

    __gm__ uint64_t* peerCommInfoBasePtr_{nullptr};
    __gm__ uint64_t* outputAddrSlotPtr_{nullptr};
    uint64_t notifyBase_{0};

    __gm__ AivUrmaChannelInfo* channelInfoPtr_{nullptr};
    __gm__ uint32_t* subCommRanksPtr_{nullptr};
    uint32_t mySubCommRankIdx_{0};

    uint32_t assignedRankStart_{0};
    uint32_t assignedRankEnd_{0};

    using HcommBatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;
    HcommBatchHandle activeBatchHandle_{};
    uint32_t batchCount_{0};

private:
    __aicore__ inline void GetTag()
    {
        __gm__ uint32_t* tagGM = (__gm__ uint32_t*)((__gm__ uint8_t*)buffIn_ + AIV_FLAG_CLEAR_OFFSET);
        LocalTensor<uint32_t> tmp = dataBuf_.GetWithOffset<uint32_t>(1, 0);
        GlobalTensor<uint32_t> tagGT;
        tagGT.SetGlobalBuffer(tagGM, 1);
        DataCopyGM2UB(tmp, tagGT, 1);
        PipeBarrier<PIPE_ALL>();
        tag_ = tmp.GetValue(0);
        SyncAll<true>();
        tag_ = (tag_ >= TAG_RESET_COUNT) ? TAG_INIT_VALUE : tag_ + 1;
        if (blockIdx_ == 0) {
            tmp.SetValue(0, tag_);
            PipeBarrier<PIPE_ALL>();
            DataCopyUB2GM(tagGT, tmp, 1);
            PipeBarrier<PIPE_ALL>();
        }
        SyncAll<true>();
    }

    __aicore__ inline void LoadChannelInfo()
    {
        channelInfoPtr_ = (__gm__ AivUrmaChannelInfo*)((__gm__ uint8_t*)buffIn_ + AIV_URMA_HANDLE_ADDR_OFFSET);
    }

    __aicore__ inline void LoadSubCommRanks()
    {
        uint32_t levelOffset
            = AIV_URMA_SUBCOMM_RANKS_START_OFFSET + subCommRankLevel_ * AIV_URMA_SUBCOMM_RANKS_PER_LEVEL_BYTES;
        subCommRanksPtr_ = (__gm__ uint32_t*)((__gm__ uint8_t*)buffIn_ + levelOffset);
        mySubCommRankIdx_ = 0;
        bool found = false;
        for (uint32_t i = 0; i < subCommRankSize_; i++) {
            if (subCommRanksPtr_[i] == rank_) {
                mySubCommRankIdx_ = i;
                found = true;
                break;
            }
        }
        AIV_INFO(
            "[AivUrmaCommBase][LoadSubCommRanks] rank=%u level=%u subRankSize=%u found=%d mySubRankIdx=%u\n", rank_,
            subCommRankLevel_, subCommRankSize_, found, mySubCommRankIdx_);
    }

public:
    __aicore__ inline uint32_t GetSubCommRank(uint32_t idx) { return subCommRanksPtr_[idx]; }

    __aicore__ inline uint32_t GetChannelsPerRank(uint32_t peerRank)
    {
        return channelInfoPtr_->channelsPerRank[peerRank];
    }

    __aicore__ inline uint64_t GetChannelHandle(uint32_t peerRank, uint32_t ch)
    {
        return channelInfoPtr_->channelHandle[peerRank * MAX_CHANNELS_PER_RANK + ch];
    }

    __aicore__ inline uint32_t GetPortNum(uint32_t peerRank, uint32_t ch)
    {
        return channelInfoPtr_->portNum[peerRank * MAX_CHANNELS_PER_RANK + ch];
    }
};
#endif // (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)

#endif // AIV_URMA_COMM_BASE_H
