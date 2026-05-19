/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_communication_base_v2.h"

using namespace AscendC;

template<typename T>
class AivScatterMesh1D : public AivCommBase {
public:
    __aicore__ inline AivScatterMesh1D() {}

    __aicore__ inline void InitCommon(uint32_t sliceId)
    {
        uint64_t smallDataSize = 512 * 1024;
        dataSize_ = len_ * sizeof(T);
        coreIdx_ = GetBlockIdx();
        coreNum_ = block_num;
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
    }

    __aicore__ inline void Process()
    {
        ProcessCoreCtrl();
    }

private:
    __aicore__ inline void ProcessCoreCtrl()
    {
        uint32_t remainRankSize = rankSize_ % coreNum_;
        uint32_t copyNumThisCore = coreIdx_ < remainRankSize ? rankSize_ / coreNum_ + 1 : rankSize_ / coreNum_;

        if(copyNumThisCore == 0){
            return;
        }
        // PreCopy阶段
        uint32_t dstRank;
        if (rank_ == root_) {
            for (int i = 0; i < copyNumThisCore; i++) {
                dstRank = i * coreNum_ + coreIdx_;
                srcOffset_ = input_ + dstRank * inputSliceStride_;
                dstOffset_ = reinterpret_cast<uint64_t>(GM_IN[dstRank]);
                CpGM2GM((__gm__ T *)dstOffset_, (__gm__ T *)srcOffset_, len_);
                pipe_barrier(PIPE_ALL);
            }
        }

        for (int i = 0; i < copyNumThisCore; i++) {
            dstRank = i * coreNum_ + coreIdx_;
            Record(dstRank, rank_, curTag_);
        }

        for (int i = 0; i < copyNumThisCore; i++) {
            dstRank = i * coreNum_ + coreIdx_;
            WaitFlag(rank_, dstRank, curTag_);
            if (dstRank == root_) {
                srcOffset_ = reinterpret_cast<uint64_t>(GM_IN[rank_]);
                dstOffset_ = output_;
                CpGM2GM((__gm__ T *)dstOffset_, (__gm__ T *)srcOffset_, len_);
                pipe_barrier(PIPE_ALL);
            }
        }
    }

    uint32_t coreNum_;
    uint32_t coreIdx_;

    uint64_t dataSize_;  // 要给每个rank搬运的数据大小

    uint64_t srcOffset_;
    uint64_t dstOffset_;
};

template<typename T>
__aicore__ inline void AivScatterV2Mesh1D(KERNEL_ARGS_DEF)
{
    AivScatterMesh1D<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.InitCommon(sliceId);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    op.Process();
    op.BarrierAll();
}