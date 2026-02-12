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
class AivReduceMesh1D : public AivCommBase {
public:
    __aicore__ inline AivReduceMesh1D() {}
 
    __aicore__ inline void InitCoreInfo()
    {
        dataSize_ = len_ * sizeof(T);
        SplitData(len_, sliceLen_, offsetLen_);
        offsetSize_ = offsetLen_ * sizeof(T);
        if (rank_ < root_) {
            srcOffset_ = input_ + offsetSize_;
            dstOffset_ = reinterpret_cast<uint64_t>(GM_IN[root_]) + offsetSize_ + rank_ * dataSize_;
        } else if (rank_ > root_) {
            srcOffset_ = input_ + offsetSize_;
            dstOffset_ = reinterpret_cast<uint64_t>(GM_IN[root_]) + offsetSize_ + (rank_ - 1) * dataSize_;
        } else {
            srcOffset_ = input_ + offsetSize_;
            dstOffset_ = output_ + offsetSize_;
        }
    }
 
    __aicore__ inline void Process(int32_t tag)
    {
        tag_ = tag;
        if (rank_ != root_) {
            // 写远端：将自身core负责的Input数据搬运至root的Scratch上
            if (sliceLen_ > 0) {
                CpGM2GM((__gm__ T *)dstOffset_, (__gm__ T *)srcOffset_, sliceLen_);
                PipeBarrier<PIPE_ALL>();
            }
            // 写同步：将aivTag写入root上的数据同步标志位，表示数据搬运完成
            uint64_t flagOffset;
            flagOffset = rank_ * block_num + block_idx;
            Record(root_, flagOffset, tag_);
        } else {
            // 本地拷贝：将自身core负责的Input数据搬运至本地Output上
            if (sliceLen_ > 0) {
                CpGM2GM((__gm__ T *)dstOffset_, (__gm__ T *)srcOffset_, sliceLen_);
                PipeBarrier<PIPE_ALL>();
            }
            uint32_t sliceIdx = 0;
            for (uint32_t dataRank = 0; dataRank < rankSize_; dataRank++) {
                if (dataRank == rank_) {
                    continue;
                }
                // 读同步：阻塞读取本地数据同步标志位，当前aivTag等于读取值时，继续步骤
                uint64_t flagOffset = dataRank * block_num + block_idx;
                WaitFlag(rank_, flagOffset, tag_);
                // 本地规约：将本地ScratchBuffer上的数据Reduce到本地OutputBuffer上
                if (sliceLen_ > 0) {
                    srcOffset_ = reinterpret_cast<uint64_t>(GM_IN[root_]) + sliceIdx * dataSize_ + offsetSize_;
                    CpGM2GM((__gm__ T *)dstOffset_, (__gm__ T *)srcOffset_, sliceLen_, reduceOp_);
                    PipeBarrier<PIPE_ALL>();
                }
                sliceIdx++;
            }
        }
    }
 
    __aicore__ inline void SplitData(uint64_t dataLen, uint64_t& sliceLen, uint64_t& offsetLen)
    {
        uint64_t sliceLenMin = dataLen / block_num;
        uint64_t remainLen = dataLen % block_num;
        // remainLen必然小于dataLen，均分给前remainLen个aiv处理
        if (block_idx < remainLen) {
            sliceLen = sliceLenMin + 1;
            offsetLen = block_idx * (sliceLenMin + 1);
        } else {
            sliceLen = sliceLenMin;
            offsetLen = remainLen * (sliceLenMin + 1) + (block_idx - remainLen) * sliceLenMin;
        }
    }
 
    uint64_t dataSize_;
    uint64_t sliceLen_;
    uint64_t offsetLen_;
    uint64_t offsetSize_;
    uint64_t srcOffset_;
    uint64_t dstOffset_;
};
 
template<typename T>
__aicore__ inline void AivReduceV2Mesh1D(EXTERN_KERNEL_ARGS_DEF_V2)
{
    AivReduceMesh1D<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.InitCoreInfo();
    SyncAll<true>();
    if (block_idx == 0 && tag >> AIV_TAG_MOVE_RIGHT_BITS == 1 && (tag & LOW_16_BITS) == 1) {
        op.BarrierForFirstOP();
    }
    SyncAll<true>();
    op.Process(tag);
    op.BarrierAll();
    SyncAll<true>();
}