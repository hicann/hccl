/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_SYNC_INTERFACE_H
#define AIV_SYNC_INTERFACE_H

#include "kernel_operator.h"
using namespace AscendC;

constexpr uint64_t UB_FLAG_PAD_COUNT = 8;
constexpr uint64_t UB_ADDRESS_PAD_COUNT = 4;

// 在gm上设置同步信号的值
__aicore__ inline void SetSignalValue(__gm__ int32_t* gmSignalAddr, LocalTensor<int32_t>& localTensor, int32_t value)
{
    GlobalTensor<int32_t> globalTensor;
    globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(gmSignalAddr));

    localTensor.SetValue(0, value);
    //SyncFunc<HardEvent::S_MTE3>();
    pipe_barrier(PIPE_ALL);
    DataCopy(globalTensor, localTensor, 32);
    pipe_barrier(PIPE_ALL);
    return;
}

// 随路原子更新gm上同步信号的值，为加法，如果要减去某个值，输入值为负数即可
__aicore__ inline void AddSignalValue(__gm__ int32_t* gmSignalAddr, LocalTensor<int32_t>& localTensor, uint64_t value)
{
    GlobalTensor<int32_t> globalTensor;
    globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(gmSignalAddr));

    Duplicate<int32_t>(localTensor, value, UB_FLAG_PAD_COUNT);
    SetAtomicAdd<int32_t>();
    PipeBarrier<PIPE_ALL>();

    DataCopy(globalTensor, localTensor, UB_FLAG_PAD_COUNT);
    SetAtomicNone();
    return;
}

// 等待同步信号变成某个预期的值
__aicore__ inline void WaitSignalValue(__gm__ int32_t *gmSignalAddr, LocalTensor<int32_t>& localTensor, int32_t expectedValue)
{
    GlobalTensor<int32_t> globalTensor;
    globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(gmSignalAddr));
    while (true) {
        DataCopy(localTensor, globalTensor, 32);
        //SyncFunc<HardEvent::MTE2_S>();
        pipe_barrier(PIPE_ALL);
        if (localTensor.GetValue(0) == expectedValue) {
            break;
        }
    }
    return;
}

// 等待同步信号大于等于某个预期的值
__aicore__ inline void WaitSignalGEValue(__gm__ int32_t *gmSignalAddr, LocalTensor<int32_t>& localTensor, int32_t value)
{
    GlobalTensor<int32_t> globalTensor;
    globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(gmSignalAddr));

    while (true) {
        DataCopy(localTensor, globalTensor, 32);
        //SyncFunc<HardEvent::MTE2_S>();
        pipe_barrier(PIPE_ALL);
        if (localTensor.GetValue(0) >= value) {
            break;
        }
    }
    return;
}

__aicore__ inline uint64_t GetSignalValue(__gm__ int32_t *gmSignalAddr, LocalTensor<int32_t>& localTensor)
{
    GlobalTensor<int32_t> globalTensor;
    globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(gmSignalAddr));
    DataCopy(localTensor, globalTensor, UB_FLAG_PAD_COUNT);
    //SyncFunc<HardEvent::MTE2_S>();
    pipe_barrier(PIPE_ALL);
    int32_t ret = localTensor.GetValue(0);
    return ret;
}

__aicore__ inline void WaitSignalNotEqValue(__gm__ int32_t *gmSignalAddr, LocalTensor<int32_t>& localTensor, uint64_t value)
{
    GlobalTensor<int32_t> globalTensor;
    globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(gmSignalAddr));

    while (true) {
        DataCopy(localTensor, globalTensor, UB_FLAG_PAD_COUNT);
        //SyncFunc<HardEvent::MTE2_S>();
        pipe_barrier(PIPE_ALL);
        if (localTensor.GetValue(0) != value) {
            break;
        }
    }
    return;
}

__aicore__ inline void SetFlagBatchValue(__gm__ int32_t *ctrlFlagGM, TQue<QuePosition::VECOUT, 1> &batchQue, uint64_t setValue, uint64_t count)
{
    GlobalTensor<int32_t> globalBatchSet;
    globalBatchSet.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ctrlFlagGM));
    LocalTensor<int32_t> localBatchSet = batchQue.AllocTensor<int32_t>();

    for (int32_t i = 0; i < count; i++) {
        localBatchSet.SetValue(i * UB_FLAG_PAD_COUNT, setValue);
    }

    //SyncFunc<HardEvent::S_MTE3>();
    pipe_barrier(PIPE_ALL);

    DataCopy(globalBatchSet, localBatchSet, UB_FLAG_PAD_COUNT * count);

    batchQue.FreeTensor(localBatchSet);
}

#endif