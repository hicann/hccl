/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_AIV_TEST_COMMUNICATION_BASE_H
#define ALL_GATHER_AIV_TEST_COMMUNICATION_BASE_H

#include "aiv_defines.h"

// 执行真实kernel并记录数据流；CPU调度器模拟flag依赖和独立UB快照，不模拟硬件DMA时序。
class AivCommBase {
public:
    template <typename... ExtraArgs>
    void Init(
        GM_ADDR, uint64_t input, uint64_t output, uint32_t rank, uint32_t, uint32_t ranks, uint64_t len, uint32_t,
        uint32_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, bool,
        uint32_t blocks, ExtraArgs...)
    {
        input_ = input;
        output_ = output;
        rank_ = rank;
        rankSize_ = ranks;
        numBlocks_ = blocks;
        blockIdx_ = all_gather_aiv_test::currentTrace->block;
        myGmIn_ = GetGmIn(rank);
    }
    GM_ADDR GetGmIn(uint32_t rank)
    {
        return reinterpret_cast<GM_ADDR>(all_gather_aiv_test::Address(all_gather_aiv_test::CCL_BASE, rank));
    }
    template <typename T>
    void CpGM2GM(T* destination, T* source, uint64_t count)
    {
        if (count == 0) {
            return;
        }
        all_gather_aiv_test::currentTrace->events.push_back(
            {all_gather_aiv_test::EventKind::COPY, reinterpret_cast<uint64_t>(destination),
             reinterpret_cast<uint64_t>(source), count * sizeof(T)});
    }
    template <typename T>
    void
    DataCopyGM2UB(const AscendC::LocalTensor<T>& destination, const AscendC::GlobalTensor<T>& source, uint32_t count)
    {
        if (count * sizeof(T) > UB_DB_DATA_BATCH_SIZE) {
            throw std::logic_error("cached piece exceeds the fast-path UB budget");
        }
        all_gather_aiv_test::currentTrace->events.push_back(
            {all_gather_aiv_test::EventKind::LOAD, destination.address, source.address, count * sizeof(T)});
    }
    template <typename T>
    void
    DataCopyUB2GM(const AscendC::GlobalTensor<T>& destination, const AscendC::LocalTensor<T>& source, uint32_t count)
    {
        all_gather_aiv_test::currentTrace->events.push_back(
            {all_gather_aiv_test::EventKind::STORE, destination.address, source.address, count * sizeof(T)});
    }
    void Record(uint32_t target, uint64_t slot, int32_t tag)
    {
        all_gather_aiv_test::currentTrace->events.push_back(
            {all_gather_aiv_test::EventKind::RECORD, 0, 0, 0, target, slot, tag});
    }
    void WaitFlag(uint32_t target, uint64_t slot, int32_t tag)
    {
        all_gather_aiv_test::currentTrace->events.push_back(
            {all_gather_aiv_test::EventKind::WAIT, 0, 0, 0, target, slot, tag});
    }
    template <bool lock = false>
    void SyncAll()
    {
        all_gather_aiv_test::currentTrace->events.push_back({all_gather_aiv_test::EventKind::SYNC});
    }
    bool IsFirstOP(uint32_t) const { return false; }
    void BarrierForFirstOP() {}
    void BarrierAll() {}

    uint64_t input_ = 0;
    uint64_t output_ = 0;
    uint32_t rank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t numBlocks_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t tag_ = 1;
    uint32_t curTag_ = 0;
    bool useDoubleBuffer_ = true;
    GM_ADDR myGmIn_ = nullptr;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 1> inOutQue;
};
#endif
