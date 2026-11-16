/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_primitives_dl.hpp"

namespace hcomm {
namespace ccu {

    int32_t g_reduceRet = 0;
    int32_t g_waitRet = 0;
    int32_t g_eventAllocCount = 0;
    bool g_allocThrow = false;
    bool g_allocThrowNonStd = false;
    std::vector<ReduceRecord> g_reduceRecords;
    std::vector<WaitRecord> g_waitRecords;

    CcuResult LocalReduce(
        LocalAddr dst, LocalAddr src, Variable len, HcclDataType dataType, HcclReduceOp reduceOp, Event& evt,
        uint16_t mask)
    {
        ReduceRecord record;
        record.dstAddr = dst.addr;
        record.dstToken = dst.token;
        record.srcAddr = src.addr;
        record.srcToken = src.token;
        record.len = len;
        record.dataType = static_cast<int>(dataType);
        record.reduceOp = static_cast<int>(reduceOp);
        record.mask = mask;
        g_reduceRecords.push_back(record);
        (void)evt;
        return g_reduceRet;
    }

    CcuResult EventWait(Event& evt, uint16_t mask)
    {
        WaitRecord record;
        record.eventId = evt.id;
        record.mask = mask;
        g_waitRecords.push_back(record);
        return g_waitRet;
    }

} // namespace ccu
} // namespace hcomm
