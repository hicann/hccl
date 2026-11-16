/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_STUCK_DETECT_CCU_PRIMITIVES_DL_HPP
#define UT_STUCK_DETECT_CCU_PRIMITIVES_DL_HPP

#include <cstdint>
#include <stdexcept>
#include <vector>

using CcuResult = int;
constexpr CcuResult CCU_SUCCESS = 0;
constexpr CcuResult CCU_E_INTERNAL = 4;

using CcuKernelArg = void*;
using CcuKernelFunc = CcuResult (*)(CcuKernelArg);

enum class HcclDataType { HCCL_DATA_TYPE_INT32 = 0 };

enum class HcclReduceOp { HCCL_REDUCE_SUM = 0 };

namespace hcomm {
namespace ccu {

    struct LocalAddr {
        uint64_t addr{0};
        uint64_t token{0};
    };

    using Variable = uint64_t;

    extern int32_t g_eventAllocCount;
    extern bool g_allocThrow; // 注入 CcuEventAlloc 失败（构造抛 CcuException 同形的 std::runtime_error）
    extern bool g_allocThrowNonStd; // 注入非 std::exception 派生异常，验证 catch(...) 同样降级

    struct Event {
        int id{0};
        Event() : id(++g_eventAllocCount)
        {
            if (g_allocThrowNonStd) {
                throw 0x5A5A; // 非标准异常类型
            }
            if (g_allocThrow) {
                throw std::runtime_error("CcuEventAlloc: failed");
            }
        }
    };

    struct ReduceRecord {
        uint64_t dstAddr{0};
        uint64_t dstToken{0};
        uint64_t srcAddr{0};
        uint64_t srcToken{0};
        uint64_t len{0};
        int dataType{0};
        int reduceOp{0};
        uint16_t mask{0};
    };

    struct WaitRecord {
        int eventId{0};
        uint16_t mask{0};
    };

    extern int32_t g_reduceRet;
    extern int32_t g_waitRet;
    extern std::vector<ReduceRecord> g_reduceRecords;
    extern std::vector<WaitRecord> g_waitRecords;

    CcuResult LocalReduce(
        LocalAddr dst, LocalAddr src, Variable len, HcclDataType dataType, HcclReduceOp reduceOp, Event& evt,
        uint16_t mask);
    CcuResult EventWait(Event& evt, uint16_t mask);

} // namespace ccu
} // namespace hcomm

#endif // UT_STUCK_DETECT_CCU_PRIMITIVES_DL_HPP
