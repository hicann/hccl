/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file ccu_stuck_counter.h
 * @brief CCU 卡死检测计数：trampoline 与 kernel 内 mission mark 共用同一份 LocalReduce 实现。
 *
 * Event 申请说明：
 *  - CcuEventAlloc 是“当前 kernel build”内的资源，会追加到 CcuKernel::res_.blockCompletedEvent，
 *    由 GetResourceRequest() 计入 blockCkeReq，随资源查询写入资源描述，注册时按该口径申请实例事件；
 *  - 句柄是 build 内索引，不能跨 build（query / register）复用，因此本类在每次 build 首次 Emit 前申请 1 个，
 *    head / tail 复用同一个；
 *  - 申请失败（如不在 build 上下文）只降级为“不计数”，不允许把异常抛给业务算子；
 *    捕获范围覆盖 CcuException（std::exception 派生）与任意未知类型，避免 hcomm 异常类型演进破坏 fail-open。
 */

#ifndef CCU_STUCK_COUNTER_H
#define CCU_STUCK_COUNTER_H

#include <cstdint>
#include <exception>
#include <optional>

#include "alg_param.h"
#include "ccu_primitives_dl.hpp"
#include "log.h"

namespace ops_hccl {

constexpr uint64_t CCU_STUCK_COUNTER_BYTES = sizeof(int32_t);
constexpr uint16_t CCU_STUCK_COUNTER_EVT_MASK = 1;

class CcuStuckCounter {
public:
    explicit CcuStuckCounter(const CcuKernelArgBase* arg) : arg_(arg) {} // 构造不申请 Event，避免无打点时白占资源

    CcuResult Head() { return Emit(arg_ == nullptr ? 0 : arg_->stuckHeadAddr); }

    CcuResult Tail() { return Emit(arg_ == nullptr ? 0 : arg_->stuckTailAddr); }

private:
    bool CanEmit(uint64_t dstAddr) const
    {
        return arg_->stuckCounterEnabled && dstAddr != 0 && arg_->stuckAddOneAddr != 0;
    }

    // 每次 build 最多申请 1 个 Event（head/tail 复用）；失败只降级，不抛给业务。
    // 除 CcuException（std::exception 派生）外，未知类型异常同样吞掉：fail-open 是硬约束，
    // 不能因为 hcomm 侧异常类型演进就把异常抛进 CCU 注册/build 流程。
    bool EnsureEvent()
    {
        if (evt_.has_value()) {
            return true;
        }
        if (allocFailed_ || arg_ == nullptr || !(CanEmit(arg_->stuckHeadAddr) || CanEmit(arg_->stuckTailAddr))) {
            return false;
        }
        try {
            evt_.emplace(); // 内部 CcuEventAlloc，失败会抛 CcuException
        } catch (const std::exception& err) {
            allocFailed_ = true; // 本次 build 不再重试，避免每次打点都抛一次
            HCCL_DEBUG("[StuckDetect][CcuEmit] event alloc failed, skip counting, why[%s].", err.what());
            return false;
        } catch (...) {
            allocFailed_ = true; // 非 std::exception 派生异常同样降级，绝不抛给业务算子
            HCCL_DEBUG("[StuckDetect][CcuEmit] event alloc failed(unknown), skip counting.");
            return false;
        }
        return true;
    }

    CcuResult Emit(uint64_t dstAddr)
    {
        if (arg_ == nullptr || !CanEmit(dstAddr) || !EnsureEvent()) {
            return CCU_SUCCESS; // DFX 降级：不计数，不影响算子
        }
        hcomm::ccu::LocalAddr dst;
        dst.addr = dstAddr;
        dst.token = arg_->stuckCounterToken;
        hcomm::ccu::LocalAddr src;
        src.addr = arg_->stuckAddOneAddr;
        src.token = arg_->stuckCounterToken;
        hcomm::ccu::Variable len;
        len = CCU_STUCK_COUNTER_BYTES;
        CcuResult ret = hcomm::ccu::LocalReduce(
            dst, src, len, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM, *evt_,
            CCU_STUCK_COUNTER_EVT_MASK);
        if (ret != CCU_SUCCESS) {
            return ret;
        }
        return hcomm::ccu::EventWait(*evt_, CCU_STUCK_COUNTER_EVT_MASK);
    }

    const CcuKernelArgBase* arg_{nullptr};
    std::optional<hcomm::ccu::Event> evt_; // 按需申请，head/tail 复用
    bool allocFailed_{false};              // 本次 build 申请失败后不再重试
};

} // namespace ops_hccl

#endif // CCU_STUCK_COUNTER_H
