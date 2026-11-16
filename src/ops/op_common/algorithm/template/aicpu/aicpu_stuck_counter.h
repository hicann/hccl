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
 * @file aicpu_stuck_counter.h
 * @brief AICPU 卡死检测计数下发：以同流 inline reduce SQE 对 head/tail 执行 counter += addOne。
 */

#ifndef AICPU_STUCK_COUNTER_H
#define AICPU_STUCK_COUNTER_H

#include <cstdint>

#include "alg_param.h"
#include "hcomm_primitives.h"
#include "log.h"

namespace ops_hccl {

// 卡死检测计数语义（检测侧按此设定超时阈值）：
//   head 在算子真正下发点（OpOrchestrate / OrchestrateWithThread）之前发射，
//   tail 在其返回后立即发射（成功/失败都发）。
//   因此 head-tail 窗口 = “下发 → 完成”，包含等待 host stream 通知（NotifyWait）
//   与队列排队的时间；检测阈值应大于正常排队等待时长，否则可能早报。
inline bool AicpuStuckCounterEmit(ThreadHandle thread, const OpParam& param, bool isHead)
{
    if (param.isEnableCounter == 0) {
        return false;
    }
    // 配对优先：head/tail/addOne 必须同时可用；任一为 0 时两侧都不发射，
    // 避免单侧累加导致 head != tail 的持续误报（比漏检更糟）
    if (thread == 0 || param.counterHeadAddr == 0 || param.counterTailAddr == 0 || param.counterAddOneAddr == 0) {
        HCCL_DEBUG(
            "[StuckDetect][AicpuEmit] skip, thread[%llu], head[0x%llx], tail[0x%llx], addOne[0x%llx].", thread,
            param.counterHeadAddr, param.counterTailAddr, param.counterAddOneAddr);
        return false;
    }
    uint64_t countAddr = isHead ? param.counterHeadAddr : param.counterTailAddr;
    int32_t ret = HcommLocalReduceOnThread(
        thread, reinterpret_cast<void*>(countAddr), reinterpret_cast<const void*>(param.counterAddOneAddr), 1,
        HCOMM_DATA_TYPE_INT32, HCOMM_REDUCE_SUM);
    if (ret != 0) {
        HCCL_DEBUG("[StuckDetect][AicpuEmit] LocalReduce failed, thread[%llu], dst[0x%llx].", thread, countAddr);
    }
    return true;
}

inline bool AicpuStuckCounterHead(ThreadHandle thread, const OpParam& param)
{
    return AicpuStuckCounterEmit(thread, param, true);
}

inline bool AicpuStuckCounterTail(ThreadHandle thread, const OpParam& param)
{
    return AicpuStuckCounterEmit(thread, param, false);
}

} // namespace ops_hccl

#endif // AICPU_STUCK_COUNTER_H
