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
 * @file aiv_stuck_counter_util.h
 * @brief AIV 计数参数填充：普通 kernel args / SK-lite args 共用一份字段映射。
 */

#ifndef AIV_STUCK_COUNTER_UTIL_H
#define AIV_STUCK_COUNTER_UTIL_H

namespace ops_hccl {

template <typename ArgsT, typename CounterT>
inline void FillAivStuckCounter(ArgsT& args, const CounterT& counter)
{
    args.headCountMem = reinterpret_cast<decltype(args.headCountMem)>(counter.headCountMem);
    args.tailCountMem = reinterpret_cast<decltype(args.tailCountMem)>(counter.tailCountMem);
    args.addOneMem = reinterpret_cast<decltype(args.addOneMem)>(counter.addOneMem);
    args.counterMemSize = static_cast<decltype(args.counterMemSize)>(counter.memSize);
    args.isEnableCounter = static_cast<decltype(args.isEnableCounter)>(counter.isEnableCounter);
}

} // namespace ops_hccl

#endif // AIV_STUCK_COUNTER_UTIL_H
