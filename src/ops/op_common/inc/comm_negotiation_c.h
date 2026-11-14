/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_NEGOTIATION_C_H
#define COMM_NEGOTIATION_C_H

#include <stdbool.h>
#include <stdint.h>
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 公共协商机制的 C 接口（弱符号，默认实现转调 comm_negotiation.cc 的 C++ 本体）：
 * - extern "C" 符号名固定，UT/ST 可用强符号覆盖打桩，不受 C++ 签名演进的 mangled 漂移影响；
 * - 单入口 + 算子枚举分发：后续新增协商算子（如 ReduceScatter）只追加枚举值与模块内实现，
 *   接口签名、导出符号、共享桩结构均不变。
 */

// 协商算子类型：枚举值只增不改不复用（新增算子向后追加，保持既有值稳定）
typedef enum {
    NEGOTIATE_OP_ALL_REDUCE_MIN = 0, // 全域AND聚合：sendBuf/recvBuf各1字节(bool)，模块内部走int32 MIN编排
    NEGOTIATE_OP_ALL_GATHER = 1,     // 全域收集：recvBuf布局 = rankSize * sendSize字节
} NegotiateOpC;

// 协商统一入口：按 (comm, typeTag) 懒创建/复用协商子通信域资源，执行 op 指定的协商算子。
// 收发布局由 op 决定：
// - NEGOTIATE_OP_ALL_REDUCE_MIN：sendSize/recvSize 须为 sizeof(bool)，全部 rank 为 true 才输出 true；
// - NEGOTIATE_OP_ALL_GATHER：recvBufSize 须不小于 rankSize * sendSize，recvBuf[i * sendSize] 即 rank i 载荷。
// 失败返回非 HCCL_SUCCESS，由调用方决定降级策略（协商类失败按不支持处理，禁止上抛）。
HcclResult NegotiateCommC(
    HcclComm comm, const char* typeTag, NegotiateOpC op, const void* sendBuf, uint64_t sendSize, void* recvBuf,
    uint64_t recvSize);

#ifdef __cplusplus
}
#endif

#endif // COMM_NEGOTIATION_C_H
