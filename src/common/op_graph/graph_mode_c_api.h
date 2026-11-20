/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_GRAPH_MODE_C_API_H
#define HCCL_GRAPH_MODE_C_API_H

// GraphMode C 接口统一前向声明
// 权威声明: src/ops/interface_graph_mode/calc_resource_graph_mode.h (OpParamGraphMode 位于
// namespace ops_hccl) 与 GraphMode 通信接口 (hcom/hccl 头侧)。不直接 include 权威头是避免
// alg_param.h 依赖链; 本头作为单一事实源, 签名逐条对照权威头维护 — 新增/变更接口时只改这里
#include <stddef.h>
#include <stdint.h>

#include "acl/acl_rt.h"
#include "hccl.h"
#include "hccl/hccl_types.h"

namespace ops_hccl {
struct OpParamGraphMode;
} // namespace ops_hccl
using ops_hccl::OpParamGraphMode;

extern "C" {
// ---- opParam 组装/销毁/在线资源 (calc_resource_graph_mode.h 同签名) ----
HcclResult HcclCreateOpParamGraphMode(OpParamGraphMode** opParam);
HcclResult HcclDestroyOpParamGraphMode(OpParamGraphMode* opParam);
HcclResult HcclSetOpParamGraphModeOpType(OpParamGraphMode* opParam, const char* opType);
HcclResult HcclSetOpParamGraphModeDataCount(OpParamGraphMode* opParam, const uint64_t* dataCount);
HcclResult HcclSetOpParamGraphModeRankSize(OpParamGraphMode* opParam, const uint32_t* rankSize);
HcclResult HcclSetOpParamGraphModeHCCLBufferSize(OpParamGraphMode* opParam, const uint64_t* hcclBufferSize);
HcclResult HcclSetOpParamGraphModeDataType(OpParamGraphMode* opParam, HcclDataType dataType);
HcclResult HcclSetAivSelectOpParamGraphMode(OpParamGraphMode* opParam, uint32_t aivCoreLimit);
HcclResult HcclSelectAlgGraphMode(
    const char* group, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclCMDType opType,
    uint32_t aivCoreLimit, bool* ifAiv, char* algName);
HcclResult HcclCalcOpResOnlineGraphMode(
    OpParamGraphMode* opParam, uint64_t* opMemSize, uint32_t* streamNum, uint32_t* taskNum, uint32_t* aivCoreNum);
HcclResult HcclSetAivCoreLimitGraphMode(const char* group, uint32_t aivCoreLimit);

// ---- AllToAll 族 GraphMode 通信入口 ----
HcclResult HcclAlltoAllGraphMode(
    const void* sendBuf, uint64_t sendCount, HcclDataType sendType, const void* recvBuf, uint64_t recvCount,
    HcclDataType recvType, const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount,
    void* scratchMemAddr, uint64_t scratchMemSize);
HcclResult HcclAlltoAllVGraphMode(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, const char* group, aclrtStream stream,
    const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize);
HcclResult HcclAlltoAllVCGraphMode(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,
    uint64_t scratchMemSize);

} // extern "C"

#endif // HCCL_GRAPH_MODE_C_API_H
