/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_NEGOTIATION_H
#define COMM_NEGOTIATION_H

#include <string>
#include "hccl.h"
#include "alg_param.h"

namespace ops_hccl {
/*
 * 公共跨rank协商机制（从 CCU NegotiationCtx 提炼，机制与策略分离）：
 * 本模块只拥有「如何协商」（资源生命周期 + 全域子通信域 + 原语 + CommWorker + EngineCtx缓存 + 防嵌套），
 * 调用方各自提供「协商什么」（门控/载荷/聚合/动作/缓存粒度）。
 *
 * 生命周期按 (comm, typeTag) 懒创建：negTag = commName + "_" + typeTag + "_negotiation"，
 * 各 typeTag（如 "ccu" / "hostdpu_inline"）拥有独立子通信域与生命周期；
 * 通信域销毁前(DESTROY_PRE)级联清理、快恢后(RESUME_POST)级联恢复。
 */

// 协商子通信域名称后缀：negTag = commName + "_" + typeTag + 后缀（子通信域名同构）。
// 唯一来源：生成方（本模块拼装协商子通信域名）与识别方（op_common.cc IsNegotiationCommName，
// profiling按NEGOTIATIONOP上报、selector日志区分）同源引用，防两处漂移静默失效
const std::string NEGOTIATION_COMM_SUFFIX = "_negotiation";

// 判断comm是否为协商子通信域（防嵌套主防线）：协商原语自身是集合算子，其 Selector 再次进入
// 协商门控时命中本判定即跳过，递归在协商入口被切断。
// 标记在创建协商子通信域时打入 EngineCtx（随 comm 生命周期销毁，无句柄复用陈旧问题）；
// "_negotiation" 名称后缀作模块内快速路径（子通信域名截断逻辑保证后缀保留）。
bool IsNegotiationComm(HcclComm comm);

// 全域 AND 协商：全域子通信域 AllReduce(INT32, MIN) 聚合本端能力位，全部 rank 为 true 才输出 true。
// 失败返回非 HCCL_SUCCESS，由调用方决定降级策略（协商类失败按不支持处理，禁止上抛）。
HcclResult
NegotiateAllReduceMin(HcclComm comm, const std::string& typeTag, bool localCapability, bool& allRanksCapable);

// 全域检视协商：全域子通信域 AllGather(UINT8) 收集各 rank 载荷（CCU 参数检视迁移时使用）。
// recvBuf 由调用方提供，容量须不小于 rankSize * payloadSize 字节。
HcclResult NegotiateAllGather(
    HcclComm comm, const std::string& typeTag, const void* localPayload, uint64_t payloadSize, void* recvBuf,
    uint64_t recvBufSize);
} // namespace ops_hccl

#endif // COMM_NEGOTIATION_H
