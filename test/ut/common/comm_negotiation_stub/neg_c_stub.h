/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_COMM_NEGOTIATION_C_STUB_H
#define UT_COMM_NEGOTIATION_C_STUB_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "comm_negotiation_c.h"

/*
 * 公共协商机制共享 UT 桩（消费方：ccu_fallback、hostdpu_inline_reduce 等策略层 UT）：
 * - NegotiateCommC 以强符号覆盖产品 C 弱壳：符号名固定，不受 C++ 本体签名演进的
 *   mangled 漂移影响；
 * - IsNegotiationComm（C++本体）由桩提供同签名定义（消费方UT不链接 comm_negotiation.cc）；
 * - 数据级注入、op 无关：ALL_REDUCE_MIN 注入返回码与全域能力位；ALL_GATHER 按
 *   "复制本端载荷 + 逐槽位字段覆盖"保真回填，rank 数由收发尺寸推导，与被测方零共享配置；
 * - 后续协商新增算子（枚举值追加）时本桩零改动：预填内容由各测试按需自定
 *   （只有测试知道期望的远端数据）。
 */
namespace ut {
struct NegCStubConfig {
    /* ---- IsNegotiationComm：防嵌套门控 ---- */
    bool isNegotiationComm = false; // 返回值（true = 协商子通信域，被测方应跳过协商防递归）

    /* ---- NEGOTIATE_OP_ALL_REDUCE_MIN：全域 AND 聚合 ---- */
    HcclResult reduceMinRet = HCCL_SUCCESS; // 注入返回值（协商基础设施失败 / CommWorker 并发拒绝）
    bool allRanksCapable = true;            // 注入全域能力位出参（false = 存在不支持的 rank）
    int reduceMinCnt = 0;                   // 调用计数
    bool lastLocalCapability = false;       // 观测：最近一次传入的本端能力位

    /* ---- NEGOTIATE_OP_ALL_GATHER：全域收集 ---- */
    HcclResult gatherRet = HCCL_SUCCESS; // 注入返回值（协商基础设施失败）
    int gatherCnt = 0;                   // 调用计数
    // 保真回填的槽位内覆盖项（载荷内偏移, 覆盖字节），应用于每个 rank 槽位；
    // 空列表 = 全域同源（每个 rank 槽位均为本端载荷副本），覆盖项经 NegCAddGatherOverride 追加
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> gatherOverrides;

    /* ---- 调用观测 ---- */
    NegotiateOpC lastOp = NEGOTIATE_OP_ALL_REDUCE_MIN; // 最近一次调用的协商算子
    std::string lastTypeTag;                           // 最近一次调用的 typeTag（nullptr 记为空串）
};

extern NegCStubConfig g_negStub;

// 重置桩配置（消费方自身 Reset 函数内调用，保持各 UT 复位语义不变）
void NegCStubReset();

// 追加 ALL_GATHER 槽位字段覆盖：将 value 的字节写入每个 rank 载荷的 offset 处
template <typename T>
inline void NegCAddGatherOverride(NegCStubConfig& cfg, uint64_t offset, T value)
{
    const auto* p = reinterpret_cast<const uint8_t*>(&value);
    cfg.gatherOverrides.push_back({offset, std::vector<uint8_t>(p, p + sizeof(T))});
}
} // namespace ut

#endif // UT_COMM_NEGOTIATION_C_STUB_H
