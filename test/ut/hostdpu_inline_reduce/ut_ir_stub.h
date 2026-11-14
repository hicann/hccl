/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_HOSTDPU_INLINE_REDUCE_STUB_H
#define UT_HOSTDPU_INLINE_REDUCE_STUB_H

#include <cstdint>
#include <set>
#include <string>
#include "hccl.h"
#include "hccl_rank_graph.h" // EndpointDesc（与HCOMM侧hcomm_res_defs.h同布局的公共结构体）
#include "neg_c_stub.h"      // 公共协商 C 接口共享桩（协商注入/观测字段统一在 g_negStub）

/*
 * hostDPU inline reduce 能力协商 UT 桩配置（与公共协商模块UT test/ut/comm_negotiation 解耦）：
 * - comm_negotiation（IsNegotiationComm C++本体 / NegotiateCommC）由共享桩打桩
 *   （test/ut/common/comm_negotiation_stub），不链接公共协商模块实现；
 * - dlsym 层真实源码（hcomm_primitives_dl.cc）参与编译，符号可用性经 HcclDlsym 桩 +
 *   HcommPrimitivesDlInit 控制，HcommEndpointCheckFeature 以强符号桩覆盖弱默认实现；
 * - HcclEngineCtx* 用 (comm, tag) 内存 KV 保真实现，语义与 HCOMM 一致（含 HCCL_E_NOT_FOUND）。
 */
namespace ut {
// 能力缓存三态值：与 hostdpu_inline_reduce.cc 内 InlineReduceCap 逐值锁定（UT 锚定缓存ABI）；
// 取值排布 fail-safe：0 必须是不支持语义（EngineCtx 底层 calloc 清零，防清零默认值假启用）
constexpr int8_t CAP_NEGOTIATION_DEGRADED = 0;  // 协商失败，按不支持降级
constexpr int8_t CAP_SOME_RANK_UNSUPPORTED = 1; // 协商成功，存在不支持的rank
constexpr int8_t CAP_ALL_RANKS_SUPPORTED = 2;   // 协商成功，全域支持

// SUT 内部缓存 tag（与 hostdpu_inline_reduce.cc 内 hostDpuInlineCapTag 锁定）
constexpr char HOSTDPU_INLINE_CAP_TAG[] = "hostdpu_inline_reduce_cap";

struct IrStubConfig {
    /* ---- 图模式捕获态桩（IsStreamInCaptureMode，真实定义在 op_common.cc，本 UT 不编译该文件） ---- */
    bool streamInCapture = false; // IsStreamInCaptureMode 返回值（true=录制态，协商跳过且不写缓存）

    /* ---- dlsym 层桩 ---- */
    // HcclDlsym 桩按集合返回非空句柄；改动后需重新调用 HcommPrimitivesDlInit 生效
    std::set<std::string> dlsymSymbols = {"HcommEndpointCheckFeature", "HcommWriteReduceWithNotifyOnThread"};
    // HcommEndpointCheckFeature 强符号桩（覆盖 hcomm_primitives_dl.cc 的弱默认实现）
    HcclResult checkFeatureRet = HCCL_SUCCESS;
    bool checkFeatureValue = true;
    int checkFeatureCnt = 0;
    int32_t lastFeatureType = -1;
    const EndpointDesc* lastEndpointDesc = nullptr;

    /* ---- HcclEngineCtx 桩观测/故障注入 ---- */
    bool engineCtxCreateFail = false; // true 时 HcclEngineCtxCreate 返回 HCCL_E_INTERNAL
    bool engineCtxCreateExist = false; // true 时 HcclEngineCtxCreate 返回 HCCL_E_PARA（模拟并发竞争同名已存在）
    int engineCtxCreateCnt = 0;

    /* ---- 并发时序注入（写前二次Get场景） ---- */
    int engineCtxGetCnt = 0; // HcclEngineCtxGet 调用计数（时序锚点）
    bool injectCapOnSecondGet = false; // true 时第2次Get（SUT写前二次Get）注入条目：模拟协商期间并发算子写入
    int8_t injectCapValue = CAP_ALL_RANKS_SUPPORTED; // 注入条目的能力值（默认并发者已写入 ALL_RANKS_SUPPORTED）
};

extern IrStubConfig g_irStub;

// 重置桩配置与 EngineCtx KV（每个用例 SetUp 调用）
void IrStubReset();

// 观测：comm 的 EngineCtx 中是否存在指定 tag
bool IrStubHasCtx(HcclComm comm, const std::string& tag);

// 观测：读取指定 tag 缓存首字节（三态能力值）；tag 不存在时返回 -1
int8_t IrStubGetCapValue(HcclComm comm, const std::string& tag);

// 预置能力缓存（cache-hit 用例）：按指定值写入 SUT 读取的 per-comm 缓存（含非法枚举值容错用例）
HcclResult IrStubPresetCap(HcclComm comm, int8_t capVal, uint64_t size);
} // namespace ut

#endif // UT_HOSTDPU_INLINE_REDUCE_STUB_H
