/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_CCU_FALLBACK_STUB_H
#define UT_CCU_FALLBACK_STUB_H

#include <cstddef>
#include <cstdint>
#include <string>
#include "op_common.h"  // OpExecuteConfig / FallbackCtxData（缓存数据布局）
#include "neg_c_stub.h" // 公共协商 C 接口共享桩（协商注入/观测字段统一在 g_negStub）

namespace ut {
// 与 ccu_fallback.cc 内部常量锁定：协商 typeTag、参数协商缓存 key 后缀与载荷 magic
constexpr char CCU_NEGOTIATION_TAG[] = "ccu";
constexpr char CCU_PARAM_CACHE_SUFFIX[] = "_ccuParamCache";
constexpr uint32_t CCU_PARAM_MAGIC_LOCK = 0x5a5a5a5a;

// 与 ccu_fallback.cc 内 static CheckParamInfo 逐字段锁定（UT 锚定协商载荷 ABI）；
// ALL_GATHER 槽位覆盖的载荷内偏移经 offsetof 引用，由共享桩按 (offset, bytes) 回填
struct CheckParamInfoAbi {
    uint64_t count;
    uint32_t opExecuteConfig;
    uint32_t dataType;
    uint32_t opType;
    uint32_t magic;
};
static_assert(sizeof(CheckParamInfoAbi) == 24, "CheckParamInfo ABI locked");

// 桩态配置：用例在 SetUp 复位后按需覆写单个字段
struct CcuStubConfig {
    /* ---- HcclGetRankSize 桩 ---- */
    uint32_t rankSize = 8;

    /* ---- IsCommExpansionModeCcu 桩（经 DlHcommFunction 单例 dlHcclConfigGetInfo 注入） ---- */
    uint32_t expansionMode = HCCL_OP_EXPANSION_EXT_CCU_MS; // 默认 ccu 模式

    /* ---- 回退动作桩（ReSelector / SetCommEngine / SetOpParamAlgTag / 一致性检视） ---- */
    int reSelectorCnt = 0;
    ops_hccl::OpExecuteConfig lastReSelectorConfig = ops_hccl::OpExecuteConfig::DEFAULT;
    int setCommEngineCnt = 0;
    int setAlgTagCnt = 0;
    int inconsistentCheckCnt = 0;

    /* ---- HcclEngineCtx 桩观测 ---- */
    int engineCtxCreateCnt = 0;
};

extern CcuStubConfig g_ccuStub;

// 重置桩配置与 EngineCtx KV，并复位 DlHcommFunction 单例成员（每个用例 SetUp 调用）
void CcuStubReset();

// 观测：指定 (comm, ctxTag) 的 EngineCtx 条目是否存在
bool CcuStubHasCtx(HcclComm comm, const std::string& tag);

// 观测：读取 EngineCtx 条目内的 FallbackCtxData（条目不存在或尺寸不足返回 false）
bool CcuStubGetCtx(HcclComm comm, const std::string& tag, ops_hccl::FallbackCtxData& out);

// 预置参数协商结果缓存（cache-hit 用例）：按 SUT 读取的 KV 语义写入回退终态
HcclResult
CcuStubPresetCtx(HcclComm comm, const std::string& tag, const std::string& algName, ops_hccl::OpExecuteConfig cfg);
} // namespace ut

#endif
