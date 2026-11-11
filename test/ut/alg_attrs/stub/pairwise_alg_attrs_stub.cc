/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// UT 桩：pairwise_alg_attrs_test 的最小链接符号集。
// 被桩函数仅在 executor attrs 注册 lambda 体内或 alg_parse.cc 的环境变量/HCCL_ALGO
// 解析路径中引用，本 UT 只断言 topoCustomCheck / topoPriorityCheck，不触发对它们的
// 运行时调用，桩实现仅需满足链接。
//
// 可注入性：HcclGetHcclAlgo / GetEnv 返回值可通过 SetHcclAlgoOverride / SetEnvOverride
// 全局注入，默认空串保持"用户未指定算法 / 环境变量未设置"语义。
// 注意：HCCL_ALGO 否定 Pairwise 的选路语义发生在 cost_model.cc::ApplyTopoPriority
// 与 cost_table.cc::ConvergeHcclAlgoTier 中，这两个 TU 未编入本 UT 目标，
// 因此该高风险路径不在本 UT 自动化覆盖范围内。可注入桩为后续补充覆盖预留入口。

#include <map>
#include <string>
#include "auto_selector_base.h"
#include "hccl/hccl_res.h"
#include "op_common.h"

namespace ops_hccl {
// 可注入返回值：测试通过以下 accessor 设定 HcclGetHcclAlgo / GetEnv 的返回内容。
// 默认空串 = "用户未指定算法 / 环境变量未设置"，保持原有语义。
static std::string g_hcclAlgoOverride;
static std::map<std::string, std::string> g_envOverride;

void SetHcclAlgoOverride(const std::string& algo) { g_hcclAlgoOverride = algo; }

void SetEnvOverride(const std::string& key, const std::string& value) { g_envOverride[key] = value; }

void ResetAlgAttrsStub()
{
    g_hcclAlgoOverride.clear();
    g_envOverride.clear();
}
// CcuSched attrs 的 topoCustomCheck lambda 引用（auto_selector_base.cc 未编入本 UT 目标）
bool AutoSelectorBase::IsLayerAllConnetedWithTopo(
    const TopoInfoWithNetLayerDetails* topoInfo, const u32 netLayer, const CommTopo topoType)
{
    (void)topoInfo;
    (void)netLayer;
    (void)topoType;
    return true;
}

// alg_parse.cc 引用（定义在 op_common.cc / alg_env_config.cc，未编入本 UT 目标）
// 返回值由 g_hcclAlgoOverride 注入，默认空串 = "用户未指定算法"。
HcclResult HcclGetHcclAlgo(HcclComm comm, std::string& hcclAlgo)
{
    (void)comm;
    hcclAlgo = g_hcclAlgoOverride;
    return HCCL_SUCCESS;
}

// 返回值由 g_envOverride 注入，默认空串 = "环境变量未设置"。
std::string GetEnv(std::string idName)
{
    auto it = g_envOverride.find(idName);
    return (it != g_envOverride.end()) ? it->second : "";
}
} // namespace ops_hccl

// Aiv attrs 的 opCustomCheck lambda 引用（真实定义在 CANN 运行时，声明见 hccl/hccl_res.h）
HcclResult HcclGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size)
{
    (void)comm;
    if (buffer != nullptr) {
        *buffer = nullptr;
    }
    if (size != nullptr) {
        *size = 0;
    }
    return HCCL_SUCCESS;
}
