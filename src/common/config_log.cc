/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "config_log.h"
#include "mmpa_api.h"

namespace ops_hccl {

static u64 g_debugConfig = 0ULL;

u64 GetDebugConfig()
{
    return g_debugConfig;
}

HcclResult InitDebugConfigByEnv()
{
    g_debugConfig = 0;
    char* env = nullptr; // 环境变量值
    MM_SYS_GET_ENV(MM_ENV_HCCL_DEBUG_CONFIG, env);
    if (env == nullptr) {
        HCCL_RUN_INFO("HCCL_DEBUG_CONFIG is not set, debugConfig set by default to 0x%llx", g_debugConfig);
        return HCCL_SUCCESS;
    }

    bool invert = (env[0] == '^');
    g_debugConfig = invert ? ~0ULL : 0ULL; // 第一个字符是'^', 使用取反模式, 用户配置的项关闭, 未配置的项打开
    char* configValue = (env[0] == '^') ? env + 1 : env; // 去掉'^'符号
    char* configDup = strdup(configValue); // 需要使用strdup避免修改字符串常量
    CHK_PTR_NULL(configDup);

    char* left = nullptr;
    char* subConfig = strtok_r(configDup, ",", &left); // 按逗号分割
    while (subConfig != nullptr) {
        u64 mask = 0;
        if (strcasecmp(subConfig, "ALG") == 0) {
            mask = HCCL_ALG;
        } else if (strcasecmp(subConfig, "TASK") == 0) {
            mask = HCCL_TASK;
        } else if (strcasecmp(subConfig, "RESOURCE") == 0) {
            mask = HCCL_RES;
        } else if (strcasecmp(subConfig, "AIV_OPS_EXC") == 0) {
            mask = HCCL_AIV_OPS_EXC;
        } else {
            HCCL_ERROR("HCCL_DEBUG_CONFIG:%s is invalid, subConfig:%s is not supported", env, subConfig);
            free(configDup);
            return HCCL_E_PARA;
        }
        g_debugConfig = invert ? (g_debugConfig & (~mask)) : (g_debugConfig | mask);
        subConfig = strtok_r(nullptr, ",", &left);
    }
    free(configDup);
    HCCL_RUN_INFO("HCCL_DEBUG_CONFIG[%s], set debugConfig[0x%llx]", env, g_debugConfig);
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl