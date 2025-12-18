/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_ADAPTER_ERROR_MANAGER_PUB_H
#define HCCL_INC_ADAPTER_ERROR_MANAGER_PUB_H

#include "log.h"
#include <string>
#include <vector> 

__attribute__((weak)) void RptInputErr(std::string error_code, std::vector<std::string> key,
    std::vector<std::string> value);
__attribute__((weak)) void RptEnvErr(std::string error_code, std::vector<std::string> key,
    std::vector<std::string> value);

#define RPT_INPUT_ERR(result, error_code, key, value) do { \
    if (UNLIKELY(result) && RptInputErr != nullptr) {     \
        RptInputErr(error_code, key, value);          \
    }                                                       \
} while (0)

#define RPT_ENV_ERR(result, error_code, key, value) do { \
    if (UNLIKELY(result) && RptEnvErr != nullptr) {                               \
        RptEnvErr(error_code, key, value);        \
    }                                                    \
} while (0)

#endif