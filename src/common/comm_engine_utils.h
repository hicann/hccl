/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_ENGINE_UTILS_H
#define COMM_ENGINE_UTILS_H

#include <string>
#include <unordered_map>
#include "hccl_res.h"

template <typename MapType>
inline std::string GetEnumToString(const MapType& mappingMap, typename MapType::key_type enumValue)
{
    auto iter = mappingMap.find(enumValue);
    if (iter != mappingMap.end()) {
        return iter->second;
    }
    return "Unknown";
}

inline const std::unordered_map<CommEngine, std::string>& GetCommEngineStatusStrMap()
{
    static const std::unordered_map<CommEngine, std::string> COMMENGINE_STATUS_STR_MAP{
        {COMM_ENGINE_RESERVED, "RESERVED"},
        {COMM_ENGINE_CPU, "CPU"},
        {COMM_ENGINE_CPU_TS, "CPU_TS"},
        {COMM_ENGINE_AICPU, "AICPU"},
        {COMM_ENGINE_AICPU_TS, "AICPU_TS"},
        {COMM_ENGINE_AIV, "AIV"},
        {COMM_ENGINE_CCU, "CCU"}};
    return COMMENGINE_STATUS_STR_MAP;
}

#endif
