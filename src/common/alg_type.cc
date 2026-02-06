/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_type.h"

namespace ops_hccl {

template<typename keyType>
std::string GetAlgoString(const std::map<keyType, std::string>& levelMap, keyType key) {
    auto iter = levelMap.find(key);
    if (iter == levelMap.end()) {
        return "invalid algo type";
    } else {
        return iter->second;
    }
}

std::string AlgTypeToStr(const AlgType algType)
{
    AlgTypeLevel0 algTypeLevel0 = algType.algoLevel0;
    AlgTypeLevel1 algTypeLevel1 = algType.algoLevel1;
    AlgTypeLevel2 algTypeLevel2 = algType.algoLevel2;
    std::string algStrLevel0 = GetAlgoString(HCCL_ALGO_LEVEL0_NAME_MAP, algTypeLevel0);
    std::string algStrLevel1 = GetAlgoString(HCCL_ALGO_LEVEL1_NAME_MAP, algTypeLevel1);
    std::string algStrLevel2 = GetAlgoString(HCCL_ALGO_LEVEL2_NAME_MAP, algTypeLevel2);
    std::string algStr;
    algStr.append("level0:").append(algStrLevel0).append(",level1:").append(algStrLevel1).append(",level2:").append(algStrLevel2);
    return algStr;
}

std::string TransferAlgTypeStr(AlgType algType)
{
    std::string algTypeStr;
    auto level0Iter = HCCL_ALGO_LEVEL0_NAME_MAP.find(algType.algoLevel0);
    auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType.algoLevel1);
    auto level2Iter = HCCL_ALGO_LEVEL2_NAME_MAP.find(algType.algoLevel2);
    
    if (level0Iter == HCCL_ALGO_LEVEL0_NAME_MAP.end() || 
        level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end() || 
        level2Iter == HCCL_ALGO_LEVEL2_NAME_MAP.end()) {
        algTypeStr = "not found";
    } else {
        // 将枚举值转换为整数进行拼接
        algTypeStr.append(std::to_string(static_cast<int>(algType.algoLevel0)))
                  .append("-")
                  .append(std::to_string(static_cast<int>(algType.algoLevel1)))
                  .append("-")
                  .append(std::to_string(static_cast<int>(algType.algoLevel2)));
    }
    return algTypeStr;
}

std::string TransferAlgType(AlgType algType)
{
    std::string algTypeStr;
    auto level0Iter = HCCL_ALGO_LEVEL0_NAME_MAP.find(algType.algoLevel0);
    auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType.algoLevel1);
    auto level2Iter = HCCL_ALGO_LEVEL2_NAME_MAP.find(algType.algoLevel2);
    
    if (level0Iter == HCCL_ALGO_LEVEL0_NAME_MAP.end() || 
        level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end() || 
        level2Iter == HCCL_ALGO_LEVEL2_NAME_MAP.end()) {
        algTypeStr = "not found";
    } else {
        // 将枚举值转换为整数进行拼接
        algTypeStr.append(static_cast<std::string>(level0Iter->second))
                  .append("-")
                  .append(static_cast<std::string>(level1Iter->second))
                  .append("-")
                  .append(static_cast<std::string>(level2Iter->second));
    }
    return algTypeStr;
}
}