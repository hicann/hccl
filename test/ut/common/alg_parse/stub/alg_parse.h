/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_COMMON_ALG_PARSE
#define OPS_HCCL_SRC_COMMON_ALG_PARSE

#include <string>
#include <vector>
#include <hccl/hccl_types.h>
#include "log.h"

namespace ops_hccl {
#define HCCL_INFO(format, ...) fprintf(stderr, "[INFO] " format "\n", ##__VA_ARGS__)
#define HCCL_ERROR(format, ...) fprintf(stderr, "[ERROR] " format "\n", ##__VA_ARGS__)
#define HCCL_WARNING(format, ...) fprintf(stderr, "[WARNING] " format "\n", ##__VA_ARGS__)
#define HCCL_DEBUG(format, ...) fprintf(stderr, "[DEBUG] " format "\n", ##__VA_ARGS__)

// mock
HcclResult static HcclGetHcclAlgo(HcclComm comm, std::string& hcclAlgo) { return HCCL_SUCCESS; }

std::string static GetEnv(std::string IdName) { return ""; }

struct HcclAlgo {
    std::string algoType; // templateType 名称
    bool enable = true;   // 是否启用该算法
};

struct HcclAlgoExecutor {
    std::string opType;             // OpType 为空 = 全局
    std::string executorType;       // executorType 名称
    std::vector<HcclAlgo> algoList; // 按 level 升序存储
    bool enable = true;             // 是否启用该算子
};

struct HcclAlgoParser {
    std::vector<HcclAlgoExecutor> executorList;
    HcclResult Parser(const std::string& algoConfig);
    // 调试用
    std::string ToString() const;
};

// CostModel 外部结构体（由其他模块传递，此处仅声明以便编译）
// ---------------------------------------------------------------------------
typedef struct {
    float A; // 用来描述跨卡传输的时间随DataSize变化的趋势，会受到UB带宽利用率的影响
    float B; // 用来描述本地传输的时间随DataSize变化的趋势，不受到UB带宽利用率的影响
    float C; // 用来描述一些基本时延的常数项
} CostModelParam;

typedef struct {
    const char* algName;         //
    const CostModelParam* param; //
    int count;                   //
} CostAlgoParams;

typedef struct {
    CostAlgoParams* costAlgoParams;
    int count;
} CostModel;

HcclResult FilterCmByHcclAlgo(HcclComm comm, CostModel& cm);

HcclResult FilterCmByHcclAlgo(HcclComm comm, CostModel& cm, const std::vector<std::string>& candidateEngineNames);

HcclResult UpdateCostModelWithAlgo(
    const HcclAlgoParser& algoParser, CostModel& model, const std::vector<std::string>& engineTypes);

std::string UnderscoreToCamelCase(const std::string& name);

struct AlgoDimEntry {
    const char* key;
    const char* pascal;
};

// 从 map 派生的数组访问接口（首次调用时构建，线程安全）
const AlgoDimEntry* GetAlgoEngines(int& count);
const AlgoDimEntry* GetAlgoExecutors(int& count);
const AlgoDimEntry* GetAlgoTemplates(int& count);

HcclResult static SetHcclAlgoConfig(const std::string& config) { return HCCL_SUCCESS; }
} // namespace ops_hccl

#endif // OPS_HCCL_SRC_COMMON_ALG_PARSE
