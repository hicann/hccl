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
#include "hccl_common.h"
#include "op_common.h"
#include "alg_env_config.h"
#include "cost_model.h"

namespace ops_hccl {

// ---------------------------------------------------------------------------
// 算法（template）条目
// algoType: 算法名称（驼峰命名），如 "mesh", "nhr", "ring", "meshMultiLink"
// enable:   是否启用；not() 取非时 enable=false
// level 由 algoList 中的位置（index）隐式表达：algoList[0]=level0, [1]=level1, ...
// ---------------------------------------------------------------------------
struct HcclAlgo {
    std::string algoType; // templateType 名称
    bool enable = true;   // 是否启用该算法
};

// ---------------------------------------------------------------------------
// 执行器（executor）条目
// opType:       为空表示全局配置（对所有 OpType 生效），否则对指定 OpType 生效
// executorType: 执行器类型名称，如 "sole", "parallel", "sequence"
// algoList:     按 level 升序存储的算法列表
// enable:       是否启用；not(executor{}) 取非时 enable=false
// ---------------------------------------------------------------------------
struct HcclAlgoExecutor {
    std::string opType;             // OpType 为空 = 全局
    std::string executorType;       // executorType 名称
    std::vector<HcclAlgo> algoList; // 按 level 升序存储
    bool enable = true;             // 是否启用该算子
};

// ---------------------------------------------------------------------------
// 解析器
// executorList: 按配置顺序存储，越靠后优先级越高
// Parser():     解析入口，结果存入 executorList
// ---------------------------------------------------------------------------
struct HcclAlgoParser {
    std::vector<HcclAlgoExecutor> executorList;
    HcclResult Parser(const std::string& algoConfig);
    // 调试用
    std::string ToString() const;
};

// 带 candidateEngineNames 重载: selector 传入候选引擎前缀(如 {"CcuMS","CcuSched","Aiv","Aicpu"})
// algo 模块据此过滤掉不在候选引擎中的算法
HcclResult FilterCmByHcclAlgo(HcclComm comm, CostModel& cm, const std::vector<std::string>& candidateEngineNames);

// ---------------------------------------------------------------------------
// 根据 HcclAlgoParser 解析结果刷新 CostModel
// 参数：
//   algoParser - 解析后的算法配置
//   model      - CostModel 结构体（输入输出）
//   engineTypes - 引擎类型列表（如 {"aicpu", "aiv", "ccu"}）
// 规则：
//   1. 反向遍历 executorList（后面的优先级高）
//   2. 按 OpType 维度匹配，已匹配的 OpType 不再参与后续匹配
//   3. 所有 OpType 都匹配成功后提前退出
//   4. enable=false 为排除算法，设置 count=0
// ---------------------------------------------------------------------------
HcclResult UpdateCostModelWithAlgo(
    const HcclAlgoParser& algoParser, CostModel& model, const std::vector<std::string>& engineTypes);

// ---------------------------------------------------------------------------
// 工具函数：下划线标识符转驼峰命名
// "mesh_multi_link" → "meshMultiLink"
// "nhr_chuck"       → "nhrChuck"
// ---------------------------------------------------------------------------
std::string UnderscoreToCamelCase(const std::string& name);

// ---------------------------------------------------------------------------
// 算法维度条目（key=user 小写名, pascal=驼峰名）
// 从 ENGINE_TYPES/EXECUTOR_TYPES/ALGO_TYPES 派生，供 selector 遍历
// ---------------------------------------------------------------------------
struct AlgoDimEntry {
    const char* key;
    const char* pascal;
};

// 从 map 派生的数组访问接口（首次调用时构建，线程安全）
const AlgoDimEntry* GetAlgoEngines(int& count);
const AlgoDimEntry* GetAlgoExecutors(int& count);
const AlgoDimEntry* GetAlgoTemplates(int& count);

} // namespace ops_hccl

#endif // OPS_HCCL_SRC_COMMON_ALG_PARSE
