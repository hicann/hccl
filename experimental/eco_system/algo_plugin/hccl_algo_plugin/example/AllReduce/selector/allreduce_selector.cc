/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * ============================================================================================
 * 验证场景一：【多个算法 对应 一个实现so】
 * 本文件演示AllReduce算子下注册两个自定义算法（AllReduceAlgoSmall / AllReduceAlgoLarge），
 * 它们的 soPath 均指向同一个 libAllReduceCustomAlgosImpl.so，仅 fnSymbol 不同。
 * PluginBroker::ExecuteAlg() 命中任一算法时，都应从这同一个so里dlsym出各自不同的执行函数。
 *
 * 本示例的目的仅为验证"插件框架本身的注册/选择/派发链路是否正确"，
 * 不代表真实可用的AllReduce算法实现，因此执行函数内部只做打桩（打印日志+返回成功）。
 *
 * 编译产物：libhccl_plugin_allreduce_selector.so
 * 部署路径：${HCCL_PLUGIN_ALG_DIR}/AllReduce/libhccl_plugin_allreduce_selector.so
 * ============================================================================================
 */

#include <cstdio>
#include "hccl_algo_plugin_sdk.h"

namespace {
/* 两个算法共用同一个实现so，验证"N个算法:1个so"的注册/查找场景 */
constexpr const char* kAllReduceCustomAlgosImplSoFile = "libAllReduceCustomAlgosImpl.so";

/* 简单dataType到元素字节数的映射，仅用于Select()阶段估算数据量 */
uint64_t DataTypeSizeHint(int dataType)
{
    switch (dataType) {
        case 0:
            return 1; // int8
        case 1:
            return 2; // int16
        case 2:
            return 4; // int32
        case 3:
            return 2; // float16
        case 4:
            return 4; // float32
        case 5:
            return 8; // int64
        case 6:
            return 8; // uint64
        default:
            return 4;
    }
}
} // namespace

/* 算法1：小数据量场景命中，执行函数符号名 HcclAlgoPluginAllReduceSmall，位于kAllReduceCustomAlgosImplSoFile */
REGISTER_HCCL_ALGO("AllReduceAlgoSmall", kAllReduceCustomAlgosImplSoFile, "HcclAlgoPluginAllReduceSmall");

/* 算法2：大数据量场景命中，执行函数符号名 HcclAlgoPluginAllReduceLarge，与算法1位于【同一个】so */
REGISTER_HCCL_ALGO("AllReduceAlgoLarge", kAllReduceCustomAlgosImplSoFile, "HcclAlgoPluginAllReduceLarge");

/*
 * 选择策略示例（仅用于演示两个算法如何被区分命中，不代表真实调优结论）：
 *   数据量 < 1MB  -> AllReduceAlgoSmall
 *   数据量 >= 1MB -> AllReduceAlgoLarge
 */
extern "C" __attribute__((visibility("default"))) bool
Select(const HcclAlgoPluginParam* param, char* algName, size_t algNameLen)
{
    if (param == nullptr || algName == nullptr || algNameLen == 0) {
        return false;
    }
    if (param->magic != HCCL_ALGO_PLUGIN_PARAM_MAGIC) {
        std::fprintf(stderr, "[AllReduceSelector][Select] bad param magic, refuse to select.\n");
        return false;
    }

    uint64_t totalBytes = param->count * DataTypeSizeHint(static_cast<int>(param->dataType));
    constexpr uint64_t kSizeThreshold = 1ULL << 20; // 1MB

    const char* hit = (totalBytes < kSizeThreshold) ? "AllReduceAlgoSmall" : "AllReduceAlgoLarge";
    if (!HcclAlgoPluginCopyString(algName, algNameLen, hit)) {
        return false;
    }
    std::fprintf(
        stderr, "[AllReduceSelector][Select] totalBytes=%llu, hit=%s\n", static_cast<unsigned long long>(totalBytes),
        hit);
    return true;
}
