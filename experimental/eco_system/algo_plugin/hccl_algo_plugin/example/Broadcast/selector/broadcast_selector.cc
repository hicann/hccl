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
 * 验证场景二：【一个算法 对应 一个实现so】
 * 本文件演示Broadcast算子下仅注册一个自定义算法（BroadcastAlgoTree），
 * 其soPath指向独立的 libBroadcastCustomAlgoImpl.so（与AllReduce示例中的so完全不同）。
 *
 * 与AllReduce示例合起来看，同时验证：
 *   - 验证场景一（多算法对应一个so）：见 ../../AllReduce/
 *   - 验证场景二（一算法对应一个so）：本文件
 *   - 验证场景三（不同算子分别正确路由到各自的selector/so，互不干扰）：
 *     PluginBroker需要根据opType=Broadcast定位到本目录而不是AllReduce目录。
 *
 * 编译产物：libhccl_plugin_broadcast_selector.so
 * 部署路径：${HCCL_PLUGIN_ALG_DIR}/Broadcast/libhccl_plugin_broadcast_selector.so
 * ============================================================================================
 */

#include <cstdio>
#include "hccl_algo_plugin_sdk.h"

namespace {
constexpr const char* kBroadcastCustomAlgosImplSoFile = "libBroadcastCustomAlgoImpl.so";
}

/* 仅注册一个算法，且独占一个实现so，验证"1个算法:1个so"的注册/查找场景 */
REGISTER_HCCL_ALGO("BroadcastAlgoTree", kBroadcastCustomAlgosImplSoFile, "HcclAlgoPluginBroadcastTree");

/*
 * 选择策略示例：只要root为0就命中自定义算法，其余场景交由HCCL原有逻辑处理，
 * 用于验证"未命中时正确回退"这条路径同时对Broadcast算子也生效。
 */
extern "C" __attribute__((visibility("default"))) bool
Select(const HcclAlgoPluginParam* param, char* algName, size_t algNameLen)
{
    if (param == nullptr || algName == nullptr || algNameLen == 0) {
        return false;
    }
    if (param->magic != HCCL_ALGO_PLUGIN_PARAM_MAGIC) {
        std::fprintf(stderr, "[BroadcastSelector][Select] bad param magic, refuse to select.\n");
        return false;
    }

    if (param->root == 0) {
        if (!HcclAlgoPluginCopyString(algName, algNameLen, "BroadcastAlgoTree")) {
            return false;
        }
        std::fprintf(stderr, "[BroadcastSelector][Select] root=0, hit=BroadcastAlgoTree\n");
        return true;
    }
    std::fprintf(
        stderr, "[BroadcastSelector][Select] root=%u, not hit, fallback to HCCL original logic\n", param->root);
    return false;
}
