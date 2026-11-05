/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_TEMPLATE_SELECTOR_TEST_TEMPLATE_UTILS_H
#define AIV_TEMPLATE_SELECTOR_TEST_TEMPLATE_UTILS_H

#include "alg_param.h"

namespace ops_hccl {

inline bool
IsAllConnetedWithTopo(const TopoInfoWithNetLayerDetails* topoInfo, const u32 netLayer, const CommTopo topoType)
{
    // 仅模拟第0层是否全Mesh连通，供各算子的语义选择回归使用。
    return netLayer == 0 && topoType == CommTopo::COMM_TOPO_1DMESH && topoInfo->level0AllMesh;
}

} // namespace ops_hccl

#endif // AIV_TEMPLATE_SELECTOR_TEST_TEMPLATE_UTILS_H
