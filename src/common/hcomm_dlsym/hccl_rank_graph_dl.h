/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RANK_GRAPH_DL_H
#define HCCL_RANK_GRAPH_DL_H

#include "dlsym_common.h"
#include "hccl_rank_graph.h"   // 原头文件，包含所有类型和 inline 函数

#ifdef __cplusplus
extern "C" {
#endif

void HcclRankGraphDlInit(void* libHcommHandle);

#ifdef __cplusplus
}
#endif

#endif // HCCL_RANK_GRAPH_DL_H