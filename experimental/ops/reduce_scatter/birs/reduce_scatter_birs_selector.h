/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_HCCL_EXPERIMENTAL_REDUCE_SCATTER_BIRS_SELECTOR
#define OPS_HCCL_EXPERIMENTAL_REDUCE_SCATTER_BIRS_SELECTOR

#include <string>
#include "alg_param.h"

namespace ops_hccl_experimental {
enum class BirsSelectResult {
    kNotSelected,
    kSelected,
    kRejectRankSizeOne,
    kRejectServerNumZero,
    kRejectRanksPerServerLT4,
};

BirsSelectResult DecideReduceScatterBirsAlg(const ops_hccl::TopoInfo& topoInfo, std::string& algName);

HcclResult BirsSelectResultToCode(BirsSelectResult result);
} // namespace ops_hccl_experimental

#endif
