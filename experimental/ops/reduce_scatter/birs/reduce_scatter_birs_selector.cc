/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_birs_selector.h"

namespace ops_hccl_experimental {
using ops_hccl::TopoInfo;

BirsSelectResult DecideReduceScatterBirsAlg(const TopoInfo& topoInfo, std::string& algName)
{
    if (topoInfo.userRankSize == 1) {
        return BirsSelectResult::kRejectRankSizeOne;
    }
    if (topoInfo.deviceType != HcclDevType::DEV_TYPE_910_93 || (topoInfo.userRankSize % 2 != 0)) {
        return BirsSelectResult::kNotSelected;
    }
    if (topoInfo.serverNum == 0) {
        return BirsSelectResult::kRejectServerNumZero;
    }
    if (topoInfo.userRankSize / topoInfo.serverNum < 4) {
        return BirsSelectResult::kRejectRanksPerServerLT4;
    }
    algName = "ReduceScatterBIRSExecutor";
    return BirsSelectResult::kSelected;
}

HcclResult BirsSelectResultToCode(BirsSelectResult result)
{
    switch (result) {
        case BirsSelectResult::kSelected:
        case BirsSelectResult::kNotSelected:
            return HCCL_SUCCESS;
        case BirsSelectResult::kRejectRankSizeOne:
            return HCCL_E_INTERNAL;
        case BirsSelectResult::kRejectServerNumZero:
        case BirsSelectResult::kRejectRanksPerServerLT4:
            return HCCL_E_PARA;
    }
    return HCCL_E_INTERNAL;
}
} // namespace ops_hccl_experimental
