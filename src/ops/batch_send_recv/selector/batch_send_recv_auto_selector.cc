/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batch_send_recv_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {

SelectorStatus BatchSendRecvAutoSelector::SelectAicpuAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                                      const OpParam &opParam,
                                                      const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                      std::string &selectAlgName) const
{
    HCCL_DEBUG("[BatchSendRecvAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)opParam;
    (void)configAlgMap;

    selectAlgName = "InsBatchSendRecv";
    HCCL_DEBUG("[BatchSendRecvAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus BatchSendRecvAutoSelector::SelectDPUAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                                      const OpParam &opParam,
                                                      const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                      std::string &selectAlgName) const
{
    HCCL_DEBUG("[BatchSendRecvAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)opParam;
    (void)configAlgMap;

    selectAlgName = "InsBatchSendRecvDPU";
    HCCL_DEBUG("[BatchSendRecvAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, 18, BatchSendRecvAutoSelector);
} // namespace ops_hccl
