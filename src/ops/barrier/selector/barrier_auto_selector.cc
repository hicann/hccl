/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "barrier_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {

// Barrier 新流程支持 HostDPU 和 AICPU 两种引擎：
// - HostDPU 场景：Select() 内部 CheckHostDPUOnly → SelectDPUAlgo → InsBarrierMeshNhrDPU
// - AICPU 场景：Select() 内部 IsStarsState → SelectAicpuAlgo → InsBarrierNhrAicpu
// 其余引擎（CCU/AIV）在 BarrierOutPlace 中已提前回退到旧 HcclBarrier，不会进入算法选择。
SelectorStatus BarrierAutoSelector::SelectDPUAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    (void)opParam;
    (void)configAlgMap;
    HCCL_DEBUG("[BarrierAutoSelector][%s] start, topoLevelNums[%u]", __func__, topoInfo->topoLevelNums);
    selectAlgName = "InsBarrierMeshNhrDPU";
    HCCL_INFO("[BarrierAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus BarrierAutoSelector::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    (void)opParam;
    (void)configAlgMap;
    HCCL_INFO("[BarrierAutoSelector][SelectAicpuAlgo] topoLevelNums[%u], level0Topo[%u]",
        topoInfo->topoLevelNums, topoInfo->level0Topo);
    selectAlgName = "InsBarrierNhrAicpu";
    HCCL_INFO("[BarrierAutoSelector][SelectAicpuAlgo] Algo match[%s]", selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_BARRIER, 18, BarrierAutoSelector);

}  // namespace ops_hccl
