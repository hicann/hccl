/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor_base.h"
#include "workflow.h"
namespace ops_hccl {

ExecutorBase::ExecutorBase() : workflowMode_(GetWorkflowMode())
{
}

HcclResult ExecutorBase::CalcResRequest(HcclComm comm, const OpParam& param,
    TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    AlgResourceRequest& resourceRequest, AlgType& algType)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)algHierarchyInfo;
    (void)resourceRequest;
    (void)algType;
    return HCCL_SUCCESS;
}

HcclResult ExecutorBase::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_WARNING("[ExecutorBase][KernelRun]Using the default kernel run, nothing is done.");
    return HCCL_SUCCESS;
}

HcclResult ExecutorBase::GetSubCommInfo(const CommPlane levelIndex, SubCommInfo &info)
{
    AlgHierarchyInfo& algHierarchyInfo = algResource_->algHierarchyInfo;
    if (levelIndex >= algHierarchyInfo.levels) {
        HCCL_ERROR("[ExecutorBase][GetSubCommInfo]tag[%s], levelIndex[%u] exceeds actual levels[%u]",
            tag_.c_str(), levelIndex, algHierarchyInfo.levels);
        return HCCL_E_INTERNAL;
    }

    info = algHierarchyInfo.infos[levelIndex];
    return HCCL_SUCCESS;
}

HcclResult ExecutorBase::RefreshAlgType(AlgType& algType)
{
    const std::vector<AlgTypeLevel0> &l0Algo = desc_.level0SupportedAlgos;
    const std::vector<AlgTypeLevel1> &l1Algo = desc_.level1SupportedAlgos;
    const std::vector<AlgTypeLevel2> &l2Algo = desc_.level2SupportedAlgos;

    if (!l0Algo.empty()
        && std::find(l0Algo.begin(), l0Algo.end(), algType.algoLevel0) == l0Algo.end()) {
        HCCL_WARNING("[%s] not support level0 algo[%d], reset to algo[%d]", __func__,
            algType.algoLevel0, l0Algo[0]);
        algType.algoLevel0 = l0Algo[0];
    }
    if (!l1Algo.empty()
        && std::find(l1Algo.begin(), l1Algo.end(), algType.algoLevel1) == l1Algo.end()) {
        HCCL_WARNING("[%s] not support level1 algo[%d], reset to algo[%d]", __func__,
            algType.algoLevel1, l1Algo[0]);
        algType.algoLevel1 = l1Algo[0];
    }
    if (!l2Algo.empty()
        && std::find(l2Algo.begin(), l2Algo.end(), algType.algoLevel2) == l2Algo.end()) {
        HCCL_WARNING("[%s] not support level2 algo[%d], reset to algo[%d]", __func__,
            algType.algoLevel2, l2Algo[0]);
        algType.algoLevel2 = l2Algo[0];
    }
    return HCCL_SUCCESS;
}
}