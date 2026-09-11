/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adaptor_executor.h"
#include "log.h"
#include <cstdlib>
#include <string>

namespace ops_hccl {

bool IsRecursiveExecutorEnabled()
{
    constexpr bool recursiveExecutorEnabled = false; // 默认 false
    if (!recursiveExecutorEnabled) {
        return false;
    }
    const char* env = getenv("HCCL_EXPERIMENTAL_RECURSIVE_EXECUTOR");
    return env != nullptr && std::string(env) == "true";
}

HcclResult AdaptorExecutorBase::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    HcclAlgorithm alg;
    if (!AlgSelector::Instance().GetAlgorithm(algName_, alg)) {
        HCCL_ERROR("[AdaptorExecutorBase] algorithm [%s] not found for CalcAlgHierarchyInfo.", algName_.c_str());
        return HCCL_E_PARA;
    }
    if (alg.topoMatch == nullptr) {
        HCCL_ERROR("[AdaptorExecutorBase] algorithm [%s] topoMatch is nullptr.", algName_.c_str());
        return HCCL_E_PARA;
    }
    return alg.topoMatch->MatchTopo(topoInfo, algHierarchyInfo, alg.algAttrs);
}

HcclResult AdaptorExecutorBase::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    if (!executor_) {
        HcclAlgorithm alg;
        if (!AlgSelector::Instance().GetAlgorithm(algName_, alg)) {
            HCCL_ERROR("[AdaptorExecutorBase] algorithm [%s] not found.", algName_.c_str());
            return HCCL_E_PARA;
        }
        executor_ = alg.GetExecutor(param);
        CHK_RET(executor_->InitAlgHierarchyInfo(topoInfo, algHierarchyInfo));
    }
    return executor_->CalcRes(comm, resourceRequest);
}

HcclResult AdaptorExecutorBase::Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    if (!executor_) {
        HcclAlgorithm algo;
        if (!AlgSelector::Instance().GetAlgorithm(algName_, algo)) {
            HCCL_ERROR("[AdaptorExecutorBase] algorithm [%s] not found on kernel side.", algName_.c_str());
            return HCCL_E_PARA;
        }
        executor_ = algo.GetExecutor(param);
    }
    return executor_->Orchestrate(resCtx);
}

std::string AdaptorExecutorBase::Describe() const { return "AdaptorExecutorBase"; }

} // namespace ops_hccl
