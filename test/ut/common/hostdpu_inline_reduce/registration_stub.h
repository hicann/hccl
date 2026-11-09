/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HOSTDPU_REGISTRATION_STUB_H
#define HOSTDPU_REGISTRATION_STUB_H

#include "coll_alg_v2_exec_registry.h"

namespace inline_reduce_ut {
// Compile the production registration and ALG_ATTRS unchanged. Replace only
// executor creation: selection tests calibrate cost without allocating devices.
class CostExecutor : public ops_hccl::InsCollAlgBase {
public:
    std::vector<ops_hccl::CostModelParam>
    CalcCostCoeff(HcclComm, ops_hccl::TopoInfoWithNetLayerDetails*, const char*, const ops_hccl::OpParam&) override
    {
        return {{0, 0, 1, 0}};
    }
    HcclResult CalcAlgHierarchyInfoV2(
        ops_hccl::TopoInfoWithNetLayerDetails*, ops_hccl::AlgHierarchyInfoForAllLevel&,
        const ops_hccl::AlgAttrs&) override
    {
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult CalcRes(
        HcclComm, const ops_hccl::OpParam&, const ops_hccl::TopoInfoWithNetLayerDetails*,
        const ops_hccl::AlgHierarchyInfoForAllLevel&, ops_hccl::AlgResourceRequest&) override
    {
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult Orchestrate(const ops_hccl::OpParam&, const ops_hccl::AlgResourceCtxSerializable&) override
    {
        return HCCL_E_NOT_SUPPORT;
    }
};
template <typename T>
ops_hccl::InsCollAlgBase* MakeCostExecutor()
{
    return new CostExecutor;
}
} // namespace inline_reduce_ut

#define DefaultExecCreatorV2 inline_reduce_ut::MakeCostExecutor
#endif
