/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_TEMP_ALL_GATHER_MESH_1D_Z_AXIS_DETOUR_H
#define AIV_TEMP_ALL_GATHER_MESH_1D_Z_AXIS_DETOUR_H

#include "aiv_temp_all_gather_mesh_1D.h"

namespace ops_hccl {

class AivTempAllGatherMesh1DZAxisDetour : public AivTempAllGatherMesh1D {
public:
    AivTempAllGatherMesh1DZAxisDetour() = default;
    explicit AivTempAllGatherMesh1DZAxisDetour(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~AivTempAllGatherMesh1DZAxisDetour() override;

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);
};

} // namespace ops_hccl

#endif // AIV_TEMP_ALL_GATHER_MESH_1D_Z_AXIS_DETOUR_H
