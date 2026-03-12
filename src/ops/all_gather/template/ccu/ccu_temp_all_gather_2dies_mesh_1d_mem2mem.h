/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCL_CCU_TEMP_ALL_GATHER_2DIES_MESH_1D_MEM2MEM_H
#define HCCL_CCU_TEMP_ALL_GATHER_2DIES_MESH_1D_MEM2MEM_H

#include "ccu_alg_template_base.h"

namespace ops_hccl {
class CcuTempAllGather2DiesMeshMem2Mem1D : public CcuAlgTemplateBase {
public:
    explicit  CcuTempAllGather2DiesMeshMem2Mem1D(const OpParam& param, 
                                                const u32 rankId,
                                                const std::vector<std::vector<u32>> &subCommRanks);
 
    ~CcuTempAllGather2DiesMeshMem2Mem1D() override;
 
    std::string Describe() const override
    {
        return StringFormat("Template of all gather ccu sche mesh 1D Mem2Mem with tempRankSize [%u].",
                            subCommRanks_[0].size());
    }
 
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
 
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         const TemplateResource& templateResource) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
private:
    uint32_t mySubCommRank_ = 0;
};

}// namespace ops_hccl

#endif