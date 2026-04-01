/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef HCCL_CCU_TEMP_REDUCE_SCATTER_MESH_1D_2DIE_MEM2MEM_H
#define HCCL_CCU_TEMP_REDUCE_SCATTER_MESH_1D_2DIE_MEM2MEM_H

#include "utils.h"
#include "ccu_alg_template_base.h"
 
namespace ops_hccl {
 
class CcuTempReduceScatterMeshMem2Mem1D2Die : public CcuAlgTemplateBase {
public:
    CcuTempReduceScatterMeshMem2Mem1D2Die() = default;
    explicit  CcuTempReduceScatterMeshMem2Mem1D2Die(const OpParam& param, 
                                                const u32 rankId, // 传通信域的rankId，userRank
                                                const std::vector<std::vector<u32>> &subCommRanks);
 
    ~CcuTempReduceScatterMeshMem2Mem1D2Die() override;
 
    std::string Describe() const override
    {
        return StringFormat("Template of Reduce Scatter ccu mesh 1D 2Die Mem2Mem with tempRankSize [%u].",
                    templateRankSize_);
    }
 
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
 
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         TemplateResource& templateResource) override;
 
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
 
private:
    uint32_t localReduceOffset_ = 0;
    uint32_t mySubCommRank_ = 0; // 本卡在子通信域的rankId
};
 
}// namespace ops_hccl
 
#endif// HCCL_CCU_TEMP_REDUCE_SCATTER_MESH_1D_2DIE_MEM2MEM_H