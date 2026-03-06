/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef HCCL_CCU_TEMP_ALL_REDUCE_MESH_1D_MEM2MEM_2DIE_ONESHOT_H
#define HCCL_CCU_TEMP_ALL_REDUCE_MESH_1D_MEM2MEM_2DIE_ONESHOT_H
 
#include "ccu_alg_template_base.h"
 
namespace ops_hccl {
 
class CcuTempAllReduceMesh1DMem2Mem2DieOneShot : public CcuAlgTemplateBase {
public:
    explicit  CcuTempAllReduceMesh1DMem2Mem2DieOneShot(const OpParam& param,
                                                const u32 rankId, // 传通信域的rankId，userRank
                                                const std::vector<std::vector<u32>> &subCommRanks);
 
    ~CcuTempAllReduceMesh1DMem2Mem2DieOneShot() override;
 
    std::string Describe() const override
    {
        return StringFormat("Template of All Reduce ccu mesh 1D Mem2Mem 2Die OneShot with tempRankSize [%u].",
                            subCommRanks_[0].size());
    }
 
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
 
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         const TemplateResource& templateResource) override;
 
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
 
private:
    uint32_t mySubCommRank_ = 0; // 本卡在子通信域的rankId
};
 
}// namespace ops_hccl
 
#endif// HCCL_CCU_TEMP_ALL_REDUCE_MESH_1D_MEM2MEM_2DIE_ONESHOT_H