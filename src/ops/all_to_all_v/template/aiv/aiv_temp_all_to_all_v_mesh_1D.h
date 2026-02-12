/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_TEMP_ALL_TO_ALL_V_MESH_1D
#define AIV_TEMP_ALL_TO_ALL_V_MESH_1D

#include <cstring>
#include "aiv_alg_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class AivTempAlltoAllVMesh1D : public AivAlgTemplateBase {
public:
    explicit AivTempAlltoAllVMesh1D(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                        const std::vector<std::vector<u32>> &subCommRanks);
    ~AivTempAlltoAllVMesh1D() override;

    std::string Describe() const override
    {
        std::string info = "Template of all to all v Mesh with tempRankSize ";
        info += std::to_string(tempRankSize_);
        return info;
    }
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                        AlgResourceRequest& resourceRequest) override;
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& tempAlgParams,
                         const TemplateResource& templateResource) override;
    HcclResult CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit) override;
};
}  // namespace Hccl

#endif  // AIV_TEMP_ALL_TO_ALL_MESH_1D