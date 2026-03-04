/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_REDUCE_MESH_2D_H
#define HCCL_CCU_TEMP_REDUCE_MESH_2D_H

#include "ccu_alg_template_base.h"
#include "utils.h"

namespace ops_hccl {

class CcuTempReduceMesh2D : public CcuAlgTemplateBase {
public:
    explicit  CcuTempReduceMesh2D(const OpParam& param,
                                                const u32 rankId, // 传通信域的rankId，userRank
                                                const std::vector<std::vector<u32>> &subCommRanks);

    ~CcuTempReduceMesh2D() override;

    std::string Describe() const override
    {
        return StringFormat("Template of Reduce ccu mesh 2D with tempRankSize [%u].",
                            subCommRanks_[0].size());
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                       AlgResourceRequest& resourceRequest) override;

    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         const TemplateResource& templateResource) override;

private:
    HcclResult PartitionChannels(const std::vector<HcclChannelDesc>& channelDescs,
                                 std::vector<HcclChannelDesc>& channels_x,
                                 std::vector<HcclChannelDesc>& channels_y);
    std::vector<uint64_t> dimSize_;
    uint32_t myRankId_   = 0; // 全局rankid
    uint32_t rootId_     = 0; // 全局rootid
};

}// namespace ops_hccl

#endif// HCCL_CCU_TEMP_REDUCE_MESH_2D__H