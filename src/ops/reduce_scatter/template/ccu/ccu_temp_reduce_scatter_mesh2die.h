/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_REDUCE_SCATTER_MESH_2DIE_H
#define HCCL_CCU_TEMP_REDUCE_SCATTER_MESH_2DIE_H

#include "utils.h"
#include "ccu_alg_template_base.h"

namespace ops_hccl {
using RankId = u32;
using RankGroup = std::vector<RankId>;
class CcuTempReduceScatterMesh2Die : public CcuAlgTemplateBase{
public:
    explicit CcuTempReduceScatterMesh2Die(const OpParam &param, RankId rankId,
    const std::vector<std::vector<u32>> &subCommRanks);
    ~CcuTempReduceScatterMesh2Die() override;

    std::string Describe() const override
    {
        return StringFormat("Template of ReduceScatter ccu mesh 2Die with rankSize[%u]", templateRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        AlgResourceRequest &resourceRequest) override;

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
        const TemplateResource &templateResource) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
private:
    HcclResult PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs);

    const uint32_t DIE_NUM = 2; // 2Die
    const uint64_t SCRATCH_MULTIPLE_COUNT_2 = 2;
    std::map<uint32_t, RankGroup> rankGroup_;
    std::map<uint32_t, std::vector<HcclChannelDesc>> channels_; // key is DieId
    uint32_t mySubCommRank_ = 0;
};
}// namespace ops_hccl

#endif //HCCL_CCU_TEMP_REDUCE_SCATTER_MESH_2DIE_H