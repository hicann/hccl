/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLGATHER_MESH_TEMPLATE_H
#define ALLGATHER_MESH_TEMPLATE_H

#include "aicpu_base_template.h"

namespace ops_hccl {

// AllGather Mesh 模板
class AllGatherMeshTemplate : public AicpuBaseTemplate {
public:
    AllGatherMeshTemplate(u32 myRank, std::vector<u32> ranks, TemplateDesc templateDesc)
        : AicpuBaseTemplate(myRank, std::move(ranks), std::move(templateDesc))
    {}
    ~AllGatherMeshTemplate() = default;

protected:
    HcclResult DoCalcChannelRequest(
        HcclComm comm, const OpParam& param, TopoInfoWithNetLayerDetails* topoInfo,
        const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& levelChannels) override;
    u32 DoCalcThreadNum() const override;
    u32 DoCalcNotifyPerThread() const override;

    HcclResult
    RunAlgorithm(std::vector<DataSlicesList>& txRxSlicesLists, std::vector<u32>& ranksForOutputData) override;

    HcclResult SendAll(
        const std::vector<DataSlicesList>& txRxSlicesLists, TemplateResource& templateResource,
        const std::vector<ThreadHandle>& threads) override;

    HcclResult PostCopy(const std::vector<ThreadHandle>& threads) override;
};

} // namespace ops_hccl

#endif // ALLGATHER_MESH_TEMPLATE_H
