/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLGATHER_NHR_TEMPLATE_H
#define ALLGATHER_NHR_TEMPLATE_H

#include "aicpu_base_template.h"
#include "data_transfer.h"

namespace ops_hccl {

// AllGather NHR 模板
class AllGatherNhrTemplate : public AicpuBaseTemplate {
public:
    AllGatherNhrTemplate(u32 myRank, std::vector<u32> ranks, TemplateDesc templateDesc)
        : AicpuBaseTemplate(myRank, std::move(ranks), std::move(templateDesc))
    {
        syncAtCopyBoundary_ = true;
    }
    ~AllGatherNhrTemplate() = default;

protected:
    HcclResult DoCalcChannelRequest(
        HcclComm comm, const OpParam& param, TopoInfoWithNetLayerDetails* topoInfo,
        const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& levelChannels) override;
    u32 DoCalcThreadNum() const override;
    u32 DoCalcNotifyPerThread() const override;

    HcclResult
    RunAlgorithm(std::vector<DataSlicesList>& txRxSlicesLists, std::vector<u32>& ranksForOutputData) override;
    HcclResult PreCopy(const std::vector<ThreadHandle>& threads) override;
    HcclResult SendAll(
        const std::vector<DataSlicesList>& txRxSlicesLists, TemplateResource& templateResource,
        const std::vector<ThreadHandle>& threads) override;
    HcclResult PostCopy(const std::vector<ThreadHandle>& threads) override;
    HcclResult CopyInputToOutput(const std::vector<ThreadHandle>& threads) override;

private:
    static constexpr u64 SINGLE_CHANNEL_MAX_DATA_SIZE_ = 1 * 1024 * 1024; // 1MB
    static void DedupChannelsByRemoteRank(std::vector<HcclChannelDesc>& channels);

    HcclResult LaunchPostCopy(const std::vector<ThreadHandle>& threads, u32 postCopyChannelNum);
    TransferContext BuildTransferContext(
        const DataSlicesList& txRxSlicesList, TemplateResource& templateResource, bool isLastStep) const;

    bool canParallelPostCopy_{false};
    bool postCopyLaunched_{false};
    std::vector<u32> ranksForOutputData_{};
    std::vector<u32> lastStepRxRanks_{};
};

} // namespace ops_hccl

#endif // ALLGATHER_NHR_TEMPLATE_H
