/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allgather_nhr_template.h"

namespace ops_hccl {

REGISTER_RE_TEMPLATE(HcclCMDType::HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_NHR, AllGatherNhrTemplate)

HcclResult AllGatherNhrTemplate::DoCalcChannelRequest(
    HcclComm comm, const OpParam& param, TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& levelChannels)
{
    return HCCL_SUCCESS;
}

u32 AllGatherNhrTemplate::DoCalcThreadNum() const { return 0; }

u32 AllGatherNhrTemplate::DoCalcNotifyPerThread() const { return 0; }

void AllGatherNhrTemplate::DedupChannelsByRemoteRank(std::vector<HcclChannelDesc>& channels) {}

HcclResult
AllGatherNhrTemplate::RunAlgorithm(std::vector<DataSlicesList>& txRxSlicesLists, std::vector<u32>& ranksForOutputData)
{
    return HCCL_SUCCESS;
}

HcclResult AllGatherNhrTemplate::PreCopy(const std::vector<ThreadHandle>& threads) { return HCCL_SUCCESS; }

HcclResult AllGatherNhrTemplate::CopyInputToOutput(const std::vector<ThreadHandle>& threads) { return HCCL_SUCCESS; }

HcclResult AllGatherNhrTemplate::LaunchPostCopy(const std::vector<ThreadHandle>& threads, u32 postCopyChannelNum)
{
    return HCCL_SUCCESS;
}

TransferContext AllGatherNhrTemplate::BuildTransferContext(
    const DataSlicesList& txRxSlicesList, TemplateResource& templateResource, bool isLastStep) const
{
    return {};
}

HcclResult AllGatherNhrTemplate::SendAll(
    const std::vector<DataSlicesList>& txRxSlicesLists, TemplateResource& templateResource,
    const std::vector<ThreadHandle>& threads)
{
    return HCCL_SUCCESS;
}

HcclResult AllGatherNhrTemplate::PostCopy(const std::vector<ThreadHandle>& threads) { return HCCL_SUCCESS; }

} // namespace ops_hccl
