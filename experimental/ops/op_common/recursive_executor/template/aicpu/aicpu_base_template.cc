/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_base_template.h"

namespace ops_hccl {

void AicpuBaseTemplate::InitKernelRunParams(const DataParams& tempAlgParams) {}

HcclResult AicpuBaseTemplate::KernelRun(
    const DataParams& tempAlgParams, TemplateResource& templateResource, std::vector<u32>& ranksForOutputData)
{
    return HCCL_SUCCESS;
}

void AicpuBaseTemplate::PrepareSubThreads(
    const std::vector<ThreadHandle>& threads, std::vector<ThreadHandle>& subThreads,
    std::vector<u32>& notifyIdxMainToSub, std::vector<u32>& notifyIdxSubToMain) const
{}

HcclResult AicpuBaseTemplate::RunSingleRank(
    TemplateResource& templateResource, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxMainToSub, const std::vector<u32>& notifyIdxSubToMain,
    std::vector<u32>& ranksForOutputData)
{
    return HCCL_SUCCESS;
}

HcclResult AicpuBaseTemplate::RunMultiRank(
    TemplateResource& templateResource, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxMainToSub, const std::vector<u32>& notifyIdxSubToMain,
    std::vector<u32>& ranksForOutputData)
{
    return HCCL_SUCCESS;
}

HcclResult AicpuBaseTemplate::PrepareDataSplit(const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    return HCCL_SUCCESS;
}

HcclResult AicpuBaseTemplate::SendAll(
    const std::vector<DataSlicesList>& txRxSlicesLists, TemplateResource& templateResource,
    const std::vector<ThreadHandle>& threads)
{
    return HCCL_SUCCESS;
}

HcclResult AicpuBaseTemplate::PreSyncSubThreads(
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxMainToSub)
{
    return HCCL_SUCCESS;
}

HcclResult AicpuBaseTemplate::PostSyncSubThreads(
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxSubToMain)
{
    return HCCL_SUCCESS;
}

HcclResult AicpuBaseTemplate::CopyInputToOutput(const std::vector<ThreadHandle>& threads) { return HCCL_SUCCESS; }

HcclResult AicpuBaseTemplate::PreCopy(const std::vector<ThreadHandle>& threads) { return HCCL_SUCCESS; }

HcclResult AicpuBaseTemplate::PostCopy(const std::vector<ThreadHandle>& threads) { return HCCL_SUCCESS; }

} // namespace ops_hccl
