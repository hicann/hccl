/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_mesh_host_dpu_template.h"
#include "alg_template_register.h"
#include "hcomm_primitives.h"
#include "adapter_acl.h"

namespace ops_hccl {
ReduceScatterHostDpuTemplate::ReduceScatterHostDpuTemplate() : AlgTemplateBase()
{
}

ReduceScatterHostDpuTemplate::~ReduceScatterHostDpuTemplate()
{
}

HcclResult ReduceScatterHostDpuTemplate::RunAsync(const DPUAlgResourceCtx *dpuResCtx)
{    
    AlgHierarchyInfo algHierarchyInfo = dpuResCtx->algHierarchyInfo;

    // 展开算法
    u32 localRank = algHierarchyInfo.infos[1].localRank;
    u32 rankSize = algHierarchyInfo.infos[1].localRankSize;
    std::vector<Slice> slices(rankSize);
    for (u32 rankId = 0; rankId < rankSize; rankId++) {
        slices[rankId].size = dpuResCtx->sliceSize;
        slices[rankId].offset = dpuResCtx->sliceSize * rankId;
    }

    // 如果只有一张卡，AICPU做本地搬运，此处直接返回成功
    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }
    outputMem_ = dpuResCtx->cclInputMem;
    if (dpuResCtx->channelNum < rankSize) {
        HCCL_ERROR("[ScatterMesh][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]",
            localRank, dpuResCtx->channelNum, rankSize);
        return HCCL_E_INTERNAL;
    }

    std::vector<ChannelInfo> channelInfo(dpuResCtx->channels, dpuResCtx->channels + dpuResCtx->channelNum);

    // 遍历templateArgs中各个卡上数据的信息，完成搬运
    for (u32 rankId = 0; rankId < rankSize; rankId++) {
        CHK_RET(RunDataSend(rankId, localRank, slices, channelInfo, dpuResCtx));
    }

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHostDpuTemplate::RunDataSend(u32 dstRank,
                                                     u32 srcRank, const std::vector<Slice>& slices,
                                                     std::vector<ChannelInfo>& channels, const DPUAlgResourceCtx *dpuResCtx)
{
    if (dstRank >= channels.size()) {
        HCCL_ERROR("[Run][RecvScatter]SrcRank[%u] is out of range, linkSize[%llu]", srcRank, channels.size());
        return HCCL_E_INTERNAL;
    }

    if (srcRank == dstRank) {
        HCCL_INFO("Skip data transport for same rank [%u]", srcRank);
        return HCCL_SUCCESS;
    }

    HCCL_INFO("rank[%u] will rcv with ouput's offset[%llu], size[%llu]", dstRank, slices[dstRank].offset, slices[dstRank].size);
    // 通知对端我这边准备好写数据了
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecord(channels[dstRank].handle, 0)));
    // 等待对端准备完成
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWait(channels[dstRank].handle, 0, CUSTOM_TIMEOUT)));
    void* src = static_cast<void *>(static_cast<s8 *>(dpuResCtx->cclInputMem.addr) + slices[dstRank].offset);
    void* dst = static_cast<void *>(static_cast<s8 *>(channels[dstRank].remoteOutput.addr) + slices[dstRank].size * srcRank);
    // 开始写
    CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyNbi(channels[dstRank].handle, dst, src, slices[dstRank].size, 1)));
    // 等待对端收到写数据
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWait(channels[dstRank].handle, 1, CUSTOM_TIMEOUT)));
    // 写完通知对端
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecord(channels[dstRank].handle, 2)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWait(channels[dstRank].handle, 2, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommFlush()));
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCE_SCATTER_HOST_DPU, ReduceScatterHostDpuTemplate);
}