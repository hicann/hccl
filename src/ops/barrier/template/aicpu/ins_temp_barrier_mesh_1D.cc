/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_barrier_mesh_1D.h"
#include "alg_data_trans_wrapper.h"
#include "channel.h"
#include "alg_v2_template_register.h"

namespace ops_hccl {
InsTempBarrierMesh1D::InsTempBarrierMesh1D(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

HcclResult InsTempBarrierMesh1D::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    HCCL_INFO("[InsTempBarrierMesh1D][CalcRes] start");
    u32 level0RankSize = templateRankSize_;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    HCCL_INFO(
        "[InsTempBarrierMesh1D][CalcRes] slaveThreadNum[%u] notifyNumOnMainThread[%u] level0Channels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, level0Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempBarrierMesh1D::GetThreadNum() const { return templateRankSize_ > 1 ? templateRankSize_ - 1 : 1; }

u64 InsTempBarrierMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

HcclResult InsTempBarrierMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    (void)tempAlgParams;
    HCCL_INFO("[InsTempBarrierMesh1D] Run start, rank[%u] rankSize[%u]", myRank_, templateRankSize_);
    if (templateRankSize_ == 1) {
        return HCCL_SUCCESS;
    }
    threadNum_ = templateResource.threads.size();
    dataType_ = param.DataDes.dataType;

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    CHK_RET(RunBarrierMesh(templateResource.threads, templateResource.channels));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    HCCL_INFO("[InsTempBarrierMesh1D] Run End");
    return HCCL_SUCCESS;
}

HcclResult InsTempBarrierMesh1D::RunBarrierMesh(
    const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    HCCL_INFO("[InsTempBarrierMesh1D] RunBarrierMesh RankID[%u].", myRank_);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    for (u32 threadIdx = 0; threadIdx < subCommRanks_[0].size() - 1; threadIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + threadIdx) % subCommRanks_[0].size()];

        CHK_PRT_RET(
            threadIdx >= threads.size() || channels.count(connectedRank) == 0 || channels.at(connectedRank).empty(),
            HCCL_ERROR(
                "[InsTempBarrierMesh1D][RankID]=%u threadIdx=%u, threads.size=%u, "
                "connectedRank=%d, channels.size=%u",
                myRank_, threadIdx, threads.size(), connectedRank, channels.size()),
            HcclResult::HCCL_E_INTERNAL);

        const ChannelInfo& linkRemote = channels.at(connectedRank)[0];

        // Barrier 语义：只做前后 ACK / DATA_SIGNAL 同步对，不搬运数据。
        // SendRecvWrite 内部的 for-loop 在空 slice vector 下被跳过。
        std::vector<DataSlice> emptySlices;
        TxRxSlicesList sendRecvSlicesList({emptySlices, emptySlices}, {emptySlices, emptySlices});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
        CHK_PRT_RET(
            SendRecvWrite(sendRecvInfo, threads[threadIdx]),
            HCCL_ERROR("[InsTempBarrierMesh1D] RunBarrierMesh SendRecvWrite failed, threadIdx=%u", threadIdx),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

void InsTempBarrierMesh1D::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempBarrierMesh1D::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

REGISTER_TEMPLATE_V2("InsTempBarrierMesh1D", InsTempBarrierMesh1D);
} // namespace ops_hccl
