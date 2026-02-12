/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_alg_template_base.h"

namespace ops_hccl {


AivAlgTemplateBase::AivAlgTemplateBase(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                       const std::vector<std::vector<u32>>& subCommRanks)
{
    opMode_            = param.opMode;
    root_              = param.root;
    myRank_            = rankId;
    tempRankSize_      = subCommRanks[0].size();
    subCommRanks_      = subCommRanks;

    reduceOp_          = param.reduceType;
    // 从OpParam中获取
    enableDetour_      = param.enableDetour;
}

AivAlgTemplateBase::~AivAlgTemplateBase()
{
}

u64 AivAlgTemplateBase::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    return 1;
}

HcclResult AivAlgTemplateBase::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                                       AlgResourceRequest& resourceRequest)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)resourceRequest;
    HCCL_ERROR("[AivAlgTemplateBase] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult AivAlgTemplateBase::KernelRun(const OpParam& param,
                                         const TemplateDataParams& tempAlgParams,
                                         const TemplateResource& templateResource)
{
    (void)param;
    (void)tempAlgParams;
    (void)templateResource;
    HCCL_ERROR("[AivAlgTemplateBase] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

void AivAlgTemplateBase::IncSliceId()
{
    sliceId_++;
    return;
}

HcclResult AivAlgTemplateBase::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void) dataSize;
    if (numBlocksLimit >= tempRankSize_) {
        numBlocks = tempRankSize_;
    } else {
        numBlocks = numBlocksLimit;
    } 
    HCCL_INFO("[AivAlgTemplateBase] Actually use core num[%u]", numBlocks);
    return HCCL_SUCCESS;
}
// 可能用不到，预留
HcclResult AivAlgTemplateBase::PreSyncInterQueues(const std::vector<ThreadHandle> &threads) const
{
    for (u32 queIdx = 0; queIdx < threads.size(); queIdx++) {
        CHK_PRT_RET(PreSync(queIdx, threads) != HcclResult::HCCL_SUCCESS,
                    HCCL_ERROR("[AivAlgTemplateBase] Rank [%d], Que [%u], Semaphore Synchronization Failed.", myRank_,
                               queIdx),
                    HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivAlgTemplateBase::PostSyncInterQueues(const std::vector<ThreadHandle> &threads) const
{
    for (u32 queIdx = 0; queIdx < threads.size(); queIdx++) {
        CHK_PRT_RET(PostSync(queIdx, threads) != HcclResult::HCCL_SUCCESS,
                    HCCL_ERROR("[AivCollAlgFactory] Rank [%d], Que [%u], Semaphore Synchronization Failed.", myRank_,
                               queIdx),
                    HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivAlgTemplateBase::PreSync(const u32 queIdx, const std::vector<ThreadHandle> &threads) const
{
    ThreadHandle currThread = threads[queIdx];
    if (queIdx == 0) {
        // Semaphore Post
        for (u32 qidx = 1; qidx < threads.size(); qidx++) {
            // 主流向从流发送post
            // 第三个参数是目标流的notify的索引，在exector中计算资源的时候会确定从流需要多少索引
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(currThread, threads[qidx],
                                                                            LOCAL_NOTIFY_IDX_ZERO)));
        }
    } else {
        // Semaphore Wait
        // 从流等待主流通知
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(currThread, LOCAL_NOTIFY_IDX_ZERO,
                                                                          CUSTOM_TIMEOUT)));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivAlgTemplateBase::PostSync(const u32 queIdx, const std::vector<ThreadHandle> &threads) const
{
    ThreadHandle currThread = threads[queIdx];
    if (queIdx == 0) {
        // Semaphore Wait
        for (u32 qidx = 0; qidx < threads.size() - 1; qidx++) {
            // 主流等待所有从流通知
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[qidx], qidx, CUSTOM_TIMEOUT)));
        }
    } else {
        // Semaphore Post
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(currThread, threads[0],
                                                                        queIdx - 1))); // LOCAL_NOTIFY_IDX_ZERO
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl