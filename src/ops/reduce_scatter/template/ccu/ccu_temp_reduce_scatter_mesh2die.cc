/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "alg_data_trans_wrapper.h"

#include "ccu_temp_reduce_scatter_mesh2die.h"
#include "ccu_kernel_reduce_scatter_mesh2die.h"

namespace ops_hccl {
CcuTempReduceScatterMesh2Die::CcuTempReduceScatterMesh2Die(const OpParam &param, RankId rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempReduceScatterMesh2Die::~CcuTempReduceScatterMesh2Die()
{
}

u64 CcuTempReduceScatterMesh2Die::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return SCRATCH_MULTIPLE_COUNT_2;
}

HcclResult CcuTempReduceScatterMesh2Die::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    // 需要从流
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread.push_back(1);

    //多少个kernel
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(PartitionChannels(comm, channelDescs));
    resourceRequest.channels.emplace_back(channelDescs);

    const uint32_t rankSize = subCommRanks_[0].size();
    resourceRequest.ccuKernelNum.push_back(DIE_NUM);        // kernel数量
    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {    // 2Die算法，需要执行两次
        // 创建每个kernel的kernelArg，放入kernelInfo, 然后将kernelInfo放入resourceRequest.ccuKernelInfos
        CcuKernelInfo kernelInfo;
        kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
            return std::make_unique<CcuKernelReduceScatterMesh2Die>(arg);
        };
        const bool rmtReduceWithMyRank = channels_[dieId].size() > channels_[1 - dieId].size() ? false : true;
        auto kernelArg = std::make_shared<CcuKernelArgReduceScatterMesh2Die>(rankSize, mySubCommRank_, param, subCommRanks_,
            rmtReduceWithMyRank);
        kernelInfo.kernelArg = kernelArg;
        kernelInfo.channels = channels_[dieId];
        resourceRequest.ccuKernelInfos.emplace_back(kernelInfo);
        HCCL_DEBUG("[CcuTempReduceScatterMesh2Die][CalcRes] dieId=%u, channels=%llu, rankSize=%llu, ccuKernelInfos=%llu",
            dieId, channels_[dieId].size(), rankSize, resourceRequest.ccuKernelInfos.size());
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMesh2Die::PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs)
{
   for (const auto &channel : channelDescs) {
        using DieIdType = uint32_t;
        const RankId remoteRank = channel.remoteRank;
        const uint32_t dieIdTypeSize = sizeof(DieIdType);
        DieIdType dieId = 0;
        EndpointDesc localEndpoint = channel.localEndpoint;
        HcclResult ret = HcclRankGraphGetEndpointInfo(comm, mySubCommRank_, &localEndpoint, ENDPOINT_ATTR_DIE_ID,
            dieIdTypeSize, static_cast<void*>(&dieId));
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CcuTempReduceScatterMesh2Die][CalcRes] Rank[%d] channel to remoteRank[%d], Failed to get dieId. "
                "errNo[0x%016llx]", mySubCommRank_, remoteRank, HCCL_ERROR_CODE(ret)),
            ret);
        CHK_PRT_RET(dieId >= DIE_NUM,
            HCCL_ERROR("[CcuTempReduceScatterMesh2Die][CalcRes] Rank[%d] channel to remoteRank[%d], dieId[%u] is invalid.",
                mySubCommRank_, remoteRank, dieId),
            HCCL_E_INTERNAL);
        HCCL_INFO("[CcuTempReduceScatterMesh2Die][CalcRes] Rank[%d] channel to remoteRank[%d], insert to "
            "channels(dieId[%u]).", mySubCommRank_, channel.remoteRank, dieId);
        channels_[dieId].emplace_back(channel);
        rankGroup_[dieId].push_back(channel.remoteRank);
    }
    rankGroup_[0].push_back(mySubCommRank_);   // keep mySubCommRank_ at last, sync with kernel
    rankGroup_[1].push_back(mySubCommRank_);
    uint32_t minChannels = std::min(channels_[0].size(), channels_[1].size());
    uint32_t maxChannels = std::max(channels_[0].size(), channels_[1].size());
    CHK_PRT_RET(minChannels + 1 != maxChannels,
        HCCL_ERROR("[CcuTempReduceScatterMesh2Die][CalcRes] Rank[%d], Unexpected channels size, "
            "die0 channels[%u], die1 channels[%u].", mySubCommRank_, channels_[0], channels_[1]),
        HcclResult::HCCL_E_PARA);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMesh2Die::KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
    const TemplateResource &templateResource)
{
    HCCL_INFO("[CcuTempReduceScatterMesh2Die] Run");
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr  = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t scratchAddr = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    uint64_t sliceSize             = templateDataParams.sliceSize;
    uint64_t token = hcomm::CcuRep::GetTokenInfo(PointerToAddr(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t inputSliceStride = templateDataParams.inputSliceStride;
    uint64_t offsetSliceSize = templateDataParams.sliceSize;
   HCCL_INFO("[CcuTempReduceScatterMesh2Die] inputAddr[%llu], outputAddr[%llu], scratchAddr0[%llu], "
              "sliceSize[%llu], inputSliceStride[%llu]",
              inputAddr, outputAddr, scratchAddr, sliceSize, inputSliceStride);

    // 前流同步
    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    std::vector<u32> notifyIdxMainToSub(1, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));

    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {    // 2Die算法，需要执行两次
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgReduceScatterMesh2Die>(
            inputAddr, outputAddr, token, scratchAddr, sliceSize, inputSliceStride);
        void *taskArgPtr = static_cast<void *>(taskArg.get());
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[dieId], templateResource.ccuKernels[dieId],
            taskArgPtr));
    }

    // 后流同步
    std::vector<u32> notifyIdxSubToMain(1, 0);
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));

    DataSlice srcSlice(buffInfo_.hcclBuff.addr, buffInfo_.hcclBuffBaseOff + offsetSliceSize, sliceSize, sliceSize/DATATYPE_SIZE_TABLE[param.DataDes.dataType]);
    DataSlice dstSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, sliceSize, sliceSize/DATATYPE_SIZE_TABLE[param.DataDes.dataType]);
    LocalReduce(templateResource.threads[0], srcSlice, dstSlice, param.DataDes.dataType, param.reduceType);

    HCCL_INFO("[CcuTempReduceScatterMesh2Die] Template Run for all steps Ends.");
    return HcclResult::HCCL_SUCCESS;
}
} // namespace ops_hccl