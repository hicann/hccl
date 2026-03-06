/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_all_to_all_v_mesh2die.h"
#include "channel.h"
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "alg_data_trans_wrapper.h"
#include "kernel/ccu_kernel_all_to_all_v_mesh2die.h"

namespace ops_hccl {

CcuTempAlltoAllVMesh2Die::CcuTempAlltoAllVMesh2Die(const OpParam &param, RankId rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
}

CcuTempAlltoAllVMesh2Die::~CcuTempAlltoAllVMesh2Die()
{
}

HcclResult CcuTempAlltoAllVMesh2Die::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    CHK_PRT_RET(subCommRanks_.size() != 1 || subCommRanks_[0].empty(),
        HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][CalcRes] Invalid subCommRanks[%u] or subCommRanks empty.",
            subCommRanks_.size()), HcclResult::HCCL_E_INTERNAL);

    HCCL_DEBUG("[CcuTempAlltoAllVMesh2Die][CalcRes] rankSize[%u] subCommRanks0[%u].", templateRankSize_,
        subCommRanks_[0].size());

    // 需要从流
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumOnMainThread = resourceRequest.slaveThreadNum;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    HCCL_DEBUG("[CcuTempAlltoAllVMesh2Die][CalcRes] channelDescs size[%u].", channelDescs.size());
    CHK_RET(PartitionChannels(comm, channelDescs));
    resourceRequest.channels.emplace_back(channelDescs);

    resourceRequest.ccuKernelNum.push_back(DIE_NUM);        // kernel数量
    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {    // 2Die算法，需要执行两次
        // 创建每个kernel的kernelArg，放入kernelInfo, 然后将kernelInfo放入resourceRequest.ccuKernelInfos
        CcuKernelInfo kernelInfo;
        kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
            return std::make_unique<CcuKernelAllToAllVMesh2Die>(arg);
        };
        const bool withMyRank = channels_[dieId].size() < channels_[1 - dieId].size();
        auto kernelArg = std::make_shared<CcuKernelArgAllToAllVMesh2Die>(myRank_, param, subCommRanks_, withMyRank,
            rankGroup_[dieId]);
        kernelInfo.kernelArg = kernelArg;
        kernelInfo.channels = channels_[dieId];
        resourceRequest.ccuKernelInfos.emplace_back(kernelInfo);
        HCCL_DEBUG("[CcuTempAlltoAllVMesh2Die][CalcRes] dieId=%u, channels=%llu, withMyRank=%u, ccuKernelInfos=%llu",
            dieId, channels_[dieId].size(), withMyRank, resourceRequest.ccuKernelInfos.size());
    }

    return HcclResult::HCCL_SUCCESS;
}

// 分别记录两个Die上的channel，构造rankGroup
HcclResult CcuTempAlltoAllVMesh2Die::PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs)
{
    for (const auto &channel : channelDescs) {
        const RankId remoteRank = channel.remoteRank;
        uint32_t dieId = 0;
        HcclResult ret = GetChannelDieId(comm, myRank_, channel, dieId);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][PartitionChannels] Rank[%d] channel to remoteRank[%d], Failed to "
                "get dieId. errNo[0x%016llx]", myRank_, remoteRank, HCCL_ERROR_CODE(ret)),
            ret);
        CHK_PRT_RET(dieId >= DIE_NUM,
            HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][PartitionChannels] Rank[%d] channel to remoteRank[%d], dieId[%u] is "
                "invalid.", myRank_, remoteRank, dieId),
            HCCL_E_INTERNAL);
        HCCL_INFO("[CcuTempAlltoAllVMesh2Die][PartitionChannels] Rank[%d] channel to remoteRank[%d], insert to "
            "channels at dieId[%u].", myRank_, remoteRank, dieId);
        channels_[dieId].emplace_back(channel);
        rankGroup_[dieId].push_back(remoteRank);
    }
    uint32_t minChannels = std::min(channels_[0].size(), channels_[1].size());
    uint32_t maxChannels = std::max(channels_[0].size(), channels_[1].size());
    CHK_PRT_RET(minChannels + 1 != maxChannels,
        HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][PartitionChannels] Rank[%d], Unexpected channels size, "
            "die0 channels[%u], die1 channels[%u].", myRank_, channels_[0].size(), channels_[1].size()),
        HcclResult::HCCL_E_PARA);
    HCCL_DEBUG("[CcuTempAlltoAllVMesh2Die][PartitionChannels] Rank[%d], die0 channels[%u], die1 channels[%u].", myRank_,
        channels_[0].size(), channels_[1].size());
    // keep myRank_ at last, sync with kernel
    if (channels_[0].size() < channels_[1].size()) {
        rankGroup_[0].push_back(myRank_);
    } else {
        rankGroup_[1].push_back(myRank_);
    }
    return HcclResult::HCCL_SUCCESS;
}

// executor在orchestra中调用
void CcuTempAlltoAllVMesh2Die::SetA2ASendRecvInfo(const A2ASendRecvInfo &sendRecvInfo)
{
    localSendRecvInfo_ = sendRecvInfo;
}

HcclResult CcuTempAlltoAllVMesh2Die::KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
    const TemplateResource &templateResource)
{
    CHK_PRT_RET(subCommRanks_.empty() || subCommRanks_[0].empty(),
        HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][KernelRun] subCommRanks empty."), HcclResult::HCCL_E_INTERNAL);

    const auto &buffInfo = templateDataParams.buffInfo;
    CHK_PRT_RET(buffInfo.inputPtr == nullptr || buffInfo.outputPtr == nullptr,
        HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][KernelRun] Rank[%d] input[%#llx] or output[%#llx] is null",
            myRank_, buffInfo.inputPtr, buffInfo.outputPtr),
        HcclResult::HCCL_E_PTR);

    uint64_t inputAddr = PointerToAddr(buffInfo.inputPtr);
    uint64_t outputAddr = PointerToAddr(buffInfo.outputPtr);
    HCCL_INFO("[CcuTempAlltoAllVMesh2Die][KernelRun] begin. Rank[%d], input[%#llx/%#llx], output[%#llx/%#llx], "
        "sendType[%d], recvType[%d]", myRank_, inputAddr, param.inputPtr, outputAddr, param.outputPtr,
        param.all2AllVDataDes.sendType, param.all2AllVDataDes.recvType);

    uint64_t token = hcomm::CcuRep::GetTokenInfo(inputAddr, buffInfo.inputSize);

    // 前流同步
    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    std::vector<u32> notifyIdxMainToSub(1, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));

    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {    // 2Die算法，需要执行两次
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllToAllVMesh2Die>(
            inputAddr, outputAddr, token, localSendRecvInfo_);
        void *taskArgPtr = static_cast<void *>(taskArg.get());
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[dieId], templateResource.ccuKernels[dieId],
            taskArgPtr));
    }

    // 后流同步
    std::vector<u32> notifyIdxSubToMain(1, 0);
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));

    HCCL_DEBUG("[CcuTempAlltoAllVMesh2Die][KernelRun] end. Rank[%d]", myRank_);

    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl