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
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "kernel/ccu_kernel_all_to_all_v_mesh2die.h"
#include "ccu_temp_all_to_all_v_mesh2die.h"
#include "ccu_launch_dl.h"
#include "ccu_res_dl.h"

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
        CcuKernelInfo kernelInfo;
        CHK_SAFETY_FUNC_RET(strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllToAllVMesh2DieKernel"));
        kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllToAllVMesh2DieKernel);

        const bool withMyRank = channels_[dieId].size() < channels_[1 - dieId].size();
        auto kernelArg = std::make_shared<CcuKernelArgAllToAllVMesh2Die>();
        kernelArg->rankId = myRank_;
        kernelArg->opParam = param;
        kernelArg->subCommRanks = subCommRanks_;
        kernelArg->withMyRank = withMyRank;
        kernelArg->rankGroup = rankGroup_[dieId];
        kernelInfo.setKernelArg(kernelArg);
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

void CcuTempAlltoAllVMesh2Die::FillRankGroupInfo()
{
    uint32_t rankSize = subCommRanks_[0].size();
    rankGroup_.insert({0, RankGroup()});
    rankGroup_.insert({1, RankGroup()});
    for (uint32_t i = 0; i < rankSize / 2; i++) {
        if(i == myRank_) {
            continue;
        }
        if (myRank_ < rankSize / 2) {
            rankGroup_[1].push_back(subCommRanks_[0][i]);
        } else {
            rankGroup_[0].push_back(subCommRanks_[0][i]);
        }
    }
    for (uint32_t i = rankSize / 2; i < rankSize; i++) {
         if(i == myRank_) {
            continue;
        }
        if (myRank_ < rankSize / 2) {
            rankGroup_[0].push_back(subCommRanks_[0][i]);
        } else {
            rankGroup_[1].push_back(subCommRanks_[0][i]);
        }
    }
    if (rankGroup_[0].size() > rankGroup_[1].size()) {
        rankGroup_[1].push_back(myRank_);
    } else {
        rankGroup_[0].push_back(myRank_);
    }
    return;
}

void CcuTempAlltoAllVMesh2Die::FillRankGroupTaskArgs(uint32_t dieId, const LoopGroupConfig &config, std::vector<uint64_t> &taskArgs)
{
    for (auto peerId : rankGroup_[dieId]) {
        const uint64_t sendSize = localSendRecvInfo_.sendLength[peerId];
        const uint64_t floorLoopNum = sendSize / UB_MAX_TRANS_SIZE;
        uint64_t sendLoopNum = UINT64_MAX - 1 - floorLoopNum;
        uint64_t sendTailSize = sendSize - floorLoopNum * UB_MAX_TRANS_SIZE;
        auto sendTailGoSize = CalGoSize(sendTailSize, config);
        uint64_t sendOffset = localSendRecvInfo_.sendOffset[peerId];
        uint64_t recvOffset = localSendRecvInfo_.recvOffset[peerId];
        taskArgs.push_back(sendOffset);
        taskArgs.push_back(recvOffset);
        taskArgs.push_back(sendTailSize);
        for (auto val : sendTailGoSize) {
            taskArgs.push_back(val);
        }
        taskArgs.push_back(sendLoopNum);
    }
    return;
}

HcclResult CcuTempAlltoAllVMesh2Die::KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
    TemplateResource& templateResource)
{
    CHK_PRT_RET(subCommRanks_.empty() || subCommRanks_[0].empty(),
        HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][KernelRun] subCommRanks empty."), HcclResult::HCCL_E_INTERNAL);

    buffInfo_ = templateDataParams.buffInfo;
    CHK_PRT_RET(buffInfo_.inputPtr == nullptr || buffInfo_.outputPtr == nullptr,
        HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][KernelRun] Rank[%d] input[%#llx] or output[%#llx] is null",
            myRank_, buffInfo_.inputPtr, buffInfo_.outputPtr),
        HcclResult::HCCL_E_PTR);

    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr);
    HCCL_INFO("[CcuTempAlltoAllVMesh2Die][KernelRun] begin. Rank[%d], input[%#llx/%#llx], output[%#llx/%#llx], "
        "sendType[%d], recvType[%d]", myRank_, inputAddr, param.inputPtr, outputAddr, param.outputPtr,
        param.all2AllVDataDes.sendType, param.all2AllVDataDes.recvType);

    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    // 前流同步
    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    std::vector<u32> notifyIdxMainToSub(1, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    FillRankGroupInfo();
    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {    // 2Die算法，需要执行两次
        std::vector<uint64_t> taskArgs;
        taskArgs.push_back(inputAddr);
        taskArgs.push_back(outputAddr);
        taskArgs.push_back(token);

        LoopGroupConfig config{};
        config.msInterleave = CCU_MS_INTERLEAVE;
        config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
        config.memSlice = LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
        auto xnMaxTransportGoSize = CalGoSize(UB_MAX_TRANS_SIZE, config);
        for (auto val : xnMaxTransportGoSize) {
            taskArgs.push_back(val);
        }

        FillRankGroupTaskArgs(dieId, config, taskArgs);

        uint64_t argSize = taskArgs.size();
        CcuResult launchRet = HcommCcuKernelLaunch(
            templateResource.threads[dieId], templateResource.ccuKernels[dieId],
            taskArgs.data(), argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    // 后流同步
    std::vector<u32> notifyIdxSubToMain(1, 0);
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));

    HCCL_DEBUG("[CcuTempAlltoAllVMesh2Die][KernelRun] end. Rank[%d]", myRank_);

    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
