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
#include "ccu_temp_all_to_all_v_mesh1d_2Die.h"
#include "kernel/ccu_kernel_all_to_all_v_mesh1d_2die.h"
#include "ccu_launch_dl.h"
#include "hccl_res_dl.h"

namespace ops_hccl {

constexpr uint32_t CACHED_IN_BUFF_OFF = 0;
constexpr uint32_t CACHED_OUT_BUFF_OFF = 1;
constexpr uint32_t CACHED_TOKEN = 2;
constexpr uint32_t CACHED_GO_SIZE_BASE = 3;
constexpr uint32_t CACHED_GO_SIZE_NUM = 4;

CcuTempAlltoAllVMesh1D2Die::CcuTempAlltoAllVMesh1D2Die(
    const OpParam& param, RankId rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        myRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempAlltoAllVMesh1D2Die::~CcuTempAlltoAllVMesh1D2Die() {}

HcclResult CcuTempAlltoAllVMesh1D2Die::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_PRT_RET(
        subCommRanks_.size() != 1 || subCommRanks_[0].empty(),
        HCCL_ERROR(
            "[CcuTempAlltoAllVMesh1D2Die][CalcRes] Invalid subCommRanks[%u] or subCommRanks empty.",
            subCommRanks_.size()),
        HcclResult::HCCL_E_INTERNAL);

    HCCL_DEBUG(
        "[CcuTempAlltoAllVMesh1D2Die][CalcRes] rankSize[%u] subCommRanks0[%u].", templateRankSize_,
        subCommRanks_[0].size());

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));
    HCCL_INFO("channelDescs size[%u]", channelDescs.size());

    CHK_RET(PartitionChannels(comm, channelDescs, rankIdToChannelDesc_));

    double ratio = 1.0;
    CHK_RET(CalcDieSplitRatio(
        comm, myRank_, is2Plus6_, kernelChannels_[KERNEL_CLOS_MAJOR], kernelChannels_[KERNEL_CLOS_MINOR], ratio));
    resourceRequest.dieSplitRatio = ratio;
    dieSplitRatio_ = ratio;

    uint32_t slaveThreadNum = kernelCount_ - 1;
    resourceRequest.notifyNumOnMainThread = slaveThreadNum;
    resourceRequest.slaveThreadNum = slaveThreadNum;
    resourceRequest.notifyNumPerThread.assign(slaveThreadNum, 1);

    resourceRequest.channels.emplace_back(channelDescs);
    HCCL_INFO("resourceRequest.channels[%d]", resourceRequest.channels.size());

    resourceRequest.ccuKernelNum.push_back(kernelCount_);

    for (uint32_t i = 0; i < kernelCount_; i++) {
        CcuKernelInfo kernelInfo;
        CHK_SAFETY_FUNC_RET(
            strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAlltoAllVMesh1D2DieKernel"));
        kernelInfo.kernelFunc = reinterpret_cast<void*>(CcuAlltoAllVMesh1D2DieKernel);

        auto kernelArg = std::make_shared<CcuKernelArgAllToAllVMesh1D2Die>();
        kernelArg->rankId = myRank_;
        kernelArg->opParam = param;
        kernelArg->subCommRanks = subCommRanks_;
        kernelArg->withMyRank = kernelWithMyRank_[i];
        kernelArg->rankGroup = kernelRankGroup_[i];
        kernelInfo.setKernelArg(kernelArg);
        kernelInfo.channels = kernelChannels_[i];
        resourceRequest.ccuKernelInfos.emplace_back(kernelInfo);
        HCCL_DEBUG(
            "[CcuTempAlltoAllVMesh1D2Die][CalcRes] kernel[%u], channels=%llu, withMyRank=%u, ccuKernelInfos=%llu", i,
            kernelChannels_[i].size(), kernelWithMyRank_[i], resourceRequest.ccuKernelInfos.size());
    }

    CHK_RET(SaveCacheCtx(comm, param));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh1D2Die::PartitionChannels(
    HcclComm comm, const std::vector<HcclChannelDesc>& channelDescs,
    std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc)
{
    (void)channelDescs;
    std::map<uint32_t, std::vector<HcclChannelDesc>> singleChByDie;
    std::map<uint32_t, std::vector<HcclChannelDesc>> multiChByDie;
    CHK_RET(
        SplitChannelsByDie(comm, myRank_, rankIdToChannelDesc, singleChByDie, multiChByDie, is2Plus6_, &closPeers_));

    CHK_RET(PartitionChannelsFor2Die(
        comm, singleChByDie, multiChByDie, is2Plus6_, myRank_, kernelCount_, fullmeshDieId_, kernelChannels_,
        kernelRankGroup_, "CcuTempAlltoAllVMesh1D2Die"));

    CHK_RET(ReorderAndLogKernelGroups(
        myRank_, static_cast<uint32_t>(subCommRanks_[0].size()), is2Plus6_, kernelCount_, kernelChannels_,
        kernelRankGroup_, "CcuTempAlltoAllVMesh1D2Die"));

    return HcclResult::HCCL_SUCCESS;
}

void CcuTempAlltoAllVMesh1D2Die::SetA2ASendRecvInfo(const A2ASendRecvInfo& sendRecvInfo)
{
    localSendRecvInfo_ = sendRecvInfo;
}

HcclResult CcuTempAlltoAllVMesh1D2Die::SaveCacheCtx(HcclComm comm, const OpParam& param)
{
    Mesh2DieCacheCtx cacheCtx;
    cacheCtx.dieNum = DIE_NUM;
    cacheCtx.is2Plus6 = is2Plus6_;
    cacheCtx.kernelCount = kernelCount_;
    for (uint32_t i = 0; i < MAX_KERNEL_NUM_2DIE; i++) {
        cacheCtx.rankGroup[i] = kernelRankGroup_[i];
    }
    cacheCtx.closPeers = closPeers_;
    cacheCtx.dieSplitRatio = dieSplitRatio_;

    std::vector<char> buf = cacheCtx.Serialize();

    char cacheTag[ALG_TAG_LENGTH] = {0};
    int ret = snprintf_s(cacheTag, sizeof(cacheTag), sizeof(cacheTag) - 1, "%s_mesh2die", param.tag);
    CHK_PRT_RET(
        ret <= 0, HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][SaveCacheCtx] failed to fill cacheTag"), HCCL_E_INTERNAL);

    void* ctxPtr = nullptr;
    uint64_t ctxSize = 0;
    HcclResult getRet = HcclEngineCtxGet(comm, cacheTag, CommEngine::COMM_ENGINE_CPU_TS, &ctxPtr, &ctxSize);
    if (getRet == HCCL_SUCCESS && ctxPtr != nullptr && ctxSize == buf.size()) {
        CHK_SAFETY_FUNC_RET(memcpy_s(ctxPtr, ctxSize, buf.data(), buf.size()));
        HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][SaveCacheCtx] updated existing cacheCtx, size[%zu]", buf.size());
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_RET(HcclEngineCtxCreate(comm, cacheTag, CommEngine::COMM_ENGINE_CPU_TS, buf.size(), &ctxPtr));

    CHK_SAFETY_FUNC_RET(memcpy_s(ctxPtr, buf.size(), buf.data(), buf.size()));

    HCCL_INFO(
        "[CcuTempAlltoAllVMesh1D2Die][SaveCacheCtx] created and saved cacheCtx, size[%zu], "
        "is2Plus6[%d] kernelCount[%u] closPeers[%zu]",
        buf.size(), is2Plus6_, kernelCount_, closPeers_.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh1D2Die::LoadCacheCtx(const OpParam& param, Mesh2DieCacheCtx& cacheCtx)
{
    char cacheTag[ALG_TAG_LENGTH] = {0};
    int ret = snprintf_s(cacheTag, sizeof(cacheTag), sizeof(cacheTag) - 1, "%s_mesh2die", param.tag);
    CHK_PRT_RET(
        ret <= 0, HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][LoadCacheCtx] failed to fill cacheTag"), HCCL_E_INTERNAL);

    void* ctxPtr = nullptr;
    uint64_t ctxSize = 0;
    HcclComm comm = static_cast<HcclComm>(param.hcclComm);
    HcclResult getRet = HcclEngineCtxGet(comm, cacheTag, CommEngine::COMM_ENGINE_CPU_TS, &ctxPtr, &ctxSize);
    CHK_PRT_RET(
        getRet != HCCL_SUCCESS,
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][LoadCacheCtx] HcclEngineCtxGet failed, ret=%d", getRet), getRet);
    CHK_PRT_RET(
        ctxPtr == nullptr || ctxSize == 0,
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][LoadCacheCtx] cache ctx is null or empty"),
        HcclResult::HCCL_E_INTERNAL);

    CHK_RET(cacheCtx.Deserialize(static_cast<const char*>(ctxPtr), static_cast<size_t>(ctxSize)));

    HCCL_INFO(
        "[CcuTempAlltoAllVMesh1D2Die][LoadCacheCtx] loaded cacheCtx, size[%llu], "
        "is2Plus6[%d] kernelCount[%u] closPeers[%zu]",
        ctxSize, cacheCtx.is2Plus6, cacheCtx.kernelCount, cacheCtx.closPeers.size());
    return HcclResult::HCCL_SUCCESS;
}

void CcuTempAlltoAllVMesh1D2Die::FillRankGroupTaskArgs(
    uint32_t kernelIdx, const Mesh2DieCacheCtx& cacheCtx, const LoopGroupConfig& config,
    std::vector<uint64_t>& taskArgs)
{
    for (auto peerId : cacheCtx.rankGroup[kernelIdx]) {
        uint64_t sendSize = localSendRecvInfo_.sendLength[peerId];
        uint64_t sendOffset = localSendRecvInfo_.sendOffset[peerId];
        uint64_t recvOffset = localSendRecvInfo_.recvOffset[peerId];
        uint64_t origSendSize = sendSize;

        if (cacheCtx.is2Plus6 && cacheCtx.closPeers.count(peerId) > 0) {
            uint64_t recvLength = localSendRecvInfo_.recvLength[peerId];
            uint64_t majorSendSize = static_cast<uint64_t>(sendSize * dieSplitRatio_);
            uint64_t minorSendSize = sendSize - majorSendSize;
            uint64_t majorRecvSize = static_cast<uint64_t>(recvLength * dieSplitRatio_);
            uint64_t minorRecvSize = recvLength - majorRecvSize;
            if (kernelIdx == KERNEL_CLOS_MAJOR) {
                sendOffset += minorSendSize;
                recvOffset += minorRecvSize;
                sendSize = majorSendSize;
            } else if (kernelIdx == KERNEL_CLOS_MINOR) {
                sendSize = minorSendSize;
            }
            HCCL_DEBUG(
                "[CcuTempAlltoAllVMesh1D2Die][FillRankGroupTaskArgs] Rank[%d] kernel[%u] peer[%u] "
                "closSplit: origSendSize[%llu] -> sendSize[%llu], sendOffset[%llu], recvOffset[%llu].",
                myRank_, kernelIdx, peerId, origSendSize, sendSize, sendOffset, recvOffset);
        }

        const uint64_t floorLoopNum = sendSize / UB_MAX_TRANS_SIZE;
        uint64_t sendLoopNum = UINT64_MAX - 1 - floorLoopNum;
        uint64_t sendTailSize = sendSize - floorLoopNum * UB_MAX_TRANS_SIZE;
        auto sendTailGoSize = CalGoSize(sendTailSize, config, GetCcuVersion());
        taskArgs.push_back(sendOffset);
        taskArgs.push_back(recvOffset);
        taskArgs.push_back(sendTailSize);
        for (auto val : sendTailGoSize) {
            taskArgs.push_back(val);
        }
        taskArgs.push_back(sendLoopNum);

        HCCL_DEBUG(
            "[CcuTempAlltoAllVMesh1D2Die][FillRankGroupTaskArgs] Rank[%d] kernel[%u] peer[%u] "
            "sendOffset[%llu] recvOffset[%llu] sendTailSize[%llu] floorLoopNum[%llu] sendLoopNum[%llu].",
            myRank_, kernelIdx, peerId, sendOffset, recvOffset, sendTailSize, floorLoopNum, sendLoopNum);
    }
    return;
}

HcclResult CcuTempAlltoAllVMesh1D2Die::KernelRun(
    const OpParam& param, const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    CHK_PRT_RET(
        subCommRanks_.empty() || subCommRanks_[0].empty(),
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][KernelRun] subCommRanks empty."), HcclResult::HCCL_E_INTERNAL);

    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr);
    HCCL_INFO(
        "[CcuTempAlltoAllVMesh1D2Die][KernelRun] begin. Rank[%d], input[%#llx/%#llx], output[%#llx/%#llx], "
        "sendType[%d], recvType[%d]",
        myRank_, inputAddr, param.inputPtr, outputAddr, param.outputPtr, param.all2AllVDataDes.sendType,
        param.all2AllVDataDes.recvType);

    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    Mesh2DieCacheCtx cacheCtx;
    CHK_RET(LoadCacheCtx(param, cacheCtx));

    if (templateResource.dieSplitRatio > 0.0) {
        dieSplitRatio_ = templateResource.dieSplitRatio;
    }

    uint32_t kernelCount = cacheCtx.kernelCount;
    uint32_t subThreadCount = kernelCount - 1;

    std::vector<ThreadHandle> subThreads(
        templateResource.threads.begin() + 1, templateResource.threads.begin() + 1 + subThreadCount);
    std::vector<u32> notifyIdxMainToSub(subThreadCount, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
    auto xnMaxTransportGoSize = CalGoSize(UB_MAX_TRANS_SIZE, config, GetCcuVersion());

    for (uint32_t i = 0; i < kernelCount; i++) {
        std::vector<uint64_t> taskArgs;
        taskArgs.push_back(inputAddr);
        taskArgs.push_back(outputAddr);
        taskArgs.push_back(token);
        for (auto val : xnMaxTransportGoSize) {
            taskArgs.push_back(val);
        }

        FillRankGroupTaskArgs(i, cacheCtx, config, taskArgs);

        uint64_t argSize = taskArgs.size();
        HCCL_INFO(
            "[CcuTempAlltoAllVMesh1D2Die][KernelRun] kernel[%u] argSize[%llu] rankGroupSize[%zu] "
            "channelCount[%llu] withMyRank[%u]",
            i, argSize, cacheCtx.rankGroup[i].size(), templateResource.ccuKernels.size(), kernelWithMyRank_[i]);
        CcuResult launchRet = HcommCcuKernelLaunch(
            templateResource.threads[i], templateResource.ccuKernels[i], taskArgs.data(), argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }

        CcuKernelSubmitInfo submitInfo;
        submitInfo.kernelHandle = templateResource.ccuKernels[i];
        submitInfo.cachedArgs[CACHED_IN_BUFF_OFF] = buffInfo_.inBuffBaseOff;
        submitInfo.cachedArgs[CACHED_OUT_BUFF_OFF] = buffInfo_.outBuffBaseOff;
        submitInfo.cachedArgs[CACHED_TOKEN] = token;
        for (uint32_t j = 0; j < CACHED_GO_SIZE_NUM; j++) {
            submitInfo.cachedArgs[CACHED_GO_SIZE_BASE + j] = xnMaxTransportGoSize[j];
        }
        templateResource.submitInfos.push_back(submitInfo);
    }

    std::vector<u32> notifyIdxSubToMain(subThreadCount);
    for (uint32_t i = 0; i < subThreadCount; i++) {
        notifyIdxSubToMain[i] = i;
    }
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));

    HCCL_DEBUG("[CcuTempAlltoAllVMesh1D2Die][KernelRun] end. Rank[%d]", myRank_);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh1D2Die::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    uint32_t kernelCount = static_cast<uint32_t>(tempFastLaunchCtx.ccuKernelSubmitInfos.size());
    if (kernelCount == 0) {
        HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][FastLaunch] ccu kernel num is 0, just success.");
        return HcclResult::HCCL_SUCCESS;
    }
    HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][FastLaunch] start, kernelCount[%u]", kernelCount);

    Mesh2DieCacheCtx cacheCtx;
    CHK_RET(LoadCacheCtx(param, cacheCtx));
    dieSplitRatio_ = cacheCtx.dieSplitRatio;

    HcclDataType dataType = param.all2AllVDataDes.sendType;
    uint64_t dataTypeSize = HCCL_SIZE_TABLE[dataType];
    uint32_t rankSize = static_cast<uint32_t>(param.varMemSize / (ALL_TO_ALL_V_VECTOR_NUM * sizeof(u64)));
    CHK_PRT_RET(
        rankSize == 0,
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][FastLaunch] rankSize is 0, varMemSize[%llu]", param.varMemSize),
        HcclResult::HCCL_E_INTERNAL);

    A2ASendRecvInfo sendRecvInfo;
    sendRecvInfo.sendLength.resize(rankSize, 0);
    sendRecvInfo.sendOffset.resize(rankSize, 0);
    sendRecvInfo.recvOffset.resize(rankSize, 0);
    sendRecvInfo.recvLength.resize(rankSize, 0);

    const u64* data = reinterpret_cast<const u64*>(param.varData);
    for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * rankSize; i++) {
        u64 val = i / rankSize;
        u64 curRank = i % rankSize;
        switch (val) {
            case 0:
                sendRecvInfo.sendLength[curRank] = data[i] * dataTypeSize;
                break;
            case 1:
                sendRecvInfo.recvLength[curRank] = data[i] * dataTypeSize;
                break;
            case 2:
                sendRecvInfo.sendOffset[curRank] = data[i] * dataTypeSize;
                break;
            case 3:
                sendRecvInfo.recvOffset[curRank] = data[i] * dataTypeSize;
                break;
            default:
                break;
        }
    }

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;

    uint32_t subThreadCount = kernelCount - 1;
    std::vector<ThreadHandle> subThreads(
        tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.begin() + 1 + subThreadCount);
    std::vector<u32> notifyIdxMainToSub(subThreadCount, 0);
    CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxMainToSub));

    for (uint32_t i = 0; i < kernelCount; i++) {
        const uint64_t* args = tempFastLaunchCtx.ccuKernelSubmitInfos[i].cachedArgs;
        uint64_t inputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[CACHED_IN_BUFF_OFF];
        uint64_t outputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[CACHED_OUT_BUFF_OFF];
        uint64_t token = args[CACHED_TOKEN];

        std::vector<uint64_t> taskArgs;
        taskArgs.push_back(inputAddr);
        taskArgs.push_back(outputAddr);
        taskArgs.push_back(token);
        for (uint32_t j = 0; j < CACHED_GO_SIZE_NUM; j++) {
            taskArgs.push_back(args[CACHED_GO_SIZE_BASE + j]);
        }

        for (auto peerId : cacheCtx.rankGroup[i]) {
            uint64_t sendSize = sendRecvInfo.sendLength[peerId];
            uint64_t sendOffset = sendRecvInfo.sendOffset[peerId];
            uint64_t recvOffset = sendRecvInfo.recvOffset[peerId];

            if (cacheCtx.is2Plus6 && cacheCtx.closPeers.count(peerId) > 0) {
                uint64_t recvLength = sendRecvInfo.recvLength[peerId];
                uint64_t majorSendSize = static_cast<uint64_t>(sendSize * dieSplitRatio_);
                uint64_t minorSendSize = sendSize - majorSendSize;
                uint64_t majorRecvSize = static_cast<uint64_t>(recvLength * dieSplitRatio_);
                uint64_t minorRecvSize = recvLength - majorRecvSize;
                if (kernelType_[i] == KERNEL_CLOS_MAJOR) {
                    sendOffset += minorSendSize;
                    recvOffset += minorRecvSize;
                    sendSize = majorSendSize;
                } else if (kernelType_[i] == KERNEL_CLOS_MINOR) {
                    sendSize = minorSendSize;
                }
            }

            const uint64_t floorLoopNum = sendSize / UB_MAX_TRANS_SIZE;
            uint64_t sendLoopNum = UINT64_MAX - 1 - floorLoopNum;
            uint64_t sendTailSize = sendSize - floorLoopNum * UB_MAX_TRANS_SIZE;
            auto sendTailGoSize = CalGoSize(sendTailSize, config, GetCcuVersion());
            taskArgs.push_back(sendOffset);
            taskArgs.push_back(recvOffset);
            taskArgs.push_back(sendTailSize);
            for (auto val : sendTailGoSize) {
                taskArgs.push_back(val);
            }
            taskArgs.push_back(sendLoopNum);
        }

        uint64_t argSize = taskArgs.size();
        HCCL_INFO(
            "[CcuTempAlltoAllVMesh1D2Die][FastLaunch] kernel[%u] argSize[%llu] rankGroupSize[%zu]", i, argSize,
            cacheCtx.rankGroup[i].size());
        CcuResult launchRet = HcommCcuKernelLaunch(
            tempFastLaunchCtx.threads[i], tempFastLaunchCtx.ccuKernelSubmitInfos[i].kernelHandle, taskArgs.data(),
            argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    std::vector<u32> notifyIdxSubToMain(subThreadCount);
    for (uint32_t i = 0; i < subThreadCount; i++) {
        notifyIdxSubToMain[i] = i;
    }
    CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxSubToMain));

    HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}
} // namespace ops_hccl
