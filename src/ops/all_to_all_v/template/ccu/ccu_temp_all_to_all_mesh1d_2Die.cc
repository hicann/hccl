/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iterator>
#include <utility>
#include "channel.h"
#include "channel_request.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_launch_dl.h"

#include "ccu_temp_all_to_all_mesh1d_2Die.h"
#include "ccu_kernel_all_to_all_mesh2die.h"
#include "ccu_temp_all_to_all_mesh_1D.h"
#include "ccu_kernel_all_to_all_mesh1d.h"
#include "template_utils.h"

namespace ops_hccl {

CcuTempAllToAllMesh1D2Die::CcuTempAllToAllMesh1D2Die(
    const OpParam& param, RankId rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    for (u32 i = 0; i < subCommRanks_.size(); i++) {
        for (u32 j = 0; j < subCommRanks_[i].size(); j++) {
            HCCL_INFO("subCommRanks_[%u][%u]=%u", i, j, subCommRanks_[i][j]);
        }
    }

    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        myRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempAllToAllMesh1D2Die::~CcuTempAllToAllMesh1D2Die() {}

HcclResult CcuTempAllToAllMesh1D2Die::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_PRT_RET(
        subCommRanks_.size() != 1 || subCommRanks_[0].empty(),
        HCCL_ERROR(
            "[CcuTempAllToAllMesh1D2Die][CalcRes] Invalid subCommRanks[%u] or subCommRanks empty.",
            subCommRanks_.size()),
        HcclResult::HCCL_E_INTERNAL);
    HCCL_DEBUG(
        "[CcuTempAllToAllMesh1D2Die][CalcRes] rankSize[%u] subCommRanks0[%u].", templateRankSize_,
        subCommRanks_[0].size());

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));
    HCCL_INFO("channelDescs size[%u]", channelDescs.size());

    CHK_RET(PartitionChannels(comm, rankIdToChannelDesc_));
    double ratio = 1.0;
    CHK_RET(CalcDieSplitRatio(
        comm, myRank_, is2Plus6_, kernelChannels_[KERNEL_CLOS_MAJOR], kernelChannels_[KERNEL_CLOS_MINOR], ratio));
    resourceRequest.dieSplitRatio = ratio;
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
            strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllToAllMesh2DieKernel"));
        kernelInfo.kernelFunc = reinterpret_cast<void*>(CcuAllToAllMesh2DieKernel);
        auto kernelArg = std::make_shared<CcuKernelArgAllToAllMesh2Die>();
        kernelArg->rankId = myRank_;
        kernelArg->opParam = param;
        kernelArg->subCommRanks = subCommRanks_;
        kernelArg->withMyRank = kernelWithMyRank_[i];
        kernelArg->rankGroup = kernelRankGroup_[i];
        kernelInfo.setKernelArg(kernelArg);
        kernelInfo.channels = kernelChannels_[i];
        resourceRequest.ccuKernelInfos.emplace_back(kernelInfo);
        HCCL_DEBUG(
            "[CcuTempAllToAllMesh1D2Die][CalcRes] kernel[%u], channels=%llu, withMyRank=%u, ccuKernelInfos=%llu", i,
            kernelChannels_[i].size(), kernelWithMyRank_[i], resourceRequest.ccuKernelInfos.size());
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::PartitionChannels(
    HcclComm comm, std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc)
{
    std::map<uint32_t, std::vector<HcclChannelDesc>> singleChByDie, multiChByDie;
    CHK_RET(SplitChannelsByDie(comm, myRank_, rankIdToChannelDesc, singleChByDie, multiChByDie, is2Plus6_));
    CHK_RET(PartitionChannelsFor2Die(
        singleChByDie, multiChByDie, is2Plus6_, myRank_, kernelCount_, fullmeshDieId_, kernelChannels_,
        kernelRankGroup_, "CcuTempAllToAllMesh1D2Die"));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::CalcFillArgsInfo(uint32_t kernelIdx, uint64_t& sliceSize, uint64_t& sliceOffset)
{
    const uint64_t full = sliceSize;
    if (kernelIdx == KERNEL_FULLMESH) {
        sliceSize = full;
        sliceOffset = 0;
    } else if (is2Plus6_ && kernelIdx == KERNEL_CLOS_MAJOR) {
        sliceSize = static_cast<uint64_t>(full * dieSplitRatio_);
        sliceOffset = 0;
    } else if (is2Plus6_ && kernelIdx == KERNEL_CLOS_MINOR) {
        uint64_t majorSize = static_cast<uint64_t>(full * dieSplitRatio_);
        sliceSize = full - majorSize;
        sliceOffset = majorSize;
    } else {
        sliceSize = full;
        sliceOffset = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::LaunchKernels(
    uint32_t kernelCount, uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t sliceStride,
    const LoopGroupConfig& config, const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    for (uint32_t i = 0; i < kernelCount; i++) {
        uint64_t sliceSize = templateDataParams.sliceSize;
        uint64_t sliceOffset = 0;
        CHK_RET(CalcFillArgsInfo(i, sliceSize, sliceOffset));
        auto goSize = CalGoSize(sliceSize, config, GetCcuVersion());
        std::vector<uint64_t> taskArgs
            = {inputAddr + sliceOffset, outputAddr, token, sliceSize, sliceStride, sliceStride * myRank_ + sliceOffset};
        for (auto val : goSize) {
            taskArgs.push_back(val);
        }
        HCCL_INFO(
            "[CcuTempAllToAllMesh1D2Die][KernelRun] kernel[%u] sliceSize[%llu] sliceOffset[%llu] "
            "rankGroupSize[%zu] withMyRank[%u]",
            i, sliceSize, sliceOffset, kernelRankGroup_[i].size(), kernelWithMyRank_[i]);
        CcuResult launchRet = HcommCcuKernelLaunch(
            templateResource.threads[i], templateResource.ccuKernels[i], taskArgs.data(), taskArgs.size());
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllToAllMesh1D2Die][KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
        CcuKernelSubmitInfo submitInfo;
        CHK_RET(FillCachedArgs(
            submitInfo, taskArgs[0], taskArgs[1], taskArgs[2], taskArgs[3], taskArgs[4], taskArgs[5], taskArgs[6],
            taskArgs[7], taskArgs[8], taskArgs[9], buffInfo_.outBuffBaseOff));
        submitInfo.cachedArgs[11] = buffInfo_.inBuffBaseOff + sliceOffset;
        submitInfo.kernelHandle = templateResource.ccuKernels[i];
        templateResource.submitInfos.push_back(submitInfo);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::KernelRun(
    const OpParam& param, const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempAllToAllMesh1D2Die] Run");
    opMode_ = param.opMode;
    buffInfo_ = templateDataParams.buffInfo;
    CHK_PRT_RET(
        subCommRanks_.empty() || subCommRanks_[0].empty(),
        HCCL_ERROR("[CcuTempAllToAllMesh1D2Die][KernelRun] subCommRanks empty."), HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        buffInfo_.inputPtr == nullptr || buffInfo_.outputPtr == nullptr,
        HCCL_ERROR(
            "[CcuTempAllToAllMesh1D2Die][KernelRun] Rank[%d] input[%#llx] or output[%#llx] is null", myRank_,
            buffInfo_.inputPtr, buffInfo_.outputPtr),
        HcclResult::HCCL_E_PTR);

    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t outputSliceStride
        = templateDataParams.sdispls[1] * DATATYPE_SIZE_TABLE[param.all2AllDataDes.recvType] - buffInfo_.inBuffBaseOff;
    HCCL_INFO(
        "[CcuTempAllToAllMesh1D2Die][KernelRun] begin. Rank[%d], input[%#llx/%#llx], output[%#llx/%#llx]", myRank_,
        inputAddr, param.inputPtr, outputAddr, param.outputPtr);
    uint32_t kernelCount = templateResource.ccuKernels.size();
    is2Plus6_ = (kernelCount == MAX_KERNEL_NUM_2DIE);
    if (templateResource.dieSplitRatio > 0.0) {
        dieSplitRatio_ = templateResource.dieSplitRatio;
    }

    uint32_t subThreadCount = kernelCount - 1;
    std::vector<ThreadHandle> subThreads(
        templateResource.threads.begin() + 1, templateResource.threads.begin() + 1 + subThreadCount);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, std::vector<u32>(subThreadCount, 0)));

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
    CHK_RET(LaunchKernels(
        kernelCount, inputAddr, outputAddr, token, outputSliceStride, config, templateDataParams, templateResource));

    std::vector<u32> notifyIdxSubToMain(subThreadCount);
    for (uint32_t i = 0; i < subThreadCount; i++) {
        notifyIdxSubToMain[i] = i;
    }
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    HCCL_DEBUG("[CcuTempAlltoAllMesh1D2Die][KernelRun] end. Rank[%d]", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    constexpr u32 argInIdx = 0;
    constexpr u32 argOutIdx = 1;
    constexpr u32 metaOutBaseOffIdx = 10;
    constexpr u32 metaInCombineOffIdx = 11;
    constexpr u32 taskArgSize = 10;

    u32 kernelCount = static_cast<u32>(tempFastLaunchCtx.ccuKernelSubmitInfos.size());
    if (kernelCount == 0) {
        HCCL_INFO("[CcuTempAllToAllMesh1D2Die][FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempAllToAllMesh1D2Die][FastLaunch] start, kernelCount[%u]", kernelCount);

    uint64_t inputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr);
    uint64_t outputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr);

    if (kernelCount > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxMainToSub(kernelCount - 1, 0);
        CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxMainToSub));
    }

    for (u32 i = 0; i < kernelCount; i++) {
        uint64_t* args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[i].cachedArgs);
        args[argInIdx] = inputAddr + args[metaInCombineOffIdx];
        args[argOutIdx] = outputAddr + args[metaOutBaseOffIdx];

        CcuResult launchRet = HcommCcuKernelLaunch(
            tempFastLaunchCtx.threads[i], tempFastLaunchCtx.ccuKernelSubmitInfos[i].kernelHandle,
            reinterpret_cast<void*>(args), taskArgSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllToAllMesh1D2Die][FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    if (kernelCount > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxSubToMain(kernelCount - 1);
        for (u32 i = 0; i < kernelCount - 1; i++) {
            notifyIdxSubToMain[i] = i;
        }
        CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxSubToMain));
    }
    HCCL_DEBUG("[CcuTempAllToAllMesh1D2Die][FastLaunch] end");
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
