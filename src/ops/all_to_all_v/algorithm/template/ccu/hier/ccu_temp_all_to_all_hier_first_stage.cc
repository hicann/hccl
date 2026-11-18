/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <utility>
#include "channel.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_alg_template_base.h"
#include "ccu_launch_dl.h"
#include "ccu_alltoall_stage_template_registry.h"
#include "ccu_temp_all_to_all_hier_first_stage.h"
#include "kernel/ccu_kernel_all_to_all_hier_first_stage.h"

namespace ops_hccl {

CcuTempAlltoAllHierFirstStage::CcuTempAlltoAllHierFirstStage(
    const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks, u32 stageIndex)
    : CcuTempAlltoAllHierStageBase(param, rankId, subCommRanks, stageIndex)
{}

// 2+6 multilink：双 die 出框链路 → CLOS_MAJOR/CLOS_MINOR 双 kernel 分段并行
HcclResult CcuTempAlltoAllHierFirstStage::BuildDualKernelChannels(
    HcclComm comm, const std::map<uint32_t, std::vector<HcclChannelDesc>>& multiChByDie)
{
    auto it = multiChByDie.begin();
    kernelChannels_[0] = it->second;
    ++it;
    kernelChannels_[1] = it->second;
    kernelCount_ = 2;

    uint32_t majorBw = 0;
    uint32_t minorBw = 0;
    if (!kernelChannels_[0].empty() && !kernelChannels_[1].empty()) {
        CHK_RET(GetChannelBwCoeff(comm, myRank_, kernelChannels_[0][0], majorBw));
        CHK_RET(GetChannelBwCoeff(comm, myRank_, kernelChannels_[1][0], minorBw));
        if (majorBw < minorBw) {
            std::swap(kernelChannels_[0], kernelChannels_[1]);
        }
    }
    // 双 kernel 契约：两桶各持全部对端（组序对齐），任一桶多/少即索引错位
    uint32_t peerNum = static_cast<uint32_t>(subCommRanks_[0].size()) - 1; // 不含自身
    CHK_PRT_RET(
        kernelChannels_[0].size() != peerNum || kernelChannels_[1].size() != peerNum,
        HCCL_ERROR(
            "[CcuTempAlltoAllHierFirstStage][%s] dual-kernel channels(major[%zu] + minor[%zu]) "
            "must each equal peer count[%u].",
            __func__, kernelChannels_[0].size(), kernelChannels_[1].size(), peerNum),
        HcclResult::HCCL_E_INTERNAL);

    uint32_t peerIdx = 0;
    for (u32 rank : subCommRanks_[0]) {
        if (rank == myRank_) {
            continue;
        }
        CHK_PRT_RET(
            kernelChannels_[0][peerIdx].remoteRank != rank || kernelChannels_[1][peerIdx].remoteRank != rank,
            HCCL_ERROR(
                "[CcuTempAlltoAllHierFirstStage][%s] dual-kernel channels not in group order: peer[%u], "
                "major[%u]/minor[%u] at idx[%u].",
                __func__, rank, kernelChannels_[0][peerIdx].remoteRank, kernelChannels_[1][peerIdx].remoteRank,
                peerIdx),
            HcclResult::HCCL_E_INTERNAL);
        peerIdx++;
    }
    return CalcDieSplitRatio(comm, myRank_, true, kernelChannels_[0], kernelChannels_[1], dieSplitRatio_);
}

// 非 2+6：单 kernel 全量，按组序装配（每对端取首条 channel，无则报错）
HcclResult CcuTempAlltoAllHierFirstStage::BuildSingleKernelChannels(
    const std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc)
{
    kernelChannels_[0].clear();
    for (u32 rank : subCommRanks_[0]) {
        if (rank == myRank_) {
            continue;
        }
        auto peerIt = rankIdToChannelDesc.find(rank);
        CHK_PRT_RET(
            peerIt == rankIdToChannelDesc.end() || peerIt->second.empty(),
            HCCL_ERROR(
                "[CcuTempAlltoAllHierFirstStage][%s] no channel for peer rank[%u] in single-kernel path.", __func__,
                rank),
            HcclResult::HCCL_E_INTERNAL);
        kernelChannels_[0].push_back(peerIt->second[0]);
    }
    kernelCount_ = 1;
    kernelChannels_[1].clear();
    dieSplitRatio_ = 1.0;
    return HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllHierFirstStage::PartitionInterBoardChannels(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    std::vector<HcclChannelDesc>& allChannels)
{
    if (topoInfo != nullptr && topoInfo->topoLevelNums > totalStages_) {
        HCCL_INFO(
            "[CcuTempAlltoAllHierFirstStage][%s] flattened %u-level topology on %u-stage sequence: inter-board "
            "peers span net layers, cross-pod peers ride L2 single-port, expect bandwidth degradation.",
            __func__, topoInfo->topoLevelNums, totalStages_);
    }
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    allChannels = channelDescs;

    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc;
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc));

    std::map<uint32_t, std::vector<HcclChannelDesc>> singleChByDie;
    std::map<uint32_t, std::vector<HcclChannelDesc>> multiChByDie;
    CHK_RET(SplitChannelsByDie(comm, myRank_, rankIdToChannelDesc, singleChByDie, multiChByDie, is2Plus6_));

    if (is2Plus6_ && multiChByDie.size() >= 2 && singleChByDie.empty()) {
        CHK_RET(BuildDualKernelChannels(comm, multiChByDie));
    } else {
        CHK_RET(BuildSingleKernelChannels(rankIdToChannelDesc));
    }

    HCCL_INFO(
        "[CcuTempAlltoAllHierFirstStage][%s] rank[%u], is2Plus6[%d], kernelCount[%u], "
        "dieSplitRatio[%f], majorChannels[%zu], minorChannels[%zu]",
        __func__, myRank_, static_cast<int>(is2Plus6_), kernelCount_, dieSplitRatio_, kernelChannels_[0].size(),
        kernelChannels_[1].size());
    return HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllHierFirstStage::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_RET(ValidateStageTopo(topoInfo));
    peerSliceNum_ = topoInfo->userRankSize / templateRankSize_;

    std::vector<HcclChannelDesc> allChannels;
    CHK_RET(PartitionInterBoardChannels(comm, param, topoInfo, allChannels));

    uint32_t slaveThreadNum = kernelCount_ - 1;
    resourceRequest.slaveThreadNum = slaveThreadNum;
    resourceRequest.notifyNumOnMainThread = slaveThreadNum;
    resourceRequest.notifyNumPerThread.assign(slaveThreadNum, 1);
    resourceRequest.channels.emplace_back(allChannels);
    resourceRequest.ccuKernelNum.push_back(kernelCount_);

    for (uint32_t i = 0; i < kernelCount_; i++) {
        CcuKernelInfo kernelInfo;
        CHK_SAFETY_FUNC_RET(
            strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllToAllHierFirstStageKernel"));
        kernelInfo.kernelFunc = reinterpret_cast<void*>(CcuAllToAllHierFirstStageKernel);
        auto kernelArg = std::make_shared<CcuKernelArgAllToAllHierFirstStage>();
        kernelArg->rankId = myRank_;
        kernelArg->myAlgRank = myAlgRank_;
        kernelArg->groupSize = templateRankSize_;
        kernelArg->peerSliceNum = peerSliceNum_;
        kernelArg->rankGroup = subCommRanks_[0];
        kernelInfo.setKernelArg(kernelArg);
        kernelInfo.channels = kernelChannels_[i];
        resourceRequest.ccuKernelInfos.emplace_back(kernelInfo);
        HCCL_INFO(
            "[CcuTempAlltoAllHierFirstStage][CalcRes] kernel[%u], channels[%zu], myAlgRank[%u], "
            "groupSize[%u], peerSliceNum[%u]",
            i, kernelChannels_[i].size(), myAlgRank_, templateRankSize_, peerSliceNum_);
    }

    resourceRequest.dieSplitRatio = dieSplitRatio_;
    return HCCL_SUCCESS;
}

u64 CcuTempAlltoAllHierFirstStage::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

struct CalcParams {
    u64 totalCount = 0;
    u64 dataTypeSize = 0;
    u64 otherDim = 0;
    u64 cclCount = 0;
    u32 kernelCount = 0;
};

static HcclResult ValidateAndCalcParams(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource,
    CalcParams& calc)
{
    CHK_PRT_RET(
        tempAlgParams.buffInfo.inputPtr == nullptr || tempAlgParams.buffInfo.hcclBuff.addr == nullptr,
        HCCL_ERROR(
            "[CcuTempAlltoAllHierFirstStage][%s] inputPtr[%p] or hcclBuff[%p] is null", __func__,
            tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.hcclBuff.addr),
        HcclResult::HCCL_E_PTR);
    CHK_RET(
        CalcUniformSliceParams(param, tempAlgParams, calc.totalCount, calc.dataTypeSize, calc.otherDim, calc.cclCount));

    calc.kernelCount = static_cast<u32>(templateResource.ccuKernels.size());
    CHK_PRT_RET(
        calc.kernelCount == 0 || calc.kernelCount > 2,
        HCCL_ERROR(
            "[CcuTempAlltoAllHierFirstStage][%s] unexpected kernelCount[%u] (expect 1 or 2).", __func__,
            calc.kernelCount),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        templateResource.threads.size() < calc.kernelCount,
        HCCL_ERROR(
            "[CcuTempAlltoAllHierFirstStage][%s] threads[%zu] < kernelCount[%u].", __func__,
            templateResource.threads.size(), calc.kernelCount),
        HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

static HcclResult PreSyncSubThreads(const TemplateResource& templateResource, u32 kernelCount)
{
    u32 subThreadCount = kernelCount - 1;
    if (subThreadCount == 0) {
        return HCCL_SUCCESS;
    }
    std::vector<ThreadHandle> subThreads(
        templateResource.threads.begin() + 1, templateResource.threads.begin() + 1 + subThreadCount);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, std::vector<u32>(subThreadCount, 0)));
    return HCCL_SUCCESS;
}

static HcclResult PostSyncSubThreads(const TemplateResource& templateResource, u32 kernelCount)
{
    u32 subThreadCount = kernelCount - 1;
    if (subThreadCount == 0) {
        return HCCL_SUCCESS;
    }
    std::vector<ThreadHandle> subThreads(
        templateResource.threads.begin() + 1, templateResource.threads.begin() + 1 + subThreadCount);
    std::vector<u32> notifyIdxSubToMain(subThreadCount);
    for (u32 i = 0; i < subThreadCount; i++) {
        notifyIdxSubToMain[i] = i;
    }
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    return HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllHierFirstStage::CalcSegSize(u32 kernelIdx, u64 fullSliceSize, u64& segSize, u64& segOffset)
{
    segSize = fullSliceSize;
    segOffset = 0;
    if (!is2Plus6_) {
        return HCCL_SUCCESS;
    }
    uint64_t majorSize = static_cast<uint64_t>(fullSliceSize * dieSplitRatio_);
    if (fullSliceSize < 2 * CCU_2DIE_SPLIT_ALIGN_SIZE) {
        majorSize = fullSliceSize;
    } else {
        if (majorSize < CCU_2DIE_SPLIT_ALIGN_SIZE) {
            majorSize = CCU_2DIE_SPLIT_ALIGN_SIZE;
        }
        if (majorSize > fullSliceSize - CCU_2DIE_SPLIT_ALIGN_SIZE) {
            majorSize = fullSliceSize - CCU_2DIE_SPLIT_ALIGN_SIZE;
        }
        // 分段边界按 128B 向下对齐，保证两段偏移/长度均对齐
        majorSize = CcuAlignDown(majorSize, CCU_2DIE_SPLIT_ALIGN_SIZE);
    }
    if (kernelIdx == 0) {
        segSize = majorSize;
    } else {
        segSize = fullSliceSize - majorSize;
        segOffset = majorSize;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllHierFirstStage::LaunchKernels(
    u32 kernelCount, const LaunchSliceDesc& desc, const LoopGroupConfig& config,
    const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    for (u32 i = 0; i < kernelCount; i++) {
        u64 segSize = 0;
        u64 segOffset = 0;
        CHK_RET(CalcSegSize(i, tempAlgParams.sliceSize, segSize, segOffset));
        auto goSize = CalGoSize(segSize, config, GetCcuVersion());
        std::vector<uint64_t> taskArgs
            = {desc.inputAddr,
               desc.cclBuffAddr,
               desc.token,
               segSize,
               segOffset,
               buffInfo_.inBuffBaseOff,
               buffInfo_.hcclBuffBaseOff,
               desc.blockStride,
               desc.slotStride,
               goSize[0],
               goSize[1],
               goSize[2],
               goSize[3]};
        HCCL_INFO(
            "[CcuTempAlltoAllHierFirstStage][KernelRun] kernel[%u], segSize[%llu], sliceOffset[%llu], "
            "sliceSize[%llu], blockStride[%llu], slotStride[%llu]",
            i, segSize, segOffset, tempAlgParams.sliceSize, desc.blockStride, desc.slotStride);
        CcuResult launchRet = HcommCcuKernelLaunch(
            templateResource.threads[i], templateResource.ccuKernels[i], taskArgs.data(),
            static_cast<uint32_t>(taskArgs.size()));
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAlltoAllHierFirstStage][KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
        CcuKernelSubmitInfo submitInfo{};
        CHK_RET(FillCachedArgs(
            submitInfo, taskArgs[0], taskArgs[1], taskArgs[2], taskArgs[3], taskArgs[4], taskArgs[5], taskArgs[6],
            taskArgs[7], taskArgs[8], taskArgs[9], taskArgs[10], taskArgs[11], taskArgs[12]));
        submitInfo.kernelHandle = templateResource.ccuKernels[i];
        templateResource.submitInfos.push_back(submitInfo);
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllHierFirstStage::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    buffInfo_ = tempAlgParams.buffInfo;
    CalcParams calc{};
    CHK_RET(ValidateAndCalcParams(param, tempAlgParams, templateResource, calc));

    LaunchSliceDesc desc;
    desc.inputAddr = PointerToAddr(buffInfo_.inputPtr);
    desc.cclBuffAddr = PointerToAddr(buffInfo_.hcclBuff.addr);
    uint64_t token = 0;
    CHK_RET(GetToken(buffInfo_, token));
    desc.token = token;
    desc.blockStride = calc.totalCount * calc.dataTypeSize;
    desc.slotStride = calc.cclCount * calc.dataTypeSize;

    if (templateResource.dieSplitRatio > 0.0) {
        dieSplitRatio_ = templateResource.dieSplitRatio;
    }
    is2Plus6_ = (calc.kernelCount == 2);

    CHK_RET(PreSyncSubThreads(templateResource, calc.kernelCount));

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;

    CHK_RET(LaunchKernels(calc.kernelCount, desc, config, tempAlgParams, templateResource));

    CHK_RET(PostSyncSubThreads(templateResource, calc.kernelCount));
    return HCCL_SUCCESS;
}

HcclResult
CcuTempAlltoAllHierFirstStage::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    (void)param;
    constexpr u32 inputAddrIdx = 0;
    constexpr u32 cclBuffAddrIdx = 1;
    constexpr u32 taskArgSize = 13;

    u32 kernelCount = static_cast<u32>(tempFastLaunchCtx.ccuKernelSubmitInfos.size());
    if (kernelCount == 0) {
        HCCL_INFO("[CcuTempAlltoAllHierFirstStage][FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        tempFastLaunchCtx.threads.size() < kernelCount,
        HCCL_ERROR(
            "[CcuTempAlltoAllHierFirstStage][FastLaunch] threads[%zu] < kernelCount[%u].",
            tempFastLaunchCtx.threads.size(), kernelCount),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        tempFastLaunchCtx.buffInfo.inputPtr == nullptr || tempFastLaunchCtx.buffInfo.hcclBuff.addr == nullptr,
        HCCL_ERROR(
            "[CcuTempAlltoAllHierFirstStage][FastLaunch] inputPtr[%p] or hcclBuff[%p] is null",
            tempFastLaunchCtx.buffInfo.inputPtr, tempFastLaunchCtx.buffInfo.hcclBuff.addr),
        HcclResult::HCCL_E_PTR);

    uint64_t inputBase = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr);
    uint64_t cclBuffBase = PointerToAddr(tempFastLaunchCtx.buffInfo.hcclBuff.addr);

    if (kernelCount > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxMainToSub(kernelCount - 1, 0);
        CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxMainToSub));
    }

    for (u32 i = 0; i < kernelCount; i++) {
        uint64_t* args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[i].cachedArgs);
        args[inputAddrIdx] = inputBase;
        args[cclBuffAddrIdx] = cclBuffBase;

        CcuResult launchRet = HcommCcuKernelLaunch(
            tempFastLaunchCtx.threads[i], tempFastLaunchCtx.ccuKernelSubmitInfos[i].kernelHandle,
            reinterpret_cast<void*>(args), taskArgSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAlltoAllHierFirstStage][FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
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
    return HCCL_SUCCESS;
}

REGISTER_CCU_A2A_STAGE_TEMPLATE(FirstMeshStage, CcuTempAlltoAllHierFirstStage);

} // namespace ops_hccl
