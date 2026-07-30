/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_all_to_all_concurrent_mesh_nhr.h"
#include "alg_data_trans_wrapper.h"
#include "alg_template_base.h"
#include "channel.h"
#include "ccu_launch_dl.h"
#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_utils.h"
#include "template_utils.h"

namespace ops_hccl {

// notify indices for 2-thread concurrent sync
constexpr u32 NOTIFY_IDX_PRE_SYNC = 0;   // PreSync: threads[0] -> threads[1]
constexpr u32 NOTIFY_IDX_POST_SYNC = 0;  // PostSync: threads[1] -> threads[0]
constexpr u32 CLOS_BW_CONSTANT = 8;

CcuTempAllToAllConcurrentMeshNHR::CcuTempAllToAllConcurrentMeshNHR(
    const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    if (!subCommRanks.empty() && !subCommRanks[0].empty()) {       
        auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
        templateRankSize_ = subCommRanks[0].size();
        if (it != subCommRanks[0].end()) {
            mySubCommRank_ = static_cast<uint32_t>(std::distance(subCommRanks[0].begin(), it));
        }
    }
}

HcclResult CcuTempAllToAllConcurrentMeshNHR::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    std::vector<std::vector<u32>> meshSubCommRanks = {subCommRanks_[0]};
    std::vector<std::vector<u32>> closSubCommRanks = {subCommRanks_[1]};
    CcuTempAlltoAllMesh1D meshSub(param, myRank_, meshSubCommRanks);
    CcuTempAlltoAllMesh1D closSub(param, myRank_, closSubCommRanks);

    AlgResourceRequest meshReq;
    AlgResourceRequest closReq;
    CHK_RET(meshSub.CalcRes(comm, param, topoInfo, meshReq));
    CHK_RET(closSub.CalcRes(comm, param, topoInfo, closReq));

    // Replace clos channels with die 0 (CLOS port) channels via layer 1 links.
    // Mesh kernel uses layer 0 (die 1) channels from CalcChannelRequestMesh1D above.
    // Different die -> different CKE ID pool -> no CKE conflict between mesh and clos kernels.
    std::vector<HcclChannelDesc> closChannelDescs;
    CHK_RET(CalcChannelRequestMesh1DLevel1(comm, param, topoInfo, closSubCommRanks, closChannelDescs));
    closReq.ccuKernelInfos[0].channels = closChannelDescs;

    // 2 threads: mesh main + clos main (executor slave)
    // No outer inter-thread sync needed: each kernel has its own internal
    // channel-based PreSync/PostSync, and mesh/clos use independent channels.
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumOnMainThread = 1;  // PostSync: main WAIT + slave->main RECORD
    resourceRequest.notifyNumPerThread.emplace_back(1);  // PreSync: slave waits

    // CCU kernel: mesh 1 + clos 1
    resourceRequest.ccuKernelNum.emplace_back(meshReq.ccuKernelNum[0]);
    resourceRequest.ccuKernelNum.emplace_back(closReq.ccuKernelNum[0]);
    resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                          meshReq.ccuKernelInfos.begin(),
                                          meshReq.ccuKernelInfos.end());
    resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                          closReq.ccuKernelInfos.begin(),
                                          closReq.ccuKernelInfos.end());

    HCCL_INFO("[CcuTempAllToAllConcurrentMeshNHR][CalcRes] rank[%u] slaveThreadNum[%u], "
              "notifyNumOnMainThread[%u], ccuKernelNum[%zu]",
              myRank_, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread,
              resourceRequest.ccuKernelNum.size());
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllConcurrentMeshNHR::GetRes(AlgResourceRequest &resourceRequest) const
{
    resourceRequest.slaveThreadNum = 1;  // clos main thread
    resourceRequest.notifyNumOnMainThread = 1;  // PostSync: main WAIT + slave->main RECORD
    resourceRequest.notifyNumPerThread.emplace_back(1);  // PreSync: slave waits
    return HCCL_SUCCESS;
}

u64 CcuTempAllToAllConcurrentMeshNHR::GetThreadNum() const
{
    return 2;  // mesh main + clos main
}

u64 CcuTempAllToAllConcurrentMeshNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

void CcuTempAllToAllConcurrentMeshNHR::CalcDataSplit(
    u64 totalSize, u64 dataTypeSize, u64 &meshSize, u64 &closSize) const
{
    u32 meshBw = (templateRankSize_ > 1) ? (templateRankSize_ - 1) : 1;
    u32 closBw = CLOS_BW_CONSTANT;
    u32 factor = meshBw + closBw;
    double splitRatio = static_cast<double>(meshBw) / static_cast<double>(factor);
    u64 sliceAlign = HCCL_MIN_SLICE_ALIGN;
    if (dataTypeSize > 0) {
        u64 alignCount = sliceAlign / dataTypeSize;
        if (alignCount == 0) {
            alignCount = 1;
        }
        u64 totalCount = totalSize / dataTypeSize;
        u64 meshCount = static_cast<u64>(std::floor(splitRatio * static_cast<double>(totalCount)));
        meshCount = meshCount / alignCount * alignCount;
        meshSize = meshCount * dataTypeSize;
    } else {
        meshSize = static_cast<u64>(std::floor(splitRatio * static_cast<double>(totalSize)));
        meshSize = meshSize / sliceAlign * sliceAlign;
    }
    closSize = totalSize - meshSize;
    HCCL_INFO("[CcuTempAllToAllConcurrentMeshNHR][CalcDataSplit] totalSize[%llu], meshSize[%llu], "
              "closSize[%llu], meshBw[%u], closBw[%u]", totalSize, meshSize, closSize, meshBw, closBw);
}

HcclResult CcuTempAllToAllConcurrentMeshNHR::BuildMeshTaskArgs(
    const TemplateDataParams &templateDataParams, u64 meshSliceSize,
    uint64_t token, std::vector<uint64_t> &meshTaskArgs)
{
    const BuffInfo &buff = templateDataParams.buffInfo;
    uint64_t inputAddr = PointerToAddr(buff.inputPtr) + buff.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buff.outputPtr) + buff.outBuffBaseOff;
    uint64_t srcStride = templateDataParams.outputSliceStride;
    uint64_t srcOffset = 0;
    uint64_t dstOffset = static_cast<uint64_t>(myRank_) * srcStride;
    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE * LOCAL_COPY_MS_PER_LOOP;
    auto goSize = CalGoSize(meshSliceSize, config, GetCcuVersion());
    meshTaskArgs = {inputAddr, outputAddr, token, meshSliceSize,
                    srcStride, srcOffset, dstOffset,
                    goSize[0], goSize[1], goSize[2], goSize[3]};
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllConcurrentMeshNHR::BuildClosTaskArgs(
    const TemplateDataParams &templateDataParams, u64 meshSliceSize,
    u64 closSliceSize, uint64_t token, std::vector<uint64_t> &closTaskArgs)
{
    const BuffInfo &buff = templateDataParams.buffInfo;
    uint64_t inputAddr = PointerToAddr(buff.inputPtr) + buff.inBuffBaseOff + meshSliceSize;
    uint64_t outputAddr = PointerToAddr(buff.outputPtr) + buff.outBuffBaseOff;
    uint64_t srcStride = templateDataParams.outputSliceStride;
    uint64_t srcOffset = 0;
    uint64_t closDstOffset = static_cast<uint64_t>(myRank_) * srcStride + meshSliceSize;
    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE * LOCAL_COPY_MS_PER_LOOP;
    auto goSize = CalGoSize(closSliceSize, config, GetCcuVersion());
    closTaskArgs = {inputAddr, outputAddr, token, closSliceSize,
                    srcStride, srcOffset, closDstOffset,
                    goSize[0], goSize[1], goSize[2], goSize[3]};
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllConcurrentMeshNHR::LaunchConcurrentKernels(
    TemplateResource &templateResource, bool hasMesh, bool hasClos,
    const std::vector<uint64_t> &meshTaskArgs, const std::vector<uint64_t> &closTaskArgs)
{
    if (hasClos && templateResource.threads.size() >= 2) {
        CHK_RET(PreSyncInterThreads(templateResource.threads[0],
            {templateResource.threads[1]}, {NOTIFY_IDX_PRE_SYNC}));
    }
    if (hasMesh) {
        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0],
            templateResource.ccuKernels[0], const_cast<uint64_t*>(meshTaskArgs.data()),
            CcuAlltoAllMesh1DArgLayout::ARG_SIZE);
        CHK_PRT_RET(launchRet != CCU_SUCCESS,
            HCCL_ERROR("[CcuTempAllToAllConcurrentMeshNHR] mesh kernel launch failed, ccuRet -> %d", launchRet),
            ConvertCcuToHccl(launchRet));
    }
    if (hasClos) {
        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[1],
            templateResource.ccuKernels[1], const_cast<uint64_t*>(closTaskArgs.data()),
            CcuAlltoAllMesh1DArgLayout::ARG_SIZE);
        CHK_PRT_RET(launchRet != CCU_SUCCESS,
            HCCL_ERROR("[CcuTempAllToAllConcurrentMeshNHR] clos kernel launch failed, ccuRet -> %d", launchRet),
            ConvertCcuToHccl(launchRet));
    }
    if (hasClos && templateResource.threads.size() >= 2) {
        CHK_RET(PostSyncInterThreads(templateResource.threads[0],
            {templateResource.threads[1]}, {NOTIFY_IDX_POST_SYNC}));
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllConcurrentMeshNHR::SaveSubmitInfos(
    TemplateResource &templateResource, const std::vector<uint64_t> &meshTaskArgs,
    const std::vector<uint64_t> &closTaskArgs, u64 meshSliceSize,
    bool hasMesh, bool hasClos, const BuffInfo &buff)
{
    if (hasMesh) {
        CcuKernelSubmitInfo meshSubmit;
        meshSubmit.kernelHandle = templateResource.ccuKernels[0];
        CHK_RET(FillCachedArgs(meshSubmit,
            meshTaskArgs[CcuAlltoAllMesh1DArgLayout::INPUT],
            meshTaskArgs[CcuAlltoAllMesh1DArgLayout::OUTPUT],
            meshTaskArgs[CcuAlltoAllMesh1DArgLayout::TOKEN],
            meshTaskArgs[CcuAlltoAllMesh1DArgLayout::SLICE_SIZE],
            meshTaskArgs[CcuAlltoAllMesh1DArgLayout::SRC_STRIDE],
            meshTaskArgs[CcuAlltoAllMesh1DArgLayout::SRC_OFFSET],
            meshTaskArgs[CcuAlltoAllMesh1DArgLayout::DST_OFFSET],
            meshTaskArgs[CcuAlltoAllMesh1DArgLayout::GO_SIZE_0],
            meshTaskArgs[CcuAlltoAllMesh1DArgLayout::GO_SIZE_1],
            meshTaskArgs[CcuAlltoAllMesh1DArgLayout::GO_SIZE_2],
            meshTaskArgs[CcuAlltoAllMesh1DArgLayout::GO_SIZE_3],
            buff.inBuffBaseOff,
            buff.outBuffBaseOff));
        templateResource.submitInfos.push_back(meshSubmit);
    }
    if (hasClos) {
        CcuKernelSubmitInfo closSubmit;
        closSubmit.kernelHandle = templateResource.ccuKernels[1];
        CHK_RET(FillCachedArgs(closSubmit,
            closTaskArgs[CcuAlltoAllMesh1DArgLayout::INPUT],
            closTaskArgs[CcuAlltoAllMesh1DArgLayout::OUTPUT],
            closTaskArgs[CcuAlltoAllMesh1DArgLayout::TOKEN],
            closTaskArgs[CcuAlltoAllMesh1DArgLayout::SLICE_SIZE],
            closTaskArgs[CcuAlltoAllMesh1DArgLayout::SRC_STRIDE],
            closTaskArgs[CcuAlltoAllMesh1DArgLayout::SRC_OFFSET],
            closTaskArgs[CcuAlltoAllMesh1DArgLayout::DST_OFFSET],
            closTaskArgs[CcuAlltoAllMesh1DArgLayout::GO_SIZE_0],
            closTaskArgs[CcuAlltoAllMesh1DArgLayout::GO_SIZE_1],
            closTaskArgs[CcuAlltoAllMesh1DArgLayout::GO_SIZE_2],
            closTaskArgs[CcuAlltoAllMesh1DArgLayout::GO_SIZE_3],
            buff.inBuffBaseOff + meshSliceSize,
            buff.outBuffBaseOff));
        templateResource.submitInfos.push_back(closSubmit);
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllConcurrentMeshNHR::PatchMeshArgs(const TemplateFastLaunchCtx &ctx)
{
    uint64_t *args = const_cast<uint64_t*>(ctx.ccuKernelSubmitInfos[0].cachedArgs);
    args[CcuAlltoAllMesh1DArgLayout::INPUT] =
        PointerToAddr(ctx.buffInfo.inputPtr) + args[CcuAlltoAllMesh1DArgLayout::IN_BUFF_BASE_OFF];
    args[CcuAlltoAllMesh1DArgLayout::OUTPUT] =
        PointerToAddr(ctx.buffInfo.outputPtr) + args[CcuAlltoAllMesh1DArgLayout::OUT_BUFF_BASE_OFF];
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllConcurrentMeshNHR::PatchClosArgs(const TemplateFastLaunchCtx &ctx, u32 meshKernelNum)
{
    uint64_t *args = const_cast<uint64_t*>(ctx.ccuKernelSubmitInfos[meshKernelNum].cachedArgs);
    args[CcuAlltoAllMesh1DArgLayout::INPUT] =
        PointerToAddr(ctx.buffInfo.inputPtr) + args[CcuAlltoAllMesh1DArgLayout::IN_BUFF_BASE_OFF];
    args[CcuAlltoAllMesh1DArgLayout::OUTPUT] =
        PointerToAddr(ctx.buffInfo.outputPtr) + args[CcuAlltoAllMesh1DArgLayout::OUT_BUFF_BASE_OFF];
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllConcurrentMeshNHR::KernelRun(
    const OpParam &param, const TemplateDataParams &templateDataParams,
    TemplateResource &templateResource)
{
    HCCL_INFO("[CcuTempAllToAllConcurrentMeshNHR][KernelRun] rank[%u] start.", myRank_);

    u64 dataTypeSize = HCCL_SIZE_TABLE[param.all2AllDataDes.sendType];
    u64 meshSliceSize = 0;
    u64 closSliceSize = 0;
    CalcDataSplit(templateDataParams.sliceSize, dataTypeSize, meshSliceSize, closSliceSize);

    if (meshSliceSize == 0 && closSliceSize == 0) {
        HCCL_INFO("[CcuTempAllToAllConcurrentMeshNHR][KernelRun] both zero, skip.");
        return HCCL_SUCCESS;
    }

    bool hasMesh = (meshSliceSize > 0 && templateResource.ccuKernels.size() > 0);
    bool hasClos = (closSliceSize > 0 && templateResource.ccuKernels.size() > 1);

    uint64_t token;
    CHK_RET(GetToken(templateDataParams.buffInfo, token));

    std::vector<uint64_t> meshTaskArgs;
    if (hasMesh) {
        CHK_RET(BuildMeshTaskArgs(templateDataParams, meshSliceSize, token, meshTaskArgs));
    }
    std::vector<uint64_t> closTaskArgs;
    if (hasClos) {
        CHK_RET(BuildClosTaskArgs(templateDataParams, meshSliceSize, closSliceSize, token, closTaskArgs));
    }

    CHK_RET(LaunchConcurrentKernels(templateResource, hasMesh, hasClos, meshTaskArgs, closTaskArgs));
    CHK_RET(SaveSubmitInfos(templateResource, meshTaskArgs, closTaskArgs, meshSliceSize,
                            hasMesh, hasClos, templateDataParams.buffInfo));

    HCCL_INFO("[CcuTempAllToAllConcurrentMeshNHR][KernelRun] rank[%u] end.", myRank_);
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllConcurrentMeshNHR::FastLaunch(
    const OpParam &param, const TemplateFastLaunchCtx &tempFastLaunchCtx)
{
    (void)param;
    u32 totalKernelNum = static_cast<u32>(tempFastLaunchCtx.ccuKernelSubmitInfos.size());
    if (totalKernelNum == 0) {
        HCCL_INFO("[CcuTempAllToAllConcurrentMeshNHR::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    if (tempFastLaunchCtx.threads.size() < 1) {
        HCCL_ERROR("[CcuTempAllToAllConcurrentMeshNHR::FastLaunch] thread num is 0.");
        return HCCL_E_INTERNAL;
    }

    u32 meshKernelNum = 1;
    u32 closKernelNum = (totalKernelNum > meshKernelNum) ? (totalKernelNum - meshKernelNum) : 0;
    bool hasMesh = (meshKernelNum > 0);
    bool hasClos = (closKernelNum > 0);

    if (hasMesh) {
        CHK_RET(PatchMeshArgs(tempFastLaunchCtx));
    }
    if (hasClos) {
        CHK_RET(PatchClosArgs(tempFastLaunchCtx, meshKernelNum));
    }

    std::vector<uint64_t> meshArgs;
    std::vector<uint64_t> closArgs;
    if (hasMesh) {
        const auto &si = tempFastLaunchCtx.ccuKernelSubmitInfos[0];
        meshArgs.assign(si.cachedArgs, si.cachedArgs + CcuAlltoAllMesh1DArgLayout::ARG_SIZE);
    }
    if (hasClos) {
        const auto &si = tempFastLaunchCtx.ccuKernelSubmitInfos[meshKernelNum];
        closArgs.assign(si.cachedArgs, si.cachedArgs + CcuAlltoAllMesh1DArgLayout::ARG_SIZE);
    }

    TemplateResource tmpRes;
    tmpRes.threads = tempFastLaunchCtx.threads;
    tmpRes.ccuKernels.clear();
    for (const auto &si : tempFastLaunchCtx.ccuKernelSubmitInfos) {
        tmpRes.ccuKernels.push_back(si.kernelHandle);
    }

    CHK_RET(LaunchConcurrentKernels(tmpRes, hasMesh, hasClos, meshArgs, closArgs));

    HCCL_DEBUG("[CcuTempAllToAllConcurrentMeshNHR::FastLaunch] end");
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
