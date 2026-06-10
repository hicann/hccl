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
#include "ccu_launch_dl.h"
#include "alg_template_base.h"
#include "kernel/ccu_kernel_all_to_all_mesh1d_multi_jetty.h"
#include "ccu_temp_all_to_all_mesh1d_multi_jetty.h"

namespace ops_hccl {
constexpr uint32_t STUB_JETTY_NUM = 1;
CcuTempAllToAllMesh1dMultiJetty::CcuTempAllToAllMesh1dMultiJetty(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    for (u32 i = 0; i < subCommRanks_.size(); i++) {
        for (u32 j = 0; j < subCommRanks_[i].size(); j++) {
            HCCL_INFO("subCommRanks_[%u][%u]=%u", i, j, subCommRanks_[i][j]);
        }
    }
    jettyNums_.assign(templateRankSize_, STUB_JETTY_NUM);
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        myRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempAllToAllMesh1dMultiJetty::~CcuTempAllToAllMesh1dMultiJetty()
{
}

HcclResult CcuTempAllToAllMesh1dMultiJetty::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllToAllMesh1dMultiJetty::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllToAllMesh1DMultiJettyKernel");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllToAllMesh1DMultiJettyKernel);

    auto kernelArg = std::make_shared<CcuKernelArgAllToAllMesh1DMultiJetty>();
    kernelArg->rankSize = templateRankSize_;
    kernelArg->rankId = myRank_;
    kernelArg->opParam = param;
    kernelArg->jettyNums = jettyNums_;
    kernelInfo.setKernelArg(kernelArg);

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs,
                                                     CommTopo::COMM_TOPO_1DMESH));
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllToAllMesh1dMultiJetty::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), templateRankSize_, resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1dMultiJetty::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAllToAllMesh1dMultiJetty::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempAllToAllMesh1dMultiJetty::FastLaunch] start");
    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);

    uint64_t argSize = args[0];
    uint32_t inputIdx = 1;
    uint32_t outputIdx = 2;
    uint32_t inputOffsetIdx = argSize + 1;
    uint32_t outputOffsetIdx = argSize + 2;
    args[inputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    args[outputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];
    void *taskArgs = reinterpret_cast<void*>(args + 1);
    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
                                               tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle,
                                               taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAllToAllMesh1dMultiJetty::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[CcuTempAllToAllMesh1dMultiJetty::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

void CcuTempAllToAllMesh1dMultiJetty::CalcJettySlices(uint64_t sliceSize,
    std::vector<uint64_t>& jettySlice, std::vector<uint64_t>& jettySliceTail)
{
    for (uint32_t rank = 0; rank < templateRankSize_; rank++) {
        uint64_t quotient = sliceSize / jettyNums_[rank] / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        uint64_t tailSlice = sliceSize - quotient * (jettyNums_[rank] - 1);
        jettySlice.push_back(quotient);
        jettySliceTail.push_back(tailSlice);
    }
}

std::vector<uint64_t> CcuTempAllToAllMesh1dMultiJetty::BuildTaskArgs(
    uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
    uint64_t sliceSize, uint64_t srcStride, uint64_t srcOffset, uint64_t dstOffset,
    const std::vector<uint64_t>& goSize, const std::vector<uint64_t>& jettySlice,
    const std::vector<uint64_t>& jettySliceTail)
{
    std::vector<uint64_t> taskArgs = {
        inputAddr, outputAddr, token, sliceSize, srcStride, srcOffset, dstOffset,
        goSize[0], goSize[1], goSize[2], goSize[3]
    };
    for (uint32_t i = 0; i < templateRankSize_; i++) {
        taskArgs.push_back(jettySlice[i]);
    }
    for (uint32_t i = 0; i < templateRankSize_; i++) {
        taskArgs.push_back(jettySliceTail[i]);
    }
    return taskArgs;
}

void CcuTempAllToAllMesh1dMultiJetty::BuildSubmitInfo(TemplateResource& templateResource,
    const std::vector<uint64_t>& taskArgs, uint64_t argSize)
{
    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    uint32_t idx = 0;
    submitInfo.cachedArgs[idx++] = argSize;
    for (const auto& arg : taskArgs) {
        submitInfo.cachedArgs[idx++] = arg;
    }
    submitInfo.cachedArgs[idx++] = buffInfo_.inBuffBaseOff;
    submitInfo.cachedArgs[idx++] = buffInfo_.outBuffBaseOff;
    templateResource.submitInfos.push_back(submitInfo);
}

HcclResult CcuTempAllToAllMesh1dMultiJetty::KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;
    sendCounts_ = templateDataParams.sendCounts;
    recvCounts_ = templateDataParams.recvCounts;
    sdispls_ = templateDataParams.sdispls;
    rdispls_ = templateDataParams.rdispls;
    dataType_ = param.all2AllVDataDes.sendType;
    uint32_t dataTypeSize = SIZE_TABLE[dataType_];

    uint64_t inputAddr  = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t sliceSize     = templateDataParams.sliceSize;
    uint64_t totalSliceSize = (sdispls_[1] - sdispls_[0]) * dataTypeSize;
    uint64_t srcStride = totalSliceSize;
    uint64_t srcOffset = 0;
    uint64_t dstOffset = myRank_ * srcStride;
    HCCL_INFO("sliceSize=%llu, totalSliceSize=%llu, srcStride=%llu, srcOffset=%llu, dstOffset=%llu,"
              " dataType_=%lu, dataTypeSize=%lu", sliceSize, totalSliceSize, srcStride, srcOffset,
              dstOffset, dataType_, dataTypeSize);

    std::vector<uint64_t> jettySlice, jettySliceTail;
    CalcJettySlices(sliceSize, jettySlice, jettySliceTail);

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice     = CCU_MS_SIZE * LOCAL_COPY_MS_PER_LOOP;
    auto goSize = CalGoSize(sliceSize, config);

    uint64_t argSize = 11 + 2 * templateRankSize_;
    std::vector<uint64_t> taskArgs = BuildTaskArgs(inputAddr, outputAddr, token, sliceSize,
        srcStride, srcOffset, dstOffset, goSize, jettySlice, jettySliceTail);

    HCCL_INFO("[CcuTempAllToAllMesh1dMultiJetty::KernelRun] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
              "sliceSize[%llu], srcStride[%llu], srcOffset[%llu], dstOffset[%llu]",
              inputAddr, outputAddr, sliceSize, srcStride, srcOffset, dstOffset);
    CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0],
                                                taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAllToAllMesh1dMultiJetty::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    BuildSubmitInfo(templateResource, taskArgs, argSize);
    HCCL_DEBUG("[CcuTempAllToAllMesh1dMultiJetty::KernelRun] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllToAllMesh1dMultiJetty::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    return 0;
}

u64 CcuTempAllToAllMesh1dMultiJetty::GetThreadNum() const
{
    return 1;
}
} // namespace ops_hccl
