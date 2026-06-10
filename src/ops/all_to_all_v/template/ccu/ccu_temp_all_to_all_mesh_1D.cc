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
#include "kernel/ccu_kernel_all_to_all_mesh1d.h"
#include "ccu_temp_all_to_all_mesh_1D.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

CcuTempAlltoAllMesh1D::CcuTempAlltoAllMesh1D(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    tempRankSize_ = subCommRanks[0].size();
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
}

CcuTempAlltoAllMesh1D::~CcuTempAlltoAllMesh1D()
{
}

HcclResult CcuTempAlltoAllMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // kernel数量
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAlltoAllMesh1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    typeSize_ = DataTypeSizeGet(param.all2AllDataDes.sendType);
    CcuKernelInfo kernelInfo;

    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelAlltoAllMesh1D");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAlltoAllMesh1DKernel);
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    auto kernelArg = std::make_shared<CcuKernelArgAlltoAllMesh1D>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->loadFromMem = param.isMc2;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAlltoAllMeshMem2Mem1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

// device侧调用
void CcuTempAlltoAllMesh1D::InitInsAlgTemplate(
    std::vector<u64> &sendCounts, std::vector<u64> &recvCounts,
    std::vector<u64> &sdispls, std::vector<u64> &rdispls)
{
    sendCounts_ = sendCounts;
    recvCounts_ = recvCounts;
    sdispls_ = sdispls;
    rdispls_ = rdispls;

    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllMesh1D] InitInsAlgTemplate, sendCounts[%u] is [%u]", i, sendCounts[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllMesh1D] InitInsAlgTemplate, recvCounts[%u] is [%u]", i, recvCounts[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllMesh1D] InitInsAlgTemplate, sdispls[%u] is [%u]", i, sdispls[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllMesh1D] InitInsAlgTemplate, rdispls[%u] is [%u]", i, rdispls[i]);
    }

    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllMesh1D] InitInsAlgTemplate, sendCounts_[%u] is [%u]", i, sendCounts_[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllMesh1D] InitInsAlgTemplate, recvCounts_[%u] is [%u]", i, recvCounts_[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllMesh1D] InitInsAlgTemplate, sdispls_[%u] is [%u]", i, sdispls_[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllMesh1D] InitInsAlgTemplate, rdispls_[%u] is [%u]", i, rdispls_[i]);
    }
}

HcclResult CcuTempAlltoAllMesh1D::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAlltoAllMesh1D::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
     HCCL_INFO("[CcuTempAlltoAllMesh1D::FastLaunch] start");

    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    constexpr u32 inputIdx = 0;
    constexpr u32 outputIdx = 1;
    constexpr u32 inputOffsetIdx = 11;
    constexpr u32 outputOffsetIdx = 12;
    uint64_t argSize = 11;

    args[inputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    args[outputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];

    void *taskArgs = reinterpret_cast<void*>(args);
    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
                                            tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle,
                                            taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempReduceScatterMesh1D::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_INFO("[CcuTempAlltoAllMesh1D::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllMesh1D::KernelRun(const OpParam& param,
                                            const TemplateDataParams& templateDataParams,
                                            TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempAllToAllMesh1D] Run");
    buffInfo_ = templateDataParams.buffInfo;

    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    // 拿到input和output的首地址,和每片小数据的大小
    uint64_t curSendCounts = templateDataParams.count;
    uint64_t sliceSize = templateDataParams.sliceSize;

    uint32_t                                rankId    = myRank_;
    uint64_t                                repeatNumTmp  = templateDataParams.repeatNum;
    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    
    uint64_t srcStride = templateDataParams.outputSliceStride;
    uint64_t dstStride = templateDataParams.outputSliceStride;

    uint64_t dataType_ = param.all2AllVDataDes.sendType;
    uint64_t dataTypeSize_ = SIZE_TABLE[dataType_];
    uint64_t sliceBias = templateDataParams.processedDataCount * dataTypeSize_;

    HCCL_DEBUG("[CcuTempAlltoAllMesh1D::KernelRun] Start");
    if (tempRankSize_ == 1) {
        DataSlice usrInSlice = DataSlice(buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, sliceSize);
        DataSlice usrOutSlice = DataSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, sliceSize);
        LocalCopy(templateResource.threads[0], usrInSlice, usrOutSlice);

        HCCL_DEBUG("[CcuTempAlltoAllMesh1D::KernelRun] end");
        return HcclResult::HCCL_SUCCESS;
    }

    uint64_t srcOffset = 0;
    uint64_t dstOffset = myRank_ * dstStride;
    bool loadFromMem = false;

    HCCL_INFO("[CcuTempAllToAllMesh1D] Run Init: loadFromMem_[%d], myRank_[%d], dimSize[%llu], inputAddr[%llu],"\
        "outputAddr[%llu], sliceSize[%llu], srcOffset[%llu], dstOffset[%llu]",
        loadFromMem, myRank_, dimSize[0], inputAddr, outputAddr, sliceSize, srcOffset, dstOffset);
    LoopGroupConfig  config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_INTERLEAVE;
    config.memSlice     = CCU_MS_INTERLEAVE * CCU_MS_SIZE;
    auto     goSize     = CalGoSize(sliceSize, config);
    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, sliceSize, srcStride, srcOffset, dstOffset, goSize[0], goSize[1], goSize[2], goSize[3]};
    uint64_t argSize = 11;

    HCCL_INFO("[CcuTempAlltoAllMesh1D::KernelRun] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
            "srcStride[%llu], srcOffset[%llu],"
            "dstOffset[%llu], sliceSize[%llu], goSize: [%llu], [%llu], [%llu], [%llu]",
            inputAddr, outputAddr, srcStride, srcOffset,
            dstOffset, sliceSize, goSize[0], goSize[1], goSize[2], goSize[3]);

    CcuResult launchRet =  HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAlltoAllMesh1D::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    CcuKernelSubmitInfo subCommInfo;
    subCommInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(subCommInfo, inputAddr, outputAddr,  
        token, sliceSize, srcStride, srcOffset, dstOffset, 
        goSize[0], goSize[1], goSize[2], goSize[3],
        buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff));
    templateResource.submitInfos.push_back(subCommInfo);
     
    HCCL_DEBUG("[CcuTempAlltoAllMesh1D::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAlltoAllMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}
} // namespace ops_hccl