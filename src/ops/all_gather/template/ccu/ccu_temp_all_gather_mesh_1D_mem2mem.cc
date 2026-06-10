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
#include "ccu_kernel_all_gather_mesh1d_mem2mem.h"
#include "ccu_temp_all_gather_mesh_1D_mem2mem.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

CcuTempAllGatherMesh1DMem2Mem::CcuTempAllGatherMesh1DMem2Mem(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempAllGatherMesh1DMem2Mem::~CcuTempAllGatherMesh1DMem2Mem()
{
}

HcclResult CcuTempAllGatherMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    GetRes(resourceRequest);
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的KernelArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllGatherMesh1DMem2MemKernel");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllGatherMesh1DMem2MemKernel);

    std::vector<HcclChannelDesc> channelDescs;
    if(topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1DFullMesh(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_1DMESH));
        for(auto channel : channelDescs){
            if(channel.channelProtocol != COMM_PROTOCOL_UBC_CTP){
                HCCL_ERROR("[CcuTempAllGatherMesh1DMem2Mem][CalcRes] channelProtocol: %u", channel.channelProtocol);
                return HCCL_E_INTERNAL;
            }
        }
    }
    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2Mem::CalcRes] Get Mesh Channel Success!");

    auto kernelArg = std::make_shared<CcuKernelArgAllGatherMesh1DMem2Mem>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherMesh1DMem2Mem::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAllGatherMesh1DMem2Mem::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2Mem::FastLaunch] start");
    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    constexpr u32 inputIdx = 0;
    constexpr u32 outputIdx = 1;
    constexpr u32 currentRankSliceInputOffsetIdx = 3;
    constexpr u32 currentRankSliceOutputOffsetIdx = 4;
    constexpr u32 isInputOutputEqualIdx = 10;
    constexpr u32 inputOffsetIdx = 15;
    constexpr u32 outputOffsetIdx = 16;
    uint64_t argSize = 15;

    uint64_t inputAddr                     = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    uint64_t outputAddr                    = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];
    uint64_t currentRankSliceInputOffset   = args[currentRankSliceInputOffsetIdx];
    uint64_t currentRankSliceOutputOffset  = args[currentRankSliceOutputOffsetIdx];
    bool inputOutputEqual = (inputAddr + currentRankSliceInputOffset == outputAddr + currentRankSliceOutputOffset);

    uint64_t isInputOutputEqual = static_cast<uint64_t>(inputOutputEqual);

    args[inputIdx]  = inputAddr;
    args[outputIdx] = outputAddr;
    args[isInputOutputEqualIdx] = isInputOutputEqual;

    void *taskArgs = reinterpret_cast<void*>(args);
    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
                                               tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle,
                                               taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAllGatherMesh1DMem2Mem::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2Mem::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherMesh1DMem2Mem::PrepareLaunchArgs(const OpParam& param,
    const TemplateDataParams& templateDataParams, std::vector<uint64_t>& taskArgs, uint64_t& argSize)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    uint64_t inputSliceStride   = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride  = templateDataParams.outputSliceStride;
    uint32_t repeatNum          = templateDataParams.repeatNum;
    uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize    = templateDataParams.sliceSize;
    uint64_t lastSliceSize      = templateDataParams.tailSize;
    bool inputOutputEqual = (inputAddr + inputSliceStride * mySubCommRank_ == outputAddr + outputSliceStride * mySubCommRank_);
    uint64_t isInputOutputEqual = static_cast<uint64_t>(inputOutputEqual);
    if (templateDataParams.tailSize != 0 && mySubCommRank_ == templateRankSize_ - 1) {
        normalSliceSize = templateDataParams.tailSize;
    }
    HCCL_INFO("[CcuTempAllGatherMesh1DMem2Mem][KernelRun] normalSliceSize [%u]", normalSliceSize);

    HcclDataType dataType       = param.DataDes.dataType;
    uint64_t dataTypeSize       = DataTypeSizeGet(dataType);
    uint64_t dataCount          = normalSliceSize / dataTypeSize;
    if (dataCount == 0 && lastSliceSize == 0) {
        HCCL_INFO("[CcuTempAllGatherMesh1DMem2Mem] DataCount == 0 && lastSliceSize == 0, Template Run Ends.");
        return HcclResult::HCCL_SUCCESS;
    }

    uint64_t currentRankSliceInputOffset  = inputSliceStride * mySubCommRank_;
    uint64_t currentRankSliceOutputOffset = outputSliceStride * mySubCommRank_;
    uint64_t tmpRepeatNum                 = UINT64_MAX - repeatNum;

    LoopGroupConfig  config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice     = CCU_MS_SIZE * LOCAL_COPY_MS_PER_LOOP;
    auto  goSize        = CalGoSize(normalSliceSize, config);

    taskArgs = {inputAddr, outputAddr, token, currentRankSliceInputOffset,
                currentRankSliceOutputOffset, tmpRepeatNum, inputRepeatStride,
                outputRepeatStride, normalSliceSize, lastSliceSize,
                isInputOutputEqual, goSize[0], goSize[1], goSize[2], goSize[3]};
    argSize = 15;

    HCCL_INFO("[CcuTempAllGatherMesh1DMem2Mem::KernelRun] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
               "currentRankSliceInputOffset[%llu], currentRankSliceOutputOffset[%llu], "
               "repeatNum[%llu],inputRepeatStride[%llu], outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu]",
               inputAddr, outputAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset, tmpRepeatNum,
               inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherMesh1DMem2Mem::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    std::vector<uint64_t> taskArgs;
    uint64_t argSize = 0;
    HcclResult ret = PrepareLaunchArgs(param, templateDataParams, taskArgs, argSize);
    if (ret != HcclResult::HCCL_SUCCESS) {
        return ret;
    }
    if (taskArgs.empty()) {
        return HcclResult::HCCL_SUCCESS;
    }

    CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0],
                                                taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAllGatherMesh1DMem2Mem::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(submitInfo, taskArgs[0], taskArgs[1], taskArgs[2], taskArgs[3], taskArgs[4],
                           taskArgs[5], taskArgs[6], taskArgs[7], taskArgs[8], taskArgs[9],
                           taskArgs[10], taskArgs[11], taskArgs[12], taskArgs[13], taskArgs[14],
                           buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff));
    templateResource.submitInfos.push_back(submitInfo);

    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2Mem::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGatherMesh1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempAllGatherMesh1DMem2Mem::GetThreadNum() const
{
    return 1;
}
 
HcclResult CcuTempAllGatherMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}
} // namespace ops_hccl