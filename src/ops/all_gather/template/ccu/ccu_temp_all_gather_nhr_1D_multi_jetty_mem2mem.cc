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
#include "alg_template_base.h"
#include "ccu_kernel_all_gather_nhr1d_multi_jetty_mem2mem.h"
#include "ccu_temp_all_gather_nhr_1D_multi_jetty_mem2mem.h"
#include "ccu_launch_dl.h"

constexpr u32 JETTY_NUM = 1;

namespace ops_hccl {

CcuTempAllGatherNHR1DMultiJettyMem2Mem::CcuTempAllGatherNHR1DMultiJettyMem2Mem(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    // 获取本卡在子通信域(如果有)中的rankid, 以及子通信域内所有卡数
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    tempRankSize_ = ranks.size();
    subCommRanks_ = subCommRanks;
}

CcuTempAllGatherNHR1DMultiJettyMem2Mem::~CcuTempAllGatherNHR1DMultiJettyMem2Mem()
{
}

HcclResult CcuTempAllGatherNHR1DMultiJettyMem2Mem::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    GetRes(resourceRequest);
    // 需要1个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllGatherNHR1DMultiJettyMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的KernelArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllGatherNHR1DMultiJettyMem2MemKernel");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllGatherNHR1DMultiJettyMem2MemKernel);
    
    std::vector<HcclChannelDesc> channelDescs;
    jettyNum_ = JETTY_NUM;
    CommTopo  priorityTopo = COMM_TOPO_CLOS;
    CHK_RET(CalcChannelRequestNHRWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, priorityTopo));
    for (auto channel : channelDescs) {
        if (channel.channelProtocol != COMM_PROTOCOL_UBC_CTP) {
            HCCL_ERROR("[CcuTempAllGatherNHR1DMultiJettyMem2Mem][CalcRes] channelProtocol: %u", channel.channelProtocol);
            return HCCL_E_INTERNAL;
        }
    }
    HCCL_DEBUG("[CcuTempAllGatherNHR1DMultiJettyMem2Mem::CalcRes] Get Clos Channel Success!");

    std::vector<NHRStepInfo>     stepInfoVector;
    std::map<u32, u32>           rank2ChannelIdx;

    for (u32 i = 0; i < channelDescs.size(); ++i) {
        u32 remoteRank = channelDescs[i].remoteRank;
        rank2ChannelIdx[RemoteRankId2RankId(remoteRank)] = i;
    }

    CHK_RET(CalcNHRInfo(stepInfoVector));

    auto kernelArg = std::make_shared<CcuKernelArgAllGatherNHR1DMultiJettyMem2Mem>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->opParam = param;
    kernelArg->jettyNum = jettyNum_;
    kernelArg->stepInfoVector = stepInfoVector;
    kernelArg->rank2ChannelIdx = rank2ChannelIdx;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllGatherNHR1DMultiJettyMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherNHR1DMultiJettyMem2Mem::CalcNHRInfo(std::vector<NHRStepInfo> &stepInfoVector) const
{
    u32 nSteps = GetNHRStepNum(tempRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 CcuTempAllGatherNHR1DMultiJettyMem2Mem::GetNHRStepNum(u32 rankSize) const
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_DEBUG("[NHRBase][GetStepNumInterServer] rankSize[%u] nSteps[%u]", rankSize, nSteps);
    return nSteps;
}

HcclResult CcuTempAllGatherNHR1DMultiJettyMem2Mem::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo) const
{
    u32 rankIdx = mySubCommRank_;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step   = step;
    stepInfo.myRank = rankIdx;

    // 计算通信对象
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom  = (rankIdx + tempRankSize_ - deltaRank) % tempRankSize_;
    u32 sendTo    = (rankIdx + deltaRank) % tempRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices         = (tempRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx      = rankIdx;
    u32 rxSliceIdx      = (rankIdx - (1 << (nSteps - 1 - step)) + tempRankSize_) % tempRankSize_;

    stepInfo.nSlices  = nSlices;
    stepInfo.toRank   = sendTo;
    stepInfo.fromRank = recvFrom;
    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
        HCCL_DEBUG("[AllGatherNHR][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);
        txSliceIdx = (txSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
        rxSliceIdx = (rxSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

uint32_t CcuTempAllGatherNHR1DMultiJettyMem2Mem::RemoteRankId2RankId(const uint32_t remoteRankId) const
{
    uint32_t subCommRankId = 0;
    std::vector<u32> ranks = subCommRanks_[0];
    auto it = std::find(ranks.begin(), ranks.end(), remoteRankId);
    if (it != ranks.end()) {
        subCommRankId = std::distance(ranks.begin(), it);
    }
    return subCommRankId;
}

HcclResult CcuTempAllGatherNHR1DMultiJettyMem2Mem::PrepareLaunchArgs(const OpParam& param,
    const TemplateDataParams& templateDataParams, std::vector<uint64_t>& taskArgs, uint64_t& argSize)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr             = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr            = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    
    uint64_t sliceSize             = templateDataParams.sliceSize;
    HcclDataType dataType          = param.DataDes.dataType;
    uint64_t dataTypeSize          = DataTypeSizeGet(dataType);
    uint64_t dataCount             = sliceSize / dataTypeSize;
    jettyNum_ = JETTY_NUM;

    uint64_t sliceCountPerJetty    = dataCount / jettyNum_  / (HCCL_MIN_SLICE_ALIGN / dataTypeSize) * (HCCL_MIN_SLICE_ALIGN / dataTypeSize);
    uint64_t lastCountSizePerJetty = dataCount - sliceCountPerJetty * (jettyNum_ - 1);
    uint64_t sliceSizePerJetty     = sliceCountPerJetty * dataTypeSize;
    uint64_t lastSliceSizePerJetty = lastCountSizePerJetty * dataTypeSize;
    uint64_t inputSliceStride      = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride     = templateDataParams.outputSliceStride;
    uint64_t inputRepeatStride     = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride    = templateDataParams.outputRepeatStride;
    uint64_t repeatNumTmp          = templateDataParams.repeatNum;

    uint64_t repeatNumInv = UINT64_MAX - repeatNumTmp;

    bool inputOutputEqual = (inputAddr + inputSliceStride * mySubCommRank_ == outputAddr + outputSliceStride * mySubCommRank_);
    uint64_t isInputOutputEqual = static_cast<uint64_t>(inputOutputEqual);

    if (dataCount == 0) {
        HCCL_INFO("[CcuTempAllGatherNHR1DMultiJettyMem2Mem] DataCount == 0, Template Run End.");
        return HcclResult::HCCL_SUCCESS;
    }

    LoopGroupConfig  config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice     = CCU_MS_SIZE * LOCAL_COPY_MS_PER_LOOP;
    auto   goSize       = CalGoSize(sliceSize, config);

    taskArgs = {inputAddr, outputAddr, token, sliceSize, sliceSizePerJetty,
                lastSliceSizePerJetty, repeatNumInv, inputSliceStride,
                outputSliceStride, inputRepeatStride, outputRepeatStride,
                isInputOutputEqual, goSize[0], goSize[1], goSize[2], goSize[3]};
    argSize = 16;

    HCCL_DEBUG("[CcuTempAllGatherNHR1DMultiJettyMem2Mem] inputAddr[%llu], outputAddr[%llu],"
    "sliceSize[%llu], sliceSizePerJetty[%llu], lastSliceSizePerJetty[%llu], repeatNumInv[%llu], inputSliceStride[%llu], "
    "outputSliceStride[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu], isInputOutputEqual[%llu]",
    inputAddr, outputAddr, sliceSize, sliceSizePerJetty, lastSliceSizePerJetty, repeatNumInv, inputSliceStride,
    outputSliceStride, inputRepeatStride, outputRepeatStride, isInputOutputEqual);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherNHR1DMultiJettyMem2Mem::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempAllGatherNHR1DMultiJettyMem2Mem] Template Run start.");

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
        HCCL_ERROR("[CcuTempAllGatherNHR1DMultiJettyMem2Mem::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(submitInfo, taskArgs[0], taskArgs[1], taskArgs[2], taskArgs[3], taskArgs[4],
                           taskArgs[5], taskArgs[6], taskArgs[7], taskArgs[8], taskArgs[9],
                           taskArgs[10], taskArgs[11], taskArgs[12], taskArgs[13], taskArgs[14], taskArgs[15],
                           buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, mySubCommRank_));
    templateResource.submitInfos.push_back(submitInfo);

    HCCL_DEBUG("[CcuTempAllGatherNHR1DMultiJettyMem2Mem] Template Run end.");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherNHR1DMultiJettyMem2Mem::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAllGatherNHR1DMultiJettyMem2Mem::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempAllGatherNHR1DMultiJettyMem2Mem::FastLaunch] start");
    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    constexpr u32 inputIdx = 0;
    constexpr u32 outputIdx = 1;
    constexpr u32 inputSliceStrideIdx  = 7;
    constexpr u32 outputSliceStrideIdx  = 8;
    constexpr u32 isInputOutputEqualIdx = 11;
    constexpr u32 inputOffsetIdx = 16;
    constexpr u32 outputOffsetIdx = 17;
    constexpr u32 mySubCommRankIdx = 18;
    uint64_t argSize = 16;

    uint64_t inputAddr          = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    uint64_t outputAddr         = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];
    uint64_t inputSliceStride   = args[inputSliceStrideIdx];
    uint64_t outputSliceStride  = args[outputSliceStrideIdx];
    uint64_t mySubCommRank      = args[mySubCommRankIdx];
    bool inputOutputEqual = (inputAddr + inputSliceStride * mySubCommRank == outputAddr + outputSliceStride * mySubCommRank);

    uint64_t isInputOutputEqual = static_cast<uint64_t>(inputOutputEqual);

    args[inputIdx]  = inputAddr;
    args[outputIdx] = outputAddr;
    args[isInputOutputEqualIdx] = isInputOutputEqual;

    void *taskArgs = reinterpret_cast<void*>(args);
    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
                                               tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle,
                                               taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAllGatherNHR1DMultiJettyMem2Mem::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[CcuTempAllGatherNHR1DMultiJettyMem2Mem::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGatherNHR1DMultiJettyMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempAllGatherNHR1DMultiJettyMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempAllGatherNHR1DMultiJettyMem2Mem::GetRes(AlgResourceRequest &resourceRequest) const
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HCCL_SUCCESS;
}
} // namespace ops_hccl