/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "ccu_kernel_all_gather_omnipipe_nhr1d_mem2mem.h"
#include "ccu_temp_all_gather_omnipipe_nhr1d_mem2mem.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

CcuTempAllGatherOmniPipeNHR1DMem2Mem::CcuTempAllGatherOmniPipeNHR1DMem2Mem(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    // 获取本卡在子通信域(如果有)中的rankid, 以及子通信域内所有卡数
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    templateRankSize_ = ranks.size();
    HCCL_DEBUG("[%s] mySubCommRank[%u] rankId[%u]", __func__, mySubCommRank_, rankId);
}

CcuTempAllGatherOmniPipeNHR1DMem2Mem::~CcuTempAllGatherOmniPipeNHR1DMem2Mem()
{
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    GetRes(resourceRequest);
    // 需要1个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[%s] notifyNumOnMainThread[%u] slaveThreadNum[%u]", __func__,
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的KernelArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllGatherOmniPipeNHR1DMem2MemKernel");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllGatherOmniPipeNHR1DMem2MemKernel);

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestNhrMultiJetty(comm, param, topoInfo, subCommRanks_, channelDescs)); 
    for (auto channel : channelDescs) {
        if (channel.channelProtocol != COMM_PROTOCOL_UBC_CTP) {
            HCCL_ERROR("[%s] channelProtocol: %u", __func__, channel.channelProtocol);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    HCCL_INFO("[%s] Get Clos Channel Success!", __func__);

    std::map<u32, u32> rank2ChannelIdx; // rankId和channel匹配
    for (u32 i = 0; i < channelDescs.size(); ++i) {
        u32 remoteRank = channelDescs[i].remoteRank;
        u32 subRankIdx = RemoteRankId2RankId(remoteRank);
        rank2ChannelIdx[subRankIdx] = i;
    }

    std::vector<NHRStepInfo> stepInfoVector;
    CHK_RET(CalcNHRInfo(stepInfoVector));

    auto kernelArg = std::make_shared<CcuKernelArgAllGatherOmniPipeNHR1DMem2Mem>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->opParam = param;
    kernelArg->stepInfoVector = stepInfoVector;
    kernelArg->rank2ChannelIdx = rank2ChannelIdx;
    kernelArg->subCommRanks = subCommRanks_;

    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[%s] channelDescs.size()=%llu dimsize=%llu ccuKernelInfos.size()=%llu",
        __func__, channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMem2Mem::CalcNHRInfo(std::vector<NHRStepInfo> &stepInfoVector) const
{
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 CcuTempAllGatherOmniPipeNHR1DMem2Mem::GetNHRStepNum(u32 rankSize) const
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_DEBUG("[%s] rankSize[%u] nSteps[%u]", __func__, rankSize, nSteps);
    return nSteps;
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMem2Mem::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo) const
{
    // 将本rank号转换成算法使用的索引号
    u32 rankIdx = mySubCommRank_;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step   = step;
    stepInfo.myRank = rankIdx;

    // 计算通信对象
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom  = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 sendTo    = (rankIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices         = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx      = rankIdx;
    u32 rxSliceIdx      = (rankIdx - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices  = nSlices;
    stepInfo.toRank   = sendTo;
    stepInfo.fromRank = recvFrom;
    // 计算本rank在本轮收/发中的slice编号
    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
        HCCL_DEBUG("[%s] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", __func__, i, txSliceIdx, rxSliceIdx);
        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

uint32_t CcuTempAllGatherOmniPipeNHR1DMem2Mem::RemoteRankId2RankId(const uint32_t remoteRankId) const
{
    uint32_t subCommRankId = 0;
    std::vector<u32> ranks = subCommRanks_[0];
    auto it = std::find(ranks.begin(), ranks.end(), remoteRankId);
    if (it != ranks.end()) {
        subCommRankId = std::distance(ranks.begin(), it);
    }
    return subCommRankId;
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMem2Mem::KernelRun(
    const OpParam& param, const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);

    uint64_t inBuffBaseOff = buffInfo_.inBuffBaseOff;
    uint64_t outBuffBaseOff = buffInfo_.outBuffBaseOff;

    uint64_t inputAddr = inputAddrBase + inBuffBaseOff;
    uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    auto stepSliceInfo = templateDataParams.stepSliceInfo;
    uint32_t repeatNum = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_].size();

    uint64_t sliceStride = stepSliceInfo.stepInputSliceStride[mySubCommRank_];
    for (uint32_t rpt = 0; rpt < repeatNum; ++rpt) {
        uint64_t sliceSize = stepSliceInfo.stepSliceSize[mySubCommRank_][rpt];
        uint64_t inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_][rpt];
        std::vector<uint64_t> inputOmniSliceStrideVec;
        std::vector<uint64_t> inputOmniSliceSizeVec;
        for (uint32_t ridx = 0; ridx < stepSliceInfo.inputOmniPipeSliceStride.size(); ridx++) {
            inputOmniSliceStrideVec.push_back(
                stepSliceInfo.inputOmniPipeSliceStride[ridx][rpt] + stepSliceInfo.stepInputSliceStride[ridx]);
            inputOmniSliceSizeVec.push_back(stepSliceInfo.stepSliceSize[ridx][rpt]);
            HCCL_DEBUG("[%s] myRank[%u] stepSliceInfo.inputOmniPipeSliceStride[%d][%d]:%d", __func__,
                        myRank_, ridx, rpt, stepSliceInfo.inputOmniPipeSliceStride[ridx][rpt]);
            HCCL_DEBUG("[%s] myRank[%u] stepSliceInfo.stepInputSliceStride[%d]:%d", __func__,
                        myRank_, ridx, stepSliceInfo.stepInputSliceStride[ridx]);
            HCCL_DEBUG("[%s] myRank[%u] stepSliceInfo.stepSliceSize[%d][%d]:%d", __func__,
                        myRank_, ridx, rpt, stepSliceInfo.stepSliceSize[ridx][rpt]);
        }
        uint64_t inputSliceStride = templateDataParams.inputSliceStride;

        std::vector<uint64_t> taskArgs
            = {inputAddr, outputAddr, token, sliceSize, sliceStride, 0, inputOmniPipeSliceStride};
        taskArgs.insert(taskArgs.end(), inputOmniSliceStrideVec.begin(), inputOmniSliceStrideVec.end());
        taskArgs.push_back(inputSliceStride);
        taskArgs.insert(taskArgs.end(), inputOmniSliceSizeVec.begin(), inputOmniSliceSizeVec.end());

        HCCL_DEBUG("[%s] myRank[%u] mySubCommRank[%u] rpt[%u] inputAddrBase[%llu] outputAddrBase[%llu] "
                    "inBuffBaseOff[%llu] outBuffBaseOff[%llu] inputAddr[%llu] "
                    "outputAddr[%llu] sliceSize[%llu] sliceStride[%llu] inputOmniPipeSliceStride[%llu] localCopyFlag[%llu]",
                    __func__, myRank_, mySubCommRank_, rpt, inputAddrBase, outputAddrBase, inBuffBaseOff,
                    outBuffBaseOff, inputAddr, outputAddr, sliceSize, sliceStride, inputOmniPipeSliceStride, 0);

        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0],
                                                    taskArgs.data(), taskArgs.size());
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[%s] kernel launch failed, ccuRet -> %d", __func__, launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGatherOmniPipeNHR1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // No Scratch buff
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempAllGatherOmniPipeNHR1DMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMem2Mem::GetRes(AlgResourceRequest &resourceRequest) const
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HcclResult::HCCL_SUCCESS;
}
} // namespace ops_hccl