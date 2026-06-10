/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ccu_temp_all_reduce_nhr_mem2mem_1D_multi_jetty.h"
#include <algorithm>
#include "channel.h"
#include "alg_template_base.h"
#include "ccu_launch_dl.h"
#include "ccu_kernel_all_reduce_nhr_mem2mem_1D_multi_jetty.h"

namespace ops_hccl {
constexpr u32 PORT_NUM = 1;

CcuTempAllReduceNhrMem2Mem1DMultiJetty::CcuTempAllReduceNhrMem2Mem1DMultiJetty(const OpParam& param,
        const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks) : 
        CcuAlgTemplateBase(param, rankId, subCommRanks),
        dataType_(param.DataDes.dataType),
        localRank_(INVALID_VALUE_RANKID),
        portNum_(0)
{
    // 根据子通信域，计算localRank
    for (u32 i = 0; i < subCommRanks[0].size(); i++) {
        if (subCommRanks[0][i] == rankId) {
            localRank_ = i;
        }
        subCommRankMap_.insert(std::make_pair(subCommRanks[0][i], i));
    }
    templateRankSize_ = subCommRanks[0].size();
    portNum_ = PORT_NUM;
}

CcuTempAllReduceNhrMem2Mem1DMultiJetty::~CcuTempAllReduceNhrMem2Mem1DMultiJetty()
{
}

HcclResult CcuTempAllReduceNhrMem2Mem1DMultiJetty::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;

    // 单个kernel
    resourceRequest.ccuKernelNum.emplace_back(1);
    HCCL_DEBUG("[%s] notifyNumOnMainThread[%u] slaveThreadNum[%u]", __func__,
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 计算NHR算法编排所需信息
    std::vector<NHRStepInfo> algStepInfoList; // 描述NHR算法步骤信息
    CHK_RET(ProcessNHRStepInfo(algStepInfoList));

    std::vector<HcclChannelDesc> channelDescs, channelDescsTemp;
    CHK_RET(CalcChannelRequestNHRWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescsTemp,
                                                  CommTopo::COMM_TOPO_CLOS));

    std::copy_if(channelDescsTemp.begin(), channelDescsTemp.end(), std::back_inserter(channelDescs), 
        [](const HcclChannelDesc &c) { return c.channelProtocol == COMM_PROTOCOL_UBC_CTP; });

    std::map<u32, u32> channelIdxMap; // 存放某个通信对端对应channels的索引，例(2, 1)表示rank2的channel是channels中的[1]，rank是子通信域的rank
    for (u32 i = 0; i < channelDescs.size(); ++i) {
        const auto &channelDesc = channelDescs[i];
        channelIdxMap[subCommRankMap_[channelDesc.remoteRank]] = i;
    }

    // 创建每个kernel的KernelArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelAllReduceNhrMem2Mem1DMultiJetty");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllReduceNhrMem2Mem1DMultiJettyKernel);
    auto kernelArg = std::make_shared<CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty>();
    kernelArg->rankSize = templateRankSize_;
    kernelArg->rankId = localRank_;
    kernelArg->portNum = portNum_;
    kernelArg->algStepInfoList = algStepInfoList;
    kernelArg->channelIdxMap = channelIdxMap;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);

    kernelInfo.channels = channelDescs;

    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[%s] finish. rankSize[%u], channelDescs.size[%zu], ccuKernelInfos.size[%zu].",
               __func__, templateRankSize_, channelDescs.size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNhrMem2Mem1DMultiJetty::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAllReduceNhrMem2Mem1DMultiJetty::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[%s] begin.", __func__);
    const uint64_t *args = tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs;
    buffInfo_ = tempFastLaunchCtx.buffInfo;

    // 计算NHR Multi Jetty特有的参数
    std::vector<uint64_t> taskArgs = {
        PointerToAddr(buffInfo_.inputPtr) + args[0],
        PointerToAddr(buffInfo_.outputPtr) + args[1],
        args[2], // outputToken
        args[3], // isInplace
        args[4], // dataSizePerRank
        args[5], // dataSizePerPort
        args[6], // lastRankSliceSize
        args[7], // lastPortSliceSize
        args[8],  // localCopyGoSize[0]
        args[9],  // localCopyGoSize[1]
        args[10], // localCopyGoSize[2]
        args[11], // localCopyGoSize[3]
        args[12], // localCopyGoSizeLastSlice[0]
        args[13], // localCopyGoSizeLastSlice[1]
        args[14], // localCopyGoSizeLastSlice[2]
        args[15]  // localCopyGoSizeLastSlice[3]
    };
    uint64_t argSize = taskArgs.size();

    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
        tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[%s] kernel launch failed, ccuRet -> %d", __func__, launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[%s] end.", __func__);
    return HcclResult::HCCL_SUCCESS;
}

void CcuTempAllReduceNhrMem2Mem1DMultiJetty::CalcSliceParams(const uint64_t dataCount, 
    uint64_t& dataSizePerRank, uint64_t& dataSizePerPort, 
    uint64_t& lastRankSliceSize, uint64_t& lastPortSliceSize) const
{
    const uint64_t unitSize = DataTypeSizeGet(dataType_);
    const uint64_t sliceDivisor = templateRankSize_ * portNum_;
    const uint64_t sliceAlignCount = HCCL_MIN_SLICE_ALIGN / unitSize;
    const uint64_t dataCountPerPort = dataCount / sliceDivisor / sliceAlignCount * sliceAlignCount;
    const uint64_t dataCountPerRank = dataCountPerPort * portNum_;
    dataSizePerPort = dataCountPerPort * unitSize;
    dataSizePerRank = dataSizePerPort * portNum_;
    const uint64_t lastRankSliceCount = dataCount - dataCountPerRank * (templateRankSize_ - 1);
    const uint64_t lastPortSliceCount = lastRankSliceCount - dataCountPerPort * (portNum_ - 1);
    lastRankSliceSize = lastRankSliceCount * unitSize;
    lastPortSliceSize = lastPortSliceCount * unitSize;
}

void CcuTempAllReduceNhrMem2Mem1DMultiJetty::BuildTaskArgsAndGoSize(const uint64_t inputAddr, 
    const uint64_t outputAddr, const uint64_t outputToken, const uint64_t isInplace, 
    const uint64_t dataSizePerRank, const uint64_t dataSizePerPort, 
    const uint64_t lastRankSliceSize, const uint64_t lastPortSliceSize,
    std::vector<uint64_t>& taskArgs, std::vector<uint64_t>& localCopyGoSize,
    std::vector<uint64_t>& localCopyGoSizeLastSlice) const
{
    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_DEFAULT_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE;
    localCopyGoSize = CalGoSize(dataSizePerRank, config);
    localCopyGoSizeLastSlice = CalGoSize(lastRankSliceSize, config);

    taskArgs = {inputAddr, outputAddr, outputToken, isInplace,
        dataSizePerRank, dataSizePerPort, lastRankSliceSize, lastPortSliceSize};
    taskArgs.insert(taskArgs.end(), localCopyGoSize.cbegin(), localCopyGoSize.cend());
    taskArgs.insert(taskArgs.end(), localCopyGoSizeLastSlice.cbegin(), localCopyGoSizeLastSlice.cend());
}

void CcuTempAllReduceNhrMem2Mem1DMultiJetty::FillSubmitInfo(const uint64_t outputToken, const uint64_t isInplace,
    const uint64_t dataSizePerRank, const uint64_t dataSizePerPort, const uint64_t lastRankSliceSize,
    const uint64_t lastPortSliceSize, const std::vector<uint64_t>& localCopyGoSize,
    const std::vector<uint64_t>& localCopyGoSizeLastSlice, TemplateResource& templateResource) const
{
    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    (void)FillCachedArgs(submitInfo, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, outputToken, isInplace,
        dataSizePerRank, dataSizePerPort, lastRankSliceSize, lastPortSliceSize, localCopyGoSize[0],
        localCopyGoSize[1], localCopyGoSize[2], localCopyGoSize[3], localCopyGoSizeLastSlice[0],
        localCopyGoSizeLastSlice[1], localCopyGoSizeLastSlice[2], localCopyGoSizeLastSlice[3]);
    templateResource.submitInfos.push_back(submitInfo);
}

HcclResult CcuTempAllReduceNhrMem2Mem1DMultiJetty::KernelRun(const OpParam& param,
    const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    HCCL_DEBUG("[%s] begin.", __func__);
    buffInfo_ = templateDataParams.buffInfo;

    CHK_PRT_RET(templateDataParams.count == 0, HCCL_INFO("[%s] Count is zero, nothing to do.", __func__),
                HcclResult::HCCL_SUCCESS);

    const uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    const uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    
    uint64_t outputToken;
    {
        auto tokenBuffInfo = buffInfo_;
        tokenBuffInfo.inputPtr = nullptr;
        tokenBuffInfo.inputSize = 0;
        CHK_RET(GetToken(tokenBuffInfo, outputToken));
    }
    
    const uint64_t isInplace = inputAddr == outputAddr ? 1 : 0;
    const uint64_t dataCount = templateDataParams.count;

    uint64_t dataSizePerRank, dataSizePerPort, lastRankSliceSize, lastPortSliceSize;
    CalcSliceParams(dataCount, dataSizePerRank, dataSizePerPort, lastRankSliceSize, lastPortSliceSize);

    HCCL_DEBUG("[%s] inputAddr[%llu], outputAddr[%llu], isInplace[%llu], dataSizePerRank[%llu], "
               "dataSizePerPort[%llu], lastRankSliceSize[%llu], lastPortSliceSize[%llu]",
               __func__, inputAddr, outputAddr, isInplace, dataSizePerRank, dataSizePerPort,
               lastRankSliceSize, lastPortSliceSize);

    std::vector<uint64_t> taskArgs, localCopyGoSize, localCopyGoSizeLastSlice;
    BuildTaskArgsAndGoSize(inputAddr, outputAddr, outputToken, isInplace, dataSizePerRank, dataSizePerPort,
                           lastRankSliceSize, lastPortSliceSize, taskArgs, localCopyGoSize, localCopyGoSizeLastSlice);

    CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0],
        taskArgs.data(), taskArgs.size());
    CHK_PRT_RET(launchRet != CCU_SUCCESS, HCCL_ERROR("[%s] kernel launch failed, ccuRet -> %d", __func__, launchRet),
                ConvertCcuToHccl(launchRet));

    FillSubmitInfo(outputToken, isInplace, dataSizePerRank, dataSizePerPort, lastRankSliceSize, lastPortSliceSize,
                   localCopyGoSize, localCopyGoSizeLastSlice, templateResource);

    HCCL_DEBUG("[%s] end.", __func__);
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllReduceNhrMem2Mem1DMultiJetty::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // 不需要scratch buffer
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

HcclResult CcuTempAllReduceNhrMem2Mem1DMultiJetty::GetReduceScatterStepInfo(u32 step, NHRStepInfo &stepInfo) const
{
    CHK_PRT_RET(localRank_ == INVALID_VALUE_RANKID, HCCL_ERROR("[%s]localRank_[%u] is invalid.", __func__, localRank_),
                HcclResult::HCCL_E_INTERNAL);

    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = localRank_;

    // 计算通信对象
    u32 deltaRank = 1 << step;
    u32 sendTo = (localRank_ + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 recvFrom = (localRank_ + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);
    u32 rxSliceIdx = localRank_;
    u32 txSliceIdx = (localRank_ - (1 << step) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG("[AllReduceNHR][GetReduceScatterStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx,
                   rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNhrMem2Mem1DMultiJetty::GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo) const
{
    CHK_PRT_RET(localRank_ == INVALID_VALUE_RANKID, HCCL_ERROR("[%s]localRank_[%u] is invalid.", __func__, localRank_),
                HcclResult::HCCL_E_INTERNAL);

    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = localRank_;

    // 计算通信对象
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (localRank_ + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 sendTo = (localRank_ + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = localRank_;
    u32 rxSliceIdx = (localRank_ - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG("[AllReduceNHR][GetAllGatherStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNhrMem2Mem1DMultiJetty::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo) const
{
    u32 nStepsNHR = nSteps / 2;
    CHK_PRT_RET(nStepsNHR == 0, HCCL_ERROR("[%s]nStepsNHR[%u] is invalid.", __func__, nStepsNHR), HcclResult::HCCL_E_INTERNAL);
    u32 realStep = step;
    if (realStep < nStepsNHR) {
        CHK_RET(GetReduceScatterStepInfo(realStep, stepInfo));
    } else {
        realStep = step % nStepsNHR;
        CHK_RET(GetAllGatherStepInfo(realStep, nStepsNHR, stepInfo));
    }
    return HcclResult::HCCL_SUCCESS;
}

uint32_t CcuTempAllReduceNhrMem2Mem1DMultiJetty::GetNHRStepNum(const uint32_t rankSize) const
{
    uint32_t nSteps = 0;
    for (uint32_t tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {}
    HCCL_DEBUG("[%s] rankSize[%u] nSteps[%u]", __func__, rankSize, nSteps);

    return nSteps;
}

uint32_t CcuTempAllReduceNhrMem2Mem1DMultiJetty::localRank2UserRank(const uint32_t localRank) const
{
    return subCommRanks_[0][localRank];
}

HcclResult CcuTempAllReduceNhrMem2Mem1DMultiJetty::ProcessNHRStepInfo(std::vector<NHRStepInfo> &algStepInfoList) const
{
    const u32 nSteps = GetNHRStepNum(templateRankSize_) * 2;  // 分为RS和AG两次NHR
    algStepInfoList.reserve(nSteps);

    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        algStepInfoList.push_back(stepInfo);

        HCCL_DEBUG("[%s] step[%u], myRank[%u], nSlices[%u], toRank[%u], fromRank[%u].", __func__, stepInfo.step,
                   stepInfo.myRank, stepInfo.nSlices, stepInfo.toRank, stepInfo.fromRank);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNhrMem2Mem1DMultiJetty::GetRes(AlgResourceRequest& resourceRequest) const
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllReduceNhrMem2Mem1DMultiJetty::GetThreadNum() const
{
    return 1;
}
} // namespace ops_hccl