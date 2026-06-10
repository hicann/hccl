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
#include <algorithm>
#include "ccu_launch_dl.h"
#include "ccu_temp_reduce_scatter_nhr_1D_multi_jetty_mem2mem.h"
#include "ccu_kernel_reduce_scatter_nhr1d_multi_jetty_mem2mem.h"

namespace ops_hccl {

CcuTempReduceScatterNhrMultiJettyMem2Mem1D::CcuTempReduceScatterNhrMultiJettyMem2Mem1D(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    dataType_ = param.DataDes.dataType;
}

CcuTempReduceScatterNhrMultiJettyMem2Mem1D::~CcuTempReduceScatterNhrMultiJettyMem2Mem1D()
{
}

HcclResult CcuTempReduceScatterNhrMultiJettyMem2Mem1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempReduceScatterNhrMultiJettyMem2Mem1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelReduceScatterNhrMutilJettyMem2Mem1D");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuReduceScatterNhrMem2Mem1DMultiJettyKernel);
    
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestNHRWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_CLOS));
    std::vector<HcclChannelDesc> myChannelDescs;
    for(auto channel : channelDescs) {
        if(channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
            myChannelDescs.push_back(channel);
        }
    }
    CHK_RET(RestoreChannelMap(myChannelDescs, rankIdToChannelDesc_)); // 让rankId变成索引查询channel
    uint16_t portNum = 1;
    std::vector<NHRStepInfo> stepInfoVector;
    std::map<u32, u32> rank2ChannelIdx; // rankId和channel匹配
    std::vector<HcclChannelDesc> channelResort; // 重排channel
    GetNhrStepInfo(channelResort, stepInfoVector, rank2ChannelIdx);
    
    auto kernelArg = std::make_shared<CcuKernelArgReduceScatterNhrMutilJettyMem2Mem1D>();
    kernelArg->dimSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->portNum = portNum;
    kernelArg->stepInfoVector = stepInfoVector;
    kernelArg->rank2ChannelIdx = rank2ChannelIdx;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelResort;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempReduceScatterNhrMultiJettyMem2Mem1D::CalcRes] myChannelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               myChannelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterNhrMultiJettyMem2Mem1D::GetRes(AlgResourceRequest& resourceRequest) const
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterNhrMultiJettyMem2Mem1D::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempReduceScatterNhrMultiJettyMem2Mem1D::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempReduceScatterNhrMultiJettyMem2Mem1D::FastLaunch] start");
    const uint64_t *args = tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs;
    buffInfo_ = tempFastLaunchCtx.buffInfo;

    // 计算NHR Multi Jetty特有的参数
    std::vector<uint64_t> taskArgs = {
        PointerToAddr(buffInfo_.inputPtr) + args[0],
        PointerToAddr(buffInfo_.outputPtr) + args[1],
        args[2], // token
        args[3], // sliceSize
        args[4], // inputSliceStride
        args[5], // outputSliceStride
        args[6], // sliceOneJettySize
        args[7], // sliceLastJettySize
        args[8], // repeatNum
        args[9], // inputRepeatStride
        args[10] // outputRepeatStride
    };
    uint64_t argSize = taskArgs.size();

    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
        tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempReduceScatterNhrMultiJettyMem2Mem1D::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[CcuTempReduceScatterNhrMultiJettyMem2Mem1D::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterNhrMultiJettyMem2Mem1D::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    constexpr uint64_t hcclMinSliceAlign = 128;
    const uint64_t sliceAlignCount = hcclMinSliceAlign / DataTypeSizeGet(dataType_);
    constexpr uint16_t portNum  = 1;
    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t sliceSize          = templateDataParams.sliceSize;
    uint64_t inputSliceStride   = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride  = templateDataParams.outputSliceStride;
    uint64_t sliceOneJettySize  = templateDataParams.sliceSize / portNum / sliceAlignCount * sliceAlignCount;
    uint64_t sliceLastJettySize = templateDataParams.sliceSize - (portNum - 1) * sliceOneJettySize;
    uint64_t repeatNum          = UINT64_MAX - templateDataParams.repeatNum;
    uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        sliceSize,
        inputSliceStride,
        outputSliceStride,
        sliceOneJettySize,
        sliceLastJettySize,
        repeatNum,
        inputRepeatStride,
        outputRepeatStride
    };
    uint64_t argSize = 11;

    HCCL_INFO("[CcuTempReduceScatterNhrMultiJettyMem2Mem1D::KernelRun] inputAddr[%llx], outputAddr[%llx], sliceSize[%u]"
                "sliceOneJettySize[%u], repeatNum[%llu], inputRepeatStride[%u], outputRepeatStride[%u]", inputAddr, outputAddr,
                sliceSize, sliceOneJettySize, repeatNum, inputRepeatStride, outputRepeatStride);
    if (sliceSize == 0) {
        HCCL_INFO("[CcuTempReduceScatterNhrMultiJettyMem2Mem1D] sliceSize == 0, Template Run Ends.");
        return HcclResult::HCCL_SUCCESS;
    }
    
    CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0],
        taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempReduceScatterNhrMultiJettyMem2Mem1D::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }
    
    //所有task下发完再保存参数信息
    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(submitInfo, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, token, sliceSize, 
        inputSliceStride, outputSliceStride, sliceOneJettySize, sliceLastJettySize, repeatNum, inputRepeatStride, 
        outputRepeatStride));
    templateResource.submitInfos.push_back(submitInfo);
    
    HCCL_DEBUG("[CcuTempReduceScatterNhrMultiJettyMem2Mem1D::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceScatterNhrMultiJettyMem2Mem1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u32 CcuTempReduceScatterNhrMultiJettyMem2Mem1D::GetNhrStepNum(u32 rankSize) const
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_DEBUG("[CcuTempReduceScatterNhrMultiJettyMem2Mem1D][GetNhrStepNum] rankSize[%u] nSteps[%u]", rankSize, nSteps);
    return nSteps;
}

HcclResult CcuTempReduceScatterNhrMultiJettyMem2Mem1D::GetNhrStepInfo(std::vector<HcclChannelDesc>& channelResort,
                                                            std::vector<NHRStepInfo>& stepInfoVector,
                                                            std::map<u32, u32>& rank2ChannelIdx)
{
    u32 nSteps = GetNhrStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, stepInfo));
        stepInfoVector.push_back(stepInfo);
        if (rank2ChannelIdx.count(stepInfo.fromRank) == 0) {
            // 存储 rankid → channelIdx 的索引
            u32 curChannelIdx = channelResort.size();
            rank2ChannelIdx[stepInfo.fromRank] = curChannelIdx;
            for (HcclChannelDesc channel: rankIdToChannelDesc_.at(stepInfo.fromRank)) {
                if (channelResort.size() == curChannelIdx) {
                    channelResort.push_back(channel);
                }
            }
        }
        if (rank2ChannelIdx.count(stepInfo.toRank) == 0) {
            u32 curChannelIdx = channelResort.size();
            rank2ChannelIdx[stepInfo.toRank] = curChannelIdx;
            for (HcclChannelDesc channel: rankIdToChannelDesc_.at(stepInfo.toRank)) {
                if (channelResort.size() == curChannelIdx) {
                    channelResort.push_back(channel);
                }
            }
        }
        HCCL_DEBUG("[%s] step[%u], myRank[%u], nSlices[%u], toRank[%u], fromRank[%u].", __func__, stepInfo.step,
                   stepInfo.myRank, stepInfo.nSlices, stepInfo.toRank, stepInfo.fromRank);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterNhrMultiJettyMem2Mem1D::GetStepInfo(u32 step, NHRStepInfo &stepInfo)
{
    // 将本rank号转换成算法使用的索引号
    u32 rankIdx = mySubCommRank_; // 子通信域下的rankId，即虚拟的rankId
    std::vector<u32> ranks = subCommRanks_[0];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = mySubCommRank_;

    // 计算通信对象
    u32 deltaRank = 1 << step;
    u32 sendTo = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 recvFrom = (rankIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);
    u32 txSliceIdx = sendTo;
    u32 rxSliceIdx = rankIdx;

    stepInfo.nSlices = nSlices;

    stepInfo.toRank = ranks[sendTo];        //  从虚拟rankid转换至通信域真实rankid
    stepInfo.fromRank = ranks[recvFrom];

    // 计算本rank在本轮收/发中的slice编号
    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx); // 虚拟id
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
        HCCL_INFO("[CcuTempReduceScatterNhrMultiJettyMem2Mem1D][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);
        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceScatterNhrMultiJettyMem2Mem1D::GetThreadNum() const
{
    return 1;
}
} // namespace ops_hccl