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
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "alg_template_base.h"
#include "ccu_kernel_all_gather_nhr1d_multi_jetty_mem2mem.h"
#include "ccu_temp_all_gather_nhr_1D_multi_jetty_mem2mem.h"

constexpr u32 JETTY_NUM = 4;

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

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelAllGatherNHR1DMultiJettyMem2Mem>(arg);
    };
    std::vector<HcclChannelDesc> channelDescs;
    jettyNum_ = JETTY_NUM; // 框架传入
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

    CHK_RET(CalcNHRInfo(stepInfoVector)); // NHR算法编排参数

    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllGatherNHR1DMultiJettyMem2Mem>(subCommRanks_[0].size(),
                                                                                    mySubCommRank_,
                                                                                    param,
                                                                                    jettyNum_,
                                                                                    stepInfoVector,
                                                                                    rank2ChannelIdx,
                                                                                    subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllGatherNHR1DMultiJettyMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherNHR1DMultiJettyMem2Mem::CalcNHRInfo(std::vector<NHRStepInfo> &stepInfoVector)
{
    u32 nSteps = GetNHRStepNum(tempRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 CcuTempAllGatherNHR1DMultiJettyMem2Mem::GetNHRStepNum(u32 rankSize)
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_DEBUG("[NHRBase][GetStepNumInterServer] rankSize[%u] nSteps[%u]", rankSize, nSteps);
    return nSteps;
}

HcclResult CcuTempAllGatherNHR1DMultiJettyMem2Mem::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo)
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

uint32_t CcuTempAllGatherNHR1DMultiJettyMem2Mem::RemoteRankId2RankId(const uint32_t remoteRankId)
{
    uint32_t subCommRankId = 0;
    std::vector<u32> ranks = subCommRanks_[0];
    auto it = std::find(ranks.begin(), ranks.end(), remoteRankId);
    if (it != ranks.end()) {
        subCommRankId = std::distance(ranks.begin(), it);
    }
    return subCommRankId;
}

HcclResult CcuTempAllGatherNHR1DMultiJettyMem2Mem::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        const TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempAllGatherNHR1DMultiJettyMem2Mem] Template Run start.");
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr             = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr            = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token                 = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t isInputOutputEqual    = buffInfo_.inputPtr == buffInfo_.outputPtr ? 1 : 0;
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

    uint64_t repeatNumInv = UINT64_MAX - repeatNumTmp; // CCU硬件限制

    HCCL_DEBUG("[CcuTempAllGatherNHR1DMultiJettyMem2Mem] inputAddr[%llu], outputAddr[%llu], token[%llu],"
    "sliceSize[%llu], sliceSizePerJetty[%llu], lastSliceSizePerJetty[%llu], repeatNumInv[%llu], inputSliceStride[%llu], "
    "outputSliceStride[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu], isInputOutputEqual[%llu]",
    inputAddr, outputAddr, token, sliceSize, sliceSizePerJetty, lastSliceSizePerJetty, repeatNumInv, inputSliceStride, 
    outputSliceStride, inputRepeatStride, outputRepeatStride, isInputOutputEqual);

    if (dataCount == 0) {
        HCCL_INFO("[CcuTempAllGatherNHR1DMultiJettyMem2Mem] DataCount == 0, Template Run Ends.");
        return HcclResult::HCCL_SUCCESS;
    }

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllGatherNHR1DMultiJettyMem2Mem>(
        inputAddr, outputAddr, token, sliceSize, sliceSizePerJetty, lastSliceSizePerJetty, repeatNumInv,
        inputSliceStride, outputSliceStride, inputRepeatStride, outputRepeatStride, isInputOutputEqual);

    void* taskArgPtr = static_cast<void*>(taskArg.get());

    HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr);
    
    HCCL_DEBUG("[CcuTempAllGatherNHR1DMultiJettyMem2Mem::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGatherNHR1DMultiJettyMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // 不需要Scratch buff
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