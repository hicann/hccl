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
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
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
    
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelReduceScatterNhrMutilJettyMem2Mem1D>(arg);
                         };
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestNHRWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_CLOS));
    std::vector<HcclChannelDesc> myChannelDescs;
    for(auto channel : channelDescs) {
        if(channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
            myChannelDescs.push_back(channel);
        }
    }
    CHK_RET(RestoreChannelMap(myChannelDescs, rankIdToChannelDesc_)); // 让rankId变成索引查询channel
    uint16_t portNum = 4;
    std::vector<NHRStepInfo> stepInfoVector;
    std::map<u32, u32> rank2ChannelIdx; // rankId和channel匹配
    std::vector<HcclChannelDesc> channelResort; // 重排channel
    GetNhrStepInfo(channelResort, stepInfoVector, rank2ChannelIdx);
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgReduceScatterNhrMutilJettyMem2Mem1D>(subCommRanks_[0].size(),
                                                                                    mySubCommRank_,
                                                                                    portNum,
                                                                                    stepInfoVector,
                                                                                    rank2ChannelIdx,
                                                                                    param,
                                                                                    subCommRanks_);
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

HcclResult CcuTempReduceScatterNhrMultiJettyMem2Mem1D::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        const TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    std::vector<uint64_t> dimSize;
    dimSize.push_back(templateRankSize_);
    constexpr uint64_t hcclMinSliceAlign = 128;
    const uint64_t sliceAlignCount = hcclMinSliceAlign / DataTypeSizeGet(dataType_);
    constexpr uint16_t portNum  = 4;
    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t sliceSize          = templateDataParams.sliceSize; // 单次处理数据的长度
    uint64_t inputSliceStride   = templateDataParams.inputSliceStride; // 输入数据的长度
    uint64_t outputSliceStride  = templateDataParams.outputSliceStride;
    uint64_t sliceOneJettySize  = templateDataParams.sliceSize / portNum / sliceAlignCount * sliceAlignCount; // 每一个jetty需要的数据长度(是不是需要128对齐)
    uint64_t sliceLastJettySize = templateDataParams.sliceSize - (portNum - 1) * sliceOneJettySize; // 最后份数据的长度
    uint64_t repeatNum          = UINT64_MAX - templateDataParams.repeatNum; // 需要重复几次
    uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgReduceScatterNhrMutilJettyMem2Mem1D>(
        inputAddr, outputAddr, token, sliceSize, inputSliceStride, outputSliceStride, sliceOneJettySize,
        sliceLastJettySize, repeatNum, inputRepeatStride, outputRepeatStride);
    
    HCCL_INFO("[CcuTempReduceScatterNhrMultiJettyMem2Mem1D::KernelRun] inputAddr[%llx], outputAddr[%llx], sliceSize[%u]"
                "sliceOneJettySize[%u], repeatNum[%llu], inputRepeatStride[%u], outputRepeatStride[%u]", inputAddr, outputAddr,
                sliceSize, sliceOneJettySize, repeatNum, inputRepeatStride, outputRepeatStride);
    if (sliceSize == 0) {
        HCCL_INFO("[CcuTempReduceScatterNhrMultiJettyMem2Mem1D] sliceSize == 0, Template Run Ends.");
        return HcclResult::HCCL_SUCCESS;
    }
    void* taskArgPtr = static_cast<void*>(taskArg.get());
    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));
    
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