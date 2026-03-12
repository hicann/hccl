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
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "alg_template_base.h"

namespace ops_hccl {
constexpr u32 PORT_NUM = 4;

CcuTempAllReduceNhrMem2Mem1DMultiJetty::CcuTempAllReduceNhrMem2Mem1DMultiJetty(const OpParam& param, const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    dataType_ = param.DataDes.dataType;

    // 生成全局rank号到子通信域rank号的对应关系
    const auto& ranks = subCommRanks[0];
    for (u32 i = 0; i < ranks.size(); ++i) {
        subCommRankMap_[ranks[i]] = i;
    }
    localRank_ = subCommRankMap_[rankId];
    portNum_ = PORT_NUM;
    templateRankSize_ = subCommRanks[0].size();
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
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelAllReduceNhr1DMem2MemMultiJetty>(arg);
    };

    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty>(
        templateRankSize_, localRank_, portNum_, param, algStepInfoList, channelIdxMap, subCommRanks_);

    kernelInfo.channels = channelDescs;

    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[%s] finish. rankSize[%u], channelDescs.size[%zu], ccuKernelInfos.size[%zu].",
               __func__, templateRankSize_, channelDescs.size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNhrMem2Mem1DMultiJetty::KernelRun(const OpParam& param,
    const TemplateDataParams& templateDataParams, const TemplateResource& templateResource)
{
    HCCL_DEBUG("[%s] begin.", __func__);
    buffInfo_ = templateDataParams.buffInfo;

    // 数据量为0时，直接返回
    CHK_PRT_RET(templateDataParams.count == 0, HCCL_INFO("[%s] Count is zero, nothing to do.", __func__),
                HcclResult::HCCL_SUCCESS);

    const uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    const uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    const uint64_t outputToken = hcomm::CcuRep::GetTokenInfo(PointerToAddr(buffInfo_.outputPtr), buffInfo_.outputSize);
    const uint64_t isInplace = inputAddr == outputAddr ? 1 : 0;
    const uint64_t dataCount = templateDataParams.count; // 总count数
    const uint64_t unitSize = DataTypeSizeGet(dataType_);
    const uint64_t dataSize = dataCount * unitSize; // 总size
    const uint64_t sliceDivisor = templateRankSize_ * portNum_;
    const uint64_t sliceAlignCount = HCCL_MIN_SLICE_ALIGN / unitSize; // 保证切分的slice呈128字节对齐
    const uint64_t dataCountPerPort = dataCount / sliceDivisor / sliceAlignCount * sliceAlignCount;
    const uint64_t dataCountPerRank = dataCountPerPort * portNum_;
    const uint64_t dataSizePerPort = dataCountPerPort * unitSize;
    const uint64_t dataSizePerRank = dataSizePerPort * portNum_;
    const uint64_t lastRankSliceCount = dataCount - dataCountPerRank * (templateRankSize_ - 1);
    const uint64_t lastPortSliceCount = lastRankSliceCount - dataCountPerPort * (portNum_ - 1);
    const uint64_t lastRankSliceSize = lastRankSliceCount * unitSize;
    const uint64_t lastPortSliceSize = lastPortSliceCount * unitSize;

    HCCL_DEBUG("[%s] inputAddr[%llu], outputAddr[%llu], outputToken[%llu], isInplace[%llu], dataSize[%llu], "
               "sliceDivisor[%llu], dataSizePerPort[%llu], dataSizePerRank[%llu], lastRankSliceSize[%llu], "
               "lastPortSliceSize[%llu]",
               __func__, inputAddr, outputAddr, outputToken, isInplace, dataSize, sliceDivisor, dataSizePerPort,
               dataSizePerRank, lastRankSliceSize, lastPortSliceSize);

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllReduceNhrMem2Mem1DMultiJetty>(
        inputAddr, outputAddr, outputToken, isInplace, dataSizePerRank, dataSizePerPort, lastRankSliceSize,
        lastPortSliceSize);

    void* taskArgPtr = static_cast<void*>(taskArg.get());

    CHK_RET(
        HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));

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

HcclResult CcuTempAllReduceNhrMem2Mem1DMultiJetty::GetReduceScatterStepInfo(u32 step, NHRStepInfo &stepInfo)
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

HcclResult CcuTempAllReduceNhrMem2Mem1DMultiJetty::GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo)
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

HcclResult CcuTempAllReduceNhrMem2Mem1DMultiJetty::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo)
{
    u32 nStepsNHR = nSteps / 2;
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

HcclResult CcuTempAllReduceNhrMem2Mem1DMultiJetty::ProcessNHRStepInfo(std::vector<NHRStepInfo> &algStepInfoList)
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
} // namespace ops_hccl