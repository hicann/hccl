/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_all_reduce_nhr_1D_mem2mem.h"
#include "channel.h"
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "ccu_kernel_all_reduce_nhr1d_mem2mem.h"
#include "alg_data_trans_wrapper.h"
namespace ops_hccl {

CcuTempAllReduceNHRMem2Mem1D::CcuTempAllReduceNHRMem2Mem1D(const OpParam& param, 
                                                const u32 rankId, // 传通信域的rankId，userRank
                                                const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
    templateRankSize_ = subCommRanks[0].size();
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
}

CcuTempAllReduceNHRMem2Mem1D::~CcuTempAllReduceNHRMem2Mem1D()
{
}

HcclResult CcuTempAllReduceNHRMem2Mem1D::GetDieNumFromChannelDescs(HcclComm comm, u32 &dieNum)
{
    constexpr u32 LINK_NUM_1 = 2;
    constexpr u32 LINK_NUM_2 = 2;
    auto firstElement = rankIdToChannelDesc_.begin();
    const std::vector<HcclChannelDesc>& firstVector = firstElement->second;
    if (firstVector.size() == 1) {
        dieNum = 1;
        return HcclResult::HCCL_SUCCESS;
    } else if (firstVector.size() == LINK_NUM_2) {
        // 检查2个channel是否在2个die上
        uint32_t dieId0 = 0;
        uint32_t dieId1 = 0;
        GetChannelDieId(comm, myRank_, firstVector[0], dieId0);
        GetChannelDieId(comm, myRank_, firstVector[1], dieId1);
        if (dieId0 == dieId1) {
            dieNum = LINK_NUM_1;
        } else {
            dieNum = LINK_NUM_2;
        }
        return HcclResult::HCCL_SUCCESS;
    } else {
        HCCL_ERROR("[CcuTempReduceScatterNHR1DMem2Mem::CalcRes] get channelDescs fail: there are [] link to rank []",
                   firstVector.size(), firstElement->first);
        return HcclResult::HCCL_E_INTERNAL;
    }
}

HcclResult CcuTempAllReduceNHRMem2Mem1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                  AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));

    // 1.从获得的channelDesc，判断kernel发送到几个die上
    uint32_t enableDieNum = 0;
    CHK_RET(GetDieNumFromChannelDescs(comm, enableDieNum));
    
    if (enableDieNum < 1 || enableDieNum > CCU_DIE_NUM_MAX_2) { // 目前只支持1个或2个die
        HCCL_ERROR("[CcuTempReduceScatterNHR1DMem2Mem::CalcRes] get channelDescs fail");
        return HcclResult::HCCL_E_INTERNAL;
    }

    uint32_t kernelNum = enableDieNum;
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.ccuKernelNum.push_back(kernelNum);
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG("[CcuTempReduceScatterNHR1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 2.将channelDescs分到2个die
    std::vector<std::vector<HcclChannelDesc>> channelsPerDie;
    std::map<u32, u32> rank2ChannelIdx;
    std::vector<NHRStepInfo> stepInfoVector;
    channelsPerDie.resize(enableDieNum);
    CHK_RET(ProcessNHRStepInfo(comm, stepInfoVector, rank2ChannelIdx, enableDieNum, channelsPerDie));
    std::vector<uint64_t> dimSize;
    dimSize.emplace_back(subCommRanks_[0].size());
    for (uint32_t kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        CcuKernelInfo kernelInfo;
        kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                                return std::make_unique<CcuKernelAllReduceNHR1D>(arg);
                            };
        kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllReduceNHR1D>(dimSize,
                                                                            mySubCommRank_,
                                                                            kernelIdx,
                                                                            kernelNum,
                                                                            stepInfoVector,
                                                                            rank2ChannelIdx,
                                                                            param,
                                                                            subCommRanks_);
        kernelInfo.channels = channelsPerDie[kernelIdx];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNHRMem2Mem1D::SplitDataFor2Dies(uint64_t dataCount, uint64_t &die0Size, uint64_t &die1Size) const
{
    constexpr uint64_t MULTIPLIER = 4;
    
    if (dataCount <= templateRankSize_ * MULTIPLIER) {   // 数据量极小，不划分die
        die0Size = 0;
        die1Size = dataCount * DataTypeSizeGet(dataType_);
        return HcclResult::HCCL_SUCCESS;
    }
    u8 die0PortGroupSize = 1;
    u8 die1PortGroupSize = 1;

    die0Size = (dataCount * die0PortGroupSize / (die0PortGroupSize + die1PortGroupSize)) * DataTypeSizeGet(dataType_);
    die1Size = dataCount * DataTypeSizeGet(dataType_) - die0Size;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNHRMem2Mem1D::ProcessNHRStepInfo(HcclComm comm,
                                                            std::vector<NHRStepInfo>& stepInfoVector,
                                                            std::map<u32, u32>& rank2ChannelIdx, u32 enableDieNum,
                                                            std::vector<std::vector<HcclChannelDesc>>& channelsPerDie)
{
    u32 nSteps = 2 * GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
        u32 fromRank = subCommRanks_[0][stepInfo.fromRank];
        u32 toRank = subCommRanks_[0][stepInfo.toRank];
        if (rank2ChannelIdx.count(stepInfo.fromRank) == 0) {
            // 存储 rankid → channelIdx 的索引
            u32 curChannelIdx = channelsPerDie[0].size();
            rank2ChannelIdx[stepInfo.fromRank] = curChannelIdx;         
            for (HcclChannelDesc channel: rankIdToChannelDesc_.at(fromRank)) {
                uint32_t dieId = 0;
                CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
                // 如果是2个die的算法，则分别加入到2个vector中，否则只加入到1个vector
                uint32_t vecIdx = dieId % enableDieNum;
                // 限制只加入一个channel
                if (channelsPerDie[vecIdx].size() == curChannelIdx) {
                    channelsPerDie[vecIdx].push_back(channel);
                }
            }
        }
        if (rank2ChannelIdx.count(stepInfo.toRank) == 0) {
            u32 curChannelIdx = channelsPerDie[0].size();
            rank2ChannelIdx[stepInfo.toRank] = curChannelIdx; 
            for (HcclChannelDesc channel: rankIdToChannelDesc_.at(toRank)) {
                u32 dieId = 0;
                CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
                u32 vecIdx = dieId % enableDieNum;
                if (channelsPerDie[vecIdx].size() == curChannelIdx) {
                    channelsPerDie[vecIdx].push_back(channel);
                }
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNHRMem2Mem1D::CalcSlice(const u64 dataSize, RankSliceInfo &sliceInfoVec) const
{
    // 将数据切分为 templateRankSize_ 份，每份大小为 dataSize / templateRankSize_，最后一份需要包含尾块
    sliceInfoVec.clear();
    sliceInfoVec.resize(templateRankSize_);
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    u64 unitPerSlice = dataSize / dataSizePerVolume / templateRankSize_;

    u64       accumOff = 0;
    SliceInfo currSlice;
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; rankIdx++) {
        if (rankIdx == templateRankSize_ - 1) {
            currSlice.offset = accumOff;
            currSlice.size   = dataSize - accumOff;
        } else {
            currSlice.offset = accumOff;
            currSlice.size   = unitPerSlice * dataSizePerVolume;
        }
        CHK_PRT_RET(currSlice.size % dataSizePerVolume != 0,
                    HCCL_ERROR("[Calc][SliceInfo]rank[%u] slice size[%llu] is invalid, dataSizePerVolume[%llu]",
                               rankIdx, currSlice.size, dataSizePerVolume),
                    HcclResult::HCCL_E_INTERNAL);
        sliceInfoVec[rankIdx].push_back(currSlice);
        accumOff += currSlice.size;
    }

    CHK_PRT_RET((sliceInfoVec[templateRankSize_ - 1][0].offset + sliceInfoVec[templateRankSize_ - 1][0].size != dataSize),
                HCCL_ERROR("[CalcSliceInfoAllReduce] SliceInfo calculation error! DataSize[%llu], "
                           "lastoffset[%llu], lastsize[%llu]",
                           dataSize, sliceInfoVec[templateRankSize_ - 1][0].offset, sliceInfoVec[templateRankSize_ - 1][0].size),
                HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNHRMem2Mem1D::KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
                                                           const TemplateResource& templateResource)
{
    uint64_t dataCount = (templateDataParams.sliceSize / DataTypeSizeGet(dataType_));
    if (dataCount == 0) {
        HCCL_INFO("[CcuTempAllReduceNHRMem2Mem1D] dataCount == 0, Template Run Ends.");
        return HCCL_SUCCESS;
    } 
    u32 kernelNum = templateResource.ccuKernels.size();
    uint64_t die0Size = 0;
    uint64_t die1Size = 0;
    constexpr uint32_t MAX_DIE_NUM_2 = 2;
    if (kernelNum == MAX_DIE_NUM_2) {
        SplitDataFor2Dies(dataCount, die0Size, die1Size);
    } else {
        die0Size = templateDataParams.sliceSize;
    }
    buffInfo_ = templateDataParams.buffInfo;
    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t repeatNum = templateDataParams.repeatNum;
    uint64_t isInputOutputEqual = (inputAddr == outputAddr)? 1: 0;
    RankSliceInfo die0SliceInfoVec;
    CHK_RET(CalcSlice(die0Size, die0SliceInfoVec));
    RankSliceInfo die1SliceInfoVec;
    CHK_RET(CalcSlice(die1Size, die1SliceInfoVec));
    uint32_t axisSize = 2;
    if (die0Size == 0 || die1Size == 0) {
        axisSize = 1;
    }
    HCCL_INFO("[CcuTempAllReduceNHRMem2Mem1D] die0Size[%llu], die1Size[%llu], inputAddr[%llu],"\
        "outputAddr[%llu], repeatNum[%llu], die0Slicesize[%llu], die1Slicesize[%llu], die0LastSlicesize[%llu],"\
        "die1LastSlicesize[%llu]",
        die0Size, die1Size, inputAddr, outputAddr, repeatNum,
        die0SliceInfoVec[0][0].size, die1SliceInfoVec[0][0].size,
        die0SliceInfoVec[templateRankSize_-1][0].size, die1SliceInfoVec[templateRankSize_-1][0].size);

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);    
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    }
    for (uint32_t axisId = 0; axisId < kernelNum; axisId++) {  // 2个die上各一个mission
        if ((axisId == 0 && die0Size == 0) || (axisId == 1 && die1Size == 0)) {
            continue;
        }
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllReduceNHR1D>(
        inputAddr, outputAddr, token, isInputOutputEqual, die0Size, die1Size, die0SliceInfoVec[0][0].size, 
        die1SliceInfoVec[0][0].size, die0SliceInfoVec[templateRankSize_-1][0].size, die1SliceInfoVec[templateRankSize_-1][0].size);
        void* taskArgPtr = static_cast<void*>(taskArg.get());
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[axisId], templateResource.ccuKernels[axisId], taskArgPtr));
    }
    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }
    HCCL_INFO("[CcuTempAllReduceNHRMem2Mem1D] Template Run for all steps Ends.");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllReduceNHRMem2Mem1D::GetThreadNum() const
{
    const u64 NHR_THREAD_NUM = 2;
    return NHR_THREAD_NUM;
}
 
HcclResult CcuTempAllReduceNHRMem2Mem1D::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = 1;
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNHRMem2Mem1D::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo)
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

HcclResult CcuTempAllReduceNHRMem2Mem1D::GetReduceScatterStepInfo(u32 step, NHRStepInfo &stepInfo) const
{
    u32 virtRankIdx = mySubCommRank_;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = virtRankIdx;

    // 计算通信对象
    u32 deltaRank = 1 << step;
    u32 sendTo = (virtRankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 recvFrom = (virtRankIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);
    u32 rxSliceIdx = virtRankIdx;
    u32 txSliceIdx = (virtRankIdx - (1 << step) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG("[AllReduceNHR][GetReduceScatterStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNHRMem2Mem1D::GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo) const
{
    u32 virtRankIdx = mySubCommRank_;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = virtRankIdx;

    // 计算通信对象
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (virtRankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 sendTo = (virtRankIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = virtRankIdx;
    u32 rxSliceIdx = (virtRankIdx - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

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

} // namespace ops_hccl