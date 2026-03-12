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
#include "ccu_kernel_reduce_nhr1d_mem2mem.h"
#include "ccu_temp_reduce_nhr_1D_mem2mem.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

CcuTempReduceNHR1DMem2Mem::CcuTempReduceNHR1DMem2Mem(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域中的虚拟rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    rootId_ = param.root;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
}

CcuTempReduceNHR1DMem2Mem::~CcuTempReduceNHR1DMem2Mem()
{
}

HcclResult CcuTempReduceNHR1DMem2Mem::GetDieNumFromChannelDescs(HcclComm comm, u32 &dieNum)
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
        HCCL_ERROR("[CcuTempReduceNHR1DMem2Mem::CalcRes] get channelDescs fail: there are [] link to rank []",
                   firstVector.size(), firstElement->first);
        return HcclResult::HCCL_E_INTERNAL;
    }
}

HcclResult CcuTempReduceNHR1DMem2Mem::ProcessNHRStepInfo(HcclComm comm,
                                                         std::vector<NHRStepInfo>& stepInfoVector,
                                                         std::map<u32, u32>& rank2ChannelIdx, u32 enableDieNum,
                                                         std::vector<std::vector<HcclChannelDesc>>& channelsPerDie)
{
    u32 nSteps = GetNHRStepNum(templateRankSize_) * 2; // 分为RS和AG两次NHR
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
        u32 fromRank = subCommRanks_[0][stepInfo.fromRank];
        u32 toRank = subCommRanks_[0][stepInfo.toRank];
        if (rank2ChannelIdx.count(stepInfo.fromRank) == 0) {
            // 存储 rankid → channelIdx 的索引
            u32 curChannelIdx = channelsPerDie[0].size(); // die0的channel数量
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

void CcuTempReduceNHR1DMem2Mem::SetRoot(u32 root)
{
    return;
}

HcclResult CcuTempReduceNHR1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));

    // 1.从获得的channelDesc，判断kernel发送到几个die上
    uint32_t enableDieNum = 0;
    CHK_RET(GetDieNumFromChannelDescs(comm, enableDieNum));
    
    if (enableDieNum < 1 || enableDieNum > CCU_DIE_NUM_MAX_2) { // 目前只支持1个或2个die
        HCCL_ERROR("[CcuTempReduceNHR1DMem2Mem::CalcRes] get channelDescs fail");
        return HcclResult::HCCL_E_INTERNAL;
    }

    uint32_t kernelNum = enableDieNum;
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.ccuKernelNum.push_back(kernelNum);
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG("[CcuTempReduceNHR1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 2.将channelDescs分到2个die
    std::vector<std::vector<HcclChannelDesc>> channelsPerDie;
    channelsPerDie.resize(enableDieNum);
    std::map<u32, u32> rank2ChannelIdx;
    std::vector<NHRStepInfo> stepInfoVector;
    
    CHK_RET(ProcessNHRStepInfo(comm, stepInfoVector, rank2ChannelIdx, enableDieNum, channelsPerDie));

    // 3.构造kernelInfo
    for (uint32_t kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
        CcuKernelInfo kernelInfo;
        
        kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                                return std::make_unique<CcuKernelReduceNHR1DMem2Mem>(arg);
                            };
        kernelInfo.kernelArg = std::make_shared<CcuKernelArgReduceNHR1D>(subCommRanks_[0].size(),
                                                                         mySubCommRank_,
                                                                         rootId_,
                                                                         kernelIdx, 
                                                                         stepInfoVector, 
                                                                         rank2ChannelIdx,
                                                                         param, 
                                                                         subCommRanks_, 
                                                                         enableDieNum);
        kernelInfo.channels = channelsPerDie[kernelIdx];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }

    HCCL_DEBUG("[CcuTempReduceNHR1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceNHR1DMem2Mem::SplitDataFor2Dies(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        uint64_t& die0Size, uint64_t& die1Size) const
{
    constexpr uint64_t MULTIPLIER = 4;
    uint64_t typeSize = DataTypeSizeGet(param.DataDes.dataType);
    uint64_t dataCount = (templateDataParams.sliceSize / typeSize);
    
    if (dataCount <= templateRankSize_ * MULTIPLIER) {   // 数据量极小，不划分die
        die0Size = dataCount * typeSize;
        die1Size = 0;
        return HcclResult::HCCL_SUCCESS;
    }
    u8 die0PortGroupSize = 1;
    u8 die1PortGroupSize = 1;

    die0Size = (dataCount * die0PortGroupSize / (die0PortGroupSize + die1PortGroupSize)) * typeSize;
    die1Size = templateDataParams.sliceSize - die0Size;
    return HcclResult::HCCL_SUCCESS;
}

/*
 * 一个基本的 Allreduce 数据切分函数，用于ReduceScatter + Allgather组合成的 Allreduce 算。
 * 输入的 dataSize 是一张卡上完整的数据量
 * 函数会将 dataSize 切分成 rankSize 份，最后一份尾块可能会比其他的切分出来的子块大。
 */
HcclResult CcuTempReduceNHR1DMem2Mem::CalcSliceInfoAllReduce(const u64 dataSize, RankSliceInfo &sliceInfoVec) const
{
    sliceInfoVec.clear();
    sliceInfoVec.resize(templateRankSize_);

    u64 unitAllignSize = DataTypeSizeGet(dataType_);
    u64 unitPerSlice = dataSize / unitAllignSize / templateRankSize_;
    HCCL_DEBUG("unitAllignSize[%llu] unitPerSlice[%llu]", unitAllignSize, unitPerSlice);

    u64       accumOff = 0;
    SliceInfo currSlice;
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; rankIdx++) {
        if (rankIdx == templateRankSize_ - 1) {
            currSlice.offset = accumOff;
            currSlice.size   = dataSize - accumOff;
        } else {
            currSlice.offset = accumOff;
            currSlice.size   = unitPerSlice * unitAllignSize;
        }
        CHK_PRT_RET(currSlice.size % unitAllignSize != 0,
                    HCCL_ERROR("[Calc][SliceInfo]rank[%u] slice size[%llu] is invalid, unitAllignSize[%llu]",
                               rankIdx, currSlice.size, unitAllignSize),
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

HcclResult CcuTempReduceNHR1DMem2Mem::KernelRun(const OpParam& param,
                                                const TemplateDataParams& templateDataParams,
                                                const TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempReduceNHR1DMem2Mem] Template KernelRun start.");
    
    u32 kernelNum = templateResource.ccuKernels.size();

    if (templateDataParams.sliceSize == 0) {
        HCCL_INFO("[CcuTempReduceNHR1DMem2Mem] sliceSize is 0, no need do, just success.");
        return HCCL_SUCCESS;
    }
    uint64_t die0Size = 0;
    uint64_t die1Size = 0;
    constexpr uint32_t MAX_DIE_NUM_2 = 2;
    if (kernelNum == MAX_DIE_NUM_2) {
        SplitDataFor2Dies(param, templateDataParams, die0Size, die1Size);
    } else {
        die0Size = templateDataParams.sliceSize;
    }
    buffInfo_ = templateDataParams.buffInfo;
    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                 static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t repeatNum = templateDataParams.repeatNum;
    uint64_t isInputOutputEqual = (inputAddr == outputAddr) ? 1 : 0;

    RankSliceInfo die0SliceInfoVec;
    CHK_RET(CalcSliceInfoAllReduce(die0Size, die0SliceInfoVec));
    RankSliceInfo die1SliceInfoVec;
    CHK_RET(CalcSliceInfoAllReduce(die1Size, die1SliceInfoVec));

    HCCL_INFO("[CcuTempReduceNHRMem2Mem1D] kernelNum[%lu], dimSize[%llu], die0Size[%llu], die1Size[%llu], inputAddr[%llu],"\
        "outputAddr[%llu], repeatNum[%llu], die0Slicesize[%llu], die1Slicesize[%llu], die0LastSlicesize[%llu],"\
        "die1LastSlicesize[%llu]",
        kernelNum, templateRankSize_, die0Size, die1Size, inputAddr, outputAddr, repeatNum,
        die0SliceInfoVec[0][0].size, die1SliceInfoVec[0][0].size,
        die0SliceInfoVec[templateRankSize_-1][0].size, die1SliceInfoVec[templateRankSize_-1][0].size);

    // 前流同步
    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);
        
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    }

    for (uint32_t axisId = 0; axisId < kernelNum; axisId++) {
        if ((axisId == 0 && die0Size == 0) || (axisId == 1 && die1Size == 0)) {
            continue;
        }
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgReduceNHR1D>(
            inputAddr, outputAddr, token, isInputOutputEqual, die0Size, die1Size, die0SliceInfoVec[0][0].size, die1SliceInfoVec[0][0].size,
            die0SliceInfoVec[templateRankSize_ - 1][0].size, die1SliceInfoVec[templateRankSize_ - 1][0].size);

        void* taskArgPtr = static_cast<void*>(taskArg.get());

        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[axisId], templateResource.ccuKernels[axisId], taskArgPtr));
    }

    // 后流同步
    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }

    HCCL_INFO("[CcuTempReduceNHR1DMem2Mem] Template Run for all steps Ends.");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceNHR1DMem2Mem::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo)
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

HcclResult CcuTempReduceNHR1DMem2Mem::GetReduceScatterStepInfo(u32 step, NHRStepInfo &stepInfo) const
{
    u32 virtRankIdx = mySubCommRank_;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = virtRankIdx;

    // 计算通信对象，计算出的是虚拟rankid
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

    // 计算本rank在本轮收/发中的slice编号
    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
        HCCL_INFO("[ReduceNHR1D][GetReduceScatterStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);
        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceNHR1DMem2Mem::GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo)
{
    u32 virtRankIdx = mySubCommRank_;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step; // 1
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

        HCCL_DEBUG("[ReduceNHR1D][GetAllGatherStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceNHR1DMem2Mem::GetThreadNum() const
{
    return 2;
}

HcclResult CcuTempReduceNHR1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = 1;

    return HCCL_SUCCESS;
}

u64 CcuTempReduceNHR1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void) inBuffType;
    (void) outBuffType;
    return 0;
}

} // namespace ops_hccl 