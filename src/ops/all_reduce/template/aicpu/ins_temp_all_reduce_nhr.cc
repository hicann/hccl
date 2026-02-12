/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_reduce_nhr.h"

namespace ops_hccl {

InsTempAllReduceNHR::InsTempAllReduceNHR(const OpParam& param, const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks) : InsAlgTemplateBase(param, rankId, subCommRanks){}

InsTempAllReduceNHR::~InsTempAllReduceNHR(){}

u64 InsTempAllReduceNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void) inBuffType;
    (void) outBuffType;
    u32 multiple = 1;
    return multiple;
}

HcclResult InsTempAllReduceNHR::CalcRes(HcclComm comm, const OpParam& param, 
    const TopoInfo* topoInfo, AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;  // 不需要从流

    resourceRequest.notifyNumOnMainThread = 0;  // 不需要从流

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);

    HCCL_INFO("[InsTempAllReduceNHR] Calculate resource finished.");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceNHR::GetRes(AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

u64 InsTempAllReduceNHR::GetThreadNum()
{
    return 1;
}

void InsTempAllReduceNHR::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    // NHR算法没有从线程，不需要主从同步Notify
    notifyIdxMainToSub.clear();
}

void InsTempAllReduceNHR::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    // NHR算法没有从线程，不需要主从同步Notify
    notifyIdxSubToMain.clear();
}

HcclResult InsTempAllReduceNHR::KernelRun(const OpParam& param, 
    const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllReduceNHR] KernelRun Start.");

    threadNum_ = templateResource.threads.size();
    CHK_PRT_RET(threadNum_ != 1,
        HCCL_ERROR("[InsTempAllReduceNHR][KernelRun] thread num is invalid, need[1], actual[%u].", threadNum_), 
            HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(subCommRanks_.size() == 0,
        HCCL_ERROR("[InsTempAllReduceNHR][KernelRun] subCommRanks is empty."),
        HcclResult::HCCL_E_INTERNAL);
    rankList_ = subCommRanks_.at(0);

    CHK_PRT_RET(rankList_.size() != templateRankSize_,
        HCCL_ERROR("[InsTempAllReduceNHR][KernelRun] rank count is invalid in rank list.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    // 获取当前rank在rank列表中的序号
    CHK_RET(GetAlgRank(myRank_, rankList_, myRankIdx_));

    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = SIZE_TABLE[dataType_];

    if (count_ == 0) {
        HCCL_WARNING("[InsTempAllReduceNHR][KernelRun] data count is 0.");
        return HcclResult::HCCL_SUCCESS;
    }

    // 数据切片
    CHK_RET(SplitData());
    CHK_PRT_RET(sliceInfoList_.size() != templateRankSize_,
        HCCL_ERROR("[InsTempAllReduceNHR][GenExtIns] slice num[%u] is not equal to rank size[%u].",
            sliceInfoList_.size(), templateRankSize_), HcclResult::HCCL_E_INTERNAL);

    // 将数据从input拷贝到hcclBuffer上
    CHK_RET(PreCopy(tempAlgParams, templateResource.threads));

    // TwoShot算法，第一步ReduceScatter
    CHK_RET(RunReduceScatter(tempAlgParams, templateResource.channels, templateResource.threads));

    // TwoShot算法，第二步AllGather
    CHK_RET(RunAllGather(tempAlgParams, templateResource.channels, templateResource.threads));

    // 将数据从hcclBuffer上拷贝到output上
    CHK_RET(PostCopy(tempAlgParams, templateResource.threads));

    HCCL_INFO("[InsTempAllReduceNHR] KernelRun finished.");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceNHR::SplitData()
{
    u32 sliceNum = templateRankSize_;
    sliceInfoList_.clear();
    sliceInfoList_.reserve(sliceNum);

    u64 sliceCount = RoundUp(count_, sliceNum);
    u64 sliceSize = sliceCount * dataTypeSize_;

    u64 offsetCount = 0;
    u64 offsetSize = 0;
    for (u32 sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
        if (count_ - offsetCount > sliceCount) {
            sliceInfoList_.emplace_back(offsetSize, sliceSize, sliceCount);
            offsetCount += sliceCount;
            offsetSize = offsetCount * dataTypeSize_;
        } else {
            u64 curSliceCount = count_ - offsetCount;
            u64 curSliceSize = curSliceCount * dataTypeSize_;
            sliceInfoList_.emplace_back(offsetSize, curSliceSize, curSliceCount);
            offsetCount = count_;
            offsetSize = offsetCount * dataTypeSize_;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceNHR::PreCopy(const TemplateDataParams &tempAlgParams,
    const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllReduceNHR] PreCopy data from input to hccl buffer");

    void* localInBuffPtr = tempAlgParams.buffInfo.inputPtr;
    void* localHcclBuffPtr = tempAlgParams.buffInfo.hcclBuff.addr;
    u64 inBuffBaseOffset = tempAlgParams.buffInfo.inBuffBaseOff;
    u64 hcclBuffBaseOffset = tempAlgParams.buffInfo.hcclBuffBaseOff;

    DataSlice copySrcSlice(localInBuffPtr, inBuffBaseOffset, processSize_, count_);
    DataSlice copyDstSlice(localHcclBuffPtr, hcclBuffBaseOffset, processSize_, count_);

    CHK_RET(LocalCopy(threads.at(0), copySrcSlice, copyDstSlice));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceNHR::RunReduceScatter(const TemplateDataParams &tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads)
{
    void* localHcclBuffPtr = tempAlgParams.buffInfo.hcclBuff.addr;
    u64 hcclBuffBaseOffset = tempAlgParams.buffInfo.hcclBuffBaseOff;

    std::vector<NHRStepInfo> stepInfoList;
    CHK_RET(GetReduceScatterStepInfoList(stepInfoList));

    for (auto& stepInfo : stepInfoList) {
        CHK_PRT_RET(channels.count(rankList_.at(stepInfo.fromRank)) == 0,
            HCCL_ERROR("[InsTempAllReduceNHR][RunReduceScatter] remoteRank[%u] is not in channels.",
                rankList_.at(stepInfo.fromRank)), HcclResult::HCCL_E_INTERNAL);
        CHK_PRT_RET(channels.count(rankList_.at(stepInfo.toRank)) == 0,
            HCCL_ERROR("[InsTempAllReduceNHR][RunReduceScatter] remoteRank[%u] is not in channels.",
                rankList_.at(stepInfo.toRank)), HcclResult::HCCL_E_INTERNAL);

        const ChannelInfo &recvChannel = channels.at(rankList_.at(stepInfo.fromRank)).at(0);
        const ChannelInfo &sendChannel = channels.at(rankList_.at(stepInfo.toRank)).at(0);
        std::vector<DataSlice> sendSrcSlicesList;
        std::vector<DataSlice> sendDstSlicesList;
        std::vector<DataSlice> recvSrcSlicesList;
        std::vector<DataSlice> recvDstSlicesList;

        void* sendRemoteHcclBuffPtr = sendChannel.remoteCclMem.addr;
        void* recvRemoteHcclBuffPtr = recvChannel.remoteCclMem.addr;

        // 在 nhrInBuffType_ 上进行 ReduceScatter 操作
        for (u32 idx = 0; idx < stepInfo.nSlices; ++idx) {
            u64 sendOffset = hcclBuffBaseOffset + sliceInfoList_.at(stepInfo.txSliceIdxs.at(idx)).offset;
            u64 sendSize = sliceInfoList_.at(stepInfo.txSliceIdxs.at(idx)).size;
            u64 sendCount = sliceInfoList_.at(stepInfo.txSliceIdxs.at(idx)).count;
            DataSlice sendSrcSlice(localHcclBuffPtr, sendOffset, sendSize, sendCount);
            DataSlice sendDstSlice(sendRemoteHcclBuffPtr, sendOffset, sendSize, sendCount);
            sendSrcSlicesList.emplace_back(sendSrcSlice);
            sendDstSlicesList.emplace_back(sendDstSlice);

            u64 recvOffset = hcclBuffBaseOffset + sliceInfoList_.at(stepInfo.rxSliceIdxs.at(idx)).offset;
            u64 recvSize = sliceInfoList_.at(stepInfo.rxSliceIdxs.at(idx)).size;
            u64 recvCount = sliceInfoList_.at(stepInfo.rxSliceIdxs.at(idx)).count;
            DataSlice recvSrcSlice(recvRemoteHcclBuffPtr, recvOffset, recvSize, recvCount);
            DataSlice recvDstSlice(localHcclBuffPtr, recvOffset, recvSize, recvCount);
            recvSrcSlicesList.emplace_back(sendSrcSlice);
            recvDstSlicesList.emplace_back(sendDstSlice);
        }
        SendRecvReduceInfo sendRecvReduceInfo{{sendChannel, recvChannel}, 
            {{sendSrcSlicesList, sendDstSlicesList}, {recvSrcSlicesList, recvDstSlicesList}}, dataType_, reduceOp_};

        CHK_PRT_RET(SendRecvWriteReduce(sendRecvReduceInfo, threads.at(0)),
            HCCL_ERROR("[InsTempAllReduceNHR] RunReduceScatter SendRecvReduce failed"),
            HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceNHR::RunAllGather(const TemplateDataParams &tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads)
{
    void* localHcclBuffPtr = tempAlgParams.buffInfo.hcclBuff.addr;
    u64 hcclBuffBaseOffset = tempAlgParams.buffInfo.hcclBuffBaseOff;

    std::vector<NHRStepInfo> stepInfoList;
    CHK_RET(GetAllGatherStepInfoList(stepInfoList));

    for (auto& stepInfo : stepInfoList) {
        CHK_PRT_RET(channels.count(rankList_.at(stepInfo.fromRank)) == 0,
            HCCL_ERROR("[InsTempAllReduceNHR][RunAllGather] remoteRank[%u] is not in channels.",
                rankList_.at(stepInfo.fromRank)), HcclResult::HCCL_E_INTERNAL);
        CHK_PRT_RET(channels.count(rankList_.at(stepInfo.toRank)) == 0,
            HCCL_ERROR("[InsTempAllReduceNHR][RunAllGather] remoteRank[%u] is not in channels.",
                rankList_.at(stepInfo.toRank)), HcclResult::HCCL_E_INTERNAL);

        const ChannelInfo &recvChannel = channels.at(rankList_.at(stepInfo.fromRank)).at(0);
        const ChannelInfo &sendChannel = channels.at(rankList_.at(stepInfo.toRank)).at(0);
        std::vector<DataSlice> sendSrcSlicesList;
        std::vector<DataSlice> sendDstSlicesList;
        std::vector<DataSlice> recvSrcSlicesList;
        std::vector<DataSlice> recvDstSlicesList;

        void* sendRemoteHcclBuffPtr = sendChannel.remoteCclMem.addr;
        void* recvRemoteHcclBuffPtr = recvChannel.remoteCclMem.addr;

        for (u32 idx = 0; idx < stepInfo.nSlices; ++idx) {
            u64 sendOffset = hcclBuffBaseOffset + sliceInfoList_.at(stepInfo.txSliceIdxs.at(idx)).offset;
            u64 sendSize = sliceInfoList_.at(stepInfo.txSliceIdxs.at(idx)).size;
            u64 sendCount = sliceInfoList_.at(stepInfo.txSliceIdxs.at(idx)).count;
            DataSlice sendSrcSlice(localHcclBuffPtr, sendOffset, sendSize, sendCount);
            DataSlice sendDstSlice(sendRemoteHcclBuffPtr, sendOffset, sendSize, sendCount);
            sendSrcSlicesList.emplace_back(sendSrcSlice);
            sendDstSlicesList.emplace_back(sendDstSlice);

            u64 recvOffset = hcclBuffBaseOffset + sliceInfoList_.at(stepInfo.rxSliceIdxs.at(idx)).offset;
            u64 recvSize = sliceInfoList_.at(stepInfo.rxSliceIdxs.at(idx)).size;
            u64 recvCount = sliceInfoList_.at(stepInfo.rxSliceIdxs.at(idx)).count;
            // remote input，没法写，先用recvRemoteHcclBuffPtr替代
            DataSlice recvSrcSlice(recvRemoteHcclBuffPtr, recvOffset, recvSize, recvCount);
            DataSlice recvDstSlice(localHcclBuffPtr, recvOffset, recvSize, recvCount);
            recvSrcSlicesList.emplace_back(sendSrcSlice);
            recvDstSlicesList.emplace_back(sendDstSlice);
        }
        SendRecvInfo sendRecvInfo{{sendChannel, recvChannel}, 
            {{sendSrcSlicesList, sendDstSlicesList}, {recvSrcSlicesList, recvDstSlicesList}}};

        CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads.at(0)),
            HCCL_ERROR("[InsTempAllReduceNHR] RunAllGather SendRecv failed"),
            HcclResult::HCCL_E_INTERNAL);
    }
    
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceNHR::PostCopy(const TemplateDataParams &tempAlgParams, 
    const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllReduceNHR][PostCopy] Opbase copy from scratchBuffer to userOut");

    void* localHcclBuffPtr = tempAlgParams.buffInfo.hcclBuff.addr;
    void* localOutBuffPtr = tempAlgParams.buffInfo.outputPtr;
    u64 hcclBuffBaseOffset = tempAlgParams.buffInfo.hcclBuffBaseOff;
    u64 outBuffBaseOffset = tempAlgParams.buffInfo.outBuffBaseOff;

    DataSlice copySrcSlice(localHcclBuffPtr, hcclBuffBaseOffset, tempAlgParams.sliceSize, tempAlgParams.count);
    DataSlice copyDstSlice(localOutBuffPtr, outBuffBaseOffset, tempAlgParams.sliceSize, tempAlgParams.count);
    
    CHK_RET(LocalCopy(threads.at(0), copySrcSlice, copyDstSlice));

    return HcclResult::HCCL_SUCCESS;
}

// 计算ReduceScatter每轮收发的对端以及slice编号
HcclResult InsTempAllReduceNHR::GetReduceScatterStepInfoList(std::vector<NHRStepInfo> &stepInfoList)
{
    stepInfoList.clear();

    u32 nSteps = GetNHRStepNum();
    stepInfoList.resize(nSteps);
    for (u32 step = 0; step < nSteps; step++) {
        // 计算通信对象
        u32 deltaRank = 1 << step;
        u32 sendToIdx = (myRankIdx_ + templateRankSize_ - deltaRank) % templateRankSize_;
        u32 recvFromIdx = (myRankIdx_ + deltaRank) % templateRankSize_;

        // 数据份数和数据编号增量
        u32 nSlices = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
        u32 deltaSliceIndex = 1 << (step + 1);
        u32 txSliceIdx = sendToIdx;
        u32 rxSliceIdx = myRankIdx_;

        NHRStepInfo &currStepInfo = stepInfoList[step];
        currStepInfo.step = step;
        currStepInfo.myRank = myRankIdx_;
        currStepInfo.nSlices = nSlices;
        currStepInfo.toRank = sendToIdx;
        currStepInfo.fromRank = recvFromIdx;

        // 计算本rank在每轮收/发中的slice编号
        currStepInfo.txSliceIdxs.reserve(nSlices);
        currStepInfo.rxSliceIdxs.reserve(nSlices);
        for (u32 i = 0; i < nSlices; i++) {
            currStepInfo.txSliceIdxs.push_back(txSliceIdx);
            currStepInfo.rxSliceIdxs.push_back(rxSliceIdx);
            HCCL_DEBUG("[InsTempAllReduceNHR][GetStepInfoList] i[%u] txSliceIdx[%u] rxSliceIdx[%u]",
                i, txSliceIdx, rxSliceIdx);
            txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
            rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

// 计算AllGather每轮收发的对端以及slice编号
HcclResult InsTempAllReduceNHR::GetAllGatherStepInfoList(std::vector<NHRStepInfo> &stepInfoList)
{
    stepInfoList.clear();

    u32 nSteps = GetNHRStepNum();
    stepInfoList.resize(nSteps);
    for (u32 step = 0; step < nSteps; step++) {
        // 计算通信对象
        u32 deltaRank = 1 << (nSteps - 1 - step);
        u32 sendToIdx = (myRankIdx_ + deltaRank) % templateRankSize_;
        u32 recvFromIdx = (myRankIdx_ + templateRankSize_ - deltaRank) % templateRankSize_;

        // 数据份数和数据编号增量
        u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
        u32 deltaSliceIndex = 1 << (nSteps - step);
        u32 txSliceIdx = myRankIdx_;
        u32 rxSliceIdx = (myRankIdx_ - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

        NHRStepInfo &currStepInfo = stepInfoList[step];
        currStepInfo.step = step;
        currStepInfo.myRank = myRankIdx_;
        currStepInfo.nSlices = nSlices;
        currStepInfo.toRank = sendToIdx;
        currStepInfo.fromRank = recvFromIdx;

        // 计算本rank在每轮收/发中的slice编号
        currStepInfo.txSliceIdxs.reserve(nSlices);
        currStepInfo.rxSliceIdxs.reserve(nSlices);
        for (u32 i = 0; i < nSlices; i++) {
            currStepInfo.txSliceIdxs.push_back(txSliceIdx);
            currStepInfo.rxSliceIdxs.push_back(rxSliceIdx);
            HCCL_DEBUG("[InsTempAllReduceNHR][GetStepInfoList] i[%u] txSliceIdx[%u] rxSliceIdx[%u]",
                i, txSliceIdx, rxSliceIdx);
            txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
            rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempAllReduceNHR::GetNHRStepNum()
{
    u32 nSteps = 0;
    for (u32 tmp = templateRankSize_ - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_DEBUG("[InsTempAllReduceNHR][GetNHRStepNum] rankSize[%u] nSteps[%u]", templateRankSize_, nSteps);

    return nSteps;
}

}  // ops_hccl