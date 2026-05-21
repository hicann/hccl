/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include "reduce_mesh_1D_two_shot.h"

namespace ops_hccl {
ReduceMesh1DTwoShot::ReduceMesh1DTwoShot(const OpParam &param,
    const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

ReduceMesh1DTwoShot::~ReduceMesh1DTwoShot()
{}

void ReduceMesh1DTwoShot::SetRoot(u32 root) const
{
    (void)root;  // todo: 为啥mesh 1d不用设置root
    return;
}

HcclResult ReduceMesh1DTwoShot::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    threadNum_ = templateRankSize_ > 1 ? templateRankSize_ : 1;
    resourceRequest.slaveThreadNum = threadNum_ - 1;
    for (u32 index = 0; index < threadNum_ - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum_ - 1;

    resourceRequest.channels.emplace_back();
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, resourceRequest.channels.back()));
    return HCCL_SUCCESS;
}

u64 ReduceMesh1DTwoShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = 0;
    if (!enableRemoteMemAccess_){
        scratchMultiple = templateRankSize_;
    }
    return scratchMultiple;
}

HcclResult ReduceMesh1DTwoShot::KernelRun(
    const OpParam &param, const TemplateDataParams &tempAlgParams, TemplateResource &templateResource)
{
    HCCL_INFO("[ReduceMesh1DTwoShot] rank[%d] KernelRun start", myRank_);
    // 处理数据量为0的场景
    CHK_PRT_RET(
        tempAlgParams.sliceSize == 0, HCCL_INFO("[ReduceMesh1DTwoShot] sliceSize is 0, no need to process"), HCCL_SUCCESS);

    CHK_PRT_RET(templateRankSize_ == 0, HCCL_ERROR("[ReduceMesh1DTwoShot] rankSize is 0"), HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(root_ == UINT32_MAX, HCCL_ERROR("[ReduceMesh1DTwoShot] root is invalid"), HcclResult::HCCL_E_INTERNAL);
    
    CHK_PRT_RET(templateResource.threads.empty(), 
                HCCL_ERROR("[ReduceMesh1DTwoShot] threads is empty"), 
                HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempAlgParams.buffInfo.hcclBuff.addr);
    CHK_PTR_NULL(tempAlgParams.buffInfo.inputPtr);
    CHK_PTR_NULL(tempAlgParams.buffInfo.outputPtr);
    const std::vector<ThreadHandle> &threads = templateResource.threads;
    threadNum_ = templateRankSize_;
    CHK_PRT_RET(threads.size() != threadNum_,
        HCCL_ERROR("[ReduceMesh1DTwoShot] resource threadNum[%u] is invalid, need[%u]", threads.size(), threadNum_),
        HcclResult::HCCL_E_INTERNAL);
    rankList_ = subCommRanks_.at(0);
    
    opMode_ = param.opMode;
    GetAlgRank(myRank_, rankList_, myIdx_);
    CHK_PRT_RET(myIdx_ >= templateRankSize_,
        HCCL_ERROR("[ReduceMesh1DTwoShot] rank idx[%u] in virtRankMap is invalid, it should be less than rankSize[%u]",
            myIdx_,
            templateRankSize_),
        HcclResult::HCCL_E_INTERNAL);
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    threadNum_ = templateRankSize_ > 1 ? templateRankSize_ : 1;
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    GetNotifyIdxSubToMain(notifyIdxSubToMain_);

    HCCL_INFO(
        "[KernelRun] sliceSize: %u, count_: %u, typeSize: %u", tempAlgParams.sliceSize, count_, SIZE_TABLE[dataType_]);
    const std::map<u32, std::vector<ChannelInfo>> &channels = templateResource.channels;
    CHK_RET(CalcSlice());
    CHK_RET(RunReduceScatter(tempAlgParams, param, channels, threads));
    CHK_RET(RunGatherToRoot(tempAlgParams, channels, threads));
    
    return HCCL_SUCCESS;
}

HcclResult ReduceMesh1DTwoShot::CalcSlice()
{
    sliceInfoList_.clear();
    sliceInfoList_.reserve(templateRankSize_);
    u32 dataTypeSize = SIZE_TABLE[dataType_];
    u64 totalElements = processSize_ / dataTypeSize;
    u64 baseElements = totalElements / templateRankSize_;
    u64 remainderElements = totalElements % templateRankSize_;

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; rankIdx++) {
        u64 currSize = 0;
        u64 curCount = 0;

        if (rankIdx < remainderElements) {
            currSize = (baseElements + 1) * dataTypeSize;
        } else {
            currSize = baseElements * dataTypeSize;
        }
        curCount = currSize / dataTypeSize;
        sliceInfoList_.emplace_back(accumOff, currSize, curCount);
        accumOff += currSize;
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceMesh1DTwoShot::SendRecvDataToPeers(const TemplateDataParams &tempAlgParam,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads)
{
    void* localInBuffPtr = tempAlgParam.buffInfo.inputPtr;
    void* localHcclBuffPtr = tempAlgParam.buffInfo.hcclBuff.addr;
    u64 inBuffBaseOffset = tempAlgParam.buffInfo.inBuffBaseOff;
    u64 hcclBuffBaseOffset = tempAlgParam.buffInfo.hcclBuffBaseOff;

    const u64 recvSliceSize = sliceInfoList_.at(myIdx_).size;
    const u64 recvSliceCount = sliceInfoList_.at(myIdx_).count;
    const u64 recvSliceOffset = sliceInfoList_.at(myIdx_).offset;

    for (u32 remoteIdx = 0; remoteIdx < templateRankSize_; remoteIdx++) {
        u64 sendSliceSize = sliceInfoList_.at(remoteIdx).size;
        u64 sendSliceCount = sliceInfoList_.at(remoteIdx).count;
        u64 sendSliceOffset = sliceInfoList_.at(remoteIdx).offset;

        if (sendSliceSize == 0 && recvSliceSize == 0) {
            continue;
        }

        if (remoteIdx == myIdx_) {
            if (enableRemoteMemAccess_) {
                continue; // 图模式跳过本地拷贝到CCL Buffer
            }
            DataSlice copySrcSlice(localInBuffPtr, inBuffBaseOffset + sendSliceOffset, sendSliceSize, sendSliceCount);
            DataSlice copyDstSlice(localHcclBuffPtr, hcclBuffBaseOffset + sendSliceOffset, sendSliceSize, sendSliceCount);
            CHK_PRT_RET(LocalCopy(threads.at(remoteIdx), copySrcSlice, copyDstSlice),
                HCCL_ERROR("[InsTempReduceMesh1DTwoShot][SendRecvDataToPeers] LocalCopy failed."),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            u32 remoteRank = rankList_.at(remoteIdx);
            const ChannelInfo &sendRecvChannel = channels.at(remoteRank).at(0);
            
            //图模式下 inputBuffer -> inputBuffer, 否则为 inputBuffer -> hcclBuff
            void* remoteInBuffPtr = (!enableRemoteMemAccess_) ? sendRecvChannel.remoteCclMem.addr : sendRecvChannel.remoteInputGraphMode.addr;
            void* remoteDstBuffPtr = (!enableRemoteMemAccess_) ? sendRecvChannel.remoteCclMem.addr : sendRecvChannel.remoteInputGraphMode.addr;
            void* localDstBuffPtr = (!enableRemoteMemAccess_) ? localHcclBuffPtr : localInBuffPtr;

            u64 sendDstOffset = (!enableRemoteMemAccess_) ?   recvSliceOffset + hcclBuffBaseOffset : recvSliceOffset + inBuffBaseOffset;
            u64 recvDstOffset = (!enableRemoteMemAccess_) ?  sendSliceOffset + hcclBuffBaseOffset : sendSliceOffset + inBuffBaseOffset;

            DataSlice sendSrcSlice(localInBuffPtr, inBuffBaseOffset + sendSliceOffset, sendSliceSize, sendSliceCount);
            DataSlice sendDstSlice(remoteDstBuffPtr, sendDstOffset, sendSliceSize, sendSliceCount);
            std::vector<DataSlice> sendSrcSlicesList{sendSrcSlice};
            std::vector<DataSlice> sendDstSlicesList{sendDstSlice};

            DataSlice recvSrcSlice(remoteInBuffPtr, inBuffBaseOffset + recvSliceOffset, recvSliceSize, recvSliceCount);
            DataSlice recvDstSlice(localDstBuffPtr, recvDstOffset, recvSliceSize, recvSliceCount);
            std::vector<DataSlice> recvSrcSlicesList{recvSrcSlice};
            std::vector<DataSlice> recvDstSlicesList{recvDstSlice};

            TxRxChannels sendRecvChannels(sendRecvChannel, sendRecvChannel);
            TxRxSlicesList sendRecvSlicesList({sendSrcSlicesList, sendDstSlicesList}, {recvSrcSlicesList, recvDstSlicesList});
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);
            CHK_PRT_RET(SendRecvBatchWrite(sendRecvInfo, threads.at(remoteIdx)),
                HCCL_ERROR("[InsTempReduceMesh1DTwoShot][SendRecvDataToPeers] SendRecv failed."),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceMesh1DTwoShot::DoLocalReduce(const TemplateDataParams &tempAlgParam, const OpParam &param,
    const std::vector<ThreadHandle> &threads)
{
    // 图模式下在Input上进行reduce操作，否则在hcclBuff上进行reduce操作
    void* localBuffPtr = (!enableRemoteMemAccess_) ? tempAlgParam.buffInfo.hcclBuff.addr : tempAlgParam.buffInfo.inputPtr;
    u64 localBuffBaseOffset = (!enableRemoteMemAccess_) ? tempAlgParam.buffInfo.hcclBuffBaseOff : tempAlgParam.buffInfo.inBuffBaseOff;

    const u64 recvSliceSize = sliceInfoList_.at(myIdx_).size;
    const u64 recvSliceCount = sliceInfoList_.at(myIdx_).count;
    const u64 recvSliceOffset = sliceInfoList_.at(myIdx_).offset;

    if (recvSliceSize == 0) {
        return HCCL_SUCCESS;
    }

    u64 destOffset = recvSliceOffset + localBuffBaseOffset;
    DataSlice finalDstSlice(localBuffPtr, destOffset, recvSliceSize, recvSliceCount);

    if (dataType_ == HcclDataType::HCCL_DATA_TYPE_INT64 || dataType_ == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        dataType_ == HcclDataType::HCCL_DATA_TYPE_FP64 || param.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
        for (const auto &thread : threads) {
            CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
        }
    }

    for (u32 remoteIdx = 0; remoteIdx < templateRankSize_; remoteIdx++) {
        if (remoteIdx == myIdx_) {
            continue;
        }
        DataSlice curSrcSlice(localBuffPtr, sliceInfoList_.at(remoteIdx).offset + localBuffBaseOffset, recvSliceSize, recvSliceCount);
        CHK_PRT_RET(LocalReduce(threads.at(0), curSrcSlice, finalDstSlice, dataType_, reduceOp_),
            HCCL_ERROR("[InsTempReduceMesh1DTwoShot][DoLocalReduce] LocalReduce failed."),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceMesh1DTwoShot::RunReduceScatter(const TemplateDataParams &tempAlgParam, const OpParam &param,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads)
{
    const ThreadHandle &masterThread = threads.at(0);
    const std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.end());

    if (threads.size() > 1u) {
        CHK_RET(PreSyncInterThreads(masterThread, subThreads, notifyIdxMainToSub_));
    }

    CHK_RET(SendRecvDataToPeers(tempAlgParam, channels, threads));

    if (threads.size() > 1u) {
        CHK_RET(PostSyncInterThreads(masterThread, subThreads, notifyIdxSubToMain_));
    }

    CHK_RET(DoLocalReduce(tempAlgParam, param, threads));

    return HCCL_SUCCESS;
}

ReduceMesh1DTwoShot::LocalSliceInfo ReduceMesh1DTwoShot::GetLocalSliceInfo(const TemplateDataParams &tempAlgParam) const
{
    LocalSliceInfo info;
    info.sliceSize = sliceInfoList_.at(myIdx_).size;
    info.sliceCount = sliceInfoList_.at(myIdx_).count;
    info.sliceOffset = sliceInfoList_.at(myIdx_).offset;
    info.inBuffBaseOffset = tempAlgParam.buffInfo.inBuffBaseOff;
    info.outBuffBaseOffset = tempAlgParam.buffInfo.outBuffBaseOff;
    info.hcclBuffBaseOffset = tempAlgParam.buffInfo.hcclBuffBaseOff;
    info.localInBuffPtr = tempAlgParam.buffInfo.inputPtr;
    info.localOutBuffPtr = tempAlgParam.buffInfo.outputPtr;
    info.localHcclBuffPtr = tempAlgParam.buffInfo.hcclBuff.addr;
    return info;
}

HcclResult ReduceMesh1DTwoShot::GatherLocalData(const TemplateDataParams &tempAlgParam,
    const std::vector<ThreadHandle> &threads)
{
    LocalSliceInfo info = GetLocalSliceInfo(tempAlgParam);
    void* localBuffPtr = (!enableRemoteMemAccess_) ? info.localHcclBuffPtr : info.localInBuffPtr;
    u64 localBuffBaseOffset = (!enableRemoteMemAccess_) ? info.hcclBuffBaseOffset : info.inBuffBaseOffset;

    if (info.sliceSize == 0) {
        return HCCL_SUCCESS;
    }
    DataSlice copySrcSlice(localBuffPtr, localBuffBaseOffset + info.sliceOffset, info.sliceSize, info.sliceCount);
    DataSlice copyDstSlice(info.localOutBuffPtr, info.outBuffBaseOffset + info.sliceOffset, info.sliceSize, info.sliceCount);
    CHK_PRT_RET(LocalCopy(threads.at(myIdx_), copySrcSlice, copyDstSlice),
        HCCL_ERROR("[InsTempReduceMesh1DTwoShot][GatherLocalData] LocalCopy failed."),
        HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ReduceMesh1DTwoShot::GatherRemoteData(const TemplateDataParams &tempAlgParam,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads)
{
    // 图模式下 remoteInputBuff -> localOutputBuff, 否则为 remoteHcclBuff -> localOutputBuff
    u64 outBuffBaseOffset = tempAlgParam.buffInfo.outBuffBaseOff;
    u64 remoteBaseOffset = (!enableRemoteMemAccess_) ? tempAlgParam.buffInfo.hcclBuffBaseOff : tempAlgParam.buffInfo.inBuffBaseOff;
    void* localOutBuffPtr = tempAlgParam.buffInfo.outputPtr;

    for (u32 remoteIdx = 0; remoteIdx < templateRankSize_; remoteIdx++) {
        if (remoteIdx == myIdx_) {
            continue;
        }
        u64 curSize = sliceInfoList_.at(remoteIdx).size;
        u64 curCount = sliceInfoList_.at(remoteIdx).count;
        if (curSize == 0) {
            continue;
        }
        u64 remoteSrcOffset = sliceInfoList_.at(remoteIdx).offset;
        const ChannelInfo &channel = channels.at(subCommRanks_.at(0).at(remoteIdx)).at(0);
        void* remoteBuffPtr = (!enableRemoteMemAccess_) ? channel.remoteCclMem.addr : channel.remoteInputGraphMode.addr;

        DataSlice recvSrcSlice(remoteBuffPtr, remoteBaseOffset + remoteSrcOffset, curSize, curCount);
        DataSlice recvDstSlice(localOutBuffPtr, outBuffBaseOffset + remoteSrcOffset, curSize, curCount);
        const SlicesList recvSlicesList({recvSrcSlice}, {recvDstSlice});
        const DataInfo recvInfo(channel, recvSlicesList);
        CHK_PRT_RET(RecvRead(recvInfo, threads.at(remoteIdx)),
            HCCL_ERROR("[InsTempReduceMesh1DTwoShot][GatherRemoteData] RecvRead failed."),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceMesh1DTwoShot::SendToRoot(const TemplateDataParams &tempAlgParam,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads)
{
    LocalSliceInfo info = GetLocalSliceInfo(tempAlgParam);

    if (info.sliceSize == 0) {
        return HCCL_SUCCESS;
    }
    void* localBufferPtr = (!enableRemoteMemAccess_) ? info.localHcclBuffPtr : info.localInBuffPtr;
    u64 localBuffBaseOffset = (!enableRemoteMemAccess_) ? info.hcclBuffBaseOffset : info.inBuffBaseOffset;

    DataSlice sendSrcSlice(localBufferPtr, localBuffBaseOffset + info.sliceOffset, info.sliceSize, info.sliceCount);
    DataSlice sendDstSlice(info.localOutBuffPtr, info.outBuffBaseOffset + info.sliceOffset, info.sliceSize, info.sliceCount); // SendRead tx 信息不使用
    const SlicesList sendSlicesList({sendSrcSlice}, {sendDstSlice});
    const ChannelInfo &channel = channels.at(subCommRanks_.at(0).at(root_)).at(0);
    const DataInfo sendInfo(channel, sendSlicesList);
    CHK_PRT_RET(SendRead(sendInfo, threads.at(root_)),
        HCCL_ERROR("[InsTempReduceMesh1DTwoShot][SendToRoot] SendRead failed."),
        HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ReduceMesh1DTwoShot::RunGatherToRoot(const TemplateDataParams &tempAlgParam,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads)
{
    const ThreadHandle &masterThread = threads.at(0);
    const std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.end());

    if (threads.size() > 1u) {
        CHK_RET(PreSyncInterThreads(masterThread, subThreads, notifyIdxMainToSub_));
    }

    if (static_cast<u32>(myIdx_) == root_) {
        CHK_RET(GatherLocalData(tempAlgParam, threads));
        CHK_RET(GatherRemoteData(tempAlgParam, channels, threads));
    } else {
        CHK_RET(SendToRoot(tempAlgParam, channels, threads));
    }

    if (threads.size() > 1u) {
        CHK_RET(PostSyncInterThreads(masterThread, subThreads, notifyIdxSubToMain_));
    }
    return HCCL_SUCCESS;
}

void ReduceMesh1DTwoShot::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    if (threadNum_ == 0) {
        threadNum_ = templateRankSize_ > 1 ? templateRankSize_ : 1;
    }
    u32 slaveThreadNum = threadNum_ - 1;
    notifyIdxMainToSub = std::vector<u32>(slaveThreadNum, 0);
}

void ReduceMesh1DTwoShot::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    if (threadNum_ == 0) {
        threadNum_ = templateRankSize_ > 1 ? templateRankSize_ : 1;
    }
    u32 notifyNum = threadNum_ - 1;
    notifyIdxSubToMain.resize(notifyNum);
    std::iota(notifyIdxSubToMain.begin(), notifyIdxSubToMain.end(), 0);
}

u64 ReduceMesh1DTwoShot::GetThreadNum() const
{
    return templateRankSize_;
}

}  // namespace ops_hccl