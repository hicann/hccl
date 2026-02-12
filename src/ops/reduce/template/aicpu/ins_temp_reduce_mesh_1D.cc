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
#include "ins_temp_reduce_mesh_1D.h"

namespace ops_hccl {
InsTempReduceMesh1D::InsTempReduceMesh1D(const OpParam &param,
    const u32 rankId,  // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempReduceMesh1D::~InsTempReduceMesh1D()
{}

HcclResult InsTempReduceMesh1D::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfo *topoInfo, AlgResourceRequest &resourceRequest)
{
    threadNum_ = templateRankSize_ > 1 ? templateRankSize_ : 1;
    resourceRequest.slaveThreadNum = threadNum_ - 1;
    for (u32 index = 0; index < threadNum_ - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum_ - 1;

    resourceRequest.channels.emplace_back();
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, resourceRequest.channels.back()));
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

u64 InsTempReduceMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = templateRankSize_;
    return scratchMultiple;
}

HcclResult InsTempReduceMesh1D::KernelRun(
    const OpParam &param, const TemplateDataParams &tempAlgParams, const TemplateResource &templateResource)
{
    HCCL_INFO("[InsTempReduceMesh1D] rank[%d] KernelRun start", myRank_);
    // 处理数据量为0的场景
    CHK_PRT_RET(tempAlgParams.sliceSize == 0,
        HCCL_INFO("[InsTempReduceMesh1D] sliceSize is 0, no need to process"),
        HCCL_SUCCESS);

    CHK_PRT_RET(templateRankSize_ == 0, HCCL_ERROR("[InsTempReduceMesh1D] rankSize is 0"), HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(root_ == UINT32_MAX, HCCL_ERROR("[InsTempReduceMesh1D] root is invalid"), HcclResult::HCCL_E_INTERNAL);

    const std::vector<ThreadHandle> &threads = templateResource.threads;
    threadNum_ = templateRankSize_;
    CHK_PRT_RET(threads.size() != threadNum_,
        HCCL_ERROR("[InsTempReduceMesh1D] resource threadNum[%u] is invalid, need[%u]", threads.size(), threadNum_),
        HcclResult::HCCL_E_INTERNAL);

    myIdx_ = GetAlgRank(myRank_);
    CHK_PRT_RET(myIdx_ >= templateRankSize_,
        HCCL_ERROR("[InsTempReduceMesh1D] rank idx[%u] in virtRankMap is invalid, it should be less than rankSize[%u]",
            myIdx_,
            templateRankSize_),
        HcclResult::HCCL_E_INTERNAL);
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    threadNum_ = templateRankSize_ > 1 ? templateRankSize_ : 1;
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    GetNotifyIdxSubToMain(notifyIdxSubToMain_);

    HCCL_INFO(
        "[KernelRun] sliceSize: %u, count_: %u, typeSize: %u", tempAlgParams.sliceSize, count_, SIZE_TABLE[dataType_]);

    const std::map<u32, std::vector<ChannelInfo>> &channels = templateResource.channels;
    CHK_RET(RunReduce(channels, threads, tempAlgParams));

    HCCL_INFO("[InsTempReduceMesh1D] rank[%d] KernelRun finished", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh1D::RunReduce(const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParam)
{
    if (myRank_ == root_) {
        const ThreadHandle &masterThread = threads.at(0);
        const std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.end());
        // 主从队列同步
        if (threads.size() > 1u) {
            CHK_RET(PreSyncInterThreads(masterThread, subThreads, notifyIdxMainToSub_));
        }
        // Gather数据
        CHK_RET(GatherData(tempAlgParam, channels, threads));
        // 主从队列同步
        if (threads.size() > 1u) {
            CHK_RET(PostSyncInterThreads(masterThread, subThreads, notifyIdxSubToMain_));
        }
        // 规约数据
        CHK_RET(ReduceData(tempAlgParam, threads));
    } else {
        // Gather数据
        CHK_RET(SendData(tempAlgParam, channels, threads));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh1D::SendData(const TemplateDataParams &dataParams,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads)
{
    const BuffInfo &buffInfo = dataParams.buffInfo;
    DataSlice srcDataSlice(buffInfo.inputPtr, buffInfo.inBuffBaseOff, processSize_, count_);

    const ChannelInfo &sendChannel = channels.at(root_).at(0);
    u64 dstOffset = buffInfo.hcclBuffBaseOff + processSize_ * myIdx_;
    if ((buffInfo.inBuffType == BufferType::HCCL_BUFFER || buffInfo.outBuffType == BufferType::HCCL_BUFFER) &&
        (myIdx_ == 0)) {
        // 兼容parallel executor调用时，SendData可能存在将CCL_BUFFER上的输入冲掉的场景：
        // parallel executor调用时，输入数据再CCL_BUFFER的第0块，直接使用processSize_ * myIdx_ = 0会将输入冲掉
        // 因此这里与GetAlgRank(root_)调换一下顺序，拷贝到1D root卡的processSize_ * GetAlgRank(root_)位置
        dstOffset = buffInfo.hcclBuffBaseOff + processSize_ * GetAlgRank(root_);
    }
    DataSlice dstDataSlice(sendChannel.remoteCclMem.addr, dstOffset, processSize_, count_);
    HCCL_INFO("[SendData] %u from: %#llx offset[%u], to: %#llx, buffInfo.inBuffBaseOff:[%llu], "
              "buffInfo.hcclBuffBaseOff:[%llu]",
        myRank_,
        buffInfo.inputPtr,
        buffInfo.inBuffBaseOff,
        sendChannel.remoteCclMem.addr,
        buffInfo.inBuffBaseOff,
        buffInfo.hcclBuffBaseOff);
    SlicesList sendSlicesList({srcDataSlice}, {dstDataSlice});
    DataInfo sendInfo(sendChannel, sendSlicesList);

    CHK_PRT_RET(SendWrite(sendInfo, threads.at(0)),
        HCCL_ERROR("[InsTempReduceMesh1D] Send data failed"),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh1D::GatherData(const TemplateDataParams &dataParams,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads)
{
    const BuffInfo &buffInfo = dataParams.buffInfo;

    // 主流将数据从inBuff拷贝到outBuff
    if (buffInfo.inBuffType != buffInfo.outBuffType) {
        void *const dstPtr = static_cast<void *>(static_cast<u8 *>(buffInfo.outputPtr) + buffInfo.outBuffBaseOff);
        void *const srcPtr = static_cast<void *>(static_cast<u8 *>(buffInfo.inputPtr) + buffInfo.inBuffBaseOff);

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(threads.at(0), dstPtr, srcPtr, processSize_)));
    }

    // 单卡场景做完LocalCopy就直接返回
    if (templateRankSize_ == 1u) {
        HCCL_INFO("[InsTempReduceMesh1D] rankSize is 1, copy data from inBuff to outBuff and return");
        return HcclResult::HCCL_SUCCESS;
    }

    // 从流接收来自其它rank的数据
    u32 queId = 1;
    for (u32 idx = 0; idx < subCommRanks_[0].size(); idx++) {
        if (idx == myIdx_) {
            continue;
        }
        const ChannelInfo &channel = channels.at(subCommRanks_.at(0).at(idx)).at(0);
        const DataSlice srcDataSlice(
            buffInfo.inBuffType == BufferType::INPUT
                ? nullptr  // buffInfo.inBuffType == BufferType::INPUT无法获取其他rank的基地址
                : channel.remoteCclMem.addr,  // buffInfo.inBuffType == BufferType::HCCL_BUFFER为对端的ccl buffer
            buffInfo.inBuffBaseOff,
            processSize_,
            count_);
        u64 dstOffset = buffInfo.hcclBuffBaseOff + processSize_ * idx;
        if ((buffInfo.inBuffType == BufferType::HCCL_BUFFER || buffInfo.outBuffType == BufferType::HCCL_BUFFER) &&
            (idx == 0)) {
            // 兼容parallel executor调用的场景，对应SendData时的dstOffset修改
            dstOffset = buffInfo.hcclBuffBaseOff + processSize_ * GetAlgRank(root_);
            HCCL_DEBUG("[GatherData] parallel: %u %u %u, %llu", myRank_, root_, GetAlgRank(root_), dstOffset);
        }
        const DataSlice dstDataSlice(buffInfo.hcclBuff.addr, dstOffset, processSize_, count_);
        HCCL_INFO("[GatherData] %u from: %#llx, to: %#llx", myRank_, buffInfo.inputPtr, buffInfo.hcclBuff.addr);
        const SlicesList recvSlicesList({srcDataSlice}, {dstDataSlice});
        const DataInfo recvInfo(channel, recvSlicesList);
        CHK_PRT_RET(RecvWrite(recvInfo, threads.at(queId)),
            HCCL_ERROR("[InsTempReduceMesh1D] Recv data failed"),
            HcclResult::HCCL_E_INTERNAL);

        queId++;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh1D::ReduceData(
    const TemplateDataParams &dataParams, const std::vector<ThreadHandle> &threads)
{
    if (templateRankSize_ == 1u) {
        // 当rankSize为1时，数据已经拷贝至output，无需规约，直接返回
        return HcclResult::HCCL_SUCCESS;
    }

    const BuffInfo &buffInfo = dataParams.buffInfo;
    u64 sliceSize = dataParams.sliceSize;

    void *const dst = static_cast<void *>(static_cast<u8 *>(buffInfo.outputPtr) + buffInfo.outBuffBaseOff);

    for (u32 idx = 0; idx < subCommRanks_[0].size(); ++idx) {
        if (buffInfo.inBuffType == BufferType::INPUT && buffInfo.outBuffType == BufferType::OUTPUT) {
            if (idx == myIdx_) {
                continue;
            }
        } else {
            if (idx == 0) {
                continue;
            }
        }

        void *const src =
            static_cast<void *>(static_cast<u8 *>(buffInfo.hcclBuff.addr) + buffInfo.hcclBuffBaseOff + sliceSize * idx);

        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(threads.at(0),
            dst,
            src,
            count_,
            static_cast<HcommDataType>(dataType_),
            static_cast<HcommReduceOp>(reduceOp_))));
    }

    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempReduceMesh1D::GetAlgRank(u32 rank) const
{
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), rank);
    if (iter == subCommRanks_[0].end()) {
        throw std::runtime_error("Cannot find myRank = " + std::to_string(rank) + " in subCommRanks_[0]");
    }
    return static_cast<u32>(std::distance(subCommRanks_[0].begin(), iter));
}

void InsTempReduceMesh1D::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    if (threadNum_ == 0) {
        threadNum_ = templateRankSize_ > 1 ? templateRankSize_ : 1;
    }
    u32 slaveThreadNum = threadNum_ - 1;
    notifyIdxMainToSub = std::vector<u32>(slaveThreadNum, 0);
}

void InsTempReduceMesh1D::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    if (threadNum_ == 0) {
        threadNum_ = templateRankSize_ > 1 ? templateRankSize_ : 1;
    }
    u32 notifyNum = threadNum_ - 1;
    notifyIdxSubToMain.resize(notifyNum);
    std::iota(notifyIdxSubToMain.begin(), notifyIdxSubToMain.end(), 0);
}

u64 InsTempReduceMesh1D::GetThreadNum()
{
    return templateRankSize_;
}

}  // namespace ops_hccl