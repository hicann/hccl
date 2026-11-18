/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_v_mesh_1D_meshchunk.h"
#include "cost_model.h"

namespace ops_hccl {

InsTempReduceScatterVMesh1DMeshChunk::InsTempReduceScatterVMesh1DMeshChunk(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempReduceScatterVMesh1DMeshChunk::~InsTempReduceScatterVMesh1DMeshChunk() {}

std::vector<CostModelParam> InsTempReduceScatterVMesh1DMeshChunk::CalcCostCoeff(CalcCostCoeffParam param)
{
    // 算法由算子入口切换选中（ReduceScatterVOutPlaceCommon），不参与 cost 竞争
    (void)param;
    return {};
}

HcclResult InsTempReduceScatterVMesh1DMeshChunk::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // 每对端一个线程，步间主从同步
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    const u32 NOTIFY_NUM_PER_SLAVE_THREAD = 2;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(NOTIFY_NUM_PER_SLAVE_THREAD);
    }
    resourceRequest.notifyNumOnMainThread = (threadNum - 1) * NOTIFY_NUM_PER_SLAVE_THREAD;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterVMesh1DMeshChunk::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // 非对称路径每 rank 仅 1 个累加槽（环形在途归约）；对称路径不占 CCL（loopTimes=1 由 executor 保证）
    return 1;
}

HcclResult InsTempReduceScatterVMesh1DMeshChunk::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    CHK_PRT_RET(
        threadNum_ < 1, HCCL_ERROR("[InsTempReduceScatterVMesh1DMeshChunk][KernelRun] threads is empty"),
        HCCL_E_INTERNAL);
    dataType_ = param.vDataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(
        templateRankSize_ == 0, HCCL_ERROR("[InsTempReduceScatterVMesh1DMeshChunk] templateRankSize_ is 0"),
        HCCL_E_INTERNAL);
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], rankIdx_));
    supportSymmetricMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    // 对称直写与 PCIe read 模式互斥（HCCL_ALGO 强制指定场景的防御）
    CHK_PRT_RET(
        supportSymmetricMemAccess_ && IsPcieProtocol(templateResource.channels),
        HCCL_ERROR("[InsTempReduceScatterVMesh1DMeshChunk][KernelRun] symmetric direct write is not supported "
                   "on PCIe read-mode links"),
        HCCL_E_NOT_SUPPORT);
    allRankSliceSize_ = tempAlgParams.allRankSliceSize;
    allRankDispls_ = tempAlgParams.allRankDispls;
    if (allRankSliceSize_.size() != templateRankSize_ || allRankDispls_.size() != templateRankSize_) {
        HCCL_ERROR(
            "[InsTempReduceScatterVMesh1DMeshChunk][KernelRun] allRankSliceSize[%zu]/allRankDispls[%zu] "
            "mismatch templateRankSize_[%u]",
            allRankSliceSize_.size(), allRankDispls_.size(), templateRankSize_);
        return HCCL_E_PARA;
    }
    allRankCounts_.resize(templateRankSize_);
    for (u32 i = 0; i < templateRankSize_; ++i) {
        allRankCounts_[i] = allRankSliceSize_[i] / dataTypeSize_;
    }

    HCCL_INFO(
        "[InsTempReduceScatterVMesh1DMeshChunk] Run Start, myRank[%u] algRank[%u] symMem[%d] threads[%zu]", myRank_,
        rankIdx_, supportSymmetricMemAccess_, templateResource.threads.size());
    CHK_PRT_RET(
        tempAlgParams.buffInfo.inputPtr == nullptr || tempAlgParams.buffInfo.outputPtr == nullptr,
        HCCL_ERROR(
            "[InsTempReduceScatterVMesh1DMeshChunk][KernelRun] inputPtr[%p] or outputPtr[%p] is null",
            tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.outputPtr),
        HCCL_E_PARA);
    // 自贡献打底（wrapper ACK 保证对端打底先于本端写入）
    CHK_RET(LocalDataCopy(tempAlgParams, templateResource.threads));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(RunReduceScatterV(templateResource.channels, templateResource.threads, tempAlgParams));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    if (!supportSymmetricMemAccess_) {
        CHK_RET(PostCopy(tempAlgParams, templateResource.threads));
    }
    HCCL_INFO("[InsTempReduceScatterVMesh1DMeshChunk] Run End");
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterVMesh1DMeshChunk::LocalDataCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const
{
    const u64 mySliceSize = allRankSliceSize_[rankIdx_];
    // 0 长度分片：无自贡献数据可搬，仅握手（与环形步进的全局一致 counts 判定对齐）
    if (mySliceSize == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        tempAlgParams.buffInfo.inputPtr == nullptr,
        HCCL_ERROR("[InsTempReduceScatterVMesh1DMeshChunk][LocalDataCopy] inputPtr is null"), HCCL_E_INTERNAL);
    bool toOutput = supportSymmetricMemAccess_;
    void* dstPtr = toOutput ? tempAlgParams.buffInfo.outputPtr : tempAlgParams.buffInfo.hcclBuff.addr;
    if (toOutput && tempAlgParams.buffInfo.inputPtr == tempAlgParams.buffInfo.outputPtr) {
        return HCCL_SUCCESS;
    }
    u64 srcOffset = tempAlgParams.buffInfo.inBuffBaseOff + allRankDispls_[rankIdx_];
    u64 dstOffset = toOutput ? tempAlgParams.buffInfo.outBuffBaseOff : tempAlgParams.buffInfo.hcclBuffBaseOff;
    DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, mySliceSize, allRankCounts_[rankIdx_]);
    DataSlice dstSlice = DataSlice(dstPtr, dstOffset, mySliceSize, allRankCounts_[rankIdx_]);
    HCCL_INFO(
        "[InsTempReduceScatterVMesh1DMeshChunk][LocalDataCopy] srcOff[%llu] dstOff[%llu] size[%llu]", srcOffset,
        dstOffset, mySliceSize);
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterVMesh1DMeshChunk::CalcPeerSegList()
{
    // 每片独立切 N-1 段：前段 4K 向下对齐、尾段取余（防跨段越界）；片过小回退按元素均分。
    // counts/displs 全 rank 一致 → 段切分全 rank 一致，段偏移可直接定位对端落点
    const u32 sliceNum = templateRankSize_ - 1;
    CHK_PRT_RET(
        sliceNum == 0,
        HCCL_ERROR(
            "[InsTempReduceScatterVMesh1DMeshChunk][CalcPeerSegList] templateRankSize_[%u] is 1", templateRankSize_),
        HcclResult::HCCL_E_INTERNAL);
    peerSegSize_.assign(templateRankSize_, {});
    peerSegOff_.assign(templateRankSize_, {});
    for (u32 j = 0; j < templateRankSize_; ++j) {
        const u64 sliceSize = allRankSliceSize_[j];
        std::vector<u64>& segSize = peerSegSize_[j];
        if (sliceSize == 0) {
            // 0 长度分片：全 0 段（wrapper 对 0 尺寸 slice 仍完成握手，步序不 deadlock）
            segSize.assign(sliceNum, 0);
        } else {
            const u64 alignSeg = (sliceSize / sliceNum) / AICPU_ALIGN_SIZE * AICPU_ALIGN_SIZE;
            if (sliceNum >= 2 && alignSeg > 0) {
                for (u32 i = 0; i < sliceNum - 1; ++i) {
                    segSize.push_back(alignSeg);
                }
                segSize.push_back(sliceSize - alignSeg * (sliceNum - 1));
            } else {
                // 小片回退：按元素均分（elemCount < N-1 时小块为 0 尺寸段，仅握手）
                const u64 elemCount = sliceSize / dataTypeSize_;
                const u64 bigNum = elemCount % sliceNum;
                const u64 bigSize = (elemCount / sliceNum + 1) * dataTypeSize_;
                const u64 smallSize = (elemCount / sliceNum) * dataTypeSize_;
                for (u32 i = 0; i < sliceNum; ++i) {
                    segSize.push_back(i < bigNum ? bigSize : smallSize);
                }
            }
        }
        std::vector<u64>& segOff = peerSegOff_[j];
        u64 off = 0;
        for (u32 i = 0; i < sliceNum; ++i) {
            segOff.push_back(off);
            off += segSize[i];
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterVMesh1DMeshChunk::RunReduceScatterV(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const TemplateDataParams& tempAlgParams)
{
    CHK_RET(CalcPeerSegList());
    CHK_RET(DoMeshChunk(channels, threads, tempAlgParams));
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterVMesh1DMeshChunk::DoMeshChunk(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const TemplateDataParams& tempAlgParams)
{
    const u32 sliceNum = templateRankSize_ - 1;
    for (u32 stepIdx = 0; stepIdx < sliceNum; ++stepIdx) {
        for (u32 i = 0; i < sliceNum; ++i) {
            // 段错开：第 i 个对端由 (stepIdx+i) 轮转决定，发送该对端片的第 i 段（每段单写者）
            u32 nextNum = stepIdx + i + 1;
            if (nextNum >= templateRankSize_) {
                nextNum += 1; // 跳过自己
            }
            const u32 nextRank = (rankIdx_ + nextNum) % templateRankSize_;
            const u32 frontRank = (2 * rankIdx_ + templateRankSize_ - nextRank) % templateRankSize_;
            const u32 toRank = subCommRanks_[0][frontRank];
            CHK_PRT_RET(
                channels.find(toRank) == channels.end() || channels.at(toRank).empty(),
                HCCL_ERROR("[InsTempReduceScatterVMesh1DMeshChunk] toRank[%u] not found in channels", toRank),
                HcclResult::HCCL_E_INTERNAL);
            const ChannelInfo& linkSend = channels.at(toRank)[0];
            const ChannelInfo& linkRecv = channels.at(toRank)[0];
            const u64 segSize = peerSegSize_[frontRank][i];
            const u64 segOff = peerSegOff_[frontRank][i];
            const u64 srcOffset = tempAlgParams.buffInfo.inBuffBaseOff + allRankDispls_[frontRank] + segOff;
            const u64 segCount = dataTypeSize_ > 0 ? segSize / dataTypeSize_ : 0;
            DataSlice txSrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, segSize, segCount);
            void* dstAddr
                = supportSymmetricMemAccess_ ? linkSend.remoteOutputGraphMode.addr : linkSend.remoteCclMem.addr;
            CHK_PTR_NULL(dstAddr);
            const u64 dstOffset = supportSymmetricMemAccess_ ? segOff : tempAlgParams.buffInfo.hcclBuffBaseOff + segOff;
            DataSlice txDstSlice = DataSlice(dstAddr, dstOffset, segSize, segCount);
            std::vector<DataSlice> txSrcSlices{txSrcSlice};
            std::vector<DataSlice> txDstSlices{txDstSlice};
            // write 模式 rx 列表不被消费（DoSendRecvBatchTx 仅消费 tx），仅用对端 link 做握手配对
            std::vector<DataSlice> rxSrcSlices;
            std::vector<DataSlice> rxDstSlices;
            SendRecvReduceInfo sendRecvReduceInfo{
                {linkSend, linkRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_, reduceOp_};
            HCCL_INFO(
                "[InsTempReduceScatterVMesh1DMeshChunk] step[%u] seg[%u] to[%u] segSize[%llu]", stepIdx, i, toRank,
                segSize);
            u32 queIdx = frontRank < rankIdx_ ? frontRank : frontRank - 1; // 每对端固定一线程槽
            CHK_PRT_RET(
                queIdx >= threads.size(),
                HCCL_ERROR(
                    "[InsTempReduceScatterVMesh1DMeshChunk] queIdx[%u] >= threads.size[%zu]", queIdx, threads.size()),
                HcclResult::HCCL_E_INTERNAL);
            CHK_PRT_RET(
                SendRecvBatchWriteReduce(sendRecvReduceInfo, threads[queIdx]),
                HCCL_ERROR(
                    "[InsTempReduceScatterVMesh1DMeshChunk] step[%u] seg[%u] SendRecvBatchWriteReduce failed", stepIdx,
                    i),
                HcclResult::HCCL_E_INTERNAL);
        }
        if (threadNum_ > 1 && stepIdx + 1 < sliceNum) {
            std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.end());
            NotifyIdxSubToMainInMeshChunk(notifyIdxSubToMain_);
            CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_));
            NotifyIdxMainToSubInMeshChunk(notifyIdxMainToSub_);
            CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterVMesh1DMeshChunk::PostCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const
{
    const u64 mySliceSize = allRankSliceSize_[rankIdx_];
    if (mySliceSize == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterVMesh1DMeshChunk][PostCopy] threads is empty"), HCCL_E_PARA);
    DataSlice srcSlice = DataSlice(
        tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.hcclBuffBaseOff, mySliceSize,
        allRankCounts_[rankIdx_]);
    DataSlice dstSlice = DataSlice(
        tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, mySliceSize, allRankCounts_[rankIdx_]);
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    return HCCL_SUCCESS;
}

void InsTempReduceScatterVMesh1DMeshChunk::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempReduceScatterVMesh1DMeshChunk::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

void InsTempReduceScatterVMesh1DMeshChunk::NotifyIdxMainToSubInMeshChunk(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(1);
    }
}

void InsTempReduceScatterVMesh1DMeshChunk::NotifyIdxSubToMainInMeshChunk(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx + notifyNum);
    }
}

} // namespace ops_hccl
