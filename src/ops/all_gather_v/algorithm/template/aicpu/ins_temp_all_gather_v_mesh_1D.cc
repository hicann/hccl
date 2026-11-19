/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_v_mesh_1D.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
namespace ops_hccl {
std::vector<CostModelParam> InsTempAllGatherVMesh1D::CalcCostCoeff(CalcCostCoeffParam param)
{
    int portNum = param.portNum[0];
    int taskNum
        = CostModelManager::CalcTransTaskNum(param.rankSize) + CostModelManager::CalcSyncTaskNum(param.rankSize) * 2;
    float A = 0;
    float B = 0;
    float C = 0;
    float D = 0;

    CostModelManager::Global()->CalcMeshParam(param.dataRatio, param.netType, portNum, param.rankSize, A, param.isPod);
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B);
    }
    CostModelManager::Global()->CalcLatencyParams(1, EngineType::AICPU, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AICPU, D);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

InsTempAllGatherVMesh1D::InsTempAllGatherVMesh1D(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
    supportSymmetricMemory_ = param.supportSymmetricMemory;
}
InsTempAllGatherVMesh1D::~InsTempAllGatherVMesh1D() {}

HcclResult InsTempAllGatherVMesh1D::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // mesh 算法只做level 0 层级的
    GetResWithoutLinks(resourceRequest);

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}
HcclResult InsTempAllGatherVMesh1D::GetResWithoutLinks(AlgResourceRequest& resourceRequest) const
{
    u32 level0RankSize = templateRankSize_;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1; // 主thread可以通过接口传入的stream来做转换
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherVMesh1D::GetThreadNum() const { return templateRankSize_ > 1 ? templateRankSize_ - 1 : 1; }

u64 InsTempAllGatherVMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    if (supportSymmetricMemory_) {
        return 0;
    }
    u64 scratchMultiple = 0;
    if (opMode_ == OpMode::OPBASE) {
        scratchMultiple = templateRankSize_;
    }
    return scratchMultiple;
}

HcclResult InsTempAllGatherVMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    HCCL_INFO("[InsTempAllGatherVMesh1D] Run start");
    if (tempAlgParams.sliceSize == 0) {
        HCCL_INFO("[InsTempAllGatherVMesh1D] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }
    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.vDataDes.dataType;
    HCCL_DEBUG("[InsTempAllGatherVMesh1D] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);
    CHK_RET(LocalDataCopy(templateResource.threads));
    if (templateRankSize_ == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    CHK_RET(RunAllGatherVMesh(templateResource.threads, templateResource.channels));

    if (threadNum_ > 1) {
        // 定义在alg_v2_template_base.h中的基类
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    if (!enableRemoteMemAccess_) {
        CHK_RET(PostLocalCopy(templateResource.threads));
    }
    HCCL_INFO("[InsTempAllGatherVMesh1D] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherVMesh1D::RunAllGatherVMesh(
    const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    HCCL_INFO("[InsTempAllGatherVMesh1D] RunAllGatherVMesh RankIDs[%d].", myRank_);
    u64 dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataTypeSize_ == 0, HCCL_ERROR("[InsTempAllGatherVMesh1D] dataTypeSize is 0"), HCCL_E_INTERNAL);
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * dataTypeSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (u32 threadIdx = 0; threadIdx < subCommRanks_[0].size() - 1; threadIdx++) {
            u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + threadIdx) % subCommRanks_[0].size()];

            u32 connectedAlgRank = 0;
            CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));
            HCCL_INFO(
                "[InsTempAllGatherVMesh1D] RunAllGatherVMesh RankIDs[%d], connectedRank[%d], connectedAlgRank[%d].",
                myRank_, connectedRank, connectedAlgRank);

            // 异常检查
            CHK_PRT_RET(
                threadIdx >= threads.size() || channels.count(connectedRank) == 0 || channels.at(connectedRank).empty(),
                HCCL_ERROR(
                    "[InsTempAllGatherVMesh1D][RankID]=%u threadIdx=%u, threads.size=%zu, "
                    "connectedRank=%d, channels.size=%zu",
                    myRank_, threadIdx, threads.size(), connectedRank, channels.size()),
                HcclResult::HCCL_E_INTERNAL);

            ThreadHandle currQue = threads[threadIdx];
            // 预留兼容offload模式
            const ChannelInfo& linkRemote = channels.at(connectedRank)[0];
            void* remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
            void* remoteOutBuffAddr = linkRemote.remoteOutputGraphMode.addr;
            if (enableRemoteMemAccess_ && remoteOutBuffAddr == nullptr) {
                HCCL_ERROR(
                    "[InsTempAllGatherVMesh1D] remote output addr is null, localRank[%u], remoteRank[%u]", myRank_,
                    connectedRank);
                return HcclResult::HCCL_E_INTERNAL;
            }

            const u64 mySliceSize = tempAlgParams_.allRankSliceSize[myAlgRank];
            const u64 peerSliceSize = tempAlgParams_.allRankSliceSize[connectedAlgRank];
            const u64 mySliceCount = mySliceSize / dataTypeSize_;
            const u64 peerSliceCount = peerSliceSize / dataTypeSize_;

            u64 txOutOffset = tempAlgParams_.allRankDispls[myAlgRank] * dataTypeSize_ + outBaseOff;
            u64 txScratchOffset = scratchBase + tempAlgParams_.sliceSize * myAlgRank;
            u64 txDstOffset = (!enableRemoteMemAccess_) ? txScratchOffset : txOutOffset;

            u64 rxOutOffset = tempAlgParams_.allRankDispls[connectedAlgRank] * dataTypeSize_ + outBaseOff;
            u64 rxScratchOffset = scratchBase + tempAlgParams_.sliceSize * connectedAlgRank;
            // write 模式下 rx 地址不参与寻址；read 模式下源与目的同为对端/本地 output（或 CCL）布局，偏移一致
            u64 rxOffset = (!enableRemoteMemAccess_) ? rxScratchOffset : rxOutOffset;

            if (enableRemoteMemAccess_) {
                // 空槽（本端/对端本轮无数据）不做落点校验：多轮 loop 且各 rank counts 不等时，
                // 已传完一侧的槽位无实际写行为，按 displs+本轮进度校验会误报越界。
                const u64 outputSize = tempAlgParams_.buffInfo.outputSize;
                if ((mySliceSize > 0 && txOutOffset + mySliceSize > outputSize)
                    || (peerSliceSize > 0 && rxOutOffset + peerSliceSize > outputSize)) {
                    HCCL_ERROR(
                        "[InsTempAllGatherVMesh1D] output offset out of range, localRank[%u], remoteRank[%u], "
                        "txOffset[%llu], txSize[%llu], rxOffset[%llu], rxSize[%llu], outputSize[%llu]",
                        myRank_, connectedRank, txOutOffset, mySliceSize, rxOutOffset, peerSliceSize, outputSize);
                    return HcclResult::HCCL_E_INTERNAL;
                }
            }

            void* txSrcPtr = tempAlgParams_.buffInfo.outputPtr;
            void* txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : remoteOutBuffAddr;
            void* rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : remoteOutBuffAddr;
            void* rxDstPtr
                = (!enableRemoteMemAccess_) ? tempAlgParams_.buffInfo.hcclBuff.addr : tempAlgParams_.buffInfo.outputPtr;
            // write模式使用tx,rx地址不生效，仅使用对端link做Post/Wait
            // read 模式使用rx, tx地址不生效，仅使用对端link做Post/Wait
            std::vector<DataSlice> txSrcSlices{
                DataSlice(txSrcPtr, txOutOffset, mySliceSize, mySliceCount)}; // 本地(send)
            std::vector<DataSlice> txDstSlices{
                DataSlice(txDstPtr, txDstOffset, mySliceSize, mySliceCount)}; // 远程(send)
            // read模式使用rx
            std::vector<DataSlice> rxDstSlices{
                DataSlice(rxDstPtr, rxOffset, peerSliceSize, peerSliceCount)}; // 本地(recv)
            std::vector<DataSlice> rxSrcSlices{
                DataSlice(rxSrcPtr, rxOffset, peerSliceSize, peerSliceCount)}; // 远程(recv)

            HCCL_DEBUG(
                "[InsTempAllGatherVMesh1D][RunAllGatherVMesh] rankId [%d] connectedRank [%d] txSrcSlices: "
                "offset[%d] sliceSize[%d] count[%d].",
                myRank_, connectedRank, txOutOffset, mySliceSize, mySliceCount);

            HCCL_DEBUG(
                "[InsTempAllGatherVMesh1D][RunAllGatherVMesh] rankId [%d] connectedRank [%d] txDstSlices: "
                "offset[%d] sliceSize[%d] count[%d].",
                myRank_, connectedRank, txDstOffset, mySliceSize, mySliceCount);

            HCCL_DEBUG(
                "[InsTempAllGatherVMesh1D][RunAllGatherVMesh] rankId [%d] connectedRank [%d] rxSrcSlices: "
                "offset[%d] sliceSize[%d] count[%d].",
                myRank_, connectedRank, rxOutOffset, peerSliceSize, peerSliceCount);

            HCCL_DEBUG(
                "[InsTempAllGatherVMesh1D][RunAllGatherVMesh] rankId [%d] connectedRank [%d] rxDrcSlices: "
                "offset[%d] sliceSize[%d] count[%d].",
                myRank_, connectedRank, rxOffset, peerSliceSize, peerSliceCount);

            TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
            TxRxChannels sendRecvChannels(linkRemote, linkRemote);
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

            CHK_PRT_RET(
                SendRecvWrite(sendRecvInfo, threads[threadIdx]),
                HCCL_ERROR("[InsTempAllGatherVMesh1D] RunAllGatherV Send failed"), HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherVMesh1D::LocalDataCopy(const std::vector<ThreadHandle>& threads)
{
    HCCL_INFO("[InsTempAllGatherVMesh1D] LocalDataCopy.");
    if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr) {
        return HcclResult::HCCL_SUCCESS;
    }
    u32 myAlgRank;
    u64 dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataTypeSize_ == 0, HCCL_ERROR("[InsTempAllGatherVMesh1D] dataTypeSize is 0"), HCCL_E_INTERNAL);
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        // repeat 造成的偏移
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        // 数据块rank编号造成的偏移
        const u64 inOff = inBaseOff;
        const u64 outOff = tempAlgParams_.allRankDispls[myAlgRank] * dataTypeSize_ + outBaseOff;
        const u64 sliceSize = tempAlgParams_.allRankSliceSize[myAlgRank];
        const u64 sliceCount = sliceSize / dataTypeSize_;

        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOff, sliceSize, sliceCount);

        HCCL_DEBUG(
            "[InsTempAllGatherVMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] srcSlice: inBaseOff[%d] inOff[%d] "
            "sliceSize[%d] count[%d].",
            myRank_, myAlgRank, inBaseOff, inOff, sliceSize, sliceCount);
        HCCL_DEBUG(
            "[InsTempAllGatherVMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] dstSlice: outBaseoff[%d] "
            "outOff[%d] sliceSize[%d] count[%d].",
            myRank_, myAlgRank, outBaseOff, outOff, sliceSize, sliceCount);

        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherVMesh1D::PostLocalCopy(const std::vector<ThreadHandle>& threads)
{
    HCCL_INFO("[InsTempAllGatherVMesh1D] PostLocalCopy.");
    u64 dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataTypeSize_ == 0, HCCL_ERROR("[InsTempAllGatherVMesh1D] dataTypeSize is 0"), HCCL_E_INTERNAL);
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (auto rank : subCommRanks_[0]) {
            if (rank == myRank_) {
                continue;
            }
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
            u64 scratchOffset = tempAlgParams_.sliceSize * algRank + scratchBase;
            u64 outBaseOff = tempAlgParams_.allRankProcessedDataCount[algRank] * dataTypeSize_
                             + rpt * tempAlgParams_.outputRepeatStride;
            u64 outOffset = tempAlgParams_.allRankDispls[algRank] * dataTypeSize_ + outBaseOff;
            u64 processSize = tempAlgParams_.allRankSliceSize[algRank];
            const u64 processCount = processSize / dataTypeSize_;
            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, processSize, processCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset, processSize, processCount);
            HCCL_DEBUG(
                "[InsTempAllGatherVMesh1D] PostLocalCopy RankID [%d] dataRank [%d] dataAlgRank[%d] "
                "scratchBase[%d] outBaseOff[%d] scratchOffset[%d] outOffset[%d].",
                myRank_, rank, algRank, scratchBase, outBaseOff, scratchOffset, outOffset);
            CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherVMesh1D::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempAllGatherVMesh1D::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

} // namespace ops_hccl
