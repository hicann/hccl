/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_1D_intra.h"

namespace ops_hccl {
InsTempReduceScatterMesh1DIntra::InsTempReduceScatterMesh1DIntra(
    const OpParam& param,
    const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempReduceScatterMesh1DIntra::~InsTempReduceScatterMesh1DIntra() {}

std::vector<CostModelParam> InsTempReduceScatterMesh1DIntra::CalcCostCoeff(CalcCostCoeffParam param)
{
    int portNum = (param.portNum.size() == 1) ? param.portNum[0] : (param.portNum[0] + param.portNum[1]);
    int kernelNum = 1;
    int taskNum = 5 * (param.rankSize - 1);
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    float B1 = 0.0f;
    float B2 = 0.0f;
    CostModelManager::Global()->CalcMeshParam(param.dataRatio, param.netType, portNum, param.rankSize, A, param.isPod);
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B1);
    } else {
        B1 = 0.0f;
    }
    CostModelManager::Global()->CalcLocalReduceParams((param.rankSize - 1) * param.dataRatio, EngineType::AICPU, B2);
    B = B1 + B2;
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AICPU, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AICPU, D);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    HCCL_DEBUG("[%s] CalcCostCoeff A=%f B=%f C=%f D=%f.", __func__, A, B, C, D);
    return params;
}

HcclResult InsTempReduceScatterMesh1DIntra::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    HCCL_DEBUG(
        "[InsTempReduceScatterMesh1DIntra][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u]", myRank_,
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1DIntra::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = templateRankSize_;
    return scratchMultiple;
}

HcclResult InsTempReduceScatterMesh1DIntra::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;

    HCCL_INFO("[InsTempReduceScatterMesh1DIntra] Run Start");
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(RunReduceScatter(templateResource.channels, templateResource.threads, tempAlgParams));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    // 增加thread synchronize以支持64类数据类型
    if ((dataType_ == HCCL_DATA_TYPE_INT64) || (dataType_ == HCCL_DATA_TYPE_UINT64)
        || (dataType_ == HCCL_DATA_TYPE_FP64) || (reduceOp_ == HcclReduceOp::HCCL_REDUCE_PROD)) {
        // 启动任务并等待所有threads任务执行完成
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
        for (const auto& thread : templateResource.threads) {
            CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
        }
    }

    PostCopy(tempAlgParams, templateResource.threads);

    HCCL_INFO("[InsTempReduceScatterMesh1DIntra] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DIntra::PostCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads)
{
    // 通信结束之后，数据都在 cclBuffer 上，需要搬运到对应的输出位置。
    u32 rankIdx = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        rankIdx = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempReduceScatterMesh1DIntra][PostCopy] subCommRanks_ or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }
    // 如果是单算子模式, 并且是最后一步算子，需要将数据从 cclBuffer 拷贝到 userOut
    HCCL_INFO("[InsTempReduceScatterMesh1DIntra][PostCopy], copy from cclBuffer to userOut");
    // 先把本卡的数据从userIn搬运到userOut，然后再在userOut上做规约
    HCCL_INFO("[InsTempReduceScatterMesh1DIntra][PostCopy]tempAlgParams.repeatNum=%llu", tempAlgParams.repeatNum);
    u64 sliceSize = tempAlgParams.allRankSliceSize.at(rankIdx);
    u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(rankIdx);
    u64 sliceOffset = tempAlgParams.allRankDispls.at(rankIdx);
    // 数据量为0的数据片无需Reduce
    if (sliceSize == 0) {
        HCCL_INFO("Skip LocalReduce because there is no data in rank[%d]", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
        // 把其他卡的数据input累加到output
        for (u32 tmpRank = 0; tmpRank < templateRankSize_; tmpRank++) {
            if (tmpRank != rankIdx) {
                DataSlice srcSlice = DataSlice(
                    tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride
                        + tmpRank * sliceSize,
                    sliceSize, sliceCount);
                DataSlice dstSlice = DataSlice(
                    tempAlgParams.buffInfo.outputPtr,
                    tempAlgParams.buffInfo.outBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride, sliceSize,
                    sliceCount);
                CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], srcSlice, dstSlice, dataType_, reduceOp_)));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DIntra::RunReduceScatter(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const TemplateDataParams& tempAlgParam)
{
    u32 myAlgRank = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        myAlgRank = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempReduceScatterMesh1DIntra][RunReduceScatter] subCommRanks_ or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }

    u64 recvSize = tempAlgParam.allRankSliceSize.at(myAlgRank);
    u64 recvCount = tempAlgParam.allRankProcessedDataCount.at(myAlgRank);
    u64 recvOffset = tempAlgParam.allRankDispls.at(myAlgRank);

    if (recvSize != 0) {
        // DMA消减：让thread 0做本地拷贝
        for (u32 repeatIdx = 0; repeatIdx < tempAlgParam.repeatNum; repeatIdx++) {
            DataSlice srcSlice = DataSlice(
                tempAlgParam.buffInfo.inputPtr,
                tempAlgParam.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParam.inputRepeatStride + recvOffset, recvSize,
                recvCount);
            DataSlice dstSlice = DataSlice(
                tempAlgParam.buffInfo.outputPtr,
                tempAlgParam.buffInfo.outBuffBaseOff + repeatIdx * tempAlgParam.outputRepeatStride, recvSize,
                recvCount);
            CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
        }
    }

    if (subCommRanks_[0].size() == 1) {
        HCCL_INFO("[InsTempReduceScatterMesh1DIntra][RunReduceScatter]Skip data trans because there is only 1 rank in "
                  "current level");
        return HCCL_SUCCESS;
    }

    for (u32 queIdx = 1; queIdx < threadNum_; queIdx++) {
        u32 nextRank = (myAlgRank + queIdx) % templateRankSize_; // 这里取的虚拟rankId
        u32 remoteRank = subCommRanks_[0][nextRank];

        u64 sendSize = tempAlgParam.allRankSliceSize.at(nextRank); // 向对端发送的参数
        u64 sendCount = tempAlgParam.allRankProcessedDataCount.at(nextRank);
        u64 sendOffset = tempAlgParam.allRankDispls.at(nextRank);

        if (sendSize == 0 && recvSize == 0) { // 既不发送也不接受
            continue;
        }

        HCCL_DEBUG(
            "[InsTempReduceScatterMesh1DIntra][RunReduceScatter] myRank[%d], toRank[%d], fromRank[%d]", myRank_,
            remoteRank, remoteRank);
        const ChannelInfo& linkSend = channels.at(remoteRank)[0];
        const ChannelInfo& linkRecv = channels.at(remoteRank)[0];

        // 在 HcclBuffer 上进行 ReduceScatter 操作
        // 由于进程只能访问远端的HcclBuffer，所以只能通过write的方式将自己userIn上的数据写到远端HcclBuffer上
        for (u32 repeatIdx = 0; repeatIdx < tempAlgParam.repeatNum; repeatIdx++) {
            // 在all_reduce_op.cc的创建channels的环节中获取到了remote的HcclBuff的地址
            void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;
            // 在接收的时候接收源应该是远端地址，但是由于rs的mesh算法用的是write，所以rx不用care
            DataSlice rxSrcSlice = DataSlice(
                tempAlgParam.buffInfo.inputPtr, // 这里应该是linkSend.remoteInput.addr，但是A5用不到，先用这个本端代替
                tempAlgParam.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParam.inputRepeatStride + recvOffset, recvSize,
                recvCount); // 接收源
            DataSlice rxDstSlice = DataSlice(
                tempAlgParam.buffInfo.hcclBuff.addr,
                tempAlgParam.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParam.outputRepeatStride
                    + nextRank * recvSize,
                recvSize,
                recvCount); // 接收目标
            std::vector<DataSlice> rxSrcSlices{rxSrcSlice};
            std::vector<DataSlice> rxDstSlices{rxDstSlice};
            DataSlice txSrcSlice = DataSlice(
                tempAlgParam.buffInfo.inputPtr,
                tempAlgParam.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParam.inputRepeatStride + sendOffset, sendSize,
                sendCount); // 发送源
            DataSlice txDstSlice = DataSlice(
                remoteCclBuffAddr,
                tempAlgParam.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParam.outputRepeatStride
                    + myAlgRank * sendSize,
                sendSize,
                sendCount); // 发送目标
            std::vector<DataSlice> txSrcSlices{txSrcSlice};
            std::vector<DataSlice> txDstSlices{txDstSlice};
            if (sendSize == 0) { // 发送数据片为0时，只接收数据
                SlicesList recvSliceList(rxSrcSlices, rxDstSlices);
                DataInfo recvInfo(linkRecv, recvSliceList);
                CHK_PRT_RET(
                    RecvWrite(recvInfo, threads[queIdx]),
                    HCCL_ERROR("[InsTempReduceScatterMesh1DIntra][RunReduceScatter] Recv failed."),
                    HcclResult::HCCL_E_INTERNAL);
            } else if (recvSize == 0) { // 接收数据片为0时，只发送数据
                SlicesList sendSliceList(txSrcSlices, txDstSlices);
                DataInfo sendInfo(linkSend, sendSliceList);
                CHK_PRT_RET(
                    SendWrite(sendInfo, threads[queIdx]),
                    HCCL_ERROR("[InsTempReduceScatterMesh1DIntra][RunReduceScatter] Send failed."),
                    HcclResult::HCCL_E_INTERNAL);
            } else {
                SendRecvInfo sendRecvInfo{
                    {linkSend, linkRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
                CHK_PRT_RET(
                    SendRecvWrite(sendRecvInfo, threads[queIdx]),
                    HCCL_ERROR("[InsTempReduceScatterMesh1DIntra][RunReduceScatter] Send failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempReduceScatterMesh1DIntra::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempReduceScatterMesh1DIntra::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

u64 InsTempReduceScatterMesh1DIntra::GetThreadNum() const
{
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    return threadNum;
}

HcclResult InsTempReduceScatterMesh1DIntra::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    return HCCL_SUCCESS;
}

} // namespace ops_hccl
