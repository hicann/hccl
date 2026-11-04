/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_1d_nda.h"

namespace ops_hccl {
InsTempReduceScatterMesh1DNDA::InsTempReduceScatterMesh1DNDA() {}

InsTempReduceScatterMesh1DNDA::InsTempReduceScatterMesh1DNDA(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempReduceScatterMesh1DNDA::~InsTempReduceScatterMesh1DNDA() {}

HcclResult InsTempReduceScatterMesh1DNDA::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO(
        "[InsTempReduceScatterMesh1DNDA][CalcRes]slaveThreadNum[%u] notifyNumOnMainThread[%u]"
        " level1Channels[%zu].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, level1Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1DNDA::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void)inBufferType;
    (void)outBufferType;
    u64 scratchMultiple = subCommRanks_[0].size();
    return scratchMultiple;
}

HcclResult InsTempReduceScatterMesh1DNDA::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;

    if (threadNum_ < 1) {
        HCCL_ERROR("[InsTempReduceScatterMesh1DNDA] Rank [%d], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempReduceScatterMesh1DNDA] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    // NDA 直接执行 Mesh1D 数据交换，不走 DPU
    u32 myAlgRank = 0;
    std::vector<u32> rankIds = subCommRanks_[0];
    auto iter = std::find(rankIds.begin(), rankIds.end(), myRank_);
    if (iter != rankIds.end()) {
        myAlgRank = std::distance(rankIds.begin(), iter);
    } else {
        HCCL_ERROR("[InsTempReduceScatterMesh1DNDA][KernelRun] rankIds or myRank is error.");
        return HCCL_E_INTERNAL;
    }

    for (u32 rankIdx = 0; rankIdx < rankIds.size(); rankIdx++) {
        u32 remoteRank = rankIds[rankIdx];
        if (remoteRank == myRank_) {
            continue;
        }
        const ChannelInfo& link = templateResource.channels.at(remoteRank)[0];

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        void* remoteCclBuffAddr = link.remoteCclMem.addr;
        for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
            txSrcSlices.push_back(DataSlice(
                tempAlgParams.buffInfo.inputPtr,
                tempAlgParams.buffInfo.inBuffBaseOff + rankIdx * tempAlgParams.inputSliceStride
                    + repeatIdx * tempAlgParams.inputRepeatStride,
                tempAlgParams.sliceSize, tempAlgParams.count));
            txDstSlices.push_back(DataSlice(
                remoteCclBuffAddr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + myAlgRank * tempAlgParams.outputSliceStride
                    + repeatIdx * tempAlgParams.outputRepeatStride,
                tempAlgParams.sliceSize, tempAlgParams.count));
            rxSrcSlices.push_back(DataSlice(
                tempAlgParams.buffInfo.inputPtr,
                tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride
                    + myAlgRank * tempAlgParams.inputSliceStride,
                tempAlgParams.sliceSize, tempAlgParams.count));
            rxDstSlices.push_back(DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride
                    + rankIdx * tempAlgParams.outputSliceStride,
                tempAlgParams.sliceSize, tempAlgParams.count));
        }

        TxRxChannels sendRecvChannels(link, link);
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        CHK_PRT_RET(
            SendRecvWriteForNda(sendRecvInfo, templateResource.threads[0]),
            HCCL_ERROR("[InsTempReduceScatterMesh1DNDA] SendRecvWriteForNda failed (remoteRank=%u)", remoteRank),
            HcclResult::HCCL_E_INTERNAL);
    }

    // 将执行模式转换回到batch
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempReduceScatterMesh1DNDA] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(PostLocalReduce(param, tempAlgParams, templateResource.threads));
    HCCL_INFO("[InsTempReduceScatterMesh1DNDA] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DNDA::PostLocalReduce(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads)
{
    // 通信结束之后，数据都在 cclBuffer 上，需要搬运到对应的输出位置。
    u32 myAlgRank = 0;
    std::vector<u32> rankIds = subCommRanks_[0];
    auto iter = std::find(rankIds.begin(), rankIds.end(), myRank_);
    if (iter != rankIds.end()) {
        myAlgRank = std::distance(rankIds.begin(), iter);
    } else {
        HCCL_ERROR("[InsTempReduceScatterMesh1DNDA][PostLocalReduce] rankIds or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
        // 将本端数据从inputPtr搬运到cclBuffer上面
        DataSlice srcSlice = DataSlice(
            tempAlgParams.buffInfo.inputPtr,
            tempAlgParams.buffInfo.inBuffBaseOff + myAlgRank * tempAlgParams.inputSliceStride
                + repeatIdx * tempAlgParams.inputRepeatStride,
            processSize_, count_);
        DataSlice dstSlice = DataSlice(
            tempAlgParams.buffInfo.hcclBuff.addr,
            tempAlgParams.buffInfo.hcclBuffBaseOff + myAlgRank * tempAlgParams.outputSliceStride
                + repeatIdx * tempAlgParams.outputRepeatStride,
            processSize_, count_);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));

        // 增加thread synchronize以支持64类数据类型
        if ((dataType_ == HCCL_DATA_TYPE_INT64) || (dataType_ == HCCL_DATA_TYPE_UINT64)
            || (dataType_ == HCCL_DATA_TYPE_FP64) || (reduceOp_ == HcclReduceOp::HCCL_REDUCE_PROD)) {
            // 启动任务并等待所有threads任务执行完成
            CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
            CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
            for (const auto& thread : threads) {
                CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
            }
        }

        // 将后n-1片数据，规约到第0片数据上
        for (u32 tmpRank = 1; tmpRank < templateRankSize_; tmpRank++) {
            DataSlice srcSlice2 = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride
                    + tmpRank * tempAlgParams.outputSliceStride,
                processSize_, count_);
            DataSlice dstSlice2 = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride, processSize_,
                count_);
            CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], srcSlice2, dstSlice2, dataType_, reduceOp_)));
        }
        // 将规约后的分片，搬运到output上
        srcSlice = DataSlice(
            tempAlgParams.buffInfo.hcclBuff.addr,
            tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride, processSize_,
            count_);
        dstSlice = DataSlice(
            tempAlgParams.buffInfo.outputPtr,
            tempAlgParams.buffInfo.outBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride, processSize_, count_);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    }
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempReduceScatterMesh1DNDA", InsTempReduceScatterMesh1DNDA);
} // namespace ops_hccl
