/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_1D_nda_inter.h"

#include "alg_data_trans_wrapper.h"
#include "alg_v2_template_register.h"
#include "channel.h"
#include "dpu_alg_data_trans_wrapper.h"

namespace ops_hccl {

InsTempReduceScatterMesh1dNdaInter::InsTempReduceScatterMesh1dNdaInter(
    const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

HcclResult InsTempReduceScatterMesh1dNdaInter::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> interChannels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, interChannels));
    resourceRequest.channels.push_back(interChannels);
    HCCL_INFO(
        "[InsTempReduceScatterMesh1dNdaInter][CalcRes] slaveThreadNum[%u] "
        "notifyNumOnMainThread[%u] interChannels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, interChannels.size());
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1dNdaInter::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void)inBufferType;
    (void)outBufferType;
    return subCommRanks_[0].size();
}

HcclResult InsTempReduceScatterMesh1dNdaInter::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    dataType_ = param.DataDes.dataType;
    if (templateResource.threads.empty()) {
        HCCL_ERROR("[InsTempReduceScatterMesh1dNdaInter] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    CHK_PRT_RET(
        HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS,
        HCCL_ERROR("[InsTempReduceScatterMesh1dNdaInter] failed to end batch mode, tag[%s].", param.algTag),
        HCCL_E_INTERNAL);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u64 recvSize = tempAlgParams.allRankSliceSize.at(myAlgRank);
    const u64 recvCount = tempAlgParams.allRankProcessedDataCount.at(myAlgRank);
    const u64 recvOffset = tempAlgParams.allRankDispls.at(myAlgRank);

    for (u32 rankIdx = 0; rankIdx < subCommRanks_[0].size(); ++rankIdx) {
        const u32 remoteRank = subCommRanks_[0][rankIdx];
        if (remoteRank == myRank_) {
            continue;
        }

        const u64 sendSize = tempAlgParams.allRankSliceSize.at(rankIdx);
        const u64 sendCount = tempAlgParams.allRankProcessedDataCount.at(rankIdx);
        const u64 sendOffset = tempAlgParams.allRankDispls.at(rankIdx);
        if (sendSize == 0 && recvSize == 0) {
            continue;
        }

        auto channelIt = templateResource.channels.find(remoteRank);
        if (channelIt == templateResource.channels.end() || channelIt->second.empty()) {
            HCCL_ERROR("[InsTempReduceScatterMesh1dNdaInter] channel not found, remoteRank[%u].", remoteRank);
            return HCCL_E_PARA;
        }
        const ChannelInfo& link = channelIt->second[0];
        if (link.remoteCclMem.addr == nullptr) {
            HCCL_ERROR("[InsTempReduceScatterMesh1dNdaInter] remote ccl mem is null, remoteRank[%u].", remoteRank);
            return HCCL_E_PARA;
        }
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; ++repeatIdx) {
            if (sendSize > 0) {
                txSrcSlices.emplace_back(
                    tempAlgParams.buffInfo.inputPtr,
                    tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride + sendOffset,
                    sendSize, sendCount);
                txDstSlices.emplace_back(
                    link.remoteCclMem.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride
                        + myAlgRank * sendSize,
                    sendSize, sendCount);
            }
            if (recvSize > 0) {
                rxSrcSlices.emplace_back(
                    tempAlgParams.buffInfo.inputPtr,
                    tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride + recvOffset,
                    recvSize, recvCount);
                rxDstSlices.emplace_back(
                    tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride
                        + rankIdx * recvSize,
                    recvSize, recvCount);
            }
        }

        TxRxChannels sendRecvChannels(link, link);
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
        CHK_PRT_RET(
            SendRecvWriteForNda(sendRecvInfo, templateResource.threads[0]) != HCCL_SUCCESS,
            HCCL_ERROR("[InsTempReduceScatterMesh1dNdaInter] SendRecvWriteForNda failed, remoteRank[%u].", remoteRank),
            HCCL_E_INTERNAL);
    }

    CHK_PRT_RET(
        HcommBatchModeStart(param.algTag) != HCCL_SUCCESS,
        HCCL_ERROR("[InsTempReduceScatterMesh1dNdaInter] failed to start batch mode, tag[%s].", param.algTag),
        HCCL_E_INTERNAL);
    CHK_RET(PostLocalReduce(param, tempAlgParams, templateResource.threads));
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1dNdaInter::PostLocalReduce(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u64 sliceSize = tempAlgParams.allRankSliceSize.at(myAlgRank);
    const u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(myAlgRank);
    const u64 sliceOffset = tempAlgParams.allRankDispls.at(myAlgRank);
    if (sliceSize == 0) {
        HCCL_INFO("[InsTempReduceScatterMesh1dNdaInter][PostLocalReduce] Rank[%u] has no data.", myRank_);
        return HCCL_SUCCESS;
    }

    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; ++repeatIdx) {
        DataSlice srcSlice(
            tempAlgParams.buffInfo.inputPtr,
            tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride + sliceOffset, sliceSize,
            sliceCount);
        DataSlice dstSlice(
            tempAlgParams.buffInfo.hcclBuff.addr,
            tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride
                + sliceSize * myAlgRank,
            sliceSize, sliceCount);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));

        if (dataType_ == HCCL_DATA_TYPE_INT64 || dataType_ == HCCL_DATA_TYPE_UINT64 || dataType_ == HCCL_DATA_TYPE_FP64
            || reduceOp_ == HcclReduceOp::HCCL_REDUCE_PROD) {
            CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
            CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
            for (const auto& thread : threads) {
                CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
            }
        }

        for (u32 rankIdx = 1; rankIdx < templateRankSize_; ++rankIdx) {
            srcSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride
                    + rankIdx * sliceSize,
                sliceSize, sliceCount);
            dstSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride, sliceSize,
                sliceCount);
            CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], srcSlice, dstSlice, dataType_, reduceOp_)));
        }

        srcSlice = DataSlice(
            tempAlgParams.buffInfo.hcclBuff.addr,
            tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride, sliceSize,
            sliceCount);
        dstSlice = DataSlice(
            tempAlgParams.buffInfo.outputPtr,
            tempAlgParams.buffInfo.outBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride, sliceSize,
            sliceCount);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempReduceScatterMesh1dNdaInter", InsTempReduceScatterMesh1dNdaInter);

} // namespace ops_hccl
