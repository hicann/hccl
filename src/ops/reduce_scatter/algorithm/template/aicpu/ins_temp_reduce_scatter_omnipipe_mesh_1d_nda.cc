/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_omnipipe_mesh_1d_nda.h"
#include "alg_data_trans_wrapper.h"
#include "dpu_alg_data_trans_wrapper.h"

namespace ops_hccl {
InsTempReduceScatterOmniPipeMesh1dNDA::InsTempReduceScatterOmniPipeMesh1dNDA() {}

InsTempReduceScatterOmniPipeMesh1dNDA::InsTempReduceScatterOmniPipeMesh1dNDA(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
    templateRankSize_ = subCommRanks[0].size();
}

InsTempReduceScatterOmniPipeMesh1dNDA::~InsTempReduceScatterOmniPipeMesh1dNDA() {}

HcclResult InsTempReduceScatterOmniPipeMesh1dNDA::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    HCCL_INFO(
        "[InsTempReduceScatterOmniPipeMesh1dNDA][CalcRes]slaveThreadNum[%u]"
        "notifyNumOnMainThread[%u]"
        " level0Channels[%zu].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, level0Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterOmniPipeMesh1dNDA::GetThreadNum() const { return 1; }

HcclResult InsTempReduceScatterOmniPipeMesh1dNDA::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterOmniPipeMesh1dNDA::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    return 1;
}

void InsTempReduceScatterOmniPipeMesh1dNDA::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) { return; }

void InsTempReduceScatterOmniPipeMesh1dNDA::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) { return; }

HcclResult InsTempReduceScatterOmniPipeMesh1dNDA::DoLocalCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads)
{
    HCCL_INFO("[InsTempReduceScatterOmniPipeMesh1dNDA][DoLocalCopy] DoLocalCopy myRank_ = [%u]", myRank_);
    if (tempAlgParams.sliceSize == 0) {
        HCCL_INFO("Rank [%d], get slicesize zero. skip localcopy", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter == subCommRanks_[0].end()) {
        HCCL_ERROR("[%s]subCommRanks_ or myRank_ is error.", __func__);
        return HCCL_E_INTERNAL;
    }

    void* srcAddr;
    void* dstAddr;
    if (tempAlgParams.buffInfo.inBuffType == BufferType::INPUT) {
        srcAddr = tempAlgParams.buffInfo.inputPtr;
        dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
    } else if (tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
        srcAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        dstAddr = tempAlgParams.buffInfo.outputPtr;
    } else {
        HCCL_ERROR("[%s]InputBufferType Error.", __func__);
        return HCCL_E_PARA;
    }
    HCCL_INFO("MT tempAlgParams.sliceSize = %u", tempAlgParams.sliceSize);
    for (auto i = 0; i < tempAlgParams.repeatNum; ++i) {
        auto srcSlice = DataSlice(
            srcAddr, tempAlgParams.buffInfo.inBuffBaseOff + i * tempAlgParams.inputSliceStride, tempAlgParams.sliceSize,
            tempAlgParams.count);
        auto dstSlice = DataSlice(
            dstAddr, tempAlgParams.buffInfo.outBuffBaseOff + i * tempAlgParams.outputSliceStride,
            tempAlgParams.sliceSize, tempAlgParams.count);
        HCCL_INFO(
            "myRank[%u], i[%u],  srcSlice:%s, dstSlice:%s", myRank_, i, srcSlice.Describe().c_str(),
            dstSlice.Describe().c_str());
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOmniPipeMesh1dNDA::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    if (templateRankSize_ == 1) {
        HCCL_INFO("templateRankSize_ ==1");
        return HcclResult::HCCL_SUCCESS;
    }

    threadNum_ = templateResource.threads.size();
    dataType_ = param.DataDes.dataType;

    HCCL_INFO(
        "[%s]Run Start, threadNum_=%u, processSize_=%u, count_=%u, dataType_=%u", __func__, threadNum_, processSize_,
        count_, dataType_);

    if (threadNum_ < 1) {
        HCCL_ERROR("[InsTempReduceScatterOmniPipeMesh1dNDA] Rank [%d], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempReduceScatterOmniPipeMesh1dNDA] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(RunReduceScatter(templateResource.channels, templateResource.threads, tempAlgParams));
    CHK_RET(PostReduce(tempAlgParams, templateResource.threads));

    // 将执行模式转换回到batch
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempReduceScatterOmniPipeMesh1dNDA] failed to start batch mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[%s]Run End", __func__);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOmniPipeMesh1dNDA::PostReduce(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads)
{
    u32 rankIdx = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        rankIdx = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempReduceScatterOmniPipeMesh1dNDA][RunReduceScatter] subCommRanks_ or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[InsTempReduceScatterOmniPipeMesh1dNDA][PostReduce], copy from cclBuffer to cclBuffer");
    void* cclBuffAddr = tempAlgParams.buffInfo.hcclBuff.addr;
    HCCL_INFO("[InsTempReduceScatterOmniPipeMesh1dNDA][PostReduce]do first slice reduce.");
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.stepSliceInfo.outputOmniPipeSliceStride[rankIdx].size();
         repeatIdx++) {
        for (u32 tmpRank = 0; tmpRank < templateRankSize_; tmpRank++) {
            if (tmpRank != rankIdx) {
                u64 srcCurrent = tempAlgParams.buffInfo.hcclBuffBaseOff
                                 + tempAlgParams.stepSliceInfo.stepOutputSliceStride[tmpRank]
                                 + tempAlgParams.stepSliceInfo.outputOmniPipeSliceStride[tmpRank][repeatIdx];
                u64 dstCurrent = tempAlgParams.buffInfo.outBuffBaseOff
                                 + tempAlgParams.stepSliceInfo.stepInputSliceStride[rankIdx]
                                 + tempAlgParams.stepSliceInfo.inputOmniPipeSliceStride[rankIdx][repeatIdx];
                auto srcSlice = DataSlice(
                    cclBuffAddr, srcCurrent, tempAlgParams.stepSliceInfo.stepSliceSize[rankIdx][repeatIdx],
                    tempAlgParams.stepSliceInfo.stepCount[rankIdx][repeatIdx]);
                auto dstSlice = DataSlice(
                    cclBuffAddr, dstCurrent, tempAlgParams.stepSliceInfo.stepSliceSize[rankIdx][repeatIdx],
                    tempAlgParams.stepSliceInfo.stepCount[rankIdx][repeatIdx]);
                HCCL_DEBUG(
                    "MT srcSlice=[%s],  dstSlice=[%s]", srcSlice.Describe().c_str(), dstSlice.Describe().c_str());
                CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], srcSlice, dstSlice, dataType_, reduceOp_)));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOmniPipeMesh1dNDA::RunReduceScatter(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const TemplateDataParams& tempAlgParam)
{
    HCCL_INFO("MT start to RunReduceScatter, channels.size()=%u", channels.size());
    u32 myAlgRank = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        myAlgRank = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempReduceScatterOmniPipeMesh1dNDA][RunReduceScatter] subCommRanks_ or myRank is error.");
        return HCCL_E_INTERNAL;
    }
    std::vector<u32> rankIds = subCommRanks_[0];
    for (u32 rankIdx = 0; rankIdx < rankIds.size(); rankIdx++) {
        u32 remoteRank = rankIds[rankIdx];
        if (remoteRank == myRank_) {
            continue;
        }
        const ChannelInfo& linkRemote = channels.at(remoteRank)[0];
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        void* localCclBuffAddr = tempAlgParam.buffInfo.hcclBuff.addr;
        void* remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
        for (u32 repeatIdx = 0; repeatIdx < tempAlgParam.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank].size();
             repeatIdx++) {
            u64 txSrcCurrent = tempAlgParam.buffInfo.inBuffBaseOff
                               + tempAlgParam.stepSliceInfo.stepInputSliceStride[rankIdx]
                               + tempAlgParam.stepSliceInfo.inputOmniPipeSliceStride[rankIdx][repeatIdx];
            u64 txDstCurrent = tempAlgParam.buffInfo.hcclBuffBaseOff
                               + tempAlgParam.stepSliceInfo.stepOutputSliceStride[myAlgRank]
                               + tempAlgParam.stepSliceInfo.outputOmniPipeSliceStride[myAlgRank][repeatIdx];

            u64 rxSrcCurrent = tempAlgParam.buffInfo.inBuffBaseOff
                               + tempAlgParam.stepSliceInfo.stepInputSliceStride[myAlgRank]
                               + tempAlgParam.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank][repeatIdx];
            u64 rxDstCurrent = tempAlgParam.buffInfo.hcclBuffBaseOff
                               + tempAlgParam.stepSliceInfo.stepOutputSliceStride[rankIdx]
                               + tempAlgParam.stepSliceInfo.outputOmniPipeSliceStride[rankIdx][repeatIdx];

            DataSlice txSrcSlice = DataSlice(
                localCclBuffAddr, txSrcCurrent, tempAlgParam.stepSliceInfo.stepSliceSize[rankIdx][repeatIdx],
                tempAlgParam.stepSliceInfo.stepCount[rankIdx][repeatIdx]);
            DataSlice txDstSlice = DataSlice(
                remoteCclBuffAddr, txDstCurrent, tempAlgParam.stepSliceInfo.stepSliceSize[rankIdx][repeatIdx],
                tempAlgParam.stepSliceInfo.stepCount[rankIdx][repeatIdx]);
            DataSlice rxSrcSlice = DataSlice(
                remoteCclBuffAddr, rxSrcCurrent, tempAlgParam.stepSliceInfo.stepSliceSize[rankIdx][repeatIdx],
                tempAlgParam.stepSliceInfo.stepCount[rankIdx][repeatIdx]);
            DataSlice rxDstSlice = DataSlice(
                localCclBuffAddr, rxDstCurrent, tempAlgParam.stepSliceInfo.stepSliceSize[rankIdx][repeatIdx],
                tempAlgParam.stepSliceInfo.stepCount[rankIdx][repeatIdx]);

            rxSrcSlices.push_back(rxSrcSlice);
            rxDstSlices.push_back(rxDstSlice);
            txSrcSlices.push_back(txSrcSlice);
            txDstSlices.push_back(txDstSlice);
        }
        SendRecvInfo sendRecvInfo{{linkRemote, linkRemote}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
        CHK_PRT_RET(
            SendRecvWriteForNda(sendRecvInfo, threads[0]),
            HCCL_ERROR("[InsTempReduceScatterOmniPipeMesh1dNDA] RunReduceScatter Send failed"),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempReduceScatterOmniPipeMesh1dNDA", InsTempReduceScatterOmniPipeMesh1dNDA);

} // namespace ops_hccl
