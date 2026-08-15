/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_omnipipe_mesh_1d_dpu.h"

namespace ops_hccl {
InsTempReduceScatterOmniPipeMesh1dDpu::InsTempReduceScatterOmniPipeMesh1dDpu() {}

InsTempReduceScatterOmniPipeMesh1dDpu::InsTempReduceScatterOmniPipeMesh1dDpu(
    const OpParam& param,
    const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempReduceScatterOmniPipeMesh1dDpu::~InsTempReduceScatterOmniPipeMesh1dDpu() {}

HcclResult InsTempReduceScatterOmniPipeMesh1dDpu::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // host网卡资源，不新增从流和对应Notify，只申请DPU上面
    resourceRequest.slaveThreadNum = 0; // 主thread可以通过接口传入的stream来做转换
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    HCCL_INFO(
        "[InsTempReduceScatterOmniPipeMesh1dDpu][CalcRes]slaveThreadNum[%u] notifyNumPerThread[%u] "
        "notifyNumOnMainThread[%u]"
        " level0Channels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread, resourceRequest.notifyNumOnMainThread,
        level0Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterOmniPipeMesh1dDpu::GetThreadNum() const { return 1; }

HcclResult InsTempReduceScatterOmniPipeMesh1dDpu::GetRes(AlgResourceRequest& resourceRequest) const
{
    // host网卡资源，不新增从流和对应notify，只申请DPU上面
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

// 语义改为返回当前template的类型，mesh返回1，nhr返回0
u64 InsTempReduceScatterOmniPipeMesh1dDpu::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    return 1;
}

void InsTempReduceScatterOmniPipeMesh1dDpu::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) { return; }

void InsTempReduceScatterOmniPipeMesh1dDpu::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) { return; }

HcclResult InsTempReduceScatterOmniPipeMesh1dDpu::DoLocalCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads)
{
    HCCL_INFO("[InsTempReduceScatterOmniPipeMesh1dDpu][DoLocalCopy] DoLocalCopy myRank_ = [%u]", myRank_);
    if (tempAlgParams.sliceSize == 0) {
        HCCL_INFO("Rank [%d], get slicesize zero. skip localcopy", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter == subCommRanks_[0].end()) {
        HCCL_ERROR("[%s]subCommRanks_ or myRank_ is error.", __func__);
        return HCCL_E_INTERNAL;
    }

    // 区分前后搬运
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
    // 这里的循环precopy是ranksize-1，postcopy是1
    HCCL_INFO("MT tempAlgParams.sliceSize = %u", tempAlgParams.sliceSize);
    for (auto i = 0; i < tempAlgParams.repeatNum; ++i) {
        auto srcSlice = DataSlice(
            srcAddr, tempAlgParams.buffInfo.inBuffBaseOff + i * tempAlgParams.inputSliceStride, tempAlgParams.sliceSize,
            tempAlgParams.count);
        auto dstSlice = DataSlice(
            dstAddr, tempAlgParams.buffInfo.outBuffBaseOff + i * tempAlgParams.outputSliceStride,
            tempAlgParams.sliceSize, tempAlgParams.count);
        HCCL_INFO(
            "myRank[%u], i[%u], srcSlice:%s, dstSlice:%s", myRank_, i, srcSlice.Describe().c_str(),
            dstSlice.Describe().c_str());
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOmniPipeMesh1dDpu::KernelRun(
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
        HCCL_ERROR("[InsTempReduceScatterOmniPipeMesh1dDpu] Rank [%d], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempReduceScatterOmniPipeMesh1dDpu] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
        HCCL_ERROR("[InsTempReduceScatterOmniPipeMesh1dDpu] HcommThreadSynchronize failed");
        return HCCL_E_INTERNAL;
    }

    DPURunInfo dpuRunInfo;
    dpuRunInfo.templateName = "InsTempReduceScatterOmniPipeMesh1dDpu";
    dpuRunInfo.tempAlgParams = tempAlgParams;
    dpuRunInfo.channels = templateResource.channels;
    dpuRunInfo.myRank = myRank_;
    dpuRunInfo.subCommRanks = subCommRanks_;
    u32 sendMsgId = 0;

    auto dpuRunInfoSeqData = dpuRunInfo.Serialize();
    if (HcommSendRequest(
            reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr), param.algTag,
            static_cast<void*>(dpuRunInfoSeqData.data()), dpuRunInfoSeqData.size(), &sendMsgId)
        != 0) {
        HCCL_ERROR("[InsTempReduceScatterOmniPipeMesh1dDpu] HcommSendRequest failed");
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[InsTempReduceScatterOmniPipeMesh1dDpu] HcommSendRequest run over, sendMsgId[%u]", sendMsgId);
    // 等待DPU数据传输，然后回写结果回来
    void* recvData = nullptr;
    u32 recvMsgId = 0;

    if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("[InsTempReduceScatterOmniPipeMesh1dDpu] HcommWaitResponse failed");
        return HCCL_E_INTERNAL;
    }

    // 将执行模式转换回到batch
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempReduceScatterOmniPipeMesh1dDpu] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempReduceScatterOmniPipeMesh1dDpu] HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

    if (recvMsgId != sendMsgId) {
        HCCL_ERROR(
            "[InsTempReduceScatterOmniPipeMesh1dDpu] recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
        return HCCL_E_INTERNAL;
    }

    // 这个PostReduce处理的是当前轴的规约任务
    PostReduce(tempAlgParams, templateResource.threads);
    HCCL_INFO("[%s]Run End", __func__);
    return HcclResult::HCCL_SUCCESS;
}

// 通信结束之后，数据都在 cclBuffer 上，需要搬运到对应的输出位置（斜对角算法仍然是规约到ccl上某片位置）
HcclResult InsTempReduceScatterOmniPipeMesh1dDpu::PostReduce(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads)
{
    u32 rankIdx = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        rankIdx = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempReduceScatterOmniPipeMesh1dDpu][RunReduceScatter] subCommRanks_ or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[InsTempReduceScatterOmniPipeMesh1dDpu][PostReduce], copy from cclBuffer to cclBuffer");
    // 本卡的数据在executor中，loop刚开始时就完成了localcopy
    // 地址指针全用ccl的
    void* cclBuffAddr = tempAlgParams.buffInfo.hcclBuff.addr;
    // 第一片数据往out做reduce
    HCCL_INFO("[InsTempReduceScatterOmniPipeMesh1dDpu][PostReduce]do first slice reduce.");
    // 从param中ccl的地址往param中out的地址规约
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
                HCCL_DEBUG("MT srcSlice=[%s], dstSlice=[%s]", srcSlice.Describe().c_str(), dstSlice.Describe().c_str());
                CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], srcSlice, dstSlice, dataType_, reduceOp_)));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

// 进行数据从ccl搬运到ccl。不reduce，只涉及receive/write
HcclResult InsTempReduceScatterOmniPipeMesh1dDpu::DPUKernelRun(
    const TemplateDataParams& tempAlgParam, const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>>& subCommRanks)
{
    templateRankSize_ = subCommRanks[0].size();
    subCommRanks_ = subCommRanks;
#ifndef AICPU_COMPILE
    HCCL_INFO("MT start to RunReduceScatter, channels.size()=%u", channels.size());
    u32 myAlgRank = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank);
    if (iter != subCommRanks_[0].end()) {
        myAlgRank = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempReduceScatterOmniPipeMesh1dDpu][RunReduceScatter] subCommRanks_ or myRank is error.");
        return HCCL_E_INTERNAL;
    }
    std::vector<u32> rankIds = subCommRanks[0];
    for (u32 rankIdx = 0; rankIdx < rankIds.size(); rankIdx++) {
        u32 remoteRank = rankIds[rankIdx];
        if (remoteRank == myRank) {
            continue;
        }
        const ChannelInfo& linkRemote = channels.at(remoteRank)[0];
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        void* localCclBuffAddr = tempAlgParam.buffInfo.hcclBuff.addr;
        void* remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
        // 按照偏移数组边计算位置
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
                tempAlgParam.stepSliceInfo.stepCount[rankIdx][repeatIdx]); // 发送源
            DataSlice txDstSlice = DataSlice(
                remoteCclBuffAddr, txDstCurrent, tempAlgParam.stepSliceInfo.stepSliceSize[rankIdx][repeatIdx],
                tempAlgParam.stepSliceInfo.stepCount[rankIdx][repeatIdx]); // 发送目标
            DataSlice rxSrcSlice = DataSlice(
                remoteCclBuffAddr, rxSrcCurrent, tempAlgParam.stepSliceInfo.stepSliceSize[rankIdx][repeatIdx],
                tempAlgParam.stepSliceInfo.stepCount[rankIdx][repeatIdx]); // 接收源
            DataSlice rxDstSlice = DataSlice(
                localCclBuffAddr, rxDstCurrent, tempAlgParam.stepSliceInfo.stepSliceSize[rankIdx][repeatIdx],
                tempAlgParam.stepSliceInfo.stepCount[rankIdx][repeatIdx]); // 接收目标

            rxSrcSlices.push_back(rxSrcSlice);
            rxDstSlices.push_back(rxDstSlice);
            txSrcSlices.push_back(txSrcSlice);
            txDstSlices.push_back(txDstSlice);
        }
        SendRecvInfo sendRecvInfo{{linkRemote, linkRemote}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
        CHK_PRT_RET(
            SendRecvWrite(sendRecvInfo),
            HCCL_ERROR("[InsTempReduceScatterOmniPipeMesh1dDpu] RunReduceScatter Send failed"),
            HcclResult::HCCL_E_INTERNAL);
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempReduceScatterOmniPipeMesh1dDpu", InsTempReduceScatterOmniPipeMesh1dDpu);

} // namespace ops_hccl
