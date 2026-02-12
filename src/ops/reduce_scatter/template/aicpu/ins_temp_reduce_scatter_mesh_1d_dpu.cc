/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_1d_dpu.h"

namespace ops_hccl {
InsTempReduceScatterMesh1dDpu::InsTempReduceScatterMesh1dDpu()
{
}
// ! 已编码完成
InsTempReduceScatterMesh1dDpu::InsTempReduceScatterMesh1dDpu(const OpParam& param,
                                                        const u32 rankId, // 传通信域的rankId，userRank
                                                        const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

// ! 已编码完成
InsTempReduceScatterMesh1dDpu::~InsTempReduceScatterMesh1dDpu()
{
}

// ! 已编码完成
HcclResult InsTempReduceScatterMesh1dDpu::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                                               AlgResourceRequest& resourceRequest)
{
    // host网卡资源，不新增从流和对应Notify，只申请DPU上面
    resourceRequest.slaveThreadNum = 0;  // 主thread可以通过接口传入的stream来做转换
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO("[InsTempReduceScatterMeshSeqInter][CalcRes]slaveThreadNum[%u] notifyNumPerThread[%u] notifyNumOnMainThread[%u]"
        " level1Channels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread, resourceRequest.notifyNumOnMainThread,
        level1Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1dDpu::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void) inBufferType;
    (void) outBufferType;
    u64 scratchMultiple = subCommRanks_[0].size();
    HCCL_INFO(
        "[InsTempReduceScatterMesh1dDpu][CalcScratchMultiple] templateScratchMultiplier[%llu]", scratchMultiple);
    return scratchMultiple;
}

HcclResult InsTempReduceScatterMesh1dDpu::KernelRun(const OpParam& param,
                                                    const TemplateDataParams& tempAlgParams,
                                                    const TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;

    if (threadNum_ < 1) {
        HCCL_ERROR("[InsTempReduceScatterMesh1dDpu] Rank [%d], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
        HCCL_ERROR("HcommThreadSynchronize failed");
        return HCCL_E_INTERNAL;
    }

    DPURunInfo dpuRunInfo;
    dpuRunInfo.templateName = "InsTempReduceScatterMesh1dDpu";
    dpuRunInfo.tempAlgParams = tempAlgParams;
    dpuRunInfo.channels = templateResource.channels;
    dpuRunInfo.myRank = myRank_;
    dpuRunInfo.subCommRanks = subCommRanks_;
    u32 sendMsgId = 0;
    auto dpuRunInfoSeqData = dpuRunInfo.Serialize();

    if (HcommSendRequest(reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr), param.algTag,
        static_cast<void*>(dpuRunInfoSeqData.data()), dpuRunInfoSeqData.size(), &sendMsgId) != 0) {
        HCCL_ERROR("HcommSendRequest failed");
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("HcommSendRequest run over, sendMsgId[%u]", sendMsgId);
    // 等待DPU数据传输，然后回写结果回来
    void *recvData = nullptr;
    u32 recvMsgId = 0;

    if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("HcommWaitResponse failed");
        return HCCL_E_INTERNAL;
    }

    // 将执行模式转换回到batch
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

    if (recvMsgId != sendMsgId) {
        HCCL_ERROR("recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(PostLocalReduce(tempAlgParams, templateResource.threads));
    HCCL_INFO("[InsTempReduceScatterMesh1dDpu] Run End");
    // }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1dDpu::DPUKernelRun(const TemplateDataParams& tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank, const std::vector<std::vector<uint32_t>>& subCommRanks)
{
#ifndef AICPU_COMPILE
    u32 myAlgRank = 0;
    std::vector<u32> rankIds = subCommRanks[0];
    auto iter = std::find(rankIds.begin(), rankIds.end(), myRank);
    if (iter != rankIds.end()) {
        myAlgRank = std::distance(rankIds.begin(), iter);
    } else {
        HCCL_ERROR("[InsTempReduceScatterMesh1D][RunReduceScatter] rankIds or myRank is error.");
        return HCCL_E_INTERNAL;
    }

    for (u32 rankIdx = 0; rankIdx < rankIds.size(); rankIdx++) {
        u32 remoteRank = rankIds[rankIdx];
        if (remoteRank == myRank) {
            continue;
        }
        HCCL_DEBUG("[InsTempReduceScatterMesh1D][RunReduceScatter] myRank[%d], toRank[%d], fromRank[%d]",
                   myRank, remoteRank, myRank);
        const ChannelInfo &linkSend = channels.at(remoteRank)[0];
        const ChannelInfo &linkRecv = channels.at(remoteRank)[0];
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        // 在 HcclBuffer 上进行 ReduceScatter 操作
        // 由于进程只能访问远端的HcclBuffer，所以只能通过write的方式将自己userIn上的数据写到远端HcclBuffer上
        for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
            // 在reduce_scatter_op.cc的创建channels的环节中获取到了remote的HcclBuff的地址
            void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;
            // 在接收的时候接收源应该是远端地址，但是由于rs的mesh算法用的是write，所以rx不用care
            DataSlice rxSrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff +
                repeatIdx * tempAlgParams.inputRepeatStride + myAlgRank * tempAlgParams.inputSliceStride,
                tempAlgParams.sliceSize, tempAlgParams.count); // 接收源
            DataSlice rxDstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff +  repeatIdx * tempAlgParams.outputRepeatStride +
                rankIdx * tempAlgParams.outputSliceStride, tempAlgParams.sliceSize, tempAlgParams.count); // 接收目标

            DataSlice txSrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, 
                tempAlgParams.buffInfo.inBuffBaseOff + rankIdx * tempAlgParams.inputSliceStride + repeatIdx * tempAlgParams.inputRepeatStride,
                tempAlgParams.sliceSize, tempAlgParams.count); // 发送源
            DataSlice txDstSlice = DataSlice(remoteCclBuffAddr, 
                tempAlgParams.buffInfo.hcclBuffBaseOff + myAlgRank * tempAlgParams.outputSliceStride + repeatIdx * tempAlgParams.outputRepeatStride,
                tempAlgParams.sliceSize, tempAlgParams.count);  // 发送目标

            rxSrcSlices.push_back(rxSrcSlice);
            rxDstSlices.push_back(rxDstSlice);
            txSrcSlices.push_back(txSrcSlice);
            txDstSlices.push_back(txDstSlice);
            HCCL_WARNING("Send from src rank %u to dst rank %u, src offset %u, dst offset %u, size %u",
                myRank, remoteRank,
                tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride + rankIdx * tempAlgParams.inputSliceStride,
                tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride + myAlgRank * tempAlgParams.outputSliceStride,
                tempAlgParams.sliceSize);
            HCCL_INFO("Send from buf %p, to buf %p", tempAlgParams.buffInfo.inputPtr, remoteCclBuffAddr);
        }
        SendRecvInfo sendRecvInfo{{linkSend, linkRecv},
                             {{txSrcSlices, txDstSlices},{rxSrcSlices, rxDstSlices}}};

        CHK_PRT_RET(SendRecvWrite(sendRecvInfo),
                    HCCL_ERROR("[InsTempReduceScatterMesh1dDpu] RunReduceScatter Send failed"),
                    HcclResult::HCCL_E_INTERNAL);
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1dDpu::PostLocalReduce(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    // 通信结束之后，数据都在 cclBuffer 上，需要搬运到对应的输出位置。
    u32 myAlgRank = 0;
    std::vector<u32> rankIds = subCommRanks_[0];
    auto iter = std::find(rankIds.begin(), rankIds.end(), myRank_);
    if (iter != rankIds.end()) {
        myAlgRank = std::distance(rankIds.begin(), iter);
    } else {
        HCCL_ERROR("[InsTempReduceScatterMesh1D][PostLocalReduce] rankIds or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }
    // 如果是单算子模式, 并且是最后一步算子，需要将数据从 cclBuffer 拷贝到 userOut
    HCCL_INFO("[InsTempReduceScatterMesh1dDpu][PostCopy], copy from cclBuffer to userOut");
    // 先把本卡的数据从userIn搬运到userOut，然后再在userOut上做规约
    HCCL_INFO("[InsTempReduceScatterMesh1dDpu][PostCopy]tempAlgParams.repeatNum=%llu", tempAlgParams.repeatNum);
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
        u64 dataTypeSize =  SIZE_TABLE[dataType_];
        HCCL_INFO("Data sssize here is %u", dataTypeSize);
        u64 count = processSize_ / dataTypeSize;
        // 将本端数据从inputPtr搬运到cclBuffer上面
        DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
                            myAlgRank * tempAlgParams.inputSliceStride
                            + repeatIdx * tempAlgParams.inputRepeatStride,
                            processSize_, count_);
        DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                            myAlgRank * tempAlgParams.outputSliceStride
                            + repeatIdx * tempAlgParams.outputRepeatStride,
                            processSize_, count_);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
        HCCL_INFO("[PostLocalReduce]Local copy rank %u data from offset %u to offset %u size %u",
            myRank_,
            myAlgRank * tempAlgParams.inputSliceStride + repeatIdx * tempAlgParams.inputRepeatStride,
            myAlgRank * tempAlgParams.outputSliceStride + repeatIdx * tempAlgParams.outputRepeatStride,
            processSize_);
        HCCL_INFO("[PostLocalReduce]Local copy src addr %p, dst addr %p",
            tempAlgParams.buffInfo.inputPtr,
            tempAlgParams.buffInfo.hcclBuff.addr);
        // 将后n-1片数据，规约到第0片数据上
        for (u32 tmpRank = 1; tmpRank < templateRankSize_; tmpRank++) {
            DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                                            tempAlgParams.buffInfo.inBuffBaseOff
                                            + repeatIdx * tempAlgParams.outputRepeatStride
                                            + tmpRank * tempAlgParams.outputSliceStride,
                                            processSize_, count_);
            DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                                            tempAlgParams.buffInfo.hcclBuffBaseOff
                                            + repeatIdx * tempAlgParams.outputRepeatStride,
                                            processSize_, count_);
            HCCL_INFO("[PostLocalReduce]Local reduce on rank %u src offset %u, dst offset %u, size %u",
                myRank_,
                tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride + tmpRank * tempAlgParams.inputSliceStride,
                tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride,
                processSize_);
            HCCL_INFO("[PostLocalReduce]Local reduce src addr %p, dst addr %p",
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuff.addr);
            CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], srcSlice, dstSlice, dataType_, reduceOp_)));
        }
        // 将规约后的分片，搬运到output上
        srcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                            tempAlgParams.buffInfo.hcclBuffBaseOff
                            + repeatIdx * tempAlgParams.outputRepeatStride,
                            processSize_, count_);
        dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
                            tempAlgParams.buffInfo.outBuffBaseOff
                            + repeatIdx * tempAlgParams.outputRepeatStride,
                            processSize_, count_);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
        HCCL_INFO("[PostLocalReduce]Local copy to dst on rank %u src offset %u, dst offset %u, size %u",
                myRank_,
                tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride,
                tempAlgParams.buffInfo.outBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride,
                processSize_);
        HCCL_INFO("[PostLocalReduce]Local copy to dst src addr %p dst addr %p",
            tempAlgParams.buffInfo.hcclBuff.addr,
            tempAlgParams.buffInfo.outputPtr);
    }
    return HcclResult::HCCL_SUCCESS;
}
#ifndef AICPU_COMPILE
REGISTER_TEMPLATE_V2("InsTempReduceScatterMesh1dDpu", InsTempReduceScatterMesh1dDpu);
#endif
} // namespace Hccl