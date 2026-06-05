/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_1D_dpu_inter.h"

namespace ops_hccl {
InsTempReduceScatterMesh1dDpuInter::InsTempReduceScatterMesh1dDpuInter()
{
}

InsTempReduceScatterMesh1dDpuInter::InsTempReduceScatterMesh1dDpuInter(const OpParam& param,
                                                        const u32 rankId, // 传通信域的rankId，userRank
                                                        const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempReduceScatterMesh1dDpuInter::~InsTempReduceScatterMesh1dDpuInter()
{
}

HcclResult InsTempReduceScatterMesh1dDpuInter::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                               AlgResourceRequest& resourceRequest)
{
    // host网卡资源，不新增从流和对应Notify，只申请DPU上面
    resourceRequest.slaveThreadNum = 0;  // 主thread可以通过接口传入的stream来做转换
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO("[InsTempReduceScatterMesh1dDpuInter][CalcRes]slaveThreadNum[%u] notifyNumPerThread[%u] notifyNumOnMainThread[%u]"
        " level1Channels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread, resourceRequest.notifyNumOnMainThread,
        level1Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1dDpuInter::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void) inBufferType;
    (void) outBufferType;
    u64 scratchMultiple = subCommRanks_[0].size();
    HCCL_INFO(
        "[InsTempReduceScatterMesh1dDpuInter][CalcScratchMultiple] templateScratchMultiplier[%llu]", scratchMultiple);
    return scratchMultiple;
}

HcclResult InsTempReduceScatterMesh1dDpuInter::KernelRun(const OpParam& param,
                                                    const TemplateDataParams& tempAlgParams,
                                                    TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;

    if (threadNum_ < 1) {
        HCCL_ERROR("[InsTempReduceScatterMesh1dDpuInter] Rank [%d], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempReduceScatterMesh1dDpuInter] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
        HCCL_ERROR("[InsTempReduceScatterMesh1dDpuInter] HcommThreadSynchronize failed");
        return HCCL_E_INTERNAL;
    }

    DPURunInfo dpuRunInfo;
    dpuRunInfo.templateName = "InsTempReduceScatterMesh1dDpuInter";
    dpuRunInfo.tempAlgParams = tempAlgParams;
    dpuRunInfo.channels = templateResource.channels;
    dpuRunInfo.myRank = myRank_;
    dpuRunInfo.subCommRanks = subCommRanks_;
    u32 sendMsgId = 0;
    auto dpuRunInfoSeqData = dpuRunInfo.Serialize();
    if (HcommSendRequest(reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr), param.algTag,
        static_cast<void*>(dpuRunInfoSeqData.data()), dpuRunInfoSeqData.size(), &sendMsgId) != 0) {
        HCCL_ERROR("[InsTempReduceScatterMesh1dDpuInter] HcommSendRequest failed");
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[InsTempReduceScatterMesh1dDpuInter] HcommSendRequest run over, sendMsgId[%u]", sendMsgId);
    // 等待DPU数据传输，然后回写结果回来
    void *recvData = nullptr;
    u32 recvMsgId = 0;

    if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("[InsTempReduceScatterMesh1dDpuInter] HcommWaitResponse failed");
        return HCCL_E_INTERNAL;
    }

    // 将执行模式转换回到batch
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempReduceScatterMesh1dDpuInter] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempReduceScatterMesh1dDpuInter] HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

    if (recvMsgId != sendMsgId) {
        HCCL_ERROR("[InsTempReduceScatterMesh1dDpuInter] recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(PostLocalReduce(param, tempAlgParams, templateResource.threads));
    HCCL_INFO("[InsTempReduceScatterMesh1dDpuInter] Run End");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1dDpuInter::DPUKernelRun(const TemplateDataParams& tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank, const std::vector<std::vector<uint32_t>>& subCommRanks)
{
#ifndef AICPU_COMPILE
    u32 myAlgRank = 0;
    std::vector<u32> rankIds = subCommRanks[0];
    auto iter = std::find(rankIds.begin(), rankIds.end(), myRank);
    if (iter != rankIds.end()) {
        myAlgRank = std::distance(rankIds.begin(), iter);
    } else {
        HCCL_ERROR("[InsTempReduceScatterMesh1dDpuInter][RunReduceScatter] rankIds or myRank is error.");
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG("[InsTempReduceScatterMesh1dDpuInter][sliceNum]: [%u] ",
            tempAlgParams.allRankSliceSize.size());
    u64 recvSize = tempAlgParams.allRankSliceSize.at(myAlgRank);
    u64 recvCount = tempAlgParams.allRankProcessedDataCount.at(myAlgRank);
    u64 recvOffset = tempAlgParams.allRankDispls.at(myAlgRank);

    for (u32 rankIdx = 0; rankIdx < rankIds.size(); rankIdx++) {
        u32 remoteRank = rankIds[rankIdx];
        if (remoteRank == myRank) {
            continue;
        }
        u64 sendSize = tempAlgParams.allRankSliceSize.at(rankIdx);//向对端发送的参数
        u64 sendCount = tempAlgParams.allRankProcessedDataCount.at(rankIdx);
        u64 sendOffset = tempAlgParams.allRankDispls.at(rankIdx);
        // 只考虑发送数据为0，因为SendRecvWrite实际上只用到了本端发送对端，没有从对端接收数据到本端
        if (sendSize == 0 && recvSize ==0) {
            continue;
        }
        HCCL_DEBUG("[InsTempReduceScatterMesh1dDpuInter][DPUKernelRun] myRank[%d], toRank[%d], fromRank[%d]",
                   myRank, remoteRank, remoteRank);
        const ChannelInfo &linkSend = channels.at(remoteRank)[0];
        const ChannelInfo &linkRecv = channels.at(remoteRank)[0];

        // 在 HcclBuffer 上进行 ReduceScatter 操作
        // 由于进程只能访问远端的HcclBuffer，所以只能通过write的方式将自己userIn上的数据写到远端HcclBuffer上
        for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
            // 在reduce_scatter_op.cc的创建channels的环节中获取到了remote的HcclBuff的地址
            void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;
            // 在接收的时候接收源应该是远端地址，但是由于rs的mesh算法用的是write，所以rx不用care
            DataSlice rxSrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
                                             tempAlgParams.buffInfo.inBuffBaseOff +
                                             repeatIdx * tempAlgParams.inputRepeatStride +
                                             recvOffset,
                                             recvSize,
                                             recvCount); // 接收源
            DataSlice rxDstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                                             tempAlgParams.buffInfo.hcclBuffBaseOff +
                                             repeatIdx * tempAlgParams.outputRepeatStride +
                                             rankIdx * recvSize,
                                             recvSize,
                                             recvCount); // 接收目标
            DataSlice txSrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
                                             tempAlgParams.buffInfo.inBuffBaseOff +
                                             repeatIdx * tempAlgParams.inputRepeatStride +
                                             sendOffset,
                                             sendSize,
                                             sendCount); // 发送源
            DataSlice txDstSlice = DataSlice(remoteCclBuffAddr,
                                             tempAlgParams.buffInfo.hcclBuffBaseOff +
                                             repeatIdx * tempAlgParams.outputRepeatStride +
                                             myAlgRank * sendSize,
                                             sendSize,
                                             sendCount);  // 发送目标
            std::vector<DataSlice> txSrcSlices{txSrcSlice};
            std::vector<DataSlice> txDstSlices{txDstSlice};
            std::vector<DataSlice> rxSrcSlices{rxSrcSlice};
            std::vector<DataSlice> rxDstSlices{rxDstSlice};

            if (sendSize > 0 && recvSize >0) {
                TxRxChannels sendRecvChannels(linkSend, linkRecv);
                TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
                SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
                CHK_PRT_RET(SendRecvWrite(sendRecvInfo),
                    HCCL_ERROR("[InsTempReduceScatterMesh1dDpuInter] SendRecvWrite failed."),
                    HcclResult::HCCL_E_INTERNAL);
            } else if (sendSize > 0) {
                SlicesList sendSliceList(txSrcSlices, txDstSlices);
                DataInfo sendInfo(linkSend, sendSliceList);
                CHK_PRT_RET(SendWrite(sendInfo),
                    HCCL_ERROR("[InsTempReduceScatterMesh1dDpuInter][DPUKernelRun] Send failed."),
                    HcclResult::HCCL_E_INTERNAL);
            } else if (recvSize > 0) {
                SlicesList recvSliceList(rxSrcSlices, rxDstSlices);
                DataInfo recvInfo(linkRecv, recvSliceList);
                CHK_PRT_RET(RecvWrite(recvInfo),
                    HCCL_ERROR("[InsTempReduceScatterMesh1dDpuInter][DPUKernelRun] Recv failed."),
                    HcclResult::HCCL_E_INTERNAL);
            }
        }
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1dDpuInter::PostLocalReduce(const OpParam &param, const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    // 通信结束之后，数据都在 cclBuffer 上，需要搬运到对应的输出位置。
    u32 myAlgRank = 0;
    std::vector<u32> rankIds = subCommRanks_[0];
    auto iter = std::find(rankIds.begin(), rankIds.end(), myRank_);
    if (iter != rankIds.end()) {
        myAlgRank = std::distance(rankIds.begin(), iter);
    } else {
        HCCL_ERROR("[InsTempReduceScatterMesh1dDpuInter][RunReduceScatter] rankIds or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempReduceScatterMesh1dDpuInter][PostCopy], copy from scratch to output");
    // 先把本卡的数据从userIn搬运到userOut，然后再在userOut上做规约
    HCCL_INFO("[InsTempReduceScatterMesh1dDpuInter][PostCopy]tempAlgParams.repeatNum=%llu", tempAlgParams.repeatNum);
    u64 sliceSize = tempAlgParams.allRankSliceSize.at(myAlgRank);
    u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(myAlgRank);
    u64 sliceOffset = tempAlgParams.allRankDispls.at(myAlgRank);
    // 数据量为0的数据片无需Reduce
    if (sliceSize == 0) {
        HCCL_INFO("[InsTempReduceScatterMesh1dDpuInter][PostCopy] Rank %u has no data to process, pass", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
        // 将规约后的分片，搬运到output上
        DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
            tempAlgParams.buffInfo.inBuffBaseOff +
            repeatIdx * tempAlgParams.inputRepeatStride +
            sliceOffset,
            sliceSize,
            sliceCount);
        DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
            tempAlgParams.buffInfo.hcclBuffBaseOff +
            repeatIdx * tempAlgParams.outputRepeatStride +
            sliceSize * myAlgRank,
            sliceSize,
            sliceCount);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));

        // 增加thread synchronize以支持64类数据类型
        if ((dataType_ == HCCL_DATA_TYPE_INT64) || (dataType_ == HCCL_DATA_TYPE_UINT64) ||
            (dataType_ == HCCL_DATA_TYPE_FP64) || (reduceOp_ == HcclReduceOp::HCCL_REDUCE_PROD)) {
            // 启动任务并等待所有threads任务执行完成
            CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
            CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
            for (const auto &thread : threads) {
                CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
            }
        }

        // 将后n-1片数据，规约到第0片数据上
        for (u32 tmpRank = 1; tmpRank < templateRankSize_; tmpRank++) {
            DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                                           tempAlgParams.buffInfo.hcclBuffBaseOff
                                           + repeatIdx * tempAlgParams.outputRepeatStride
                                           + tmpRank * sliceSize,
                                           sliceSize,
                                           sliceCount);
            DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                                           tempAlgParams.buffInfo.hcclBuffBaseOff
                                           + repeatIdx * tempAlgParams.outputRepeatStride,
                                           sliceSize,
                                           sliceCount);
            CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], srcSlice, dstSlice, dataType_, reduceOp_)));
        }
        // 将规约后的分片，搬运到output上
        srcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                            tempAlgParams.buffInfo.hcclBuffBaseOff
                            + repeatIdx * tempAlgParams.outputRepeatStride,
                            sliceSize, sliceCount);
        dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
                            tempAlgParams.buffInfo.outBuffBaseOff
                            + repeatIdx * tempAlgParams.outputRepeatStride,
                            sliceSize, sliceCount);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    }
    return HcclResult::HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
REGISTER_TEMPLATE_V2("InsTempReduceScatterMesh1dDpuInter", InsTempReduceScatterMesh1dDpuInter);
#endif
} // namespace Hccl