/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_aicpu_reduce_nhr_pcie.h"

namespace ops_hccl {
InsTempReduceScatterAicpuReduceNHRPcie::InsTempReduceScatterAicpuReduceNHRPcie(
    const OpParam& param, const u32 rankId, // 传通信域的u32，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempReduceScatterAicpuReduceNHRPcie::~InsTempReduceScatterAicpuReduceNHRPcie() {}

HcclResult InsTempReduceScatterAicpuReduceNHRPcie::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // NHR 需要的 que Num 为 1
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, channels));
    resourceRequest.channels.push_back(channels);
    HCCL_INFO(
        "[InsTempReduceScatterAicpuReduceNHRPcie][CalcRes] slaveThreadNum: [%u], notifyNumOnMainThread: [%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceNHRPcie::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterAicpuReduceNHRPcie::GetThreadNum() const { return 1; }

HcclResult InsTempReduceScatterAicpuReduceNHRPcie::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempReduceScatterAicpuReduceNHRPcie] KernelRun start");

    tempAlgParams_ = tempAlgParams;
    channels_ = templateResource.channels;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    if ((tempAlgParams_.buffInfo.inBuffType == tempAlgParams_.buffInfo.hcclBuffType)
        && (tempAlgParams_.buffInfo.inBuffBaseOff == tempAlgParams_.buffInfo.hcclBuffBaseOff)) {
        // inBuff和hcclBuff地址偏移相同时，hcclBuff自行向后偏移一部分
        u64 maxSliceSize = std::max(tempAlgParams_.sliceSize, tempAlgParams_.tailSize);
        hcclBuffBaseOff_ = tempAlgParams_.buffInfo.hcclBuffBaseOff + templateRankSize_ * maxSliceSize;
    } else {
        hcclBuffBaseOff_ = tempAlgParams_.buffInfo.hcclBuffBaseOff;
    }
    bool isPcieProtocal = IsPcieProtocol(channels_); // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocal;                     // 是否使用Read模式
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));

    for (u32 sliceIdx = 0; sliceIdx < templateRankSize_; ++sliceIdx) {
        // 判断当前sliceIdx是否为尾块（最后一个rank且tailSize > 0）
        if (sliceIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize > 0) {
            currSliceSize_ = tempAlgParams_.tailSize;
        } else {
            currSliceSize_ = tempAlgParams_.sliceSize;
        }
        currSliceCount_ = currSliceSize_ / dataTypeSize_;

        // Step 1: 本地拷贝，将本rank的数据拷贝到自己hcclBuffer上相应的位置
        CHK_RET(LocalDataCopy(templateResource.threads, sliceIdx));

        // 当处理到本rank的数据时，先将其从input拷贝到output，为后续reduce初始化output
        if (sliceIdx == myAlgIdx) {
            CHK_RET(LocalCopyToOutput(templateResource.threads, sliceIdx));
        }

        // Step 2: 数据分发，每个rank通过write方式将对端rank需要的数据写到其hcclBuffer对应的槽位
        CHK_RET(RunAllGather(templateResource.threads));

        // 必须确保所有通信任务完成，因为接下来的 AICPU Reduce 运行在 CPU 上，不感知任务队列同步
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
        for (const auto& thread : templateResource.threads) {
            CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
        }

        // Step 3: 在output上做本地规约
        if (sliceIdx == myAlgIdx) {
            CHK_RET(PostLocalReduce(templateResource.threads));
        }
    }

    HCCL_INFO("[InsTempReduceScatterAicpuReduceNHRPcie] KernelRun end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceNHRPcie::LocalDataCopy(const std::vector<ThreadHandle>& threads, u32 sliceIdx)
{
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterAicpuReduceNHRPcie][LocalDataCopy] empty threads"),
        HcclResult::HCCL_E_INTERNAL);

    ThreadHandle q = threads[0];
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);

    // 获取本rank在算法中的索引
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));

    // 把属于自己的slice从input拷贝到hcclBuff的对应位置
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        // 计算input上的偏移：使用inputSliceStride计算本rank的数据偏移
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 inOff = inBaseOff + tempAlgParams_.inputSliceStride * sliceIdx;

        // 计算hcclBuffer上的偏移：使用currSliceSize_作为步长，避免尾块场景下slot重叠
        const u64 scratchBase = hcclBuffBaseOff_ + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scOff = scratchBase + currSliceSize_ * myAlgIdx;

        DataSlice src = DataSlice(tempAlgParams_.buffInfo.inputPtr, inOff, currSliceSize_, currSliceCount_);
        DataSlice dst = DataSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scOff, currSliceSize_, currSliceCount_);

        HCCL_INFO(
            "[InsTempReduceScatterAicpuReduceNHRPcie][LocalDataCopy] rpt[%u] inOff[%llu] scOff[%llu] sliceSize[%llu]",
            rpt, inOff, scOff, currSliceSize_);

        // 如果源地址和目标地址相同，则不需要做拷贝
        if (tempAlgParams_.buffInfo.inBuffType != tempAlgParams_.buffInfo.hcclBuffType || inOff != scOff) {
            CHK_RET(LocalCopy(q, src, dst));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempReduceScatterAicpuReduceNHRPcie::LocalCopyToOutput(const std::vector<ThreadHandle>& threads, u32 sliceIdx)
{
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterAicpuReduceNHRPcie][LocalCopyToOutput] empty threads"),
        HcclResult::HCCL_E_INTERNAL);
    ThreadHandle q = threads[0];
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 inOff = inBaseOff + tempAlgParams_.inputSliceStride * sliceIdx;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        // 尾块场景下使用tailSize作为数据大小
        const u64 currSize = (sliceIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize > 0) ?
                                 tempAlgParams_.tailSize :
                                 tempAlgParams_.sliceSize;
        const u64 currCount = currSize / dataTypeSize_;
        DataSlice src = DataSlice(tempAlgParams_.buffInfo.inputPtr, inOff, currSize, currCount);
        DataSlice dst = DataSlice(tempAlgParams_.buffInfo.outputPtr, outBaseOff, currSize, currCount);
        HCCL_INFO(
            "[InsTempReduceScatterAicpuReduceNHRPcie][LocalCopyToOutput] rpt[%u] inOff[%llu] outBaseOff[%llu] "
            "sliceSize[%llu]",
            rpt, inOff, outBaseOff, currSize);
        if (tempAlgParams_.buffInfo.inBuffType != tempAlgParams_.buffInfo.outBuffType || inOff != outBaseOff) {
            CHK_RET(LocalCopy(q, src, dst));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceNHRPcie::PostLocalReduce(const std::vector<ThreadHandle>& threads)
{
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterAicpuReduceNHRPcie][PostLocalReduce] empty threads"),
        HcclResult::HCCL_E_INTERNAL);

    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));
    ThreadHandle q = threads[0];

    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + myAlgIdx * tempAlgParams_.outputSliceStride
                               + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchBase = hcclBuffBaseOff_ + rpt * tempAlgParams_.outputRepeatStride;

        // 然后将其余rank的数据规约到output
        for (u32 rankIdx = 0; rankIdx < templateRankSize_; ++rankIdx) {
            if (rankIdx == myAlgIdx) {
                continue; // 跳过本rank，已经拷贝过了
            }

            const u64 otherScratchOff = scratchBase + currSliceSize_ * rankIdx;
            DataSlice reduceSrcSlice(
                tempAlgParams_.buffInfo.hcclBuff.addr, otherScratchOff, currSliceSize_, currSliceCount_);
            DataSlice reduceDstSlice(tempAlgParams_.buffInfo.outputPtr, outBaseOff, currSliceSize_, currSliceCount_);

            HCCL_INFO(
                "[InsTempReduceScatterAicpuReduceNHRPcie][PostLocalReduce] reduce from rank[%u] otherScratchOff[%llu]",
                rankIdx, otherScratchOff);

            // 使用AicpuReduce进行规约
            CHK_RET(LocalReduce(q, reduceSrcSlice, reduceDstSlice, dataType_, reduceOp_));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceNHRPcie::RunAllGather(const std::vector<ThreadHandle>& threads)
{
    const u32 nSteps = GetNHRStepNum(templateRankSize_); // NHR 通信步数， ceil(log2(rankSize))

    for (u32 step = 0; step < nSteps; ++step) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo)); // 计算当前step要通信的卡，数据

        const ChannelInfo& channelRecv = channels_.at(GetRankFromMap(stepInfo.fromRank))[0];
        const ChannelInfo& channelSend = channels_.at(GetRankFromMap(stepInfo.toRank))[0];
        // 构造SendRecv， 都是Scratch到Scratch的传输，没有DMA消减
        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        void* sendCclBuffAddr = channelSend.remoteCclMem.addr;
        void* recvCclBuffAddr = channelRecv.remoteCclMem.addr;

        for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
            const u64 scratchRepeatStride = currSliceSize_ * templateRankSize_;
            const u64 scratchBase = hcclBuffBaseOff_ + rpt * scratchRepeatStride;

            for (u32 i = 0; i < stepInfo.nSlices; ++i) {
                const u32 txIdx = stepInfo.txSliceIdxs[i];
                const u32 rxIdx = stepInfo.rxSliceIdxs[i];

                const u64 txScratchOff = scratchBase + currSliceSize_ * txIdx;

                const u64 rxScratchOff = scratchBase + currSliceSize_ * rxIdx;

                txSrcSlicesAll.emplace_back(
                    tempAlgParams_.buffInfo.hcclBuff.addr, txScratchOff, currSliceSize_, currSliceCount_);
                txDstSlicesAll.emplace_back(sendCclBuffAddr, txScratchOff, currSliceSize_, currSliceCount_);
                rxSrcSlicesAll.emplace_back(recvCclBuffAddr, rxScratchOff, currSliceSize_, currSliceCount_);
                rxDstSlicesAll.emplace_back(
                    tempAlgParams_.buffInfo.hcclBuff.addr, rxScratchOff, currSliceSize_, currSliceCount_);
            }
        }
        // write模式使用tx,rx地址不生效，仅使用对端link做Post/Wait
        // read 模式使用rx, tx地址不生效，仅使用对端link做Post/Wait
        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(channelSend, channelRecv);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        if (isDmaRead_) {
            CHK_PRT_RET(
                SendRecvRead(sendRecvInfo, threads[0]),
                HCCL_ERROR("[InsTempReduceScatterAicpuReduceNHRPcie] sendrecv failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                SendRecvWrite(sendRecvInfo, threads[0]),
                HCCL_ERROR("[InsTempReduceScatterAicpuReduceNHRPcie] sendrecv failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterAicpuReduceNHRPcie::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    HCCL_INFO(
        "[InsTempReduceScatterAicpuReduceNHRPcie][CalcScratchMultiple] templateScratchMultiplier[%llu]",
        templateRankSize_);
    return templateRankSize_;
}

u32 InsTempReduceScatterAicpuReduceNHRPcie::GetRankFromMap(const u32 algRankIdx)
{
    return subCommRanks_[0].at(algRankIdx);
}

//  计算每轮收发的对端以及slice编号
HcclResult InsTempReduceScatterAicpuReduceNHRPcie::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = myAlgRank;

    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (myAlgRank + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 sendTo = (myAlgRank + deltaRank) % templateRankSize_;

    // ReduceNHR 数据份数和数据编号增量， NHR是一个传输数据变化的
    u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = myAlgRank;
    u32 rxSliceIdx = (myAlgRank - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.fromRank = recvFrom;
    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG("[ReduceScatterAicpu][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempReduceScatterAicpuReduceNHRPcie::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub)
{
    (void)notifyIdxMianToSub;
}

void InsTempReduceScatterAicpuReduceNHRPcie::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    (void)notifyIdxSubToMain;
}
} // namespace ops_hccl
