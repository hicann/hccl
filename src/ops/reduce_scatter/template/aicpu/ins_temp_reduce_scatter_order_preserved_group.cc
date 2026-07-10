/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 包含本类的头文件声明
#include "ins_temp_reduce_scatter_order_preserved_group.h"
#include "alg_env_config.h"
#include "order_preserved_common.h"

namespace ops_hccl {

InsTempReduceScatterOrderPreservedGroup::InsTempReduceScatterOrderPreservedGroup(const OpParam &param,
    const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
    deterministicStrict_ = (GetExternalInputHcclDeterministic() == static_cast<u8>(DeterministicEnableLevel::DETERMINISTIC_STRICT));
}

InsTempReduceScatterOrderPreservedGroup::~InsTempReduceScatterOrderPreservedGroup()
{}

// 计算资源请求：确定执行所需的线程数、通知数和通道资源
// comm: HCCL通信句柄
// param: 操作参数
// topoInfo: 拓扑信息，包含网络层详细信息
// resourceRequest: 输出参数，填充资源请求信息
HcclResult InsTempReduceScatterOrderPreservedGroup::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    u32 threadNum = CalcEffectiveThreadNum(templateRankSize_);
    resourceRequest.slaveThreadNum = threadNum - 1;

    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    // 将通道请求添加到资源请求中
    resourceRequest.channels.push_back(level0Channels);

    HCCL_INFO("[InsTempReduceScatterOrderPreservedGroup][CalcRes] myRank[%u], templateRankSize[%u], "
        "ORDER_PRESERVED_MAX_THREADS[%u], threadNum[%u] (decoupled: rankSize-1 vs cap, took min), "
        "notifyNumOnMainThread[%u], slaveThreadNum[%u]",
        myRank_, templateRankSize_, ORDER_PRESERVED_MAX_THREADS, threadNum,
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterOrderPreservedGroup::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    u64 scratchMultiple = templateRankSize_;
    HCCL_INFO("[InsTempReduceScatterOrderPreservedGroup][CalcScratchMultiple] scratchMultiple[%u]", scratchMultiple);
    return scratchMultiple;
}

HcclResult InsTempReduceScatterOrderPreservedGroup::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        HCCL_DEBUG("[InsTempReduceScatterOrderPreservedGroup] myRank[%u] sliceSize and tailSize are 0, skip.", myRank_);
        return HCCL_SUCCESS;
    }

    // 初始化成员变量
    threadNum_ = templateResource.threads.size();
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.sliceSize / DATATYPE_SIZE_TABLE[dataType_];

    HCCL_INFO("[InsTempReduceScatterOrderPreservedGroup][KernelRun] Start, threadNum[%u], count[%llu], "
        "dataType[%u], deterministicStrict[%d]", threadNum_, count_, dataType_, deterministicStrict_);

    // 步骤1: 执行预处理本地拷贝（将本rank对应的数据从用户输入拷贝到临时缓冲区）
    CHK_RET(PreLocalCopy(tempAlgParams, templateResource.threads));

    // 多线程同步：如果线程数大于1，等待子线程就绪，为all2all做准备
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    // 步骤2: 执行AllToAll操作（按 distance 对称分配线程，一次性下发所有任务）
    CHK_RET(RunAllToAll(templateResource.channels, templateResource.threads, tempAlgParams));

    // 多线程同步：如果线程数大于1，需要在操作完成后同步，等待子线程完成
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    if (dataType_ == HCCL_DATA_TYPE_FP64 || reduceOp_ == HcclReduceOp::HCCL_REDUCE_PROD) {
        // 必须确保所有通信任务完成，因为接下来的 AICPU Reduce 运行在 CPU 上，不感知任务队列同步
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
        for (const auto &thread : templateResource.threads) {
            CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
        }
    }

    // 步骤3: 执行本地归约操作（将收到的所有数据在本地进行归约）
    CHK_RET(RunLocalReduce(templateResource.threads, tempAlgParams));

    // 步骤4: 执行后处理拷贝（将归约结果从临时缓冲区拷贝到用户输出缓冲区）
    CHK_RET(PostCopy(tempAlgParams, templateResource.threads));

    HCCL_INFO("[InsTempReduceScatterOrderPreservedGroup][KernelRun] End");
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedGroup::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterOrderPreservedGroup::GetThreadNum() const
{
    u32 threadNum = CalcEffectiveThreadNum(templateRankSize_);
    HCCL_INFO("[InsTempReduceScatterOrderPreservedGroup][GetThreadNum] templateRankSize[%u], "
        "effectiveThreadNum[%u] (capped at ORDER_PRESERVED_MAX_THREADS[%u])",
        templateRankSize_, threadNum, ORDER_PRESERVED_MAX_THREADS);
    return threadNum;
}

void InsTempReduceScatterOrderPreservedGroup::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempReduceScatterOrderPreservedGroup::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

u32 InsTempReduceScatterOrderPreservedGroup::CalcOutputIndex(const u32 round, const u32 localRank)
{
    // 使用取模运算确保索引在rank范围内
    // round为轮次，也为all2all结果的偏移量，localRank为本地rank
    return (round + localRank) % templateRankSize_;
}

bool InsTempReduceScatterOrderPreservedGroup::IsLastBlockData(const u32 outputIndex)
{
    return outputIndex == templateRankSize_ - 1;
}

bool InsTempReduceScatterOrderPreservedGroup::IsLastRank(const u32 rankId)
{
    return rankId == templateRankSize_ - 1;
}

// 本地拷贝：将本rank需要的数据从用户输入缓冲区拷贝到临时缓冲区
// tempAlgParams: 模板数据参数
// threads: 线程句柄列表
HcclResult InsTempReduceScatterOrderPreservedGroup::PreLocalCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    const MemBlockInfo &memBlockInfo = memBlockInfo_;
    // 获取本rank在算法中的编号
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    // 获取本rank对应的数据块大小
    u64 sliceSize = memBlockInfo.size[myAlgRank];
    if (sliceSize == 0) {
        HCCL_DEBUG("[PreLocalCopy] myAlgRank[%u] sliceSize is 0, skip.", myAlgRank);
        return HCCL_SUCCESS;
    }

    // 计算源地址偏移量（用户输入缓冲区中的偏移）
    u64 srcOffset = memBlockInfo.userInputOffsets[myAlgRank];
    // 计算输出索引（确定在临时缓冲区中的位置）
    u32 outputIndex = CalcOutputIndex(myAlgRank, myAlgRank);
    // 计算目标地址偏移量（临时缓冲区中的偏移）
    u64 dstOffset = memBlockInfo.outputOffsets[outputIndex];

    HCCL_INFO("[PreLocalCopy] myAlgRank[%u], sliceSize[%llu], srcOffset[%llu], dstOffset[%llu], outputIndex[%u]",
        myAlgRank, sliceSize, srcOffset, dstOffset, outputIndex);

    DataSlice srcSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, sliceSize);
    DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, dstOffset, sliceSize);
    CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));

    return HCCL_SUCCESS;
}

// 内部静态函数：发送单个远端分片
HcclResult InsTempReduceScatterOrderPreservedGroup::SendSingleRankSlice(u32 dstRank, u32 myAlgRank,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const TemplateDataParams &tempAlgParams, const ThreadHandle &thread)
{
    const MemBlockInfo &memBlockInfo = memBlockInfo_;
    u32 dstPhyRank = subCommRanks_[0][dstRank];
    auto chIter = channels.find(dstPhyRank);
    CHK_PRT_RET(chIter == channels.end(),
        HCCL_ERROR("[RunAllToAll] channel not found for dstRank[%u], dstPhyRank[%u]",
            dstRank, dstPhyRank), HCCL_E_INTERNAL);
    const std::vector<ChannelInfo> &chList = chIter->second;
    CHK_PRT_RET(chList.empty(),
        HCCL_ERROR("[RunAllToAll] chList empty for dstRank[%u]", dstRank), HCCL_E_INTERNAL);

    u64 sliceSize = memBlockInfo.size[dstRank];
    const ChannelInfo &ch = chList[0];
    u32 dstOutputSlot = CalcOutputIndex(dstRank, myAlgRank);
    u64 srcOffset = memBlockInfo.userInputOffsets[dstRank];
    u64 dstOffset = memBlockInfo.outputOffsets[dstOutputSlot];

    DataSlice srcSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, sliceSize);
    DataSlice dstSlice(ch.remoteCclMem.addr, dstOffset, sliceSize);
    SendRecvInfo sendInfo{
        TxRxChannels{ch, ch},
        TxRxSlicesList{SlicesList{{srcSlice}, {dstSlice}}, SlicesList{{}, {}}}
    };

    CHK_RET(SendRecvWrite(sendInfo, thread));
    HCCL_INFO("[RunAllToAll] dstRank[%u], sliceSize[%llu]", dstRank, sliceSize);
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedGroup::RunAllToAll(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    u32 slaveThreadNum = threadNum_ - 1;
    u32 totalRanks = templateRankSize_ - 1; // 需要通信的rank数（排除自己）

    HCCL_INFO("[RunAllToAll] Start, templateRankSize[%u], threadNum[%u], slaveThreadNum[%u], "
        "totalRanks[%u]", templateRankSize_, threadNum_, slaveThreadNum, totalRanks);

    u32 queIdx = 1;
    const u32 N = templateRankSize_;
    for (u32 distance = 1; distance <= N / 2; distance++) {
        u32 rightRank = (myAlgRank + distance) % N;
        u32 leftRank = (myAlgRank + N - distance) % N;

        CHK_RET(SendSingleRankSlice(rightRank, myAlgRank, channels, tempAlgParams, threads[queIdx]));
        queIdx++;
        if (queIdx >= threadNum_) {
            queIdx = 1;
        }

        if (leftRank != rightRank) {
            CHK_RET(SendSingleRankSlice(leftRank, myAlgRank, channels, tempAlgParams, threads[queIdx]));
            queIdx++;
            if (queIdx >= threadNum_) {
                queIdx = 1;
            }
        }
    }

    HCCL_INFO("[RunAllToAll] End");
    return HCCL_SUCCESS;
}

// 计算小于给定值的最大2次幂，用于树形Reduce算法中确定每一步的数据分割点
static u32 GetLargestPowerOf2LessThan(u32 value)
{
    if (value <= 1) {
        return 0;
    }
    u32 power = 1;
    while (power * 2 < value) {
        power *= 2;
    }
    return power;
}

// 执行本地归约操作：使用树形Reduce算法将临时缓冲区中所有数据块归约到本rank对应的位置
// 树形Reduce：利用数据块不重叠的特性，分步reduce
//   Step1: M=4, E->A, F->B（后面的块reduce到前面的对应位置）
//   Step2: M=2, C->(AE), D->(BF)
//   Step3: M=1, (BFD)->(AEC)
//   最终结果在虚拟索引0位置（即myAlgRank位置
HcclResult InsTempReduceScatterOrderPreservedGroup::RunLocalReduce(
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[RunLocalReduce] Start, deterministicStrict[%d], templateRankSize[%u]",
        deterministicStrict_, templateRankSize_);

    if (templateRankSize_ <= 1) {
        HCCL_INFO("[RunLocalReduce] Skip for single rank");
        return HCCL_SUCCESS;
    }

    const MemBlockInfo &memBlockInfo = memBlockInfo_;

    // 获取本rank在算法中的编号（区分通信域rank和算法内部rank）
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    HCCL_INFO("[RunLocalReduce] myAlgRank[%u]", myAlgRank);

    u64 sliceSize = memBlockInfo.size[myAlgRank];
    u64 count = sliceSize / DATATYPE_SIZE_TABLE[dataType_];

    // ========== 树形Reduce主循环 ==========
    // remainingBlocks: 当前待处理的数据块数量
    u32 remainingBlocks = templateRankSize_;
    u32 step = 0;

    while (remainingBlocks > 1) {
        u32 M = GetLargestPowerOf2LessThan(remainingBlocks);
        if (M == 0) {
            break;
        }

        HCCL_INFO("[RunLocalReduce] Step[%u]: remainingBlocks[%u], M[%u]", step, remainingBlocks, M);

        // 先规约不对齐的数据块
        for (u32 srcVirtualIdx = M; srcVirtualIdx < remainingBlocks; srcVirtualIdx++) {
            // 计算目标虚拟索引，超出的第一个块和第一个块合并，第二个超出块和第二个块合并，以此类推
            u32 dstVirtualIdx = srcVirtualIdx % M;

            // 虚拟索引映射到实际peerRank（数据来源rank）
            // peerRank表示该数据块来自哪个rank
            // 例如: myAlgRank=0, srcVirtualIdx=4 -> srcPeerRank=4
            u32 srcPeerRank = srcVirtualIdx;
            u32 dstPeerRank = dstVirtualIdx;

            u32 srcOutputIndex = CalcOutputIndex(srcPeerRank, myAlgRank);
            u32 dstOutputIndex = CalcOutputIndex(dstPeerRank, myAlgRank);

            // 规约前后的结果都在cclbuffer上，因此都用outputOffsets
            u64 srcOffset = memBlockInfo.outputOffsets[srcOutputIndex];
            u64 dstOffset = memBlockInfo.outputOffsets[dstOutputIndex];

            if (sliceSize == 0) {
                continue;
            }

            // count是必须的，不然会走另一个构造，localreduce要求count有值
            DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcOffset, sliceSize, count);
            DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, dstOffset, sliceSize, count);

            HCCL_INFO("[RunLocalReduce] Step[%u]: virtualIdx[%u]->[%u], peerRank[%u]->[%u], "
                "offset[%llu]->[%llu]", step, srcVirtualIdx, dstVirtualIdx, srcPeerRank, dstPeerRank,
                srcOffset, dstOffset);

            // 执行本地reduce操作（使用主线程threads[0]串行执行）
            CHK_RET(LocalReduce(threads[0], srcSlice, dstSlice, dataType_, reduceOp_));
        }

        // 更新剩余数据块数量，进入下一轮
        remainingBlocks = M;
        step++;
    }

    HCCL_INFO("[RunLocalReduce] End, total steps[%u], final data at virtualIdx[0] (myAlgRank[%u])",
        step, myAlgRank);
    return HCCL_SUCCESS;
}

// 后处理拷贝：将归约结果从临时缓冲区拷贝到用户输出缓冲区
HcclResult InsTempReduceScatterOrderPreservedGroup::PostCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    const MemBlockInfo &memBlockInfo = memBlockInfo_;
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    u64 sliceSize = memBlockInfo.size[myAlgRank];
    if (sliceSize == 0) {
        HCCL_DEBUG("[PostCopy] myAlgRank[%u] sliceSize is 0, skip.", myAlgRank);
        return HCCL_SUCCESS;
    }
    // 保序规约后，结果落在peerRank=0的位置（dstVirtualIdx始终为0）
    u32 outputIndex = CalcOutputIndex(0, myAlgRank);
    // 计算源地址偏移量（临时缓冲区中的归约结果位置）
    u64 srcOffset = memBlockInfo.outputOffsets[outputIndex];
    // 目标偏移量为0（用户输出缓冲区起始位置）
    u64 dstOffset = tempAlgParams.buffInfo.outBuffBaseOff;

    HCCL_INFO("[PostCopy] myAlgRank[%u], sliceSize[%llu], srcOffset[%llu], dstOffset[%llu]",
        myAlgRank, sliceSize, srcOffset, dstOffset);

    DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcOffset, sliceSize);
    DataSlice dstSlice(tempAlgParams.buffInfo.outputPtr, dstOffset, sliceSize);

    CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));

    return HCCL_SUCCESS;
}
}
