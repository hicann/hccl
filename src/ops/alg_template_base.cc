/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_base.h"

namespace ops_hccl {
AlgTemplateBase::AlgTemplateBase()
    : slices_(slicesDummy_), count_(0), dataBytes_(0), dataType_(HCCL_DATA_TYPE_RESERVED),
      reductionOp_(HCCL_REDUCE_RESERVED), root_(INVALID_VALUE_RANKID),
      baseOffset_(0), barrierSwitchOn_(true)
{
}

AlgTemplateBase::~AlgTemplateBase()
{
    slices_.clear();
}

HcclResult AlgTemplateBase::Prepare(PrepareData &param)
{
    return HCCL_E_PARA;
}

// prepare函数给需要进行集合通信操作进行参数赋值
HcclResult AlgTemplateBase::Prepare(HcclMem &inputMem, HcclMem &outputMem, HcclMem &scratchMem,
                                 const u64 count,
                                 const HcclDataType dataType, ThreadHandle thread,
                                 const HcclReduceOp reductionOp,
                                 const u32 root, const std::vector<Slice> &slices, const u64 baseOffset,
                                 const bool disableDMAReduce)
{
    HCCL_DEBUG("AlgTemplateBase prepare start");

    /* * 参数保存 */
    inputMem_ = inputMem;
    outputMem_ = outputMem;
    scratchMem_ = scratchMem;
    thread_ = thread;
    count_ = count;
    dataType_ = dataType;
    dataBytes_ = count * DataUnitSize(dataType);
    reductionOp_ = reductionOp;
    root_ = root;
    disableDMAReduce_ = disableDMAReduce;

    /* 相对用户基地址偏移 */
    baseOffset_ = baseOffset;

    if (slices.size() > 0) {
        slices_.resize(slices.size());
        slices_ = slices;
    }

    // 不带入该参数，代表数据均分，直接用count赋值
    HCCL_DEBUG("AlgTemplateBase prepare end");
    return HCCL_SUCCESS;
}

// ScatterMesh
HcclResult AlgTemplateBase::Prepare(u32 interRank, u32 interRankSize)
{
    return HCCL_E_PARA;
}

HcclResult AlgTemplateBase::Prepare(HcomCollOpInfo *opInfo, const u32 userRank, const std::vector<u32> &ringsOrders,
        const std::vector<Slice> &userMemInputSlices) {
    return HCCL_E_PARA;
}

HcclResult AlgTemplateBase::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels)
{
    (void)rank;
    (void)rankSize;
    (void)channels;
    return HCCL_SUCCESS;
}

HcclResult AlgTemplateBase::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels, std::vector<ThreadHandle> slaveThreads)
{
    (void)rank;
    (void)rankSize;
    (void)channels;
    (void)slaveThreads;
    return HCCL_SUCCESS;
}

HcclResult AlgTemplateBase::RunAsync(const DPUAlgResourceCtx *dpuResCtx)
{
    (void) dpuResCtx;
    return HCCL_SUCCESS;
}

HcclResult AlgTemplateBase::ExecuteBarrier(ChannelInfo &channel, ThreadHandle thread)
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

    return HCCL_SUCCESS;
}

HcclResult AlgTemplateBase::ExecuteBarrier(ChannelInfo &preChannel, ChannelInfo &aftChannel)
{
    return ExecuteBarrier(preChannel, aftChannel, thread_);
}

HcclResult AlgTemplateBase::ExecuteBarrier(ChannelInfo &preChannel, ChannelInfo &aftChannel, ThreadHandle thread)
{
    // 同步与preChannel保证数据收发已结束
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, preChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, aftChannel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));

    // 同步与aftChannel保证数据收发已结束
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, aftChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, preChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

    return HCCL_SUCCESS;
}

HcclResult AlgTemplateBase::ExecuteBarrier(ChannelInfo &preChannel, ChannelInfo &aftChannel, u32 notifyIdx)
{
    return ExecuteBarrier(preChannel, aftChannel, notifyIdx, thread_);
}

HcclResult AlgTemplateBase::ExecuteBarrier(ChannelInfo &preChannel, ChannelInfo &aftChannel, u32 notifyIdx, ThreadHandle thread)
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, aftChannel.handle, notifyIdx)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, preChannel.handle, notifyIdx, CUSTOM_TIMEOUT)));

    return HCCL_SUCCESS;
}

HcclResult AlgTemplateBase::ExecEmptyTask(HcclMem &inputMem, HcclMem &outputMem, ThreadHandle thread)
{
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, outputMem.addr, inputMem.addr, 0)));
    return HCCL_SUCCESS;
}

HcclResult AlgTemplateBase::CheckConcurrentDirectParameters(const u32 rank, const u32 rankSize,
                                                         std::vector<ChannelInfo> &channels)
{
    // inputMem_ == outputMem_ 是允许的, 因为ring的时候收的slice和发的slice不是同一片
    // reduce scatter用inputMem_，allgather用outputMem_
    if (!outputMem_.addr || !inputMem_.addr) {
        HCCL_ERROR("[AlgTemplateBase] rank[%u] run_async inputmem or outputmem is null", rank);
        return HCCL_E_PTR;
    }
    HCCL_INFO("AlgTemplateBase run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
              inputMem_.addr, outputMem_.addr, count_);

    // 判断channels数量是否正确
    CHK_PRT_RET(channels.size() < rankSize,
                HCCL_ERROR("[AlgTemplateBase] rank[%u] link size[%u] is less than "
                           "rank size[%u]",
                           rank, channels.size(), rankSize),
                HCCL_E_PARA);

    // 校验DataUnitSize
    if (DataUnitSize(dataType_) == 0) {
        HCCL_ERROR("[AlgTemplateBase] rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("AlgTemplateBase finished to CheckParameters");
    return HCCL_SUCCESS;
}

}
