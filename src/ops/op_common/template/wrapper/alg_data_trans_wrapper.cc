/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

HcclResult SendWrite(const DataInfo &sendInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = sendInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendInfo.slices_.dstSlices_;
    const ChannelInfo &sendChannel = sendInfo.channel_;
    u32 sliceNum = srcSlices.size();
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK,
                                                                   CUSTOM_TIMEOUT)));
    for (int i = 0; i < sliceNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlcie = dstSlices[i];
        if (srcSlice.size_ == 0) {
            HCCL_WARNING("[AlgDataTransWrapper] SendWrite: size is 0.");
            continue;;
        }
        void* dst = static_cast<void *>(static_cast<s8 *>(dstSlcie.addr_) + dstSlcie.offset_);
        void* src = static_cast<void *>(static_cast<s8 *>(srcSlice.addr_) + srcSlice.offset_);
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, sendChannel.handle, dst, src, srcSlice.size_)));
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle,
                                                                    NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult RecvWrite(const DataInfo &recvInfo, const ThreadHandle &thread)
{
    const ChannelInfo &recvChannel = recvInfo.channel_;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL,
                                                                   CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

/*
 这个SendRecv是以notify的视角去看的，针对一个thread上的notify，即有record操作也有wait操作。
 为什么是SendRecv：因为是一个双向的写，rank 0需要向rank 1写，而rank 1也需要向rank 0写，
 因此对于rank 0来说需要向rank 1 record告诉rank 1自己准备好了可以写了，
 而rank 0也需要wait一下rank 1的record知道rnak 1那边也可以写了。
*/
HcclResult SendRecvWrite(const SendRecvInfo &sendRecvInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_;
    const ChannelInfo &sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    const ChannelInfo &recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    u32 repeatNum = srcSlices.size();
    // 向write rank发送tx同步，确保该rank的hcclBuffer可用
    // 这里只是在host上向device下任务，所以实际在host侧不会因为wait而阻塞
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK,
                                                                   CUSTOM_TIMEOUT)));
    for (int i = 0; i < repeatNum; i++) {
        // tx同步完成后准备将自己的userIn上的数据写到对方的hcclBuffer上
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlcie = dstSlices[i];
        if (srcSlice.size_ == 0) {
            HCCL_WARNING("[AlgDataTransWrapper] SendRecvWrite: size is 0.");
            continue;
        }
        void* dst = static_cast<void *>(static_cast<s8 *>(dstSlcie.addr_) + dstSlcie.offset_);
        void* src = static_cast<void *>(static_cast<s8 *>(srcSlice.addr_) + srcSlice.offset_);
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, sendChannel.handle, dst, src, srcSlice.size_)));
    }
    // 写完之后做后同步告诉对面写完了
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle,
                                                                     NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL,
                                                                   CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult SendWriteReduce(const DataReduceInfo &sendInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = sendInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendInfo.slices_.dstSlices_;
    const ChannelInfo &sendChannel = sendInfo.channel_;
    u32 repeatNum = srcSlices.size();
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK,
                                                                   CUSTOM_TIMEOUT)));
    for (int i = 0; i < repeatNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlcie = dstSlices[i];
        if (srcSlice.size_ == 0) {
            HCCL_WARNING("[AlgDataTransWrapper] SendWriteReduce: size is 0.");
            continue;
        }
        CHK_PRT_RET(
            srcSlice.count_ * DATATYPE_SIZE_TABLE[sendInfo.dataType_] != srcSlice.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] SendWriteReduce: src slice count [%u] is not mate to src slice size [%u], dataType is [%d].",
                srcSlice.size_, srcSlice.size_, sendInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        CHK_PRT_RET(
            dstSlcie.count_ * DATATYPE_SIZE_TABLE[sendInfo.dataType_] != dstSlcie.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] SendWriteReduce: dst slice count [%u] is not mate to dst slice size [%u], dataType is [%d].",
                dstSlcie.size_, dstSlcie.size_, sendInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        void* dst = static_cast<void *>(static_cast<s8 *>(dstSlcie.addr_) + dstSlcie.offset_);
        void* src = static_cast<void *>(static_cast<s8 *>(srcSlice.addr_) + srcSlice.offset_);
        CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(thread, sendChannel.handle, dst, src, srcSlice.count_,
                                                                 static_cast<HcommDataType>(sendInfo.dataType_),
                                                                 static_cast<HcommReduceOp>(sendInfo.reduceType_))));
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle,
                                                                     NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult RecvWriteReduce(const DataReduceInfo &recvInfo, const ThreadHandle &thread)
{
    const ChannelInfo &recvChannel = recvInfo.channel_;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL,
                                                                   CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult SendRecvWriteReduce(const SendRecvReduceInfo &sendRecvInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_;
    const ChannelInfo &sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    const ChannelInfo &recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    u32 repeatNum = srcSlices.size();
    // 向write rank发送tx同步，确保该rank的hcclBuffer可用
    // 这里只是在host上向device下任务，所以实际在host侧不会因为wait而阻塞
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK,
                                                            CUSTOM_TIMEOUT)));
    for (int i = 0; i < repeatNum; i++) {
        // tx同步完成后准备将自己的userIn上的数据写到对方的hcclBuffer上
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlcie = dstSlices[i];
        if (srcSlice.size_ == 0) {
            HCCL_WARNING("[AlgDataTransWrapper] SendRecvWriteReduce: size is 0.");
            continue;
        }
        CHK_PRT_RET(
            srcSlice.count_ * DATATYPE_SIZE_TABLE[sendRecvInfo.dataType_] != srcSlice.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] SendWriteReduce: src slice count [%u] is not mate to src slice size [%u], dataType is [%d].",
                srcSlice.size_, srcSlice.size_, sendRecvInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        CHK_PRT_RET(
            dstSlcie.count_ * DATATYPE_SIZE_TABLE[sendRecvInfo.dataType_] != dstSlcie.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] SendWriteReduce: dst slice count [%u] is not mate to dst slice size [%u], dataType is [%d].",
                dstSlcie.size_, dstSlcie.size_, sendRecvInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        void* dst = static_cast<void *>(static_cast<s8 *>(dstSlcie.addr_) + dstSlcie.offset_);
        void* src = static_cast<void *>(static_cast<s8 *>(srcSlice.addr_) + srcSlice.offset_);
        CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(thread, sendChannel.handle, dst, src, srcSlice.count_,
                                                                 static_cast<HcommDataType>(sendRecvInfo.dataType_),
                                                                 static_cast<HcommReduceOp>(sendRecvInfo.reduceType_))));
    }
    // 写完之后做后同步告诉对面写完了
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle,
                                                                     NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL,
                                                                   CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult SendRead(const DataInfo &sendInfo, const ThreadHandle &thread)
{
    const ChannelInfo &sendChannel = sendInfo.channel_;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL,
                                                                   CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult RecvRead(const DataInfo &recvInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = recvInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = recvInfo.slices_.dstSlices_;
    const ChannelInfo &recvChannel = recvInfo.channel_;
    u32 repeatNum = srcSlices.size();
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK,
                                                                   CUSTOM_TIMEOUT)));
    for (int i = 0; i < repeatNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlcie = dstSlices[i];
        if (srcSlice.size_ == 0) {
            HCCL_WARNING("[AlgDataTransWrapper] RecvRead: size is 0.");
            continue;
        }
        void* dst = static_cast<void *>(static_cast<s8 *>(dstSlcie.addr_) + dstSlcie.offset_);
        void* src = static_cast<void *>(static_cast<s8 *>(srcSlice.addr_) + srcSlice.offset_);
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, recvChannel.handle, dst, src, srcSlice.size_)));
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult SendRecvRead(const SendRecvInfo &sendRecvInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.dstSlices_;
    const ChannelInfo &sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    const ChannelInfo &recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    u32 repeatNum = srcSlices.size();
    // 向read rank发送rx同步，确保该rank的hcclBuffer可用
    // 这里只是在host上向device下任务，所以实际在host侧不会因为wait而阻塞
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK,
                                                                   CUSTOM_TIMEOUT)));
    for (int i = 0; i < repeatNum; i++) {
        // rx同步完成后准备将数据从对方的hcclBuffer上读到自己的userIn上
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlcie = dstSlices[i];
        if (srcSlice.size_ == 0) {
            HCCL_WARNING("[AlgDataTransWrapper] SendRecvRead: size is 0.");
            continue;
        }
        void* dst = static_cast<void *>(static_cast<s8 *>(dstSlcie.addr_) + dstSlcie.offset_);
        void* src = static_cast<void *>(static_cast<s8 *>(srcSlice.addr_) + srcSlice.offset_);
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, recvChannel.handle, dst, src, srcSlice.size_)));
    }
    // 写完之后做后同步告诉对面写完了
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle,
                                                                     NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL,
                                                                   CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult SendReadReduce(const DataReduceInfo &sendInfo, const ThreadHandle &thread)
{
    const ChannelInfo &sendChannel = sendInfo.channel_;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL,
                                                                   CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult RecvReadReduce(const DataReduceInfo &recvInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = recvInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = recvInfo.slices_.dstSlices_;
    const ChannelInfo &recvChannel = recvInfo.channel_;
    u32 repeatNum = srcSlices.size();
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK,
                                                                   CUSTOM_TIMEOUT)));
    for (int i = 0; i < repeatNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlcie = dstSlices[i];
        if (srcSlice.size_ == 0) {
            HCCL_WARNING("[AlgDataTransWrapper] RecvReadReduce: size is 0.");
            continue;
        }
        CHK_PRT_RET(
            srcSlice.count_ * DATATYPE_SIZE_TABLE[recvInfo.dataType_] != srcSlice.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] SendWriteReduce: src slice count [%u] is not mate to src slice size [%u], dataType is [%d].",
                srcSlice.size_, srcSlice.size_, recvInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        CHK_PRT_RET(
            dstSlcie.count_ * DATATYPE_SIZE_TABLE[recvInfo.dataType_] != dstSlcie.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] SendWriteReduce: dst slice count [%u] is not mate to dst slice size [%u], dataType is [%d].",
                dstSlcie.size_, dstSlcie.size_, recvInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        void* dst = static_cast<void *>(static_cast<s8 *>(dstSlcie.addr_) + dstSlcie.offset_);
        void* src = static_cast<void *>(static_cast<s8 *>(srcSlice.addr_) + srcSlice.offset_);
        CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread, recvChannel.handle, dst, src, srcSlice.count_,
                                                                 static_cast<HcommDataType>(recvInfo.dataType_),
                                                                 static_cast<HcommReduceOp>(recvInfo.reduceType_))));
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult SendRecvReadReduce(const SendRecvReduceInfo &sendRecvInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.dstSlices_;
    const ChannelInfo &sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    const ChannelInfo &recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    u32 repeatNum = srcSlices.size();
    // 向write rank发送tx同步，确保该rank的hcclBuffer可用
    // 这里只是在host上向device下任务，所以实际在host侧不会因为wait而阻塞
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK,
                                                            CUSTOM_TIMEOUT)));
    for (int i = 0; i < repeatNum; i++) {
        // tx同步完成后准备将自己的userIn上的数据写到对方的hcclBuffer上
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlcie = dstSlices[i];
        if (srcSlice.size_ == 0) {
            HCCL_WARNING("[AlgDataTransWrapper] SendRecvReadReduce: size is 0.");
            continue;
        }
        CHK_PRT_RET(
            srcSlice.count_ * DATATYPE_SIZE_TABLE[sendRecvInfo.dataType_] != srcSlice.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] SendWriteReduce: src slice count [%u] is not mate to src slice size [%u], dataType is [%d].",
                srcSlice.size_, srcSlice.size_, sendRecvInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        CHK_PRT_RET(
            dstSlcie.count_ * DATATYPE_SIZE_TABLE[sendRecvInfo.dataType_] != dstSlcie.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] SendWriteReduce: dst slice count [%u] is not mate to dst slice size [%u], dataType is [%d].",
                dstSlcie.size_, dstSlcie.size_, sendRecvInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        void* dst = static_cast<void *>(static_cast<s8 *>(dstSlcie.addr_) + dstSlcie.offset_);
        void* src = static_cast<void *>(static_cast<s8 *>(srcSlice.addr_) + srcSlice.offset_);
        CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread, recvChannel.handle, dst, src, srcSlice.count_,
                                                                 static_cast<HcommDataType>(sendRecvInfo.dataType_),
                                                                 static_cast<HcommReduceOp>(sendRecvInfo.reduceType_))));
    }
    // 写完之后做后同步告诉对面写完了
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle,
                                                                     NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL,
                                                                   CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult LocalCopy(const ThreadHandle &thread, const DataSlice &srcSlice, const DataSlice &dstSlice)
{
    CHK_PRT_RET(
        srcSlice.size_ == 0,
        HCCL_WARNING(
            "[AlgDataTransWrapper] LocalCopy: src slice size is [%u].",
            srcSlice.size_),
        HcclResult::HCCL_SUCCESS);
    
    CHK_PRT_RET(
        srcSlice.size_ != dstSlice.size_,
        HCCL_ERROR(
            "[AlgDataTransWrapper] LocalCopy: src slice size [%u] is not equal to dst slice size [%u].",
            srcSlice.size_, dstSlice.size_),
        HcclResult::HCCL_E_INTERNAL);
    void* srcIn = static_cast<void *>(static_cast<u8 *>(srcSlice.addr_) + srcSlice.offset_);
    void* dstOut = static_cast<void *>(static_cast<u8 *>(dstSlice.addr_) + dstSlice.offset_);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dstOut, srcIn, srcSlice.size_)));
    return HCCL_SUCCESS;
}

HcclResult LocalReduce(const ThreadHandle &thread, const DataSlice &srcSlice, const DataSlice &dstSlice,
                       const HcclDataType dataType, const HcclReduceOp reduceOp)
{
    CHK_PRT_RET(
        srcSlice.size_ == 0,
        HCCL_WARNING(
            "[AlgDataTransWrapper] LocalReduce: src slice size is [%u].",
            srcSlice.size_),
        HcclResult::HCCL_SUCCESS);

    CHK_PRT_RET(
        srcSlice.size_ != dstSlice.size_,
        HCCL_ERROR(
            "[InsCollAlgFactory] LocalReduce: src slice size [%u] is not equal to dst slice size [%u].",
            srcSlice.size_, dstSlice.size_),
        HcclResult::HCCL_E_INTERNAL);
    void* src = static_cast<void *>(static_cast<u8 *>(srcSlice.addr_) + srcSlice.offset_);
    void* dst = static_cast<void *>(static_cast<u8 *>(dstSlice.addr_) + dstSlice.offset_);
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread, dst, src, srcSlice.count_,
                                                             static_cast<HcommDataType>(dataType),
                                                             static_cast<HcommReduceOp>(reduceOp))));
    return HCCL_SUCCESS;
}

HcclResult LocalCopySlices(const ThreadHandle &thread, const std::vector<DataSlice> &srcSlices,
                           const std::vector<DataSlice> &dstSlices)
{
    CHK_PRT_RET(srcSlices.size() != dstSlices.size(),
                HCCL_ERROR("[InsCollAlgFactory] [AlgDataTrans] LocalCopySlices: num of src slices [%u], is not equal "
                           "to num of dst slices [%u].",
                           srcSlices.size(), dstSlices.size()),
                HcclResult::HCCL_E_INTERNAL);

    // tmpSlices: slices to be transfer in this loop
    DataSlice tmpSrcSlice = srcSlices[0];
    DataSlice tmpDstSlice = dstSlices[0];

    for (u32 sliceIdx = 0; sliceIdx < srcSlices.size(); sliceIdx++) {
        if (srcSlices[sliceIdx].size_ == 0) {
            HCCL_WARNING("[AlgDataTransWrapper] LocalCopySlices: size is 0.");
            continue;
        }
        CHK_PRT_RET(srcSlices[sliceIdx].size_ != dstSlices[sliceIdx].size_,
                    HCCL_ERROR("[InsCollAlgFactory] [AlgDataTransWrapper] LocalCopySlices: [%u]-th slice, src slice size [%u] "
                               "is not equal to dst slice size [%u].",
                               sliceIdx, srcSlices[sliceIdx].size_, dstSlices[sliceIdx].size_),
                    HcclResult::HCCL_E_INTERNAL);

        if (sliceIdx == (srcSlices.size() - 1)) {
            // last slice
            void* src = static_cast<void *>(static_cast<u8 *>(tmpSrcSlice.addr_) + tmpSrcSlice.offset_);
            void* dst = static_cast<void *>(static_cast<u8 *>(tmpDstSlice.addr_) + tmpDstSlice.offset_);
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dst, src, tmpSrcSlice.size_)));
        } else if (IsContinuousSlice(srcSlices[sliceIdx + 1], tmpSrcSlice)
                   && IsContinuousSlice(dstSlices[sliceIdx + 1], tmpDstSlice)) {
            // nxtSlice is continuous with tmpSlice, update tmpSlice
            u64 newTmpSize = tmpSrcSlice.size_ + srcSlices[sliceIdx + 1].size_;
            tmpSrcSlice    = DataSlice(tmpSrcSlice.addr_, tmpSrcSlice.offset_, newTmpSize);
            tmpDstSlice    = DataSlice(tmpDstSlice.addr_, tmpDstSlice.offset_, newTmpSize);
        } else {
            // nxtSlice is not continuous with tmpSlice, copy tmpSlice, update tmpSlice with nxtSlice
            void* src = static_cast<void *>(static_cast<u8 *>(tmpSrcSlice.addr_) + tmpSrcSlice.offset_);
            void* dst = static_cast<void *>(static_cast<u8 *>(tmpDstSlice.addr_) + tmpDstSlice.offset_);
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dst, src, tmpSrcSlice.size_)));

            tmpSrcSlice = srcSlices[sliceIdx + 1];
            tmpDstSlice = dstSlices[sliceIdx + 1];
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

bool IsContinuousSlice(const DataSlice &nxtSlice, const DataSlice &currSlice)
{
    if (nxtSlice.addr_ != currSlice.addr_) {
        return false;
    }
    if (nxtSlice.offset_ != currSlice.offset_ + currSlice.size_) {
        return false;
    }
    return true;
}

HcclResult PreSyncInterThreads(const ThreadHandle &mainThread, const std::vector<ThreadHandle> &subThreads,
                               const std::vector<u32> &notifyIdxMainToSub)
{
    CHK_PRT_RET(subThreads.size() == 0 || notifyIdxMainToSub.size() == 0,
                HCCL_ERROR("[AlgDataTransWrapper] [PreSyncInterThreads] subThreads size: [%u], notifyIdxMainToSub size [%u] "
                            "0 is not correct.",
                            subThreads.size(), notifyIdxMainToSub.size()),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(subThreads.size() != notifyIdxMainToSub.size(),
                HCCL_ERROR("[AlgDataTransWrapper] [PreSyncInterThreads] subThreads size: [%u], notifyIdxMainToSub size [%u] "
                            "is not equal.",
                            subThreads.size(), notifyIdxMainToSub.size()),
                HcclResult::HCCL_E_INTERNAL);
    // 主thread向从thread发送record
    for (u32 tidx = 0; tidx < subThreads.size(); tidx++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(mainThread, subThreads[tidx],
                                                                        notifyIdxMainToSub[tidx])));
    }

    // 从thread等待主thread的record
    for (u32 tidx = 0; tidx < subThreads.size(); tidx++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(subThreads[tidx], notifyIdxMainToSub[tidx],
                                                                      CUSTOM_TIMEOUT)));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult PostSyncInterThreads(const ThreadHandle &mainThread, const std::vector<ThreadHandle> &subThreads,
                                const std::vector<u32> &notifyIdxSubToMain)
{
    CHK_PRT_RET(subThreads.size() == 0 || notifyIdxSubToMain.size() == 0,
            HCCL_ERROR("[AlgDataTransWrapper] [PreSyncInterThreads] subThreads size: [%u], notifyIdxSubToMain size [%u] "
                        "0 is not correct.",
                        subThreads.size(), notifyIdxSubToMain.size()),
            HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(subThreads.size() != notifyIdxSubToMain.size(),
                HCCL_ERROR("[AlgDataTransWrapper] [PreSyncInterThreads] subThreads size: [%u], notifyIdxSubToMain size [%u] "
                            "is not equal.",
                            subThreads.size(), notifyIdxSubToMain.size()),
                HcclResult::HCCL_E_INTERNAL);
    // 主thread等待所有从thread的record
    for (u32 tidx = 0; tidx < subThreads.size(); tidx++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(mainThread, notifyIdxSubToMain[tidx], CUSTOM_TIMEOUT)));
    }

    // 从thread向主thread发送record
    for (u32 tidx = 0; tidx < subThreads.size(); tidx++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(subThreads[tidx], mainThread,
                                                                        notifyIdxSubToMain[tidx])));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult AicpuReduce(const ThreadHandle &thread, const DataSlice &srcSlice, const DataSlice &dstSlice,
                       const HcclDataType dataType, const HcclReduceOp reduceOp)
{
    CHK_PRT_RET(
        srcSlice.size_ != dstSlice.size_,
        HCCL_ERROR(
            "[AlgDataTransWrapper] [AicpuReduce] AicpuReduce: src slice size [%u] is not equal to dst slice size [%u].",
            srcSlice.size_, dstSlice.size_),
        HcclResult::HCCL_E_INTERNAL);
    
    auto ret = HcclResult::HCCL_SUCCESS;
    switch (dataType) {
        case HcclDataType::HCCL_DATA_TYPE_INT64:
            AicpuReduceTemplate<int64_t>((int64_t*)(dstSlice.addr_), dstSlice.size_,
                                         (int64_t*)(srcSlice.addr_), srcSlice.size_, reduceOp);
            break;
        case HcclDataType::HCCL_DATA_TYPE_UINT64:
            AicpuReduceTemplate<uint64_t>((uint64_t*)(dstSlice.addr_), dstSlice.size_,
                                          (uint64_t*)(srcSlice.addr_), srcSlice.size_, reduceOp);
            break;
        case HcclDataType::HCCL_DATA_TYPE_FP64:
            AicpuReduceTemplate<double>((double*)(dstSlice.addr_), dstSlice.size_, 
                                        (double*)(srcSlice.addr_), srcSlice.size_, reduceOp);
            break;
        default:
            HCCL_ERROR("DataType[%d] not support", int(dataType));
            ret = HCCL_E_INTERNAL;
            break;
    }
    return ret;
}
 
template <typename T>
HcclResult AicpuReduceTemplate(T* dst, u64 dstSize, T* src, u64 srcSize, const HcclReduceOp reduceOp)
{
    if (dstSize != srcSize) {
        HCCL_ERROR("srcSize[%llu] should be equal to dstSize[%llu]", srcSize, dstSize);
        return HcclResult::HCCL_E_INTERNAL;
    }
    auto ret = HcclResult::HCCL_SUCCESS;
    u64 count = dstSize / u64(sizeof(T));
    for (u64 i = 0; i < count; ++i) {
        T dstData = *(dst + i);
        T srcData = *(src + i);
        switch (reduceOp) {
            case HcclReduceOp::HCCL_REDUCE_SUM:
                *(dst + i) = srcData + dstData;
                break;
            case HcclReduceOp::HCCL_REDUCE_PROD:
                *(dst + i) = srcData * dstData;
                break;
            case HcclReduceOp::HCCL_REDUCE_MAX:
                *(dst + i) = std::max(srcData, dstData);
                break;
            case HcclReduceOp::HCCL_REDUCE_MIN:
                *(dst + i) = std::min(srcData, dstData);
                break;
            default:
                HCCL_ERROR("ReduceOp[%d] not support", int(reduceOp));
                ret = HcclResult::HCCL_E_INTERNAL;
                break;   
        }
    }
    return ret;
}

}