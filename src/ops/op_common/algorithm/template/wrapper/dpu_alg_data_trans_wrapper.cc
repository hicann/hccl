/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dpu_alg_data_trans_wrapper.h"
#include "exec_timeout_manager.h"
#include "hcomm_primitives.h"
#ifndef AICPU_COMPILE
#include "dlsym_common.h"
#include "hcomm_dlsym.h"
#include "log.h"
#endif

namespace ops_hccl {

#ifndef AICPU_COMPILE
HcclResult HcommChannelDrainOnThreadWithCompat(ThreadHandle thread, ChannelHandle channel)
{
    if (channel == 0) {
        HCCL_ERROR("[HcommChannelDrainOnThread] channel is nullptr.");
        return HCCL_E_PTR;
    }

    constexpr int hostChannelDrainMinVersion = CANN_VERSION(9, 2, 0, 2);
    const int hcommVersion = GetHcommVersion();
    if (hcommVersion < hostChannelDrainMinVersion) {
        HCCL_WARNING(
            "[HcommChannelDrainOnThread] HCOMM version[%d] does not support Host Channel Drain, "
            "fallback to HcommChannelFenceOnThread.",
            hcommVersion);
        return static_cast<HcclResult>(HcommChannelFenceOnThread(thread, channel));
    }

    const HcclResult drainRet = static_cast<HcclResult>(HcommChannelDrainOnThread(thread, channel));
    if (drainRet == HCCL_E_PTR) {
        HCCL_ERROR(
            "[HcommChannelDrainOnThread] failed with HCCL_E_PTR, channel[0x%llx], HCOMM version[%d]. "
            "Check channel or update HCOMM.",
            static_cast<unsigned long long>(channel), hcommVersion);
    }
    return drainRet;
}
#endif

HcclResult SendRecvWrite(const SendRecvInfo& sendRecvInfo)
{
#ifndef AICPU_COMPILE
    const u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_;
    const ChannelInfo& sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    const ChannelInfo& recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    u32 repeatNum = srcSlices.size();
    // 向write rank发送tx同步，确保该rank的hcclBuffer可用
    // 这里只是在host上向device下任务，所以实际在host侧不会因为wait而阻塞
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, recvChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(
        static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (int i = 0; i < repeatNum; i++) {
        // tx同步完成后准备将自己的userIn上的数据写到对方的hcclBuffer上
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlcie = dstSlices[i];
        void* dst = static_cast<void*>(static_cast<s8*>(dstSlcie.addr_) + dstSlcie.offset_);
        void* src = static_cast<void*>(static_cast<s8*>(srcSlice.addr_) + srcSlice.offset_);
        CHK_RET(static_cast<HcclResult>(
            HcommWriteWithNotifyNbiOnThread(0, sendChannel.handle, dst, src, srcSlice.size_, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(0, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    }
    // 写完之后做后同步告诉对面写完了
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, sendChannel.handle, NOTIFY_IDX_FIN_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(0, recvChannel.handle, NOTIFY_IDX_FIN_ACK, execTimeout)));
    CHK_RET(HcommChannelDrainOnThreadWithCompat(0, sendChannel.handle));
    CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
#endif
    return HCCL_SUCCESS;
}

HcclResult SendWrite(const DataInfo& sendInfo)
{
#ifndef AICPU_COMPILE
    const u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    const std::vector<DataSlice> srcSlices = sendInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendInfo.slices_.dstSlices_;
    const ChannelInfo& sendChannel = sendInfo.channel_;
    u32 sliceNum = srcSlices.size();
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, sendChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(
        static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (int i = 0; i < sliceNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlcie = dstSlices[i];
        void* dst = static_cast<void*>(static_cast<s8*>(dstSlcie.addr_) + dstSlcie.offset_);
        void* src = static_cast<void*>(static_cast<s8*>(srcSlice.addr_) + srcSlice.offset_);
        CHK_RET(static_cast<HcclResult>(
            HcommWriteWithNotifyNbiOnThread(0, sendChannel.handle, dst, src, srcSlice.size_, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(0, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, sendChannel.handle, NOTIFY_IDX_FIN_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(0, sendChannel.handle, NOTIFY_IDX_FIN_ACK, execTimeout)));
    CHK_RET(HcommChannelDrainOnThreadWithCompat(0, sendChannel.handle));
    CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
#endif
    return HCCL_SUCCESS;
}

HcclResult RecvWrite(const DataInfo& recvInfo)
{
#ifndef AICPU_COMPILE
    const u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    const std::vector<DataSlice> srcSlices = recvInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = recvInfo.slices_.dstSlices_;
    const ChannelInfo& recvChannel = recvInfo.channel_;
    u32 sliceNum = srcSlices.size();
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, recvChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(
        static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, recvChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (int i = 0; i < sliceNum; i++) {
        CHK_RET(
            static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(0, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, recvChannel.handle, NOTIFY_IDX_FIN_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(0, recvChannel.handle, NOTIFY_IDX_FIN_ACK, execTimeout)));
    CHK_RET(HcommChannelDrainOnThreadWithCompat(0, recvChannel.handle));
    CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
#endif
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
