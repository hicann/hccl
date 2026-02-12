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
#include "hcomm_primitives.h"

namespace ops_hccl {

// ! 已编码完成
HcclResult SendRecvWrite(const SendRecvInfo &sendRecvInfo) {
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_;
    const ChannelInfo &sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    const ChannelInfo &recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    u32 repeatNum = srcSlices.size();
    // 向write rank发送tx同步，确保该rank的hcclBuffer可用
    // 这里只是在host上向device下任务，所以实际在host侧不会因为wait而阻塞
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecord(recvChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWait(sendChannel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    for (int i = 0; i < repeatNum; i++) {
        // tx同步完成后准备将自己的userIn上的数据写到对方的hcclBuffer上
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlcie = dstSlices[i];
        void* dst = static_cast<void *>(static_cast<s8 *>(dstSlcie.addr_) + dstSlcie.offset_);
        void* src = static_cast<void *>(static_cast<s8 *>(srcSlice.addr_) + srcSlice.offset_);
        CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyNbi(sendChannel.handle, dst, src, srcSlice.size_, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWait(recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    // 写完之后做后同步告诉对面写完了
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecord(sendChannel.handle, NOTIFY_IDX_FIN_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWait(recvChannel.handle, NOTIFY_IDX_FIN_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommFlush()));
    return HCCL_SUCCESS;
}


} // END