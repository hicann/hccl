/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_batch_send_recv_nda.h"
#include "alg_data_trans_wrapper.h"
#include "alg_v2_template_register.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "hcomm_primitives_dl.h"

namespace ops_hccl {
InsTempBatchSendRecvNda::InsTempBatchSendRecvNda(
    const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

HcclResult InsTempBatchSendRecvNda::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    CHK_PRT_RET(
        tempAlgParams.opType != BatchSendRecvOpType::SEND && tempAlgParams.opType != BatchSendRecvOpType::RECV,
        HCCL_ERROR(
            "[InsTempBatchSendRecvNda][KernelRun] unsupported operation[%u]", static_cast<u32>(tempAlgParams.opType)),
        HCCL_E_PARA);
    auto it = templateResource.channels.find(param.sendRecvRemoteRank);
    CHK_PRT_RET(
        templateResource.threads.empty() || it == templateResource.channels.end() || it->second.empty(),
        HCCL_ERROR("[InsTempBatchSendRecvNda][KernelRun] missing thread/channel resources"), HCCL_E_PARA);
    const ChannelInfo& channel = it->second.front();
    CHK_PRT_RET(
        !channel.isValid || channel.handle == 0,
        HCCL_ERROR(
            "[InsTempBatchSendRecvNda][KernelRun] invalid channel, remoteRank[%u], isValid[%d]",
            param.sendRecvRemoteRank, channel.isValid),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        channel.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE
            && channel.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_HOST,
        HCCL_ERROR(
            "[InsTempBatchSendRecvNda][KernelRun] unsupported endpoint location[%u]",
            static_cast<u32>(channel.locationType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PTR_NULL(tempAlgParams.buffInfo.hcclBuff.addr);
    CHK_PTR_NULL(channel.remoteCclMem.addr);
    const u64 stride = tempAlgParams.inputSliceStride;
    CHK_PRT_RET(
        tempAlgParams.buffInfo.hcclBuff.size != channel.remoteCclMem.size || stride == 0
            || tempAlgParams.sliceSize > stride || stride > tempAlgParams.buffInfo.hcclBuff.size / 2
            || stride > channel.remoteCclMem.size / 2,
        HCCL_ERROR("[InsTempBatchSendRecvNda][KernelRun] invalid CCL slot or slice size"), HCCL_E_PARA);
    if (tempAlgParams.opType == BatchSendRecvOpType::SEND) {
        CHK_PTR_NULL(tempAlgParams.buffInfo.inputPtr);
    } else {
        CHK_PTR_NULL(tempAlgParams.buffInfo.outputPtr);
    }
    if (channel.locationType == ENDPOINT_LOC_TYPE_HOST) {
        CHK_PRT_RET(
            channel.protocol != COMM_PROTOCOL_ROCE,
            HCCL_ERROR("[InsTempBatchSendRecvNda][KernelRun] HOST channel must use RoCE"), HCCL_E_NOT_SUPPORT);
        CHK_PRT_RET(
            channel.notifyNum <= NOTIFY_IDX_FIN_ACK,
            HCCL_ERROR("[InsTempBatchSendRecvNda][KernelRun] HOST write requires three notifications"), HCCL_E_PARA);
    }
    const ThreadHandle thread = templateResource.threads.front();
    if (tempAlgParams.opType == BatchSendRecvOpType::SEND) {
        DataSlice inputSlice(
            tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.sliceSize);
        DataSlice localTxSlice(tempAlgParams.buffInfo.hcclBuff.addr, 0, tempAlgParams.sliceSize);
        CHK_RET(LocalCopy(thread, inputSlice, localTxSlice));
    }
    if (channel.locationType == ENDPOINT_LOC_TYPE_HOST) {
        return RunNdaWriteTransfer(param, tempAlgParams, channel, thread);
    }
    return RunIntraNode(tempAlgParams, channel, thread);
}

HcclResult InsTempBatchSendRecvNda::RunNdaWriteTransfer(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const ChannelInfo& channel,
    ThreadHandle thread) const
{
    (void)param;
    if (tempAlgParams.opType == BatchSendRecvOpType::SEND) {
        // SendWriteForNda统一完成 WaitACK(对端RX槽就绪) -> RoCE WRITE -> RecordDATA_SIGNAL ->
        // WaitFIN_ACK, 与InsTempSendNda语义一致; FIN_ACK保障TX槽复用。
        DataSlice localTxSlice(tempAlgParams.buffInfo.hcclBuff.addr, 0, tempAlgParams.sliceSize);
        DataSlice remoteRxSlice(channel.remoteCclMem.addr, channel.remoteCclMem.size / 2, tempAlgParams.sliceSize);
        SlicesList sendSlicesList{{localTxSlice}, {remoteRxSlice}};
        DataInfo sendInfo{channel, sendSlicesList};
        return SendWriteForNda(sendInfo, thread);
    }
    // Preserve the serial protocol: ACK -> DATA_SIGNAL wait -> output copy -> FIN_ACK.
    SlicesList recvSlicesList({}, {});
    DataInfo recvInfo{channel, recvSlicesList};
    CHK_RET(RecvWrite(recvInfo, thread));
    DataSlice localRxSlice(
        tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.hcclBuff.size / 2, tempAlgParams.sliceSize);
    DataSlice outputSlice(
        tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, tempAlgParams.sliceSize);
    CHK_RET(LocalCopy(thread, localRxSlice, outputSlice));
    return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_FIN_ACK));
}

HcclResult InsTempBatchSendRecvNda::RunIntraNode(
    const TemplateDataParams& tempAlgParams, const ChannelInfo& channel, ThreadHandle thread) const
{
    // DEVICE and HOST sends share the same TX half at offset zero.
    if (tempAlgParams.opType == BatchSendRecvOpType::SEND) {
        DataInfo info(channel, SlicesList({}, {}), HCCL_DATA_TYPE_INT8);
        return SendRead(info, thread);
    }
    DataSlice remoteTxSlice(channel.remoteCclMem.addr, 0, tempAlgParams.sliceSize);
    DataSlice outputSlice(
        tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, tempAlgParams.sliceSize);
    DataInfo info(channel, SlicesList({remoteTxSlice}, {outputSlice}), HCCL_DATA_TYPE_INT8);
    return RecvRead(info, thread);
}

REGISTER_TEMPLATE_V2("InsTempBatchSendRecvNda", InsTempBatchSendRecvNda);
} // namespace ops_hccl
