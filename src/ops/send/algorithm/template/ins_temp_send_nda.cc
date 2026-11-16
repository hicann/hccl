/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_send_nda.h"
#include "channel.h"

namespace ops_hccl {
InsTempSendNda::InsTempSendNda() {}

InsTempSendNda::InsTempSendNda(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempSendNda::~InsTempSendNda() {}

HcclResult InsTempSendNda::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // host网卡资源，不新增从流和对应Notify
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;
    std::vector<HcclChannelDesc> level0Channels;
    static_cast<void>(topoInfo);
    CHK_RET(CreateChannelRequestByRankId(comm, param, myRank_, param.sendRecvRemoteRank, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}

HcclResult InsTempSendNda::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    dataType_ = param.DataDes.dataType;

    if (threadNum_ < 1) {
        HCCL_ERROR("[InsTempSendNda] Rank [%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    CHK_PRT_RET(
        templateResource.channels.empty() || templateResource.channels.begin()->second.empty(),
        HCCL_ERROR("[InsTempSendNda] Rank[%u], channel list is empty.", myRank_), HCCL_E_INTERNAL);
    const ChannelInfo& sendChannel = templateResource.channels.begin()->second[0];
    CHK_PRT_RET(
        !sendChannel.isValid || sendChannel.handle == 0 || sendChannel.protocol != CommProtocol::COMM_PROTOCOL_ROCE
            || sendChannel.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_HOST,
        HCCL_ERROR(
            "[InsTempSendNda] invalid NDA channel, remoteRank[%u], isValid[%d], locationType[%d], protocol[%d].",
            param.sendRecvRemoteRank, sendChannel.isValid, sendChannel.locationType, sendChannel.protocol),
        HCCL_E_INTERNAL);
    recvRank_ = param.sendRecvRemoteRank;

    const u64 transferSize = tempAlgParams.sliceSize;
    const u64 transferCount = tempAlgParams.count;
    CHK_PRT_RET(
        sendChannel.remoteCclMem.addr == nullptr || sendChannel.remoteCclMem.size < transferSize,
        HCCL_ERROR(
            "[InsTempSendNda] invalid remote CCL buffer, remoteRank[%u], addr[%p], size[%llu], transferSize[%llu].",
            recvRank_, sendChannel.remoteCclMem.addr, sendChannel.remoteCclMem.size, transferSize),
        HCCL_E_PARA);

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempSendNda] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    // 先将本片数据staging到本地CCL Buffer; 同线程队列保证staging先于通信执行
    DataSlice srcSlice(
        tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff, transferSize, transferCount);
    DataSlice localCclSlice(tempAlgParams.buffInfo.hcclBuff.addr, 0, transferSize, transferCount);
    HcclResult ret = LocalCopy(templateResource.threads[0], srcSlice, localCclSlice);
    if (ret == HCCL_SUCCESS) {
        // 经Host RoCE直写对端CCL Buffer; SendWriteForNda统一完成
        // Wait ACK(对端buffer就绪)/DATA_SIGNAL(写入完成)/FIN_ACK(对端已消费)单向握手
        DataSlice remoteCclSlice(sendChannel.remoteCclMem.addr, 0, transferSize, transferCount);
        SlicesList sendSlicesList{{localCclSlice}, {remoteCclSlice}};
        DataInfo sendInfo{sendChannel, sendSlicesList};
        ret = SendWriteForNda(sendInfo, templateResource.threads[0]);
    }
    // 将执行模式转换回到batch; 成功与失败路径均恢复，避免执行引擎停留在eager模式
    const HcclResult restoreRet = static_cast<HcclResult>(HcommBatchModeStart(param.algTag));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS || restoreRet != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsTempSendNda] NDA send failed, runRet[%d], restoreRet[%d], remoteRank[%u].", ret, restoreRet,
            recvRank_),
        ret != HCCL_SUCCESS ? ret : restoreRet);

    HCCL_INFO("[InsTempSendNda] Rank[%u] -> Rank[%u] size[%llu] Run End.", myRank_, recvRank_, transferSize);
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempSendNda", InsTempSendNda);
} // namespace ops_hccl
