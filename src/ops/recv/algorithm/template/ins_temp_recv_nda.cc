/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_recv_nda.h"
#include "channel.h"

namespace ops_hccl {
InsTempRecvNda::InsTempRecvNda() {}

InsTempRecvNda::InsTempRecvNda(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempRecvNda::~InsTempRecvNda() {}

HcclResult InsTempRecvNda::CalcRes(
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

HcclResult InsTempRecvNda::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    dataType_ = param.DataDes.dataType;

    if (threadNum_ < 1) {
        HCCL_ERROR("[InsTempRecvNda] Rank [%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    CHK_PRT_RET(
        templateResource.channels.empty() || templateResource.channels.begin()->second.empty(),
        HCCL_ERROR("[InsTempRecvNda] Rank[%u], channel list is empty.", myRank_), HCCL_E_INTERNAL);
    const ChannelInfo& recvChannel = templateResource.channels.begin()->second[0];
    CHK_PRT_RET(
        !recvChannel.isValid || recvChannel.handle == 0 || recvChannel.protocol != CommProtocol::COMM_PROTOCOL_ROCE
            || recvChannel.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_HOST,
        HCCL_ERROR(
            "[InsTempRecvNda] invalid NDA channel, remoteRank[%u], isValid[%d], locationType[%d], protocol[%d].",
            param.sendRecvRemoteRank, recvChannel.isValid, recvChannel.locationType, recvChannel.protocol),
        HCCL_E_INTERNAL);
    sendRank_ = param.sendRecvRemoteRank;

    const u64 transferSize = tempAlgParams.sliceSize;
    const u64 transferCount = tempAlgParams.count;
    CHK_PRT_RET(
        recvChannel.remoteCclMem.addr == nullptr || recvChannel.remoteCclMem.size < transferSize,
        HCCL_ERROR(
            "[InsTempRecvNda] invalid remote CCL buffer, remoteRank[%u], addr[%p], size[%llu], transferSize[%llu].",
            sendRank_, recvChannel.remoteCclMem.addr, recvChannel.remoteCclMem.size, transferSize),
        HCCL_E_PARA);

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempRecvNda] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    // 等待对端直写本端CCL Buffer; RecvWriteForNda统一完成
    // Record ACK(本端buffer就绪)/Wait DATA_SIGNAL(对端写入完成)/Record FIN_ACK 单向握手
    DataSlice remoteCclSlice(recvChannel.remoteCclMem.addr, 0, transferSize, transferCount);
    DataSlice cclSlice(tempAlgParams.buffInfo.hcclBuff.addr, 0, transferSize, transferCount);
    SlicesList recvSlicesList{{remoteCclSlice}, {cclSlice}};
    DataInfo recvInfo{recvChannel, recvSlicesList};
    CHK_PRT_RET(
        RecvWriteForNda(recvInfo, templateResource.threads[0]) != HCCL_SUCCESS,
        HCCL_ERROR("[InsTempRecvNda] RecvWriteForNda failed, remoteRank[%u].", sendRank_), HcclResult::HCCL_E_INTERNAL);

    // 将本片数据从CCL Buffer搬运到output; 同线程队列保证搬运先于下一轮ACK执行
    DataSlice outputSlice(
        tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, transferSize, transferCount);
    CHK_RET(LocalCopy(templateResource.threads[0], cclSlice, outputSlice));

    // 将执行模式转换回到batch
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempRecvNda] failed set batch mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[InsTempRecvNda] Rank[%u] <- Rank[%u] size[%llu] Run End.", myRank_, sendRank_, transferSize);
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempRecvNda", InsTempRecvNda);
} // namespace ops_hccl
