/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "alg_template_base.h"
#include "ins_temp_recv_host_nic_dpu.h"

namespace ops_hccl {
InsTempRecvHostNicDpu::InsTempRecvHostNicDpu() {}

InsTempRecvHostNicDpu::InsTempRecvHostNicDpu(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempRecvHostNicDpu::~InsTempRecvHostNicDpu() {}

HcclResult InsTempRecvHostNicDpu::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // host网卡资源，不新增从流和对应Notify，只申请DPU上面
    resourceRequest.slaveThreadNum = 0; // 主thread可以通过接口传入的stream来做转换
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO(
        "[InsTempRecvHostNicDpu][CalcRes]slaveThreadNum[%u], notifyNumPerThread [%u], notifyNumOnMainThread [%u],"
        " level1Channels [%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread, resourceRequest.notifyNumOnMainThread,
        level1Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempRecvHostNicDpu::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void)inBufferType;
    (void)outBufferType;
    u64 scratchMultiple = subCommRanks_[0].size();
    HCCL_INFO("[InsTempRecvHostNicDpu][CalcScratchMultiple] templateScratchMultiplier [%llu]", scratchMultiple);
    return scratchMultiple;
}

HcclResult
InsTempRecvHostNicDpu::KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& res)
{
    threadNum_ = res.threads.size();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;

    if (threadNum_ < 1) {
        HCCL_ERROR("[InsTempRecvHostNicDpu] Rank [%d], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("InsTempRecvHostNicDpu::KernelRun failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    if (HcommThreadSynchronize(res.threads[0]) != 0) {
        HCCL_ERROR("InsTempRecvHostNicDpu::KernelRun HcommThreadSynchronize failed");
        return HCCL_E_INTERNAL;
    }

    DPURunInfo dpuRunInfo;
    dpuRunInfo.templateName = "InsTempRecvHostNicDpu";
    dpuRunInfo.tempAlgParams = tempAlgParams;
    dpuRunInfo.channels = res.channels;
    dpuRunInfo.myRank = myRank_;
    dpuRunInfo.subCommRanks = subCommRanks_;
    u32 sendMsgId = 0;
    auto dpuRunInfoSeqData = dpuRunInfo.Serialize();
    if (HcommSendRequest(
            reinterpret_cast<uint64_t>(res.npu2DpuShmemPtr), param.algTag, static_cast<void*>(dpuRunInfoSeqData.data()),
            dpuRunInfoSeqData.size(), &sendMsgId)
        != 0) {
        HCCL_ERROR("InsTempRecvHostNicDpu HcommRecvRequest failed");
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("InsTempRecvHostNicDpu HcommRecvRequest run over, sendMsgId[%u]", sendMsgId);

    // 等待DPU数据传输，然后回写结果回来
    void* recvData = nullptr;
    u32 recvMsgId = 0;
    if (HcommWaitResponse(reinterpret_cast<uint64_t>(res.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("InsTempRecvHostNicDpu HcommWaitResponse failed");
        return HCCL_E_INTERNAL;
    }

    // 将执行模式转换回到batch
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("InsTempRecvHostNicDpu failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("InsTempRecvHostNicDpu HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

    if (recvMsgId != sendMsgId) {
        HCCL_ERROR("InsTempRecvHostNicDpu recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[InsTempRecvHostNicDpu] Run End");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempRecvHostNicDpu::DPUKernelRun(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>>& subCommRanks)
{
#ifndef AICPU_COMPILE
    std::vector<u32> rankIds = subCommRanks[0];

    for (u32 rankIdx = 0; rankIdx < rankIds.size(); rankIdx++) {
        if (rankIdx == myRank) {
            continue;
        }
        uint64_t notifyNum = 2;
        uint64_t sizePerRound = 0;
        uint64_t offset = 0;

        uint64_t cclOutputSize = tempAlgParams.buffInfo.hcclBuff.size;
        uint64_t outputSize = tempAlgParams.buffInfo.outputSize;

        for (u64 sizeResidue = outputSize; sizeResidue > 0; sizeResidue -= sizePerRound) {
            // 前同步
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, channels.at(rankIdx)[0].handle, 0)));

            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(0, channels.at(rankIdx)[0].handle, 0, CUSTOM_TIMEOUT)));

            // 等待数据接收完成
            offset += sizePerRound;
            sizePerRound = (sizeResidue > cclOutputSize) ? cclOutputSize : sizeResidue;
            HCCL_DEBUG("rx async outputmem's offset[%llu], size[%llu]", offset, sizePerRound);
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(0, channels.at(rankIdx)[0].handle, 1, CUSTOM_TIMEOUT)));

            // 后同步，通知发送端数据接收完成
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(0, channels.at(rankIdx)[0].handle, notifyNum)));

            CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, channels.at(rankIdx)[0].handle)));
        }
    }
#endif
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
REGISTER_TEMPLATE_V2("InsTempRecvHostNicDpu", InsTempRecvHostNicDpu);
#endif
} // namespace ops_hccl
