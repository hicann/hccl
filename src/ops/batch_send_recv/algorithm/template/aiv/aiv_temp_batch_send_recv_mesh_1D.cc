/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>

#include "aiv_temp_batch_send_recv_mesh_1D.h"
#include "channel_request.h"
#include "log.h"

namespace ops_hccl {

AivTempBatchSendRecvMesh1D::AivTempBatchSendRecvMesh1D(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : AivAlgTemplateBase(param, rankId, subCommRanks)
{}

HcclResult AivTempBatchSendRecvMesh1D::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_PTR_NULL(topoInfo);
    // 所有同步都在 AIV kernel 内用 flag 完成，不需要额外 thread/notify
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    // 只对本次 item 实际涉及的对端建链：BSR 每次调用的对端集合不同，按全体 rank 建链会把"本次根本不
    // 通信"的链也建出来，增量建链的收益正来自这里。
    // subCommRanks_ 必须保持全体 rank：它决定模板的 tempRankSize_，而 kernel 的地址表是全量定长数组、
    // 下标是全局 userRank，收窄会让 GmIn/GmOut 取到错位地址。
    std::vector<std::vector<u32>> activeRanks(1);
    activeRanks[0].push_back(topoInfo->userRank);
    const HcclSendRecvItem* items = param.batchSendRecvDataDes.sendRecvItemsPtr;
    const u32 itemNum = param.batchSendRecvDataDes.itemNum;
    if (itemNum > 0) {
        CHK_PTR_NULL(items);
    }
    for (u32 i = 0; i < itemNum; i++) {
        // 与 BuildTaskDeques 同一判据：count==0 的 item 不进任务表，也就不该占一条链
        if (items[i].count == 0) {
            continue;
        }
        const u32 remoteRank = static_cast<u32>(items[i].remoteRank);
        if (remoteRank == topoInfo->userRank) {
            continue; // 自收发是 GM 直拷，不经链
        }
        CHK_PRT_RET(
            remoteRank >= topoInfo->userRankSize,
            HCCL_ERROR(
                "[AivTempBatchSendRecvMesh1D][CalcRes] invalid remoteRank[%u], myRank[%u], rankSize[%u].", remoteRank,
                topoInfo->userRank, topoInfo->userRankSize),
            HCCL_E_PARA);
        if (std::find(activeRanks[0].begin(), activeRanks[0].end(), remoteRank) == activeRanks[0].end()) {
            activeRanks[0].push_back(remoteRank);
        }
    }
    HCCL_INFO(
        "[AivTempBatchSendRecvMesh1D][CalcRes] itemNum[%u], activeRankNum[%zu], rankSize[%u].", itemNum,
        activeRanks[0].size(), topoInfo->userRankSize);

    std::vector<HcclChannelDesc> level0Channels;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMeshClosMultiJetty(comm, param, topoInfo, activeRanks, myChannelDescs, true));
        // CLOS 建链的协议由 CalcChannelRequestMeshClosMultiJetty 内硬编码：AIV 恒为 UB_MEM，
        // 匹配不到直接报错。所以这里过滤是防御性的，正常不会丢弃任何一条。
        for (auto& channel : myChannelDescs) {
            if (channel.channelProtocol == COMM_PROTOCOL_UB_MEM) {
                level0Channels.push_back(channel);
            }
        }
        HCCL_DEBUG(
            "[AivTempBatchSendRecvMesh1D][CalcRes] get MESH_1D_CLOS channel success, size[%zu].",
            level0Channels.size());
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, activeRanks, level0Channels));
        HCCL_DEBUG(
            "[AivTempBatchSendRecvMesh1D][CalcRes] get MESH_1D channel success, size[%zu].", level0Channels.size());
    }
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}

HcclResult AivTempBatchSendRecvMesh1D::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void)dataSize;
    // 收发各占一半核，至少需要 2 个核；核预算不足属资源问题而非入参错误，故与其它 AIV 模板一致报 NOT_SUPPORT。
    CHK_PRT_RET(
        numBlocksLimit < 2,
        HCCL_ERROR("[AivTempBatchSendRecvMesh1D][CalNumBlocks] numBlocksLimit[%u] is less than 2.", numBlocksLimit),
        HcclResult::HCCL_E_NOT_SUPPORT);

    numBlocks = numBlocksLimit;
    HCCL_INFO(
        "[AivTempBatchSendRecvMesh1D][CalNumBlocks] actually use block num[%u], limit[%u].", numBlocks, numBlocksLimit);
    return HCCL_SUCCESS;
}

HcclResult AivTempBatchSendRecvMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    // 不再调基类的 IncSliceId()：本算子的 sliceId 承载 tag 值（见下面 roundTag），由 executor 通过
    // SetRoundIdx 给出轮次、这里换算，各 rank 由同一套切片规则算出同一个值。

    CHK_PRT_RET(
        extraArgs_.itemNum > AIV_BSR_ITEM_MAX,
        HCCL_ERROR(
            "[AivTempBatchSendRecvMesh1D][KernelRun] itemNum[%llu] exceeds limit[%u].", extraArgs_.itemNum,
            AIV_BSR_ITEM_MAX),
        HcclResult::HCCL_E_PARA);
    // tag 在 [AIV_BSR_TAG_BASE + 1, AIV_BSR_TAG_MAX] 内随轮次推进，越界回绕（其它算子是到 1000 回绕），
    // 回绕后最小值仍是 1025，不会掉回其它算子的 tag 值域（隔离理由见头文件）。实际到不了环长：单轮最坏
    // 512 条 item，轮次数受 cclBuffer 容量约束在几十量级，所以这里只回绕、不报错。
    const u32 roundTag = AIV_BSR_TAG_BASE + 1U + roundIdx_ % AIV_BSR_TAG_SPAN;

    AivOpArgs aivBsrArgs;
    aivBsrArgs.cmdType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    aivBsrArgs.input = 0;
    aivBsrArgs.output = 0;
    aivBsrArgs.rank = myRank_;
    aivBsrArgs.rankSize = tempRankSize_;
    aivBsrArgs.count = tempAlgParams.buffInfo.hcclBuff.size;
    // DFX 上报本轮真正发出的字节数（totalDataCount 不参与 kernel launch，只喂 taskInfo.size）。
    // 回退不在本处，而在 AIV 公共 DFX 路径的消费侧：hccl_aiv_utils.cc 组 taskInfo 时取
    // (totalDataCount != 0) ? totalDataCount : count（与 AlltoAllV 共用）。故算出 0 的轮次
    // （只含 RECV / SELF_RECV）上报的仍是整个 cclBuffer 大小，与不设该字段时行为相同。
    aivBsrArgs.totalDataCount = dfxTotalBytes_;
    // kernel 按字节搬运，固定用 INT8 作为选核 key：device 侧只生成并登记 aiv_batchsendrecv_int8_t
    // 一个入口（op.h 与 aiv_kernel_def.h 必须严格对应），GetFuncKey 无 dtype 维度回退。
    aivBsrArgs.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    // sliceId 在本算子承载 tag 值（见上面的 roundTag），kernel 直接拿它当 tag：不读也不写通信域里那个
    // 全 rank 共享的 tag 计数器，于是各 rank 的 tag 只取决于自己本轮是第几轮，天然一致。
    // 其它算子的 sliceId 语义是片号，这里不同，不要照抄。
    aivBsrArgs.sliceId = roundTag;
    aivBsrArgs.buffersIn = templateResource.aivCommInfoPtr;
    aivBsrArgs.stream = param.stream;
    aivBsrArgs.hcclComm = static_cast<HcclComm>(param.hcclComm);
    aivBsrArgs.isOpBase = (param.opMode == OpMode::OPBASE);

    CHK_RET(CalNumBlocks(aivBsrArgs.numBlocks, 0, param.numBlocksLimit));
    // 任务表经 BSR 专用入口下发，不写进 AivOpArgs：它由本模板持有，launch 期间有效
    CHK_RET(ExecuteKernelLaunchBsr(aivBsrArgs, extraArgs_));

    HCCL_INFO(
        "[AivTempBatchSendRecvMesh1D][KernelRun] finished, round[%u], tag[%u], itemNum[%llu], numBlocks[%u].",
        roundIdx_ + 1, aivBsrArgs.sliceId, extraArgs_.itemNum, aivBsrArgs.numBlocks);
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
