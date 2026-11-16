/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_scatter_rhd_dpu_inter_node.h"
#include "dpu_alg_nhr_opt_wrapper.h"

namespace ops_hccl {
InsTempScatterRhdDpuInterNode::InsTempScatterRhdDpuInterNode(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempScatterRhdDpuInterNode::~InsTempScatterRhdDpuInterNode() {}

u64 InsTempScatterRhdDpuInterNode::GetThreadNum() const
{
    u64 threadNum = 1;
    return threadNum;
}

void InsTempScatterRhdDpuInterNode::SetRoot(u32 root)
{
    HCCL_INFO("[InsTempScatterRhdDpuInterNode][SetRoot] myRank_ [%u], set root_ [%u] ", myRank_, root);
    root_ = root;
}

HcclResult InsTempScatterRhdDpuInterNode::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level0Channels;
    // host RoCE 专用建链（只认 ROCE + HOST；复用通用建链会 core dump）。连接集取 Mesh 全互联
    // （同 InsTempReduceScatterMesh1dDpuInter 生产先例）。不做按 root 的步骤对端收敛：资源按
    // (comm, algTag, engine) 复用（root 不在键内），per-root 通道集在后续换 root 的 op 上缺链
    // （E_INTERNAL + 对端挂死）。收敛的正解是 RhdCalcConnectRanks 全 root 并集（AICPU 侧
    // InsTempScatterRHD::CalcRes 同款，≈2·log p+1 对端）+ 公共层导出带自定义对端列表的
    // HostRoce 建链变体（CalcHighestHostRoceChannels 现为 static，见 channel.cc）——待该
    // 公共层入口落地后切换，届时多建约 d2-1-2·log(d2) 条链消除（d2≥64 规模收益显著）
    CHK_RET(CalcChannelRequestMesh1DHighestHostRoce(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}

u64 InsTempScatterRhdDpuInterNode::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

HcclResult InsTempScatterRhdDpuInterNode::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempScatterRhdDpuInterNode] Run Start");

    threadNum_ = subCommRanks_.size();
    count_ = tempAlgParams.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    SetRoot(tempAlgParams.root);

    // 防御校验：非法入参报 HCCL_E_NOT_SUPPORT（AICPU 侧前置，快速失败）
    u32 rootAlgRank = 0;
    CHK_RET(GetAlgRank(root_, subCommRanks_[0], rootAlgRank));
    CHK_PRT_RET(
        !RhdScatterParamValid(templateRankSize_, rootAlgRank),
        HCCL_ERROR(
            "[InsTempScatterRhdDpuInterNode][KernelRun] invalid params: rankSize[%u] root_[%u] rootAlgRank[%u].",
            templateRankSize_, root_, rootAlgRank),
        HCCL_E_NOT_SUPPORT);

    HCCL_INFO(
        "[InsTempScatterRhdDpuInterNode] queNum_ = [%u], threads size = [%zu]", threadNum_,
        templateResource.threads.size());

    if (templateResource.threads.size() < 1) {
        HCCL_ERROR("[InsTempScatterRhdDpuInterNode] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    // 非 2^N 多级流输入桥：置换入队于 eager 舞步之前，由其排空（暂存借用见 RearrangeToSigma）
    CHK_RET(RearrangeToSigma(param, tempAlgParams, templateResource.threads));

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempScatterRhdDpuInterNode] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    // eager 窗口内的失败路径统一走单出口，确保 BatchModeStart 无条件恢复——
    // 否则 algTag 停留在 eager 模式，后续同 tag 算子的 batch 语义被破坏
    HcclResult eagerRet = HCCL_SUCCESS;
    do {
        if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
            HCCL_ERROR("[InsTempScatterRhdDpuInterNode] HcommThreadSynchronize failed");
            eagerRet = HCCL_E_INTERNAL;
            break;
        }

        // 单 request 覆盖全部步骤：Scatter 为搬运型无步间交替，与既有 NHR DPU 模式一致
        DPURunInfo dpuRunInfo;
        dpuRunInfo.execTimeout = param.opConfig.execTimeout;
        dpuRunInfo.templateName = "InsTempScatterRhdDpuInterNode";
        dpuRunInfo.tempAlgParams = tempAlgParams;
        // sequence 执行器不赋 dataType（默认 INT8），序列化前按 param 口径回填，device 侧 count 折算不失准
        dpuRunInfo.tempAlgParams.dataType = param.DataDes.dataType;
        dpuRunInfo.channels = templateResource.channels;
        dpuRunInfo.myRank = myRank_;
        dpuRunInfo.subCommRanks = subCommRanks_;
        auto dpuRunInfoSeqData = dpuRunInfo.Serialize();

        u32 sendMsgId = 0;
        if (HcommSendRequest(
                reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr), param.algTag,
                static_cast<void*>(dpuRunInfoSeqData.data()), dpuRunInfoSeqData.size(), &sendMsgId)
            != 0) {
            HCCL_ERROR("[InsTempScatterRhdDpuInterNode] HcommSendRequest failed");
            eagerRet = HCCL_E_INTERNAL;
            break;
        }
        HCCL_INFO("[InsTempScatterRhdDpuInterNode] HcommSendRequest run over, sendMsgId[%u]", sendMsgId);

        // 等待DPU数据传输，然后回写结果回来
        void* recvData = nullptr;
        u32 recvMsgId = 0;
        if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId)
            != 0) {
            HCCL_ERROR("[InsTempScatterRhdDpuInterNode] HcommWaitResponse failed");
            eagerRet = HCCL_E_INTERNAL;
            break;
        }
        HCCL_INFO("[InsTempScatterRhdDpuInterNode] HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

        if (recvMsgId != sendMsgId) {
            HCCL_ERROR(
                "[InsTempScatterRhdDpuInterNode] recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
            eagerRet = HCCL_E_INTERNAL;
            break;
        }
    } while (false);

    // 将执行模式转换回到batch（无条件——eager 窗口内任一失败也须恢复）
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempScatterRhdDpuInterNode] failed set batch mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }
    CHK_RET(eagerRet);

    // 融合直通（fusedWithNextAgPhase，载体 localCopyFlag bit1）：跳过 PostLocalCopy——
    // 代表的 σ 双份布局留在 ccl 里给相邻 AG 续跑；不跳则被合并方读到未写入的 ccl
    if (!RhdFusedWithNextAg(tempAlgParams)) {
        CHK_RET(PostLocalCopy(tempAlgParams, templateResource));
    }

    HCCL_INFO("[InsTempScatterRhdDpuInterNode] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterRhdDpuInterNode::RearrangeToSigma(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const
{
    // 多级流：上游按自然布局写 ccl（块内 algRank j 在 j·sliceSize），RHD 交换区段连续性要求
    // σ 布局。仅交换 root（唯一持有数据方）需要置换；2^N 时 σ=恒等零开销。
    const bool isPowerOfTwo = (templateRankSize_ & (templateRankSize_ - 1)) == 0;
    if (isPowerOfTwo || count_ == 0 || u32(myRank_) != root_
        || tempAlgParams.buffInfo.inBuffType != BufferType::HCCL_BUFFER) {
        return HCCL_SUCCESS;
    }

    // σ 环移位暂存借本 rank 接收缓冲尾段（param.outputPtr + [每rank总长−unit, 总长)），同设备侧
    // 模板：recvbuf 全 rank 有效；借首份 [0,unit) 在多 loop 下不安全（loop≥1 时是前序 chunk
    // 已写出的用户输出，暂存脏写后无人回写，非平凡 σ×多 loop 时行 root 的 chunk0 静默损坏）；
    // 尾段由最后一个 loop 的终态 post-copy 覆盖，对最终层与非最终层通用，与 ccl 置换区无重叠。
    CHK_PTR_NULL(param.outputPtr);
    // 变长契约：repeat 步长用 totalUnitBytes（Σ unitSizes）——含尾块时与 rankSize*sliceSize
    // 不等，按均匀步长会使 r≥1 的 block 错位（RunRhdSteps 同口径）
    std::vector<u64> unitSizes;
    CHK_RET(RhdBuildUnitSizes(templateRankSize_, tempAlgParams.allRankSliceSize, unitSizes));
    u64 totalUnitBytes = 0;
    for (u64 sz : unitSizes) {
        totalUnitBytes += sz;
    }
    // 环搬移不变量：移动区成员（σ(j)≠j，恒 ⊂ [0, rankSize-1)）须均匀——尾块（末 algRank）
    // 是 σ 不动点永不入环、尺寸任意；移动区非均匀属契约外形态（自然/σ 两分区边界错位，
    // 环内原地搬移不适用），显式拒绝
    for (u32 j = 0; j < templateRankSize_; j++) {
        if (RhdUnitStartOfRank(templateRankSize_, j) != j
            && tempAlgParams.allRankSliceSize[j] != tempAlgParams.sliceSize) {
            HCCL_ERROR(
                "[InsTempScatterRhdDpuInterNode][RearrangeToSigma] non-uniform mover algRank[%u] size[%llu] "
                "sliceSize[%llu].",
                j, tempAlgParams.allRankSliceSize[j], tempAlgParams.sliceSize);
            return HcclResult::HCCL_E_PARA;
        }
    }
    const u64 perRankTotalBytes = static_cast<u64>(param.DataDes.count) * dataTypeSize_;
    const u64 tmpOffset = perRankTotalBytes > tempAlgParams.sliceSize ? perRankTotalBytes - tempAlgParams.sliceSize : 0;
    DataSlice tmpSlice = DataSlice(param.outputPtr, tmpOffset, tempAlgParams.sliceSize, count_);
    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        // done 逐 block 重置：r=0 的环发现会把全部位置标记，带入 r≥1 会使后续 block 一次拷贝
        // 都不做、保持自然布局，RHD 按 σ 区段搬运 → 静默数据错位
        std::vector<bool> done(templateRankSize_, false);
        const u64 blockBase = tempAlgParams.buffInfo.hcclBuffBaseOff + r * totalUnitBytes;
        for (u32 j = 0; j < templateRankSize_; j++) {
            if (done[j]) {
                continue;
            }
            // σ 环分解：c_0=j → c_1=σ(j) → … → c_0；目标 new[c_{k+1}] = old[c_k]
            std::vector<u32> cyc;
            u32 cur = j;
            while (!done[cur]) {
                done[cur] = true;
                cyc.push_back(cur);
                cur = RhdUnitStartOfRank(templateRankSize_, cur);
            }
            if (cyc.size() < 2) {
                continue; // 不动点
            }
            DataSlice lastSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr, blockBase + cyc.back() * tempAlgParams.sliceSize,
                tempAlgParams.sliceSize, count_);
            DataSlice firstSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr, blockBase + cyc.front() * tempAlgParams.sliceSize,
                tempAlgParams.sliceSize, count_);
            // 暂存 old[c_last]（其归宿是 c_0，但反向移位会先覆盖它）
            CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), lastSlice, tmpSlice)));
            // 反向移位：c_k ← c_{k-1}（k = m-1..1，读侧恒为未被覆盖的原始数据）
            for (u32 k = cyc.size() - 1; k >= 1; k--) {
                DataSlice srcSlice = DataSlice(
                    tempAlgParams.buffInfo.hcclBuff.addr, blockBase + cyc[k - 1] * tempAlgParams.sliceSize,
                    tempAlgParams.sliceSize, count_);
                DataSlice dstSlice = DataSlice(
                    tempAlgParams.buffInfo.hcclBuff.addr, blockBase + cyc[k] * tempAlgParams.sliceSize,
                    tempAlgParams.sliceSize, count_);
                CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), srcSlice, dstSlice)));
            }
            // c_0 ← 暂存（old[c_last]）
            CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), tmpSlice, firstSlice)));
        }
    }
    HCCL_INFO(
        "[InsTempScatterRhdDpuInterNode][RearrangeToSigma] rankSize[%u] repeats[%llu] natural layout rearranged "
        "to sigma.",
        templateRankSize_, tempAlgParams.repeatNum);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterRhdDpuInterNode::DPUKernelRun(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>>& subCommRanks)
{
#ifndef AICPU_COMPILE
    myRank_ = myRank;
    templateRankSize_ = subCommRanks[0].size();
    subCommRanks_ = subCommRanks;
    CHK_RET(RunRhdSteps(channels, tempAlgParams));
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterRhdDpuInterNode::PostLocalCopy(
    const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    if (tempAlgParams.buffInfo.outputPtr == nullptr) {
        return HcclResult::HCCL_SUCCESS;
    }

    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    // 变长契约：own unit 的 packed 偏移（unitSizes 前缀和）与实际尺寸（末位可为尾块，
    // ≥ sliceSize）；均匀退化时与 σ(me)*sliceSize / sliceSize 逐字节一致
    std::vector<u64> unitSizes;
    CHK_RET(RhdBuildUnitSizes(templateRankSize_, tempAlgParams.allRankSliceSize, unitSizes));
    const u32 ownUnit = RhdUnitStartOfRank(templateRankSize_, myAlgRank);
    u64 ownOffset = 0;
    for (u32 u = 0; u < ownUnit; u++) {
        ownOffset += unitSizes[u];
    }
    const u64 ownSize = unitSizes[ownUnit];
    const u64 ownCount = dataTypeSize_ != 0 ? ownSize / dataTypeSize_ : 0;
    u64 totalUnitBytes = 0;
    for (u64 sz : unitSizes) {
        totalUnitBytes += sz;
    }
    for (u64 repeat = 0; repeat < tempAlgParams.repeatNum; ++repeat) {
        // σ 布局中的本 rank 槽位：CCL 基址偏移 + σ owner 槽位 + repeat 块偏移。
        // Broadcast 中间层的 hcclBuffBaseOff 对应 L0 分片基址，独立 Scatter 为 0。
        const u64 scratchOffset = repeat * totalUnitBytes + tempAlgParams.buffInfo.hcclBuffBaseOff + ownOffset;
        // 输出侧自然布局槽位：output 基址偏移 + 本 rank 槽位 + repeat 块偏移。
        // 独立 Scatter 的 outputSliceStride 为 0，公式退化为原 outBuffBaseOff 单槽位行为。
        const u64 outOffset = tempAlgParams.buffInfo.outBuffBaseOff + myAlgRank * tempAlgParams.outputSliceStride
                              + repeat * tempAlgParams.sliceSize;
        // 多级编排中 outputPtr 可能等于 CCL。源/目的完全重合时不做冗余 LocalCopy，
        // 避免同队列对同一段地址产生重叠写；独立 Scatter 输出用户 buffer，不会命中此分支。
        if (tempAlgParams.buffInfo.outputPtr == tempAlgParams.buffInfo.hcclBuff.addr && outOffset == scratchOffset) {
            continue;
        }
        DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr, scratchOffset, ownSize, ownCount);
        DataSlice dstSlice(tempAlgParams.buffInfo.outputPtr, outOffset, ownSize, ownCount);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterRhdDpuInterNode::RunRhdSteps(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const TemplateDataParams& tempAlgParams)
{
#ifndef AICPU_COMPILE
    SetRoot(tempAlgParams.root);
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    u32 rootAlgRank = 0;
    CHK_RET(GetAlgRank(root_, subCommRanks_[0], rootAlgRank));
    // 防御校验：非法入参报 HCCL_E_NOT_SUPPORT
    CHK_PRT_RET(
        !RhdScatterParamValid(templateRankSize_, rootAlgRank),
        HCCL_ERROR(
            "[InsTempScatterRhdDpuInterNode][RunRhdSteps] invalid params: rankSize[%u] root_[%u] rootAlgRank[%u].",
            templateRankSize_, root_, rootAlgRank),
        HCCL_E_NOT_SUPPORT);
    // count == 0：全组无数据量，跳过通信
    if (tempAlgParams.count == 0) {
        HCCL_INFO("[InsTempScatterRhdDpuInterNode][RunRhdSteps] count is 0, skip transfer.");
        return HCCL_SUCCESS;
    }
    std::vector<RhdScatterStep> steps;
    CHK_RET(GenRhdScatterSteps(templateRankSize_, rootAlgRank, myAlgRank, steps, RhdFusedWithNextAg(tempAlgParams)));
    // allRank* 契约 → unit 空间尺寸表（σ 反查；均匀分片为退化情形，尾块天然含于变长契约）
    std::vector<u64> unitSizes;
    CHK_RET(RhdBuildUnitSizes(templateRankSize_, tempAlgParams.allRankSliceSize, unitSizes));
    u64 totalUnitBytes = 0;
    for (u64 sz : unitSizes) {
        totalUnitBytes += sz;
    }
    const u64 dataTypeSize = HCCL_SIZE_TABLE[tempAlgParams.dataType];
    HCCL_INFO(
        "[InsTempScatterRhdDpuInterNode][RunRhdSteps] rootAlgRank[%u] myAlgRank[%u] rankSize[%u] steps[%zu]",
        rootAlgRank, myAlgRank, templateRankSize_, steps.size());
    for (u32 repeat = 0; repeat < tempAlgParams.repeatNum; repeat++) {
        for (const auto& step : steps) {
            // 合并段字节区间（变长契约经 unitSizes；均匀退化与 RhdSegToBytesUniform 输出一致）。
            // 空区段返回 HCCL_SUCCESS + {0,0}（2^N PRE 常态）；unitSizes 越界返回
            // HCCL_E_PARA 上抛（防御性，正常构造不触发）。
            RhdSegBytes seg;
            CHK_RET(RhdSegToBytesAllRank(step.segStartUnit, step.segEndUnit, unitSizes, seg));
            // 零字节区段跳过（判据=区段实际字节数，含空区段 [s,s)）：主体步区段横跨多 rank 的
            // unit（代表承载自身+被合并方两份额），按对端单一 rank 的 count==0 判断会误跳非空
            // 区段（被合并方 POST 阶段读到未写入的 ccl——静默数据错误）；seg.len==0 语义精确。
            if (seg.len == 0) {
                HCCL_INFO(
                    "[InsTempScatterRhdDpuInterNode][RunRhdSteps] segment[%u,%u) covers 0 byte, skip step[%u] "
                    "phase[%u].",
                    step.segStartUnit, step.segEndUnit, step.step, static_cast<u32>(step.phase));
                continue;
            }
            const u32 peerRank = subCommRanks_[0].at(step.peerAlgRank);
            auto peerIt = channels.find(peerRank);
            CHK_PRT_RET(
                peerIt == channels.end() || peerIt->second.empty(),
                HCCL_ERROR(
                    "[InsTempScatterRhdDpuInterNode][RunRhdSteps] remoteRank[%u] not found in channels", peerRank),
                HCCL_E_INTERNAL);
            const ChannelInfo& link = peerIt->second[0];
            CHK_PTR_NULL(link.remoteCclMem.addr);
            const u64 srcDstOffset = tempAlgParams.buffInfo.hcclBuffBaseOff + repeat * totalUnitBytes + seg.offset;
            const u64 segCount = dataTypeSize != 0 ? seg.len / dataTypeSize : 0;
            // 每步恰一个 DpuTransferCtx：单对端、单条连续大段（合并发送约束：每步每对端一条任务）
            DpuTransferCtx ctx;
            if (step.isSend) {
                ctx.txCh = &link;
                ctx.txSrcSlices.push_back(
                    DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcDstOffset, seg.len, segCount));
                ctx.txDstSlices.push_back(DataSlice(link.remoteCclMem.addr, srcDstOffset, seg.len, segCount));
            } else {
                ctx.rxCh = &link;
                ctx.rxSrcSlices.push_back(DataSlice(link.remoteCclMem.addr, srcDstOffset, seg.len, segCount));
                ctx.rxDstSlices.push_back(
                    DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcDstOffset, seg.len, segCount));
            }
            // 观测口径：每步每对端任务数 = 1
            HCCL_INFO(
                "[RHD-Scatter] step=%u phase=%u peer=%u seg=[%u,%u) bytes=[%llu,%llu)", step.step,
                static_cast<u32>(step.phase), step.peerAlgRank, step.segStartUnit, step.segEndUnit, seg.offset,
                seg.len);
            std::vector<DpuTransferCtx> pairs = {ctx};
            CHK_RET(DpuBatchTransfer(pairs));
        }
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempScatterRhdDpuInterNode", InsTempScatterRhdDpuInterNode);

} // namespace ops_hccl
