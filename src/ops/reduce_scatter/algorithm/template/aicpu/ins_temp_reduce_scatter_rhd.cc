/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_rhd.h"

namespace ops_hccl {

std::vector<CostModelParam> InsTempReduceScatterRHD::CalcCostCoeff(CalcCostCoeffParam param)
{
    // preferred-only：不标定，返回固定占位系数（对齐 Dpu 家族先例）。不可返回空——空系数会使算法在
    // cost 表注册阶段即被跳过，显式 HCCL_ALGO 配置也无法选中（显式配置下 RHD 为唯一候选、
    // 系数不实际参与比价）；设备侧选路管线同样调用本函数，占位返回无 AICPU_COMPILE 分裂、零
    // uncalibrated WARNING。
    (void)param;
    std::vector<CostModelParam> params;
    params.push_back({0.0f, 0.0f, 1.0f, 0.0f});
    HCCL_DEBUG("[InsTempReduceScatterRHD][CalcCostCoeff] placeholder cost (preferred-only, no calibration).");
    return params;
}

InsTempReduceScatterRHD::InsTempReduceScatterRHD(
    const OpParam& param, const u32 rankId, // 传通信域的u32，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempReduceScatterRHD::~InsTempReduceScatterRHD() {}

HcclResult InsTempReduceScatterRHD::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // 通道请求对齐 InsTempReduceScatterNHR / HD 模板：UBX 拓扑走 MultiJettyUbx（每对 rank 多 jetty
    // 连接），其余拓扑复用 NHR 全连接并过滤 ROCE 通道（本模板为 device 链路模板）
    std::vector<HcclChannelDesc> channels;
    bool isUBX = topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix;
    if (isUBX) {
        CHK_RET(CalcChannelRequestNhrMultiJettyUbx(comm, param, topoInfo, subCommRanks_, channels));
    } else {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, myChannelDescs));
        for (auto channel : myChannelDescs) {
            if (channel.channelProtocol != CommProtocol::COMM_PROTOCOL_ROCE) {
                channels.push_back(channel);
            }
        }
    }
    resourceRequest.channels.push_back(channels);
    u32 channelsPerRank = CalcChannelsPerRank(channels);
    channelsPerRank_ = channelsPerRank;
    // 非 UBX 分支过滤 ROCE 通道后可能为空，channelsPerRank 为 0 时 GetRes 的 threadNum-1 会 u32 下溢
    CHK_PRT_RET(
        channelsPerRank_ == 0, HCCL_ERROR("[InsTempReduceScatterRHD][CalcRes] channelsPerRank_ is 0."),
        HcclResult::HCCL_E_INTERNAL);
    if (isUBX && channelsPerRank_ > MAX_JETTY_NUM) {
        HCCL_ERROR(
            "[InsTempReduceScatterRHD][CalcRes] channelsPerRank_ %u is greater than MAX_JETTY_NUM %u.",
            channelsPerRank_, MAX_JETTY_NUM);
        // jetty 越界后 GetRes 按 channelsPerRank_ 建线程/notify 仍会返回成功并继续运行，属硬失败
        // 口径（对齐上方 channelsPerRank_ == 0 的 CHK_PRT_RET 拦截）
        return HcclResult::HCCL_E_INTERNAL;
    } else {
        HCCL_DEBUG("[InsTempReduceScatterRHD][CalcRes] channelsPerRank_ is %u.", channelsPerRank_);
    }
    GetRes(resourceRequest);
    HCCL_INFO(
        "[InsTempReduceScatterRHD][CalcRes] channelsPerRank_: [%u], slaveThreadNum: [%u], "
        "notifyNumOnMainThread: [%u].",
        channelsPerRank_, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterRHD::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 threadNum = 1 * channelsPerRank_;
    CHK_PRT_RET(
        threadNum == 0, HCCL_ERROR("[InsTempReduceScatterRHD][GetRes] threadNum is 0, channelsPerRank_ invalid."),
        HcclResult::HCCL_E_INTERNAL);
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    HCCL_INFO(
        "[InsTempReduceScatterRHD][GetRes] channelsPerRank_: [%u], slaveThreadNum: [%u].", channelsPerRank_,
        resourceRequest.slaveThreadNum);
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterRHD::GetThreadNum() const { return 1 * channelsPerRank_; }

HcclResult InsTempReduceScatterRHD::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempReduceScatterRHD] KernelRun start");

    tempAlgParams_ = tempAlgParams;
    channels_ = templateResource.channels;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(
        dataTypeSize_ == 0, HCCL_ERROR("[InsTempReduceScatterRHD][KernelRun] dataTypeSize is 0."),
        HcclResult::HCCL_E_INTERNAL);

    // 边界守卫：空数据直接成功（executor 保证 count > 0，此处防御）
    if (tempAlgParams_.count == 0) {
        HCCL_INFO("[InsTempReduceScatterRHD][KernelRun] count is 0, nothing to do.");
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        templateResource.threads.empty(), HCCL_ERROR("[InsTempReduceScatterRHD][KernelRun] empty threads"),
        HcclResult::HCCL_E_INTERNAL);
    // 校验先于单 rank 分支：SingleRankCopy 内 GetAlgRank 消费 subCommRanks_[0]，
    // 空表越界属 UB 而非可诊断错误（对齐 DPU 模板 HandleSingleRank 内部校验）
    CHK_PRT_RET(
        subCommRanks_.empty() || subCommRanks_[0].size() != templateRankSize_,
        HCCL_ERROR("[InsTempReduceScatterRHD][KernelRun] subCommRanks mismatch, tempRankSize[%u].", templateRankSize_),
        HcclResult::HCCL_E_INTERNAL);
    // 单 rank 防御分支：selector 层 SingleRankProc 已拦截，此处直拷 IN -> OUT（单线程，不走通道切分）
    if (templateRankSize_ == 1) {
        return HandleSingleRank(templateResource.threads);
    }
    CHK_PRT_RET(
        channels_.empty(), HCCL_ERROR("[InsTempReduceScatterRHD][KernelRun] channels is empty."),
        HcclResult::HCCL_E_INTERNAL);
    for (const auto& channelsByRank : channels_) {
        CHK_PRT_RET(
            channelsByRank.second.empty(),
            HCCL_ERROR(
                "[InsTempReduceScatterRHD][KernelRun] channel list of remote rank[%u] is empty.", channelsByRank.first),
            HcclResult::HCCL_E_INTERNAL);
    }

    // 步骤表预计算（单一事实来源：template_utils RHD 函数族，模板不自研步骤数学）
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));
    CHK_RET(CalcRhdPart1Info(templateRankSize_, myAlgIdx, INVALID_VALUE_RANKID, part1Info_));
    // σ unit 偏移构建（tailSize != sliceSize 时末 unit 更大；均匀退化与原行为逐字节一致）
    CHK_RET(BuildUnitOffsets());
    const bool isMergedParty = part1Info_.inPart1 && !part1Info_.isRepresentative; // 被合并方退出主体
    if (!isMergedParty) {
        CHK_RET(GenRhdReduceScatterSteps(part1Info_, rsSteps_));
    }
    // RS 族前后处理模式：pre = FULL（代表须收被合并方全量贡献）、post = SHARE（只需自己分片结果）
    CHK_RET(GenRhdPrePostSteps(part1Info_, preStep_, postStep_, RhdSegMode::FULL, RhdSegMode::SHARE));
    HCCL_INFO(
        "[InsTempReduceScatterRHD][KernelRun] p[%u] r[%u] P[%u] myAlgIdx[%u] v[%u] inPart1[%u] isRep[%u] "
        "pair[%u] unitStart[%u] mainSteps[%zu] pre[%u,%u) post[%u,%u).",
        templateRankSize_, part1Info_.r, part1Info_.effectiveRankSize, myAlgIdx, part1Info_.virtualRank,
        static_cast<u32>(part1Info_.inPart1), static_cast<u32>(part1Info_.isRepresentative), part1Info_.pairedAlgRank,
        part1Info_.unitStart, rsSteps_.size(), preStep_.txSegStart, preStep_.txSegEnd, postStep_.txSegStart,
        postStep_.txSegEnd);

    // 槽级通道切分（镜像/终态回搬消费）：按端口组把【单槽】数据水平切分到各通道（本地拷贝，
    // 家族既有模式）。交换相位（PRE/MAIN/POST）的通道切分按【合并后大段】比例进行、
    // 每通道单条连续子段（BuildSegSlices 内实现，需求 5.1.2 多通道口径；单通道即整段一条）
    std::vector<u64> elemCountOut;
    std::vector<u64> sizeOut;
    std::vector<u64> elemOffset;
    const u64 totalDataCount = tempAlgParams_.sliceSize / dataTypeSize_;
    CHK_RET(CalcDataSplitByPortGroup(
        totalDataCount, dataTypeSize_, channels_.begin()->second, elemCountOut, sizeOut, elemOffset));
    sizeOut_ = sizeOut;
    elemOffset_ = elemOffset;
    // 通道切分表与 channelsPerRank_ 一致性校验（防止后续按 channelIdx 下标访问越界）
    CHK_PRT_RET(
        channelsPerRank_ > sizeOut_.size() || channelsPerRank_ > elemOffset_.size(),
        HCCL_ERROR(
            "[InsTempReduceScatterRHD][KernelRun] split size[%zu]/offset size[%zu] less than "
            "channelsPerRank_[%u].",
            sizeOut_.size(), elemOffset_.size(), channelsPerRank_),
        HcclResult::HCCL_E_INTERNAL);

    threadNum_ = static_cast<u32>(GetThreadNum());
    CHK_PRT_RET(
        threadNum_ > templateResource.threads.size(),
        HCCL_ERROR(
            "[InsTempReduceScatterRHD][KernelRun] thread num[%u] more than threads size[%zu].", threadNum_,
            templateResource.threads.size()),
        HcclResult::HCCL_E_INTERNAL);
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    // 阶段一：全量镜像（p 槽，algIdx i 的输入片进槽 σ(i)；槽内即本 rank 对该份额的贡献）
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(LocalDataCopy(templateResource.threads, channelIdx));
    }
    // 镜像→交换线程屏障（多通道正确性）：镜像按【单槽】切分到各通道，交换相位按
    // 【合并后大段】比例切分——两套口径边界不一致，通道 c 的交换（PRE/MAIN/POST）可能读到
    // 通道 c' 尚未落地的镜像字节（静默少一份贡献）。镜像完成后补一次 Sub→Main→Sub 屏障，
    // 保证全部通道镜像落地先于任一通道交换启动。单通道（threadNum==1）无子线程、屏障退化为
    // 空操作，既有行为零差异（检视意见 chenyia #4）
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    // 阶段二：前处理（仅非 2 幂：被合并方全量段整块交给代表，代表侧等待）
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(RunPre(templateResource.threads, channelIdx));
    }
    // 阶段三：主体折半（代表 + block；每步每对端单条连续大块，步内归约 DMA 侧完成，无 barrier）
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(RunMain(templateResource.threads, channelIdx));
    }
    // 阶段四：后处理（仅非 2 幂：代表把被合并方份额槽终态整块写回）。rhdSkipPost 置位时跳过
    // POST 交换（Reduce/AllReduce 复用 RS 相位：被合并方结果由上层算子后续相位承担，POST 交换为
    // 冗余通信）。skipPost 口径（device 相位模型）：postStep_ 仍由 template_utils 公共函数
    // GenRhdPrePostSteps 无条件生成、RunPost 仅动作被门控——即「保留 POST 步骤、仅跳过动作」，
    // 相位结构不因 skipPost 变长；与 DPU inter 模板「POST 不入扁平步骤表」机制不同但行为等价
    // （均跳过 POST 交换 + 被合并方终态回搬，见 template_utils.h rhdSkipPost 字段注释）。
    if (!tempAlgParams_.rhdSkipPost) {
        for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
            CHK_RET(RunPost(templateResource.threads, channelIdx));
        }
    }
    // 阶段五：终态回搬（各 rank 从自己份额槽 σ(myAlgIdx) 回搬输出；skipPost 下被合并方槽内
    // 仍为自身贡献非终态，跳过回搬——结果由上层算子承担，代表/block 成员不受影响）
    const bool skipMergedCopy = tempAlgParams_.rhdSkipPost && part1Info_.inPart1 && !part1Info_.isRepresentative;
    if (!skipMergedCopy) {
        for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
            CHK_RET(PostLocalCopy(templateResource.threads, channelIdx));
        }
    }

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[InsTempReduceScatterRHD] KernelRun end");
    return HcclResult::HCCL_SUCCESS;
}

// σ unit 偏移构建（均匀退化时 unitOffsets_[u] == u*sliceSize，与原行为逐字节一致）
HcclResult InsTempReduceScatterRHD::BuildUnitOffsets()
{
    // 末 unit 长 tailSize（无尾块传 tailSize == sliceSize），其余长 sliceSize
    tailLen_ = (tempAlgParams_.tailSize != 0) ? tempAlgParams_.tailSize : tempAlgParams_.sliceSize;
    unitOffsets_.resize(templateRankSize_);
    u64 offset = 0;
    for (u32 u = 0; u < templateRankSize_; ++u) {
        unitOffsets_[u] = offset;
        offset += (u == templateRankSize_ - 1) ? tailLen_ : tempAlgParams_.sliceSize;
    }
    totalUnitBytes_ = offset;
    HCCL_INFO(
        "[InsTempReduceScatterRHD][BuildUnitOffsets] p[%u] sliceSize[%llu] tailLen[%llu] totalUnitBytes[%llu] "
        "uniform[%u]",
        templateRankSize_, tempAlgParams_.sliceSize, tailLen_, totalUnitBytes_,
        static_cast<u32>(tailLen_ == tempAlgParams_.sliceSize));
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterRHD::RptScratchBase(u64 rpt) const
{
    // 每 repeat 块占用 totalUnitBytes_（= Σ unit 尺寸；均匀退化 = p*SS 与原行为一致）
    const u64 rptScratchStride = totalUnitBytes_;
    return tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * rptScratchStride;
}

// 构造槽位段 [segStart, segStart+segSlots) 的交换切片（合并发送核心，需求 5.1.2）：
// - 段偏移/长度经 RhdSegToBytesUniform 推导（需求 5.1.1，末 unit 长 tailSize——本家族恒
//   tailSize==sliceSize，与整槽算术等价）；
// - 单通道：整段一条大 slice（每步每对端 1 条 WQE）；多通道：对合并后大段按端口组比例切分，
//   每通道单条连续子段；空子段不产 slice 但原语调用恒发生（wrapper 零长跳过 + 握手完整）。
//   RHD 注册 MESH_1D 准入（需求 4.3）下每对端恒 1 通道，多通道臂按条款构造正确备用。
HcclResult InsTempReduceScatterRHD::BuildSegSlices(
    void* addr, const u64 base, const u32 segStart, const u32 segSlots, const u32 channelIdx,
    std::vector<DataSlice>& slices) const
{
    slices.clear();
    const u64 sliceSize = tempAlgParams_.sliceSize;
    CHK_PRT_RET(
        segSlots == 0, HCCL_ERROR("[InsTempReduceScatterRHD][BuildSegSlices] empty seg [%u, +%u).", segStart, segSlots),
        HcclResult::HCCL_E_INTERNAL);
    // 无尾块契约：tailSize == sliceSize；防御框架未填（0）时同样按整槽长度，不缩小
    const u64 tailLen = (tempAlgParams_.tailSize != 0) ? tempAlgParams_.tailSize : sliceSize;
    const RhdSegBytes seg
        = RhdSegToBytesUniform(segStart, segStart + segSlots, sliceSize, tailLen, templateRankSize_ - 1);
    // [AllReduce 复用] 段偏移改用 σ unit 前缀（tailSize != sliceSize 时末 unit 槽位前移；
    // 均匀退化时 unitOffsets_[segStart] == segStart*sliceSize 与 RhdSegToBytesUniform 的
    // seg.offset 逐字节一致）；段长度 seg.len 已由 RhdSegToBytesUniform 尾块精确
    const u64 segOffset = unitOffsets_[segStart];
    CHK_PRT_RET(
        seg.len == 0,
        HCCL_ERROR("[InsTempReduceScatterRHD][BuildSegSlices] seg [%u, +%u) len is 0.", segStart, segSlots),
        HcclResult::HCCL_E_INTERNAL);

    if (channelsPerRank_ == 1) {
        // 整段连续大块（段槽数 > 1 时即合并发送的兑现点）
        slices.emplace_back(addr, base + segOffset, seg.len, seg.len / dataTypeSize_);
        return HcclResult::HCCL_SUCCESS;
    }

    // 段级比例切分：每通道一条连续子段（切分输入用首对端通道表，与镜像/回搬的槽级切分同源）
    std::vector<u64> elemCountOut;
    std::vector<u64> sizeOut;
    std::vector<u64> elemOffset;
    CHK_RET(CalcDataSplitByPortGroupCommon(
        seg.len / dataTypeSize_, dataTypeSize_, channels_.begin()->second, elemCountOut, sizeOut, elemOffset,
        channelsPerRank_));
    CHK_PRT_RET(
        channelIdx >= sizeOut.size() || channelIdx >= elemOffset.size(),
        HCCL_ERROR(
            "[InsTempReduceScatterRHD][BuildSegSlices] split size[%zu]/offset size[%zu] less than "
            "channelIdx[%u].",
            sizeOut.size(), elemOffset.size(), channelIdx),
        HcclResult::HCCL_E_INTERNAL);
    if (sizeOut[channelIdx] == 0) {
        return HcclResult::HCCL_SUCCESS; // 空子段：不产 slice；原语调用由调用方恒发起（握手完整）
    }
    slices.emplace_back(
        addr, base + seg.offset + elemOffset[channelIdx], sizeOut[channelIdx], sizeOut[channelIdx] / dataTypeSize_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterRHD::HandleSingleRank(const std::vector<ThreadHandle>& threads)
{
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        CHK_RET(SingleRankCopy(threads, rpt));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterRHD::SingleRankCopy(const std::vector<ThreadHandle>& threads, u64 rpt)
{
    ThreadHandle q = threads[0];
    // N==1 时唯一 algIdx==0：输出按通用公式 OUT[base+rpt*ORS+myAlgIdx*OSS] 补 OSS 项
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));
    const u64 inOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
    const u64 outOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride
                       + myAlgIdx * tempAlgParams_.outputSliceStride;
    DataSlice src = DataSlice(tempAlgParams_.buffInfo.inputPtr, inOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
    DataSlice dst
        = DataSlice(tempAlgParams_.buffInfo.outputPtr, outOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
    // 真同址才跳过（F2）：须一并比较基址指针——同块复用（同 buffType）时 inputPtr/outputPtr 可能为不同
    // 基址（对照下方 LocalDataCopy 的 inputPtr≠hcclBuff.addr 同款指针比较口径），offset 数值巧合相等不
    // 代表同址；仅按 buffType+offset 判同址会误跳过本地拷贝、rank 输出区保持陈旧数据。
    const bool sameAddr = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr) && (inOff == outOff);
    if (tempAlgParams_.buffInfo.inBuffType != tempAlgParams_.buffInfo.outBuffType || !sameAddr) {
        CHK_RET(LocalCopy(q, src, dst));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterRHD::LocalDataCopy(const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterRHD][LocalDataCopy] empty threads"),
        HcclResult::HCCL_E_INTERNAL);
    ThreadHandle q = threads[channelIdx];
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);

    // 全量镜像：algIdx i 的输入片进槽 σ(i)（配对两方份额相邻，代表持 [2t,2t+2)）。
    // 镜像后每槽 = 本 rank 对该份额的贡献，主体 write-reduce 直接 RMW 累加；被合并方另经 PRE
    // 把整段 [0,p) write-reduce 给代表（代表每槽 = 自身 ⊕ 被合并方，主体前即 ×2）。
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 scratchBase = RptScratchBase(rpt);
        for (u32 i = 0; i < templateRankSize_; ++i) {
            const u32 slot = RhdUnitStartOfRank(templateRankSize_, i); // σ(i)
            const u64 inOff = inBaseOff + tempAlgParams_.inputSliceStride * i + elemOffset_[channelIdx];
            const u64 scOff = scratchBase + unitOffsets_[slot] + elemOffset_[channelIdx];
            // [AllReduce 复用] 末 unit（tail）长度用 tailLen 通道份；均匀退化时与原行为一致。
            // 尾槽判定位为 σ 槽位（unit 空间，与 BuildUnitOffsets/PostLocalCopy 同口径）——
            // σ(i)≠i（非 2 幂重排）时 alg 下标 i 与尾槽不对应，按 i 判尾会写错末槽长度
            const u64 unitSize = (slot == templateRankSize_ - 1 && tailLen_ != tempAlgParams_.sliceSize) ?
                                     tailLen_ :
                                     tempAlgParams_.sliceSize;
            const u64 copyLen = (channelsPerRank_ == 1) ?
                                    unitSize :
                                    sizeOut_[channelIdx]; // 多通道暂按均匀口径（AllReduce 复用为单通道）

            DataSlice src = DataSlice(tempAlgParams_.buffInfo.inputPtr, inOff, copyLen, copyLen / dataTypeSize_);
            DataSlice dst = DataSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scOff, copyLen, copyLen / dataTypeSize_);

            HCCL_DEBUG(
                "[InsTempReduceScatterRHD][LocalDataCopy] rpt[%llu] algIdx[%u] slot[%u] inOff[%llu] scOff[%llu] "
                "channelIdx[%u]",
                rpt, i, slot, inOff, scOff, channelIdx);

            // 真同址才跳过：须一并比较基址指针——三级 Aicpu 语境输入区与 scratch
            // 同块复用（同 buffType）时 offset 数值相等不代表同址（inputPtr≠hcclBuff.addr），
            // 仅按 buffType+offset 判同址会误跳过（对齐 DPU 模板 MirrorToScratch 同款修正）
            if (tempAlgParams_.buffInfo.inputPtr != tempAlgParams_.buffInfo.hcclBuff.addr || inOff != scOff) {
                CHK_RET(LocalCopy(q, src, dst));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterRHD::RunPre(const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    // 2 的幂 / block 成员：空步骤（nSlices == 0 且 peer 无效），直接跳过
    if (!part1Info_.inPart1) {
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterRHD][RunPre] empty threads"), HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        part1Info_.pairedAlgRank == INVALID_VALUE_RANKID,
        HCCL_ERROR("[InsTempReduceScatterRHD][RunPre] pairedAlgRank invalid."), HcclResult::HCCL_E_INTERNAL);
    const u32 pairedUserRank = subCommRanks_[0].at(part1Info_.pairedAlgRank);
    CHK_PRT_RET(
        channels_.count(pairedUserRank) == 0 || channelIdx >= channels_.at(pairedUserRank).size(),
        HCCL_ERROR(
            "[InsTempReduceScatterRHD][RunPre] link missing: paired=%u channelIdx=%u", pairedUserRank, channelIdx),
        HcclResult::HCCL_E_INTERNAL);
    const ChannelInfo& channel = channels_.at(pairedUserRank).at(channelIdx);
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    const u64 segStart = preStep_.txSegStart; // FULL 模式 = 0，SHARE 模式 = 被合并方份额槽
    const u64 segEnd = preStep_.txSegEnd;     // FULL 模式 = p

    HCCL_INFO(
        "[InsTempReduceScatterRHD][RunPre] isRep[%u] pairedUserRank[%u] preSeg[%llu,%llu) channelIdx[%u]",
        static_cast<u32>(part1Info_.isRepresentative), pairedUserRank, segStart, segEnd, channelIdx);

    if (part1Info_.isRepresentative) {
        // 代表：接收镜像（push 模型数据由被合并方写入本端槽位，仅需握手等待）
        DataReduceInfo recvInfo(channel, SlicesList({}, {}), dataType_, reduceOp_);
        CHK_PRT_RET(
            RecvWriteReduce(recvInfo, threads[channelIdx]),
            HCCL_ERROR(
                "[InsTempReduceScatterRHD][RunPre] RecvWriteReduce failed, pairedUserRank[%u] channelIdx[%u]",
                pairedUserRank, channelIdx),
            HcclResult::HCCL_E_INTERNAL);
        return HcclResult::HCCL_SUCCESS;
    }

    // 被合并方：全量镜像段 [segStart,segEnd) 在槽位空间整段连续——合并发送（每 repeat 至多
    // 一条/通道；单通道下即单条大块），write-reduce 进代表同号槽（代表槽内已有自身贡献，
    // RMW 后 = ×2）；本通道空子段时仍发起原语（握手完整）
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 scratchBase = RptScratchBase(rpt);
        std::vector<DataSlice> segSrc;
        std::vector<DataSlice> segDst;
        CHK_RET(BuildSegSlices(
            tempAlgParams_.buffInfo.hcclBuff.addr, scratchBase, static_cast<u32>(segStart),
            static_cast<u32>(segEnd - segStart), channelIdx, segSrc));
        CHK_RET(BuildSegSlices(
            channel.remoteCclMem.addr, scratchBase, static_cast<u32>(segStart), static_cast<u32>(segEnd - segStart),
            channelIdx, segDst));
        srcSlices.insert(srcSlices.end(), segSrc.begin(), segSrc.end());
        dstSlices.insert(dstSlices.end(), segDst.begin(), segDst.end());
    }
    DataReduceInfo sendInfo(channel, SlicesList(srcSlices, dstSlices), dataType_, reduceOp_);
    CHK_PRT_RET(
        SendWriteReduce(sendInfo, threads[channelIdx]),
        HCCL_ERROR(
            "[InsTempReduceScatterRHD][RunPre] SendWriteReduce failed, pairedUserRank[%u] channelIdx[%u]",
            pairedUserRank, channelIdx),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterRHD::RunMain(const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    // 被合并方退出主体（份额由代表代持，PRE 后只等 POST）
    if (part1Info_.inPart1 && !part1Info_.isRepresentative) {
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterRHD][RunMain] empty threads"), HcclResult::HCCL_E_INTERNAL);
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);

    // 步 k：dist = P>>(k+1)、伙伴 = v ⊕ dist；tx/rx 各一条 DataSlice 覆盖整步区段（合并发送，
    // COMMON-2），write-reduce 双向 RMW 对端同号槽；步间无 barrier（同队列定序 + ACK 链传先行步）
    for (const auto& step : rsSteps_) {
        const u32 peerAlgRank = RhdAlgRankOfVirtual(templateRankSize_, INVALID_VALUE_RANKID, step.peer);
        CHK_PRT_RET(
            peerAlgRank == INVALID_VALUE_RANKID,
            HCCL_ERROR("[InsTempReduceScatterRHD][RunMain] peer virtualRank[%u] resolve failed.", step.peer),
            HcclResult::HCCL_E_INTERNAL);
        const u32 peerUserRank = subCommRanks_[0].at(peerAlgRank);
        CHK_PRT_RET(
            channels_.count(peerUserRank) == 0 || channelIdx >= channels_.at(peerUserRank).size(),
            HCCL_ERROR(
                "[InsTempReduceScatterRHD][RunMain] link missing: peer=%u channelIdx=%u", peerUserRank, channelIdx),
            HcclResult::HCCL_E_INTERNAL);
        const ChannelInfo& channel = channels_.at(peerUserRank).at(channelIdx);

        const u64 txSlots = step.txSegEnd - step.txSegStart;
        const u64 rxSlots = step.rxSegEnd - step.rxSegStart;
        CHK_PRT_RET(
            txSlots == 0 || rxSlots == 0,
            HCCL_ERROR(
                "[InsTempReduceScatterRHD][RunMain] empty seg: tx[%u,%u) rx[%u,%u).", step.txSegStart, step.txSegEnd,
                step.rxSegStart, step.rxSegEnd),
            HcclResult::HCCL_E_INTERNAL);

        HCCL_INFO(
            "[InsTempReduceScatterRHD][RunMain] step[%u/%zu] peerV[%u] peerAlgRank[%u] peerUserRank[%u] "
            "txSeg[%u,%u) rxSeg[%u,%u) channelIdx[%u]",
            step.step + 1, rsSteps_.size(), step.peer, peerAlgRank, peerUserRank, step.txSegStart, step.txSegEnd,
            step.rxSegStart, step.rxSegEnd, channelIdx);

        void* localCclBuffAddr = tempAlgParams_.buffInfo.hcclBuff.addr;
        void* remoteCclBuffAddr = channel.remoteCclMem.addr;
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        txSrcSlices.reserve(rptNum);
        txDstSlices.reserve(rptNum);
        rxSrcSlices.reserve(rptNum);
        rxDstSlices.reserve(rptNum);

        for (u64 rpt = 0; rpt < rptNum; ++rpt) {
            const u64 scratchBase = RptScratchBase(rpt);
            // 合并发送核心（需求 5.1.2）：整步区段单条连续大块（单通道下一条大 slice 覆盖整段 =
            // 1 条 WQE；多通道每通道一条连续子段），write 模式下 tx/rx 双端同号槽位对称
            std::vector<DataSlice> txSrc;
            std::vector<DataSlice> txDst;
            std::vector<DataSlice> rxSrc;
            std::vector<DataSlice> rxDst;
            CHK_RET(BuildSegSlices(
                localCclBuffAddr, scratchBase, step.txSegStart, static_cast<u32>(txSlots), channelIdx, txSrc));
            CHK_RET(BuildSegSlices(
                remoteCclBuffAddr, scratchBase, step.txSegStart, static_cast<u32>(txSlots), channelIdx, txDst));
            CHK_RET(BuildSegSlices(
                remoteCclBuffAddr, scratchBase, step.rxSegStart, static_cast<u32>(rxSlots), channelIdx, rxSrc));
            CHK_RET(BuildSegSlices(
                localCclBuffAddr, scratchBase, step.rxSegStart, static_cast<u32>(rxSlots), channelIdx, rxDst));
            txSrcSlices.insert(txSrcSlices.end(), txSrc.begin(), txSrc.end());
            txDstSlices.insert(txDstSlices.end(), txDst.begin(), txDst.end());
            rxSrcSlices.insert(rxSrcSlices.end(), rxSrc.begin(), rxSrc.end());
            rxDstSlices.insert(rxDstSlices.end(), rxDst.begin(), rxDst.end());
        }

        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        TxRxChannels sendRecvChannels(channel, channel);
        SendRecvReduceInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_, reduceOp_);
        CHK_PRT_RET(
            SendRecvBatchWriteReduce(sendRecvInfo, threads[channelIdx]),
            HCCL_ERROR(
                "[InsTempReduceScatterRHD][RunMain] SendRecvBatchWriteReduce failed, peerUserRank[%u] "
                "txSeg[%u,%u) channelIdx[%u]",
                peerUserRank, step.txSegStart, step.txSegEnd, channelIdx),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterRHD::RunPost(const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    if (!part1Info_.inPart1) {
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterRHD][RunPost] empty threads"), HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        part1Info_.pairedAlgRank == INVALID_VALUE_RANKID,
        HCCL_ERROR("[InsTempReduceScatterRHD][RunPost] pairedAlgRank invalid."), HcclResult::HCCL_E_INTERNAL);
    const u32 pairedUserRank = subCommRanks_[0].at(part1Info_.pairedAlgRank);
    CHK_PRT_RET(
        channels_.count(pairedUserRank) == 0 || channelIdx >= channels_.at(pairedUserRank).size(),
        HCCL_ERROR(
            "[InsTempReduceScatterRHD][RunPost] link missing: paired=%u channelIdx=%u", pairedUserRank, channelIdx),
        HcclResult::HCCL_E_INTERNAL);
    const ChannelInfo& channel = channels_.at(pairedUserRank).at(channelIdx);
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);

    if (part1Info_.isRepresentative) {
        // 代表：被合并方份额槽（SHARE 段，1 槽）终态单条整块覆盖写回被合并方同号槽
        std::vector<DataSlice> srcSlices;
        std::vector<DataSlice> dstSlices;
        srcSlices.reserve(rptNum);
        dstSlices.reserve(rptNum);
        for (u64 rpt = 0; rpt < rptNum; ++rpt) {
            const u64 scratchBase = RptScratchBase(rpt);
            std::vector<DataSlice> segSrc;
            std::vector<DataSlice> segDst;
            CHK_RET(BuildSegSlices(
                tempAlgParams_.buffInfo.hcclBuff.addr, scratchBase, postStep_.txSegStart,
                postStep_.txSegEnd - postStep_.txSegStart, channelIdx, segSrc));
            CHK_RET(BuildSegSlices(
                channel.remoteCclMem.addr, scratchBase, postStep_.txSegStart, postStep_.txSegEnd - postStep_.txSegStart,
                channelIdx, segDst));
            srcSlices.insert(srcSlices.end(), segSrc.begin(), segSrc.end());
            dstSlices.insert(dstSlices.end(), segDst.begin(), segDst.end());
        }
        DataInfo sendInfo(channel, SlicesList(srcSlices, dstSlices), dataType_);
        HCCL_INFO(
            "[InsTempReduceScatterRHD][RunPost] rep send postSeg[%u,%u) to pairedUserRank[%u] channelIdx[%u]",
            postStep_.txSegStart, postStep_.txSegEnd, pairedUserRank, channelIdx);
        CHK_PRT_RET(
            SendWrite(sendInfo, threads[channelIdx]),
            HCCL_ERROR(
                "[InsTempReduceScatterRHD][RunPost] SendWrite failed, pairedUserRank[%u] channelIdx[%u]",
                pairedUserRank, channelIdx),
            HcclResult::HCCL_E_INTERNAL);
        return HcclResult::HCCL_SUCCESS;
    }

    // 被合并方：接收镜像（push 模型数据由代表写入本端同号槽，握手等待后回搬）
    DataInfo recvInfo(channel, SlicesList({}, {}), dataType_);
    HCCL_INFO(
        "[InsTempReduceScatterRHD][RunPost] merged recv postSeg[%u,%u) from pairedUserRank[%u] channelIdx[%u]",
        postStep_.rxSegStart, postStep_.rxSegEnd, pairedUserRank, channelIdx);
    CHK_PRT_RET(
        RecvWrite(recvInfo, threads[channelIdx]),
        HCCL_ERROR(
            "[InsTempReduceScatterRHD][RunPost] RecvWrite failed, pairedUserRank[%u] channelIdx[%u]", pairedUserRank,
            channelIdx),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterRHD::PostLocalCopy(const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterRHD][PostLocalCopy] empty threads"),
        HcclResult::HCCL_E_INTERNAL);
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));
    ThreadHandle q = threads[channelIdx];
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    // 终态槽 = 自己份额槽 σ(myAlgIdx)（代表 = 配对块起点槽、被合并方 = 自身份额槽、block = 恒等槽）
    const u32 mySlot = RhdUnitStartOfRank(templateRankSize_, myAlgIdx);

    // rpt 递减序：inter 语境输出区与输入区同块复用，可能落入后续 rpt 的紧凑 scratch 块内——递增序
    // 会使低 rpt 输出写先冲掉高 rpt 未读终态槽（框间数据交叉）；递减序恒安全（output(A) ≥
    // j + A·m0·m1 > scratch(B≤A-1) 上界），同块写读同址由同队列 FIFO 保证。镜像保持递增序
    // （高 rpt scratch 写会覆盖低 rpt 已消费输入，输入必须先读——两序不可混用）。
    for (u64 rpt = rptNum; rpt > 0; --rpt) {
        const u64 stateOff = RptScratchBase(rpt - 1) + unitOffsets_[mySlot] + elemOffset_[channelIdx];
        // 输出按通用五参数公式 OUT[base+rpt*ORS+myAlgIdx*OSS]
        const u64 outOff = tempAlgParams_.buffInfo.outBuffBaseOff + (rpt - 1) * tempAlgParams_.outputRepeatStride
                           + myAlgIdx * tempAlgParams_.outputSliceStride + elemOffset_[channelIdx];
        // [AllReduce 复用] 末 rank（mySlot == p-1）输出用 tailLen；均匀退化时与原行为一致
        const u64 myUnitSize = (mySlot == templateRankSize_ - 1 && tailLen_ != tempAlgParams_.sliceSize) ?
                                   tailLen_ :
                                   tempAlgParams_.sliceSize;
        const u64 copyLen = (channelsPerRank_ == 1) ? myUnitSize : sizeOut_[channelIdx];

        DataSlice src = DataSlice(tempAlgParams_.buffInfo.hcclBuff.addr, stateOff, copyLen, copyLen / dataTypeSize_);
        DataSlice dst = DataSlice(tempAlgParams_.buffInfo.outputPtr, outOff, copyLen, copyLen / dataTypeSize_);

        HCCL_DEBUG(
            "[InsTempReduceScatterRHD][PostLocalCopy] rpt[%llu] myAlgIdx[%u] mySlot[%u] stateOff[%llu] "
            "outOff[%llu] channelIdx[%u]",
            rpt - 1, myAlgIdx, mySlot, stateOff, outOff, channelIdx);

        // 真同址才跳过：须一并比较基址指针——三级 out-of-place 且 offset 数值
        // 巧合相等时会静默跳过终态回搬、rank 输出区保持陈旧数据（对齐 LocalDataCopy :391 与 DPU
        // 版 PostLocalCopy :513 的指针比较口径）
        const bool sameAddr
            = (tempAlgParams_.buffInfo.hcclBuff.addr == tempAlgParams_.buffInfo.outputPtr) && (stateOff == outOff);
        if (tempAlgParams_.buffInfo.hcclBuffType != tempAlgParams_.buffInfo.outBuffType || !sameAddr) {
            CHK_RET(LocalCopy(q, src, dst));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterRHD::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // p 槽单缓冲（槽位总数 = 物理规模 p，σ 布局）：镜像初始化后槽内即归约态（对端 write-reduce
    // 直接 RMW 同号槽），无独立接收区；按单次 repeat 报。
    // 均匀分片（tailSize == sliceSize，RS 现状）下 p 恒正确 = 本模板可提供的最优值；非均匀
    // tailSize（AllReduce 复用）须由 consuming executor 在调用前注入尺寸并重估乘数——本模板签名
    // 无尺寸入参、不读未赋值成员（tempAlgParams_ 仅在 KernelRun 赋值，编排期恒默认 0，尾块
    // 补偿分支为死代码）
    const u64 templateScratchMultiplier = templateRankSize_;
    HCCL_INFO(
        "[InsTempReduceScatterRHD][CalcScratchMultiple] templateScratchMultiplier[%llu]", templateScratchMultiplier);
    return templateScratchMultiplier;
}

void InsTempReduceScatterRHD::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = 1 * channelsPerRank_;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempReduceScatterRHD::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = 1 * channelsPerRank_;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

} // namespace ops_hccl
