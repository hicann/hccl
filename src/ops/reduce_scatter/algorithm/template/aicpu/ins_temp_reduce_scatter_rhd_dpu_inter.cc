/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_rhd_dpu_inter.h"
#include "dpu_alg_nhr_opt_wrapper.h"
#include "channel.h"

namespace ops_hccl {

std::vector<CostModelParam> InsTempReduceScatterRhdDpuInter::CalcCostCoeff(CalcCostCoeffParam param)
{
    // preferred-only：不标定，返回固定占位系数（对齐 device 基线 InsTempReduceScatterRHD 口径）。
    // 不可返回空——空系数会使算法在 cost 表注册阶段即被跳过，显式 HCCL_ALGO 配置也无法选中
    // （显式配置下 RHD 为唯一候选、系数不实际参与比价）；DPU 规约类本迭代不设性能要求
    // （需求 2.9），占位恒非空即可。
    (void)param;
    std::vector<CostModelParam> params;
    params.push_back({0.0f, 0.0f, 1.0f, 0.0f});
    HCCL_DEBUG("[InsTempReduceScatterRhdDpuInter][CalcCostCoeff] placeholder cost (preferred-only, no calibration).");
    return params;
}

InsTempReduceScatterRhdDpuInter::InsTempReduceScatterRhdDpuInter(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempReduceScatterRhdDpuInter::~InsTempReduceScatterRhdDpuInter() {}

// 步骤对端解析（AICPU/DPU 两侧共用）：MAIN = 主体有效序承载 algRank；PRE/POST = 配对 algRank
static HcclResult ResolveStepPeer(u32 rankSize, const RhdPart1Info& info, const RhdStepInfo& step, u32& peerAlgRank)
{
    if (step.phase == RhdPhase::MAIN_RS) {
        peerAlgRank = RhdAlgRankOfVirtual(rankSize, INVALID_VALUE_RANKID, step.peer);
        CHK_PRT_RET(
            peerAlgRank == INVALID_VALUE_RANKID,
            HCCL_ERROR(
                "[InsTempReduceScatterRhdDpuInter][ResolveStepPeer] peer virtualRank[%u] resolve failed.", step.peer),
            HcclResult::HCCL_E_INTERNAL);
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        !info.inPart1 || info.pairedAlgRank == INVALID_VALUE_RANKID,
        HCCL_ERROR(
            "[InsTempReduceScatterRhdDpuInter][ResolveStepPeer] PRE/POST step without valid pair, inPart1[%u].",
            static_cast<u32>(info.inPart1)),
        HcclResult::HCCL_E_INTERNAL);
    peerAlgRank = info.pairedAlgRank;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterRhdDpuInter::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // host 网卡资源：仅主流（主线程经接口传入的 stream 做转换），不新增从流与对应 notify。
    // 建链走 NHR 系专用 host RoCE 建链（只认 ROCE+HOST，复用通用建链会在 DPU 侧 core dump）；
    // 全连接申请 = 逻辑 peer 全集（PRE/POST 配对 + MAIN 各步对端覆盖全部 p-1 个远端）。
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhrHighestHostRoce(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO(
        "[InsTempReduceScatterRhdDpuInter][CalcRes]slaveThreadNum[%u] notifyNumOnMainThread[%u] "
        "level1Channels[%zu].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, level1Channels.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterRhdDpuInter::GetRes(AlgResourceRequest& resourceRequest) const
{
    // 与 CalcRes 的线程/notify 需求保持一致（对齐既有 DPU inter 家族）：编排期无 comm/拓扑可查，
    // 按同口径重建请求；不 override 会落到基类默认实现（打 ERROR 并返回 INTERNAL）。
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterRhdDpuInter::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void)inBufferType;
    (void)outBufferType;
    // acc+recv 双区 = 2p：DPU plain write 无随路归约，对端数据须落独立 recv 区（直接落 acc 会冲掉
    // 自身贡献），每步后 LocalReduce 消费；recv 最大段 = PRE 的 FULL [0,p)。按单次 repeat 报。
    // allRank* 契约下乘数单位 = 单 rank 份额（均匀退化 sliceSize）；非均匀 fill 落地时须重估乘数
    const u64 templateScratchMultiplier = 2 * static_cast<u64>(templateRankSize_);
    HCCL_INFO(
        "[InsTempReduceScatterRhdDpuInter][CalcScratchMultiple] templateScratchMultiplier[%llu]",
        templateScratchMultiplier);
    return templateScratchMultiplier;
}

HcclResult InsTempReduceScatterRhdDpuInter::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempReduceScatterRhdDpuInter] KernelRun start");

    // DPU Thread 四步握手新路径未在本迭代实现（见下方门控注释），框架置
    // isDpuThreadEnabled 时快速失败——避免静默落老路径导致运行期挂起（新路径需线程级 notify 原语
    // 与 DPU 线程句柄透传，非本迭代交付范围）
    CHK_PRT_RET(
        templateResource.isDpuThreadEnabled,
        HCCL_ERROR("[InsTempReduceScatterRhdDpuInter][KernelRun] isDpuThreadEnabled DPU thread path not supported "
                   "in this iteration."),
        HcclResult::HCCL_E_NOT_SUPPORT);

    tempAlgParams_ = tempAlgParams;
    channels_ = templateResource.channels;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];

    // allRank* 表由本模板在 KernelRun 入口按 stride 契约（sliceSize/inputSliceStride）自推导均匀表，
    // 忽略 executor 预填的 allRank* 表：共享二级 Sequence executor 同时承载 NhrDpu L1（其 KernelRun
    // 按 allRankSliceSize 是否为空判定 stride/table 布局、table 口径为 outputSliceStride），executor
    // 预填（inputSliceStride 口径，≠outputSliceStride）会把 NhrDpu 切到表布局并致槽位错位，故该
    // executor 不再预填、本模板与 NhrDpu 各管各口径、互不污染（仿 NhrDpu::NormalizeStrideLayout
    // 的均匀退化自推导口径）。真正的变长契约校验（三向量等长）在 DPU 侧 DPUKernelRun 对反序列化
    // 值做；本入口反序列化 allRank* 恒空，下方 assign/resize 为 stride→均匀表退化而非校验。
    tempAlgParams_.allRankSliceSize.assign(templateRankSize_, tempAlgParams_.sliceSize);
    tempAlgParams_.allRankProcessedDataCount.assign(templateRankSize_, tempAlgParams_.count);
    tempAlgParams_.allRankDispls.resize(templateRankSize_);
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; ++rankIdx) {
        tempAlgParams_.allRankDispls[rankIdx] = static_cast<u64>(rankIdx) * tempAlgParams_.inputSliceStride;
    }

    // σ unit 尺寸经 RhdBuildUnitSizes 反查构建（均匀退化：allRankSliceSize[j] ≡ sliceSize，
    // unitSizes_ 全等——行为与 stride 契约逐字节一致）
    CHK_RET(RhdBuildUnitSizes(templateRankSize_, tempAlgParams_.allRankSliceSize, unitSizes_));
    RhdSegBytes totalSeg;
    CHK_RET(SegBytesOfUnits(0, templateRankSize_, totalSeg));
    accRegionBytes_ = totalSeg.len;

    // 边界守卫：全零份额（均匀退化下即 count==0 语义）直接成功；本 rank 份额为零不提前退出——
    // 仍须参与步骤序列（对端等待本端各步请求/响应与 STEP_SYNC 配对，需求 5.2 count==0 仅跳过通信）
    if (accRegionBytes_ == 0) {
        HCCL_INFO("[InsTempReduceScatterRhdDpuInter][KernelRun] total share bytes is 0, nothing to do.");
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        templateResource.threads.empty(), HCCL_ERROR("[InsTempReduceScatterRhdDpuInter][KernelRun] empty threads"),
        HcclResult::HCCL_E_INTERNAL);
    // 单 rank 防御分支：hostdpu 组网下 d1/d2>=2 结构性不达（CheckHostDPUOnly 要求 serverNum>1），
    // selector 层已拦截；此处五参数直拷兜底（单线程，无 DPU 请求）
    if (templateRankSize_ == 1) {
        return HandleSingleRank(templateResource.threads);
    }
    CHK_PRT_RET(
        subCommRanks_.empty() || subCommRanks_[0].size() != templateRankSize_,
        HCCL_ERROR(
            "[InsTempReduceScatterRhdDpuInter][KernelRun] subCommRanks mismatch, tempRankSize[%u].", templateRankSize_),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        channels_.empty(), HCCL_ERROR("[InsTempReduceScatterRhdDpuInter][KernelRun] channels is empty."),
        HcclResult::HCCL_E_INTERNAL);
    for (const auto& channelsByRank : channels_) {
        CHK_PRT_RET(
            channelsByRank.second.empty(),
            HCCL_ERROR(
                "[InsTempReduceScatterRhdDpuInter][KernelRun] channel list of remote rank[%u] is empty.",
                channelsByRank.first),
            HcclResult::HCCL_E_INTERNAL);
    }
    CHK_PRT_RET(
        dataTypeSize_ == 0, HCCL_ERROR("[InsTempReduceScatterRhdDpuInter][KernelRun] dataTypeSize is 0."),
        HcclResult::HCCL_E_INTERNAL);

    // 步骤表预计算（单一事实来源：template_utils RHD 函数族；AICPU/DPU 两侧同源重算）
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx_));
    CHK_RET(CalcRhdPart1Info(templateRankSize_, myAlgIdx_, INVALID_VALUE_RANKID, part1Info_));
    CHK_RET(BuildRhdFlatSteps(templateRankSize_, myAlgIdx_, tempAlgParams_.rhdSkipPost, flatSteps_));
    HCCL_INFO(
        "[InsTempReduceScatterRhdDpuInter][KernelRun] p[%u] r[%u] P[%u] myAlgIdx[%u] v[%u] inPart1[%u] isRep[%u] "
        "pair[%u] unitStart[%u] stepCnt[%zu] scratchMultiple[%llu].",
        templateRankSize_, part1Info_.r, part1Info_.effectiveRankSize, myAlgIdx_, part1Info_.virtualRank,
        static_cast<u32>(part1Info_.inPart1), static_cast<u32>(part1Info_.isRepresentative), part1Info_.pairedAlgRank,
        part1Info_.unitStart, flatSteps_.size(), 2 * static_cast<u64>(templateRankSize_));

    // 新路径（DPU Thread 四步握手）门控分支位预留：本迭代按需求仅交付老路径，DPU Thread 四步
    // 握手新路径未实现（后续阶段）；框架置 isDpuThreadEnabled 时由 KernelRun 入口快速失败守卫
    // 拦截，置 false（本迭代现状）则按老路径执行。后续以 TemplateResource.isDpuThreadEnabled 门控
    // 字段切换（需求 SD §2.2 双路径结构）。

    // 阶段一：全量镜像（rpt 升序：σ(i)·SS 恒落后于输入游标 (i+1)·ISS，三级位输入区与 scratch
    // 同块复用无读后写冲突——继承 device 基线已验证结论）
    CHK_RET(MirrorToScratch(templateResource.threads));

    // 交换段前一次 batch 切换：镜像落地 + 保持 eager 至交换段结束（老路径无开工握手，步间定序
    // 由每步完成屏障保证，DESIGN §3-D4）。切换后各错误返回前恢复 batch 模式
    // （对齐 ins_temp_scatter_omnipipe_nhr_dpu.cc:166 先例——eager 泄漏影响后续算子批量提交性能）
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempReduceScatterRhdDpuInter] failed set eager mode, tag is %s.", param.algTag);
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
        HCCL_ERROR("[InsTempReduceScatterRhdDpuInter] HcommThreadSynchronize failed");
        (void)HcommBatchModeStart(param.algTag);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 阶段二~四：步进交替（每步「DPU 单条连续大块搬运 → 请求/响应握手 → AICPU LocalReduce → 完成屏障」；
    // PRE/MAIN 落 recv 区后归约、POST 代表整块覆盖写直达被合并方 acc 槽）
    if (RunStepsOldPath(param, templateResource) != HcclResult::HCCL_SUCCESS) {
        (void)HcommBatchModeStart(param.algTag);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 阶段五：终态回搬（所有 rpt 统一在步进后递减序回搬——交叠证明继承 device 基线：输出区
    // output(A) ≥ j + r1·m0 + A·m0·m1 ≥ (2B+2)·m1 对 B ≤ A-1（需 m0 ≥ 2，框内 ≥ 2 rank 恒真），
    // recv 区扩张 2× 后证明仍成立；镜像升序 + 回搬递减两序不可混用）
    if (PostLocalCopy(templateResource.threads) != HcclResult::HCCL_SUCCESS) {
        (void)HcommBatchModeStart(param.algTag);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 交换段结束，回到 batch 模式（回搬已入队；后续算子任务由队列 FIFO 保序）
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempReduceScatterRhdDpuInter] failed set batch mode, tag is %s.", param.algTag);
        return HcclResult::HCCL_E_INTERNAL;
    }

    HCCL_INFO("[InsTempReduceScatterRhdDpuInter] KernelRun end");
    return HcclResult::HCCL_SUCCESS;
}

// 扁平步骤表：[PRE（仅 inPart1）， MAIN×log₂P（被合并方跳过）， POST（仅 inPart1）]。
// RS 族前后处理模式：pre = FULL（代表须收被合并方全量贡献 [0,p)）、post = SHARE（被合并方自身份额槽）。
// AICPU 侧（本文件 KernelRun）与 DPU 侧（DPUKernelRun）均经本函数重算，保证两侧步骤序一致
// （STEP_SYNC 跨 rank 配对依赖全 rank 同序）与 rhdStepCnt 一致性校验基础。
HcclResult InsTempReduceScatterRhdDpuInter::BuildRhdFlatSteps(
    u32 rankSize, u32 myAlgRank, bool skipPost, std::vector<RhdStepInfo>& flatSteps)
{
    flatSteps.clear();
    RhdPart1Info info;
    CHK_RET(CalcRhdPart1Info(rankSize, myAlgRank, INVALID_VALUE_RANKID, info));
    const bool isMergedParty = info.inPart1 && !info.isRepresentative; // 被合并方退出主体
    RhdStepInfo preStep;
    RhdStepInfo postStep;
    CHK_RET(GenRhdPrePostSteps(info, preStep, postStep, RhdSegMode::FULL, RhdSegMode::SHARE));
    if (info.inPart1) {
        flatSteps.push_back(preStep);
    }
    if (!isMergedParty) {
        std::vector<RhdStepInfo> mainSteps;
        CHK_RET(GenRhdReduceScatterSteps(info, mainSteps));
        flatSteps.insert(flatSteps.end(), mainSteps.begin(), mainSteps.end());
    }
    if (info.inPart1 && !skipPost) {
        // skipPost（Reduce/AllReduce 复用 RS 相位）：POST 交换为冗余通信，不入扁平步骤表——
        // 本模板（DPU inter 步骤表 + 逐步游标协议）用「POST 不入表」替代 device 相位模型的
        // 「POST 动作门控」，避免涌入幽灵轮询步骤；两引擎行为等价（均跳过 POST 交换 + 被合并方
        // 终态回搬，见 template_utils.h rhdSkipPost 字段注释）。两侧（AICPU KernelRun / DPU
        // DPUKernelRun）以同值 flag 构建保证 rhdStepCnt 一致与 STEP_SYNC 配对。
        flatSteps.push_back(postStep);
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterRhdDpuInter::RptScratchBase(u64 rpt) const
{
    // 每 repeat 块占用 2×accRegionBytes（acc p 槽 + recv p 槽双区，unit 尺寸前缀布局；均匀退化时
    // = 2p·SS 与 stride 契约一致）；RPT=1 时退化为 hcclBuffBaseOff。同层 subComm 组内全员
    // hcclBuffBaseOff 一致（代表组构造性质），对端 remoteCclMem 同偏移即对端槽位
    const u64 rptScratchStride = 2 * accRegionBytes_;
    return tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * rptScratchStride;
}

HcclResult InsTempReduceScatterRhdDpuInter::SegBytesOfUnits(u32 unitStart, u32 unitEnd, RhdSegBytes& bytes) const
{
    // 需求 5.1.1：区段字节经 RhdSegToBytesAllRank（σ unit 前缀）推导，不做模板内私有换算
    return RhdSegToBytesAllRank(unitStart, unitEnd, unitSizes_, bytes);
}

HcclResult InsTempReduceScatterRhdDpuInter::HandleSingleRank(const std::vector<ThreadHandle>& threads)
{
    // N==1 时唯一 algIdx==0；先解析并校验 subCommRanks（对齐 device 基线 SingleRankCopy 的防御）
    CHK_PRT_RET(
        subCommRanks_.empty() || subCommRanks_[0].empty(),
        HCCL_ERROR("[InsTempReduceScatterRhdDpuInter][HandleSingleRank] subCommRanks is empty."),
        HcclResult::HCCL_E_INTERNAL);
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx_));
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        CHK_RET(SingleRankCopy(threads, rpt));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterRhdDpuInter::SingleRankCopy(const std::vector<ThreadHandle>& threads, u64 rpt)
{
    ThreadHandle q = threads[0];
    // N==1 时唯一 algIdx==0：本 rank 份额 = allRank[0]（allRankDispls[0]=0 退化），输出按通用公式
    // OUT[base+rpt*ORS+myAlgIdx*OSS] 补 OSS 项（myAlgIdx==0 恒零项）
    const u64 mySize = tempAlgParams_.allRankSliceSize.at(0);
    const u64 myCount = tempAlgParams_.allRankProcessedDataCount.at(0);
    if (mySize == 0) {
        return HcclResult::HCCL_SUCCESS;
    }
    const u64 inOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
    const u64 outOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
    DataSlice src = DataSlice(tempAlgParams_.buffInfo.inputPtr, inOff, mySize, myCount);
    DataSlice dst = DataSlice(tempAlgParams_.buffInfo.outputPtr, outOff, mySize, myCount);
    const bool sameAddr = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr) && (inOff == outOff);
    if (!sameAddr) {
        CHK_RET(LocalCopy(q, src, dst));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterRhdDpuInter::MirrorToScratch(const std::vector<ThreadHandle>& threads)
{
    ThreadHandle q = threads[0];
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);

    // 全量镜像：algIdx i 的输入片（allRankDispls[i] 起、allRankSliceSize[i] 字节）进 acc 槽 σ(i)，
    // 镜像后每槽 = 本 rank 对该份额的贡献（DPU 交换发送源）；被合并方另经 PRE 把整段 [0,p)
    // write-reduce 给代表。零份额（count==0）无镜像数据，跳过（需求 5.2）
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 scratchBase = RptScratchBase(rpt);
        for (u32 i = 0; i < templateRankSize_; ++i) {
            const u32 slot = RhdUnitStartOfRank(templateRankSize_, i); // σ(i)
            const u64 inSize = tempAlgParams_.allRankSliceSize.at(i);
            if (inSize == 0) {
                continue; // 零份额：无镜像数据（段字节推导自然为 0，acc 槽位保留不被写）
            }
            const u64 inOff = inBaseOff + tempAlgParams_.allRankDispls.at(i);
            RhdSegBytes unit; // unit 偏移/尺寸（σ 前缀）
            CHK_RET(SegBytesOfUnits(slot, slot + 1, unit));

            DataSlice src = DataSlice(
                tempAlgParams_.buffInfo.inputPtr, inOff, inSize, tempAlgParams_.allRankProcessedDataCount.at(i));
            DataSlice dst = DataSlice(
                tempAlgParams_.buffInfo.hcclBuff.addr, scratchBase + unit.offset, unit.len, unit.len / dataTypeSize_);

            HCCL_DEBUG(
                "[InsTempReduceScatterRhdDpuInter][MirrorToScratch] rpt[%llu] algIdx[%u] slot[%u] inOff[%llu] "
                "scOff[%llu]",
                rpt, i, slot, inOff, scratchBase + unit.offset);

            // 真同址才跳过：二级绑定位 inputPtr 与 scratch 不同基址，offset 数值相等不是同址，
            // 须比较基址指针（否则 i=0 与 σ(0)=0 数值相等会误跳过）
            const bool sameAddr = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.hcclBuff.addr)
                                  && (inOff == scratchBase + unit.offset);
            if (!sameAddr) {
                CHK_RET(LocalCopy(q, src, dst));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempReduceScatterRhdDpuInter::RunStepsOldPath(const OpParam& param, const TemplateResource& templateResource)
{
    const u32 stepCnt = static_cast<u32>(flatSteps_.size());
    CHK_PRT_RET(
        stepCnt == 0, HCCL_ERROR("[InsTempReduceScatterRhdDpuInter][RunStepsOldPath] empty flat steps."),
        HcclResult::HCCL_E_INTERNAL);

    for (u32 idx = 0; idx < stepCnt; idx++) {
        const RhdStepInfo& step = flatSteps_[idx];
        u32 peerAlgRank = INVALID_VALUE_RANKID;
        CHK_RET(ResolveStepPeer(templateRankSize_, part1Info_, step, peerAlgRank));
        const u32 peerUserRank = subCommRanks_[0].at(peerAlgRank);
        // 段字节经 σ unit 前缀推导（需求 5.1.1；均匀退化 = 段槽数·SS）
        RhdSegBytes txSeg;
        RhdSegBytes rxSeg;
        CHK_RET(SegBytesOfUnits(step.txSegStart, step.txSegEnd, txSeg));
        CHK_RET(SegBytesOfUnits(step.rxSegStart, step.rxSegEnd, rxSeg));

        // DFX 步骤表日志（AC-4 任务数口径：每相位每对端 1 条 WQE，单条长度 = 段字节数）
        HCCL_INFO(
            "[InsTempReduceScatterRhdDpuInter][RunStep] step[%u/%u] phase[%u] peerAlgRank[%u] peerUserRank[%u] "
            "txSeg[%u,%u) rxSeg[%u,%u) txLen[%llu] rxLen[%llu] wqeNum[1/peer].",
            idx + 1, stepCnt, static_cast<u32>(step.phase), peerAlgRank, peerUserRank, step.txSegStart, step.txSegEnd,
            step.rxSegStart, step.rxSegEnd, txSeg.len, rxSeg.len);

        // DPURunInfo 携带步骤游标（rhdStepIdx/rhdStepCnt）序列化透传，DPU 侧按名分发后重算步骤表
        // （单一事实来源）并校验一致；每步一个请求，请求内含该步全部 rpt 段（RPT=1 现值即单段）
        TemplateDataParams stepParams = tempAlgParams_;
        stepParams.rhdStepIdx = idx;
        stepParams.rhdStepCnt = stepCnt;
        // dataType 由模板在序列化前补填（executor 链路从不赋值该字段）：
        // DPU 侧按 tempAlgParams.dataType 查 DATATYPE_SIZE_TABLE 推 count 元数据，默认 INT8 会使
        // 非 8bit 类型下字节数静默错位（对齐 NHR DPU 先例 ins_temp_reduce_scatter_nhr_dpu.cc:145）
        stepParams.dataType = param.DataDes.dataType;
        DPURunInfo dpuRunInfo;
        dpuRunInfo.execTimeout = param.opConfig.execTimeout;
        dpuRunInfo.templateName = "InsTempReduceScatterRhdDpuInter";
        dpuRunInfo.tempAlgParams = stepParams;
        dpuRunInfo.channels = channels_;
        dpuRunInfo.myRank = myRank_;
        dpuRunInfo.subCommRanks = subCommRanks_;
        u32 sendMsgId = 0;
        auto dpuRunInfoSeqData = dpuRunInfo.Serialize();
        if (HcommSendRequest(
                reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr), param.algTag,
                static_cast<void*>(dpuRunInfoSeqData.data()), dpuRunInfoSeqData.size(), &sendMsgId)
            != 0) {
            HCCL_ERROR("[InsTempReduceScatterRhdDpuInter][RunStep] HcommSendRequest failed, step[%u].", idx);
            return HcclResult::HCCL_E_INTERNAL;
        }

        // 等待 DPU 单步传输完成（DpuBatchTransfer 三阶段：STEP_SYNC 前同步 → 单条大块写 + DATA_SIGNAL
        // notify → Drain/Fence + 完工等待；跨 rank 定序由 STEP_SYNC 协议保证）
        void* recvData = nullptr;
        u32 recvMsgId = 0;
        if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId)
            != 0) {
            HCCL_ERROR("[InsTempReduceScatterRhdDpuInter][RunStep] HcommWaitResponse failed, step[%u].", idx);
            return HcclResult::HCCL_E_INTERNAL;
        }
        if (recvMsgId != sendMsgId) {
            HCCL_ERROR(
                "[InsTempReduceScatterRhdDpuInter][RunStep] recvMsgId[%u] not equal to sendMsgId[%u], step[%u].",
                recvMsgId, sendMsgId, idx);
            return HcclResult::HCCL_E_INTERNAL;
        }

        // AICPU 步间归约（搬运 → 握手 → LocalReduce 交替；B = reduce(B, A)）：
        // PRE 代表：recv 全段 [0,p) → acc 同段（acc 内已有自身贡献，累加后 = ×2，同 device 基线语义）；
        // MAIN：recv[rxSeg] → acc[rxSeg]；PRE 发送方 / POST 双方无本地归约（POST 覆盖写直达 acc 槽）
        CHK_RET(StepLocalReduce(templateResource.threads, step));

        // 步间完成屏障：本步 acc 更新落地先于下一步 DPU 读取（老路径无开工握手，显式同步兜底；
        // 对端 acc 只被对端自家 DPU 读取、本端只写对端 recv/acc 预约区，无第三写者）
        if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
            HCCL_ERROR("[InsTempReduceScatterRhdDpuInter][RunStep] HcommThreadSynchronize failed, step[%u].", idx);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempReduceScatterRhdDpuInter::StepLocalReduce(const std::vector<ThreadHandle>& threads, const RhdStepInfo& step)
{
    // PRE 发送方（被合并方）/ POST 双方：无本地归约动作（POST 代表覆盖写直达被合并方 acc 槽）
    bool needReduce = false;
    u64 segStart = 0;
    u64 segEnd = 0;
    if (step.phase == RhdPhase::PRE && part1Info_.isRepresentative) {
        // 代表侧 PRE：接收镜像 FULL 段 [0, p)
        needReduce = true;
        segStart = step.rxSegStart;
        segEnd = step.rxSegEnd;
    } else if (step.phase == RhdPhase::MAIN_RS) {
        // 主体步：recv rxSeg → acc rxSeg（归约对端贡献）
        needReduce = true;
        segStart = step.rxSegStart;
        segEnd = step.rxSegEnd;
    }
    if (!needReduce) {
        return HcclResult::HCCL_SUCCESS;
    }

    ThreadHandle q = threads[0];
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    CHK_PRT_RET(
        segEnd <= segStart, HCCL_ERROR("[InsTempReduceScatterRhdDpuInter][StepLocalReduce] empty seg."),
        HcclResult::HCCL_E_INTERNAL);

    // 段字节与偏移经 σ unit 前缀推导（需求 5.1.1；recv 区基址 = acc 区总字节之后，均匀退化 =
    // segStart·SS / p·SS + segStart·SS 与 stride 契约一致）
    RhdSegBytes seg;
    CHK_RET(SegBytesOfUnits(static_cast<u32>(segStart), static_cast<u32>(segEnd), seg));
    CHK_PRT_RET(
        seg.len == 0, HCCL_ERROR("[InsTempReduceScatterRhdDpuInter][StepLocalReduce] seg bytes is 0."),
        HcclResult::HCCL_E_INTERNAL);

    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 scratchBase = RptScratchBase(rpt);
        const u64 accOff = scratchBase + seg.offset;
        const u64 recvOff = scratchBase + accRegionBytes_ + seg.offset;
        DataSlice src = DataSlice(tempAlgParams_.buffInfo.hcclBuff.addr, recvOff, seg.len, seg.len / dataTypeSize_);
        DataSlice dst = DataSlice(tempAlgParams_.buffInfo.hcclBuff.addr, accOff, seg.len, seg.len / dataTypeSize_);
        CHK_RET(LocalReduce(q, src, dst, dataType_, reduceOp_));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterRhdDpuInter::PostLocalCopy(const std::vector<ThreadHandle>& threads)
{
    ThreadHandle q = threads[0];
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    // 终态槽 = 自己份额槽 σ(myAlgIdx)（代表 = 配对块起点槽、被合并方 = 自身份额槽、block = 恒等槽；
    // 被合并方该槽已由 POST 覆盖写直达填充）；份额字节 = unitSizes_[mySlot]（allRank 契约）
    // skipPost 下被合并方槽内仍为自身贡献非终态，跳过回搬（结果由上层算子承担；代表/block 不受影响）
    if (tempAlgParams_.rhdSkipPost && part1Info_.inPart1 && !part1Info_.isRepresentative) {
        return HcclResult::HCCL_SUCCESS;
    }
    const u32 mySlot = RhdUnitStartOfRank(templateRankSize_, myAlgIdx_);
    RhdSegBytes myUnit;
    CHK_RET(SegBytesOfUnits(mySlot, mySlot + 1, myUnit));
    const u64 mySize = myUnit.len;
    const u64 myCount = tempAlgParams_.allRankProcessedDataCount.at(myAlgIdx_);
    if (mySize == 0) {
        return HcclResult::HCCL_SUCCESS; // 零份额：无终态可回搬（count==0 语义，需求 5.2）
    }

    // rpt 递减序：三级 L2 位输出区与 scratch 同块复用，交叠由递减序消解（output(A) ≥
    // j + A·m0·m1 > scratch(B≤A-1) 上界，继承 device 基线证明）；递增序会使低 rpt 输出写先冲掉
    // 高 rpt 未读终态槽。同 rpt 写读同址由同队列 FIFO 保证；镜像保持递增序（两序不可混用）。
    for (u64 rpt = rptNum; rpt > 0; --rpt) {
        const u64 stateOff = RptScratchBase(rpt - 1) + myUnit.offset;
        // 终写 OUT[base+(rpt-1)*ORS] 无 alg-rank 项——对齐全部既有 DPU inter 模板
        // 终写先例（Mesh1dDpu/Mesh1dNDA/Mesh1dDpuInter 均无）。二级 fill 的 outputSliceStride 是
        // scratch 槽位步长语义非输出偏移，原通用式的 OSS 项在二级位消费 SS 语义即错位
        // （rD1/rD3 WQE 实证 dstOffset=SS 越出自身份额输出区）；三级位 OSS=0 零行为差。
        const u64 outOff = tempAlgParams_.buffInfo.outBuffBaseOff + (rpt - 1) * tempAlgParams_.outputRepeatStride;

        DataSlice src = DataSlice(tempAlgParams_.buffInfo.hcclBuff.addr, stateOff, mySize, myCount);
        DataSlice dst = DataSlice(tempAlgParams_.buffInfo.outputPtr, outOff, mySize, myCount);

        HCCL_DEBUG(
            "[InsTempReduceScatterRhdDpuInter][PostLocalCopy] rpt[%llu] myAlgIdx[%u] mySlot[%u] stateOff[%llu] "
            "outOff[%llu]",
            rpt - 1, myAlgIdx_, mySlot, stateOff, outOff);

        // 真同址才跳过：scratch 与输出基址指针不同时（二级位 cclIn vs user 输出）offset 数值相等
        // 不是同址，须比较基址指针（同 MirrorToScratch 的双区隔离修正）
        const bool sameAddr
            = (tempAlgParams_.buffInfo.hcclBuff.addr == tempAlgParams_.buffInfo.outputPtr) && (stateOff == outOff);
        if (!sameAddr) {
            CHK_RET(LocalCopy(q, src, dst));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

// DPU 侧单步执行（host 进程内按名分发：HcclLaunchDPUKernel → REGISTER_TEMPLATE_V2 实例）。
// 校验步骤游标后与本端同源重算步骤表，构造单对端 DpuTransferCtx（单条连续大块 = 1 条 WQE），
// 委托 DpuBatchTransfer 三阶段协议（STEP_SYNC 前同步 → 写 + DATA_SIGNAL notify → Drain/Fence + 完工等待）。
HcclResult InsTempReduceScatterRhdDpuInter::DPUKernelRun(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>>& subCommRanks)
{
    // 防御校验置于 AICPU_COMPILE 守卫之外（不依赖 DPU 传输上下文，仅消费 template_utils RHD
    // 函数族与下发参数，对齐 NhrDpu::DPUKernelRun 的入口校验位置）：host 侧单测可经
    // AICPU_COMPILE 编译本模板并直接覆盖游标/步骤表防御路径；主机侧行为不变（校验恒执行）。
    CHK_PRT_RET(
        subCommRanks.empty() || subCommRanks[0].empty(),
        HCCL_ERROR("[InsTempReduceScatterRhdDpuInter][DPUKernelRun] subCommRanks is empty."), HcclResult::HCCL_E_PARA);
    const u32 rankSize = static_cast<u32>(subCommRanks[0].size());
    CHK_PRT_RET(
        rankSize < 2,
        HCCL_ERROR(
            "[InsTempReduceScatterRhdDpuInter][DPUKernelRun] rankSize[%u] < 2, unreachable (N==1 is "
            "intercepted on AICPU side).",
            rankSize),
        HcclResult::HCCL_E_PARA);
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank, subCommRanks[0], myAlgRank));

    // 步骤游标一致性校验（同版本配套部署约束内的断言级防御：版本错配 → 显式报错而非静默错解）
    CHK_PRT_RET(
        tempAlgParams.rhdStepCnt == 0 || tempAlgParams.rhdStepIdx >= tempAlgParams.rhdStepCnt,
        HCCL_ERROR(
            "[InsTempReduceScatterRhdDpuInter][DPUKernelRun] invalid step cursor rhdStepIdx[%u] rhdStepCnt[%u].",
            tempAlgParams.rhdStepIdx, tempAlgParams.rhdStepCnt),
        HcclResult::HCCL_E_PARA);
    // allRank* 变长分片契约校验（需求 5.2）：三向量等长 = rankSize
    CHK_PRT_RET(
        tempAlgParams.allRankSliceSize.size() != rankSize || tempAlgParams.allRankDispls.size() != rankSize
            || tempAlgParams.allRankProcessedDataCount.size() != rankSize,
        HCCL_ERROR(
            "[InsTempReduceScatterRhdDpuInter][DPUKernelRun] allRank vectors size[%zu/%zu/%zu] != rankSize[%u].",
            tempAlgParams.allRankSliceSize.size(), tempAlgParams.allRankDispls.size(),
            tempAlgParams.allRankProcessedDataCount.size(), rankSize),
        HcclResult::HCCL_E_PARA);
    // 元素字节数直接查表（dataType 由模板在序列化前补填——见
    // RunStepsOldPath 组装 stepParams 时置 stepParams.dataType = param.DataDes.dataType，随
    // tempAlgParams 序列化下发，非 executor 填），与 device 侧取值同源——避免从 allRank 份额除法
    // 反推（变长份额带对齐填充时商≠真实宽度，静默错）
    // 越界前置校验：DATATYPE_SIZE_TABLE 定长于 HCCL_DATA_TYPE_RESERVED，缺位时裸查表为数组越界读
    // （默认构造的 dataType 即 RESERVED）；HcclDataType 为有符号底层枚举，反序列化裸读可落入负值
    // 区间，经无符号化后一并落入拦截（同时封住上下界两个方向）
    CHK_PRT_RET(
        static_cast<u32>(tempAlgParams.dataType) >= HCCL_DATA_TYPE_RESERVED,
        HCCL_ERROR(
            "[InsTempReduceScatterRhdDpuInter][DPUKernelRun] dataType[%d] out of range.",
            static_cast<int>(tempAlgParams.dataType)),
        HcclResult::HCCL_E_PARA);
    const u64 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    CHK_PRT_RET(
        dataTypeSize == 0,
        HCCL_ERROR(
            "[InsTempReduceScatterRhdDpuInter][DPUKernelRun] dataTypeSize is 0, dataType[%d].",
            static_cast<int>(tempAlgParams.dataType)),
        HcclResult::HCCL_E_PARA);

    // 步骤表两侧同源重算（template_utils RHD 函数族，单一事实来源）；
    // 表长不一致 = 配套部署错配或序列化错位，显式报错
    std::vector<RhdStepInfo> flatSteps;
    CHK_RET(BuildRhdFlatSteps(rankSize, myAlgRank, tempAlgParams.rhdSkipPost, flatSteps));
    CHK_PRT_RET(
        static_cast<u32>(flatSteps.size()) != tempAlgParams.rhdStepCnt,
        HCCL_ERROR(
            "[InsTempReduceScatterRhdDpuInter][DPUKernelRun] step table mismatch: local[%zu] remote[%u].",
            flatSteps.size(), tempAlgParams.rhdStepCnt),
        HcclResult::HCCL_E_INTERNAL);

#ifndef AICPU_COMPILE
    const RhdStepInfo& step = flatSteps[tempAlgParams.rhdStepIdx];

    // 角色与对端解析（与 AICPU 侧共用 ResolveStepPeer）
    RhdPart1Info info;
    CHK_RET(CalcRhdPart1Info(rankSize, myAlgRank, INVALID_VALUE_RANKID, info));
    u32 peerAlgRank = INVALID_VALUE_RANKID;
    CHK_RET(ResolveStepPeer(rankSize, info, step, peerAlgRank));
    const u32 peerUserRank = subCommRanks[0].at(peerAlgRank);

    // 链路校验 + 取每对端首通道（host RoCE 只认 ROCE+HOST，对齐 Mesh DPU inter 模板链路校验式；
    // 标准 hostdpu 组网每对端 1 通道，多通道 DPU 切分为非目标）
    auto channelIt = channels.find(peerUserRank);
    CHK_PRT_RET(
        channelIt == channels.end() || channelIt->second.empty(),
        HCCL_ERROR("[InsTempReduceScatterRhdDpuInter][DPUKernelRun] channel to rank[%u] is missing.", peerUserRank),
        HcclResult::HCCL_E_INTERNAL);
    const ChannelInfo& link = channelIt->second[0];
    CHK_PRT_RET(
        !link.isValid || link.handle == 0 || link.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_HOST
            || link.protocol != CommProtocol::COMM_PROTOCOL_ROCE,
        HCCL_ERROR(
            "[InsTempReduceScatterRhdDpuInter][DPUKernelRun] invalid DPU channel to rank[%u], isValid[%d], "
            "handle[0x%llx], locationType[%d], protocol[%d].",
            peerUserRank, link.isValid, link.handle, link.locationType, link.protocol),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        link.remoteCclMem.addr == nullptr,
        HCCL_ERROR(
            "[InsTempReduceScatterRhdDpuInter][DPUKernelRun] remote CCL buffer of rank[%u] is null.", peerUserRank),
        HcclResult::HCCL_E_INTERNAL);

    // 单步传输段构造（每步每对端单条连续大块 = 1 条 WQE，合并发送硬约束）；段偏移/长度经 σ unit
    // 前缀推导（RhdBuildUnitSizes/RhdSegToBytesAllRank，需求 5.1.1/5.2——不消费 sliceSize 寻址；
    // 均匀退化 = 段槽数·SS 与 stride 契约一致）；双端同号槽位对称：tx 落对端 recv 同号槽段，POST
    // 例外直达对端 acc shareSlot 覆盖写。零字节段（count==0 对端语义）不产 WQE。
    // 五个相位分支抽为 BuildDpuStepCtx，DPUKernelRun 保留校验/解析/分派
    std::vector<u64> unitSizes;
    CHK_RET(RhdBuildUnitSizes(rankSize, tempAlgParams.allRankSliceSize, unitSizes));
    RhdSegBytes totalSeg;
    CHK_RET(RhdSegToBytesAllRank(0, rankSize, unitSizes, totalSeg));
    DpuStepCtxParam ctxParam;
    ctxParam.step = &step;
    ctxParam.info = &info;
    ctxParam.unitSizes = &unitSizes;
    ctxParam.dataTypeSize = dataTypeSize;
    ctxParam.localCclBuffAddr = tempAlgParams.buffInfo.hcclBuff.addr;
    ctxParam.remoteCclBuffAddr = link.remoteCclMem.addr;
    ctxParam.accRegionBytes = totalSeg.len;
    ctxParam.rptNum = std::max<u64>(1, tempAlgParams.repeatNum);
    ctxParam.scratchStrideBase = tempAlgParams.buffInfo.hcclBuffBaseOff;
    DpuTransferCtx ctx;
    ctx.txCh = &link;
    ctx.rxCh = &link; // samePeer
    CHK_RET(BuildDpuStepCtx(ctx, ctxParam));

    // 全部段零字节（count==0 对端语义）：跳过传输，请求/响应握手仍由框架完成（对端同段同判，
    // STEP_SYNC 配对不破坏——段字节由全 rank 一致的 allRankSliceSize 推导，双端同值）
    if (ctx.txSrcSlices.empty() && ctx.rxSrcSlices.empty()) {
        HCCL_INFO(
            "[InsTempReduceScatterRhdDpuInter][DPUKernelRun] step[%u/%u] all segments are zero bytes, skip transfer.",
            tempAlgParams.rhdStepIdx + 1, tempAlgParams.rhdStepCnt);
        return HcclResult::HCCL_SUCCESS;
    }

    // DFX 单步传输日志（AC-4 口径：每相位每对端 1 条 WQE、单条长度 = 段槽数·SS）
    HCCL_INFO(
        "[InsTempReduceScatterRhdDpuInter][DPUKernelRun] step[%u/%u] myAlgRank[%u] peerAlgRank[%u] "
        "peerUserRank[%u] txSeg[%u,%u) rxSeg[%u,%u) txSlices[%zu] rxSlices[%zu] singleLen[%llu].",
        tempAlgParams.rhdStepIdx + 1, tempAlgParams.rhdStepCnt, myAlgRank, peerAlgRank, peerUserRank, step.txSegStart,
        step.txSegEnd, step.rxSegStart, step.rxSegEnd, ctx.txSrcSlices.size(), ctx.rxSrcSlices.size(),
        (ctx.txSrcSlices.empty() ? ctx.rxSrcSlices[0].size_ : ctx.txSrcSlices[0].size_));

    std::vector<DpuTransferCtx> pairs = {ctx};
    CHK_RET(DpuBatchTransfer(pairs));
#endif
    return HcclResult::HCCL_SUCCESS;
}

// 五相位切片构造（每相位每 rpt 至多各一条 tx/rx 大块；零字节段不产 WQE）
HcclResult InsTempReduceScatterRhdDpuInter::BuildDpuStepCtx(DpuTransferCtx& ctx, const DpuStepCtxParam& p)
{
#ifndef AICPU_COMPILE
    const RhdStepInfo& step = *p.step;
    const RhdPart1Info& info = *p.info;
    const std::vector<u64>& unitSizes = *p.unitSizes;
    for (u64 rpt = 0; rpt < p.rptNum; rpt++) {
        const u64 scratchBase = p.scratchStrideBase + rpt * (2 * p.accRegionBytes);
        const u64 recvBaseOff = scratchBase + p.accRegionBytes; // recv 区基址（acc 区之后）
        if (step.phase == RhdPhase::PRE && !info.isRepresentative) {
            // 被合并方 PRE：全量镜像段 [0,p) acc → 代表 recv 区（单条大块 = Σ unit 尺寸，write 单边）
            RhdSegBytes seg;
            CHK_RET(RhdSegToBytesAllRank(step.txSegStart, step.txSegEnd, unitSizes, seg));
            if (seg.len == 0) {
                continue; // 全零段：不产 WQE（请求/响应握手由 HcclLaunchDPUKernel 框架完成）
            }
            ctx.txSrcSlices.emplace_back(
                p.localCclBuffAddr, scratchBase + seg.offset, seg.len, seg.len / p.dataTypeSize);
            ctx.txDstSlices.emplace_back(
                p.remoteCclBuffAddr, recvBaseOff + seg.offset, seg.len, seg.len / p.dataTypeSize);
        } else if (step.phase == RhdPhase::MAIN_RS) {
            // 主体步：tx = 对端对齐半区（本端 acc → 对端 recv 同号槽段）；rx = 己方对齐半区
            // （对端 acc → 本端 recv；rxSrc/rxDst 成对填充，rxSrc 同时是 Phase3 DATA_SIGNAL 等待计数）
            RhdSegBytes txSeg;
            CHK_RET(RhdSegToBytesAllRank(step.txSegStart, step.txSegEnd, unitSizes, txSeg));
            RhdSegBytes rxSeg;
            CHK_RET(RhdSegToBytesAllRank(step.rxSegStart, step.rxSegEnd, unitSizes, rxSeg));
            if (txSeg.len == 0 && rxSeg.len == 0) {
                continue; // 双零段：不产 WQE（段内全部 unit 为零份额）
            }
            if (txSeg.len > 0) {
                ctx.txSrcSlices.emplace_back(
                    p.localCclBuffAddr, scratchBase + txSeg.offset, txSeg.len, txSeg.len / p.dataTypeSize);
                ctx.txDstSlices.emplace_back(
                    p.remoteCclBuffAddr, recvBaseOff + txSeg.offset, txSeg.len, txSeg.len / p.dataTypeSize);
            }
            if (rxSeg.len > 0) {
                ctx.rxSrcSlices.emplace_back(
                    p.remoteCclBuffAddr, scratchBase + rxSeg.offset, rxSeg.len, rxSeg.len / p.dataTypeSize);
                ctx.rxDstSlices.emplace_back(
                    p.localCclBuffAddr, recvBaseOff + rxSeg.offset, rxSeg.len, rxSeg.len / p.dataTypeSize);
            }
        } else if (step.phase == RhdPhase::POST && info.isRepresentative) {
            // 代表 POST：被合并方份额槽（1 槽）acc → 对端 acc 同号槽（覆盖写直达，非 recv——旧值 =
            // 被合并方自身 ×1 贡献，被 ×P 终态覆盖，语义同 device 基线 SendWrite 同号槽）
            RhdSegBytes seg;
            CHK_RET(RhdSegToBytesAllRank(step.txSegStart, step.txSegEnd, unitSizes, seg));
            if (seg.len == 0) {
                continue; // 被合并方零份额：无 POST 数据可回传（count==0 语义，需求 5.2）
            }
            ctx.txSrcSlices.emplace_back(
                p.localCclBuffAddr, scratchBase + seg.offset, seg.len, seg.len / p.dataTypeSize);
            ctx.txDstSlices.emplace_back(
                p.remoteCclBuffAddr, scratchBase + seg.offset, seg.len, seg.len / p.dataTypeSize);
        } else if (step.phase == RhdPhase::PRE && info.isRepresentative) {
            // 代表 PRE：仅接收（被合并方全量段落本端 recv 区 [0,p)，rxSrc 为对端发送源镜像/等待计数）
            RhdSegBytes seg;
            CHK_RET(RhdSegToBytesAllRank(step.rxSegStart, step.rxSegEnd, unitSizes, seg));
            if (seg.len == 0) {
                continue;
            }
            ctx.rxSrcSlices.emplace_back(
                p.remoteCclBuffAddr, scratchBase + seg.offset, seg.len, seg.len / p.dataTypeSize);
            ctx.rxDstSlices.emplace_back(
                p.localCclBuffAddr, recvBaseOff + seg.offset, seg.len, seg.len / p.dataTypeSize);
        } else if (step.phase == RhdPhase::POST && !info.isRepresentative) {
            // 被合并方 POST：仅接收（代表覆盖写直达本端 acc shareSlot，非 recv；无本地归约动作）
            RhdSegBytes seg;
            CHK_RET(RhdSegToBytesAllRank(step.rxSegStart, step.rxSegEnd, unitSizes, seg));
            if (seg.len == 0) {
                continue;
            }
            ctx.rxSrcSlices.emplace_back(
                p.remoteCclBuffAddr, scratchBase + seg.offset, seg.len, seg.len / p.dataTypeSize);
            ctx.rxDstSlices.emplace_back(
                p.localCclBuffAddr, scratchBase + seg.offset, seg.len, seg.len / p.dataTypeSize);
        } else {
            CHK_PRT_RET(
                true,
                HCCL_ERROR(
                    "[BuildDpuStepCtx] step has no transfer action for this role, phase[%u].",
                    static_cast<u32>(step.phase)),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
REGISTER_TEMPLATE_V2("InsTempReduceScatterRhdDpuInter", InsTempReduceScatterRhdDpuInter);
#endif
} // namespace ops_hccl
