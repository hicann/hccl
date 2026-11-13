/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_rhd_dpu.h"
#include "alg_data_trans_wrapper.h"
#include "alg_v2_template_register.h"
#include "channel.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {

std::vector<CostModelParam> InsTempAllGatherRHDDPU::CalcCostCoeff(CalcCostCoeffParam param)
{
    (void)param;
    // preferred-only（topo.topoCustomCheck 恒 false，未配置时被软过滤，不参与默认竞争）；占位系数
    // 非空是硬要求：空返回会使算法在 cost 表注册阶段被跳过（cost_model.cc:423-427），显式配置时
    // 无法命中。对齐 RHD 基类与 Dpu 家族先例恒返回 {0, 0, 1, 0}（rhd.cc CalcCostCoeff 同款口径）
    std::vector<CostModelParam> params;
    params.push_back({0.0f, 0.0f, 1.0f, 0.0f});
    return params;
}

InsTempAllGatherRHDDPU::InsTempAllGatherRHDDPU(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsTempAllGatherRHD(param, rankId, subCommRanks)
{}

InsTempAllGatherRHDDPU::~InsTempAllGatherRHDDPU() {}

HcclResult InsTempAllGatherRHDDPU::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // host 网卡资源，不新增从流与对应 notify：DPU 数据面同步由 DpuBatchTransfer 三阶段
    // （STEP_SYNC 前同步 + 写 + DATA_SIGNAL 后同步）承担，主从同步面不存在
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    // 全互连申请（rev2 D1 修正）：对 subcommInfo[COMM_LEVEL0] 域内全部 rank 申请最高 HOST+ROCE 层
    // 通道，全互连 ⊇ RHD 每步对端集（PRE/POST 配对端 t↔t+r 与 MAIN 对端 phys(v⊕2^k) 均为域内
    // rank）；对齐全互连家族先例（ins_temp_reduce_scatter_mesh_1d_dpu.cc 同款调用）。域内全部对端
    // 须存在 HOST+ROCE 链路（缺即 HCCL_E_INTERNAL，与 Nhr 变体同款守卫、仅申请集更大）——
    // host-DPU 拓扑约束（最外层最高 HOST+ROCE 全互连）已蕴含该前提
    CHK_RET(CalcChannelRequestMesh1DHighestHostRoce(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO(
        "[InsTempAllGatherRHDDPU][CalcRes]slaveThreadNum[%u] notifyNumOnMainThread[%u] level1Channels[%zu].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, level1Channels.size());
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRHDDPU::GetRes(AlgResourceRequest& resourceRequest) const
{
    // 与 CalcRes 的线程/notify 需求保持一致：仅主线程，无从线程与 notify 需求。
    // AgSeq3LevelPrepareResForTemplate 通过本接口获取各层线程数做线程分配，缺省实现会报错。
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherRHDDPU::GetThreadNum() const
{
    // 恒单线程：AICPU 侧仅主线程执行装载/卸载与 DPU 协议交互，网络数据面由 DPU 侧执行
    return 1;
}

HcclResult InsTempAllGatherRHDDPU::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllGatherRHDDPU] Run start");
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        HCCL_INFO("[InsTempAllGatherRHDDPU] Rank [%u], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }
    // remote 直写（symMem/图模式）不接入 DPU 路径：三级 Orchestrate 入口已显式拒绝 symMem，
    // 二级 opMode=OFFLOAD 场景由本守卫阻断（数据流 spec §7 支持性契约，防御性拒绝）
    CHK_PRT_RET(
        tempAlgParams.enableRemoteMemAccess,
        HCCL_ERROR(
            "[InsTempAllGatherRHDDPU] Rank [%u], remote memory access is not supported by DPU template.", myRank_),
        HcclResult::HCCL_E_NOT_SUPPORT);
    // 与基类 CheckKernelRunParams 同口径的防御：层内单 rank 由 executor 层跳过本层通信，
    // 到达即参数错误（N==1 无 RHD 网络边）；RHD 无幂守卫（非幂正是目标场景）
    const u32 scopeRankSize = subCommRanks_.empty() ? 0 : static_cast<u32>(subCommRanks_[0].size());
    CHK_PRT_RET(
        scopeRankSize < 2,
        HCCL_ERROR(
            "[InsTempAllGatherRHDDPU] Rank [%u], rankSize[%u] is less than 2, not supported.", myRank_, scopeRankSize),
        HcclResult::HCCL_E_PARA);
    if (templateResource.threads.size() < 1) {
        HCCL_ERROR("[InsTempAllGatherRHDDPU] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(LocalDataCopy(tempAlgParams, templateResource));

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempAllGatherRHDDPU] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
        HCCL_ERROR("[InsTempAllGatherRHDDPU] HcommThreadSynchronize failed");
        return HCCL_E_INTERNAL;
    }

    DPURunInfo dpuRunInfo;
    dpuRunInfo.execTimeout = param.opConfig.execTimeout;
    dpuRunInfo.templateName = "InsTempAllGatherRHDDPU";
    dpuRunInfo.tempAlgParams = tempAlgParams;
    // 回填真实 dtype（对齐 ins_temp_reduce_scatter_nhr_dpu.cc 先例）：两绑定 executor 不显式赋值
    // TemplateDataParams.dataType，基类口径为 param.DataDes.dataType——序列化前回填对齐口径，
    // DPU 侧 dataType_ 与 count_ 折算随之回到真实元素数语义；dataType 本就在序列化字段清单
    // （template_utils.h 读写对），流格式零变化
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
        HCCL_ERROR("[InsTempAllGatherRHDDPU] HcommSendRequest failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempAllGatherRHDDPU] HcommSendRequest run over, sendMsgId[%u]", sendMsgId);

    // 等待DPU数据传输，然后回写结果回来
    void* recvData = nullptr;
    u32 recvMsgId = 0;
    if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("[InsTempAllGatherRHDDPU] HcommWaitResponse failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempAllGatherRHDDPU] HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

    if (recvMsgId != sendMsgId) {
        HCCL_ERROR("[InsTempAllGatherRHDDPU] recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
        return HCCL_E_INTERNAL;
    }

    // 将执行模式转换回到batch
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempAllGatherRHDDPU] failed set batch mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(PostLocalCopy(tempAlgParams, templateResource));

    HCCL_INFO("[InsTempAllGatherRHDDPU] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRHDDPU::DPUKernelRun(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>>& subCommRanks)
{
    // 实例由注册表默认构造（无参），成员状态按本次卸载请求重建（对齐 InsTempAllGatherNHRDPU
    // 同款模式）；AICPU 侧 KernelRun 已保证 scopeRankSize>=2 才卸载
    myRank_ = myRank;
    templateRankSize_ = subCommRanks[0].size();
    subCommRanks_ = subCommRanks;
    CHK_RET(RunAllGatherRHDDpu(tempAlgParams, channels));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempAllGatherRHDDPU::LocalDataCopy(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    u32 algRankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], algRankIdx));
    // 落位公式与基类 LocalDataCopy 同源（数据面恒单 channel [0] 整份：partial 子段偏移/长度退化为
    // 0/sliceSize，尾块 Sequence 绑定恒 tailSize==sliceSize 退化同构）；remote 分支由 KernelRun
    // 入口防御拒绝（不可达），此处双分支：CCL 别名（三级 L2）直写输出网格槽位 / 真拷贝（二级
    // inter）紧凑 CCL 中转——owner 槽位=自身 algRank 槽位（σ 由网络区段 owner 反查消化）
    const bool isOutputCclBuff = tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER;
    for (u64 rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams.buffInfo.inBuffBaseOff + rpt * tempAlgParams.inputRepeatStride;
        // 输入地址含算法 rank 项（对齐基类公式：ISS=0 时为加零项，行为等价）
        const u64 inOff = tempAlgParams.inputSliceStride * algRankIdx + inBaseOff;
        u64 scOff;
        if (isOutputCclBuff) {
            scOff = tempAlgParams.outputSliceStride * algRankIdx + tempAlgParams.buffInfo.hcclBuffBaseOff
                    + rpt * tempAlgParams.outputRepeatStride;
        } else {
            const u64 scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
            const u64 scratchBaseOff = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
            scOff = tempAlgParams.sliceSize * algRankIdx + scratchBaseOff;
        }
        // IN 已在 CCL buffer 且与落点同址同长时跳过本边（基类同款守卫）
        if (tempAlgParams.buffInfo.inputPtr == tempAlgParams.buffInfo.hcclBuff.addr && inOff == scOff) {
            continue;
        }
        DataSlice srcSlice(tempAlgParams.buffInfo.inputPtr, inOff, tempAlgParams.sliceSize, tempAlgParams.count);
        DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, scOff, tempAlgParams.sliceSize, tempAlgParams.count);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempAllGatherRHDDPU::PostLocalCopy(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    if (tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        // CCL 别名绑定（三级 L2）：三相位直写 OUT 网格即最终布局（σ 已由 owner 反查消化），跳过
        // 后拷贝（基类/nhr.cc 先例同款）
        HCCL_INFO("[InsTempAllGatherRHDDPU] PostLocalCopy skip because output is scratch");
        return HcclResult::HCCL_SUCCESS;
    }
    // 真拷贝绑定（二级 inter：OUT 为用户输出网格）——恒等回搬：src/dst 槽位一致（owner 槽位与 OUT
    // 槽位同序，一对一仿射映射；partial 子段退化同 LocalDataCopy 口径）
    for (u64 rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
        const u64 outBaseOff = tempAlgParams.buffInfo.outBuffBaseOff + rpt * tempAlgParams.outputRepeatStride;
        const u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (auto rank : subCommRanks_[0]) {
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
            u64 scratchOffset = tempAlgParams.sliceSize * algRank + scratchBase;
            u64 outOffset = tempAlgParams.outputSliceStride * algRank + outBaseOff;
            DataSlice srcSlice(
                tempAlgParams.buffInfo.hcclBuff.addr, scratchOffset, tempAlgParams.sliceSize, tempAlgParams.count);
            DataSlice dstSlice(
                tempAlgParams.buffInfo.outputPtr, outOffset, tempAlgParams.sliceSize, tempAlgParams.count);
            CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRHDDPU::RunAllGatherRHDDpu(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels)
{
#ifndef AICPU_COMPILE
    // 域信息重建（镜像基类 PrepareKernelRun；DPU 侧实例无参构造，按本次卸载请求重建，无实例级
    // 状态残留）：remote 分支不可达（AICPU 侧 KernelRun 已拒绝）、isDmaRead_ 恒 false（DPU 数据面
    // 恒写模式，host-DPU 链路非 PCIE）
    tempAlgParams_ = tempAlgParams;
    dataType_ = tempAlgParams.dataType;
    isOutputCclBuff_ = (tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER);
    // 显式置否（而非沿袭默认值）：remote 直写分支不接入 DPU 路径（AICPU 侧 KernelRun 已拒绝，
    // 卸载请求不可能携带 remote=true），RhdSlotOffset 恒走非 remote 分支——基类同款
    // PrepareKernelRun 重建语义的 DPU 侧收窄
    enableRemoteMemAccess_ = false;
    scopeRankSize_ = static_cast<u32>(subCommRanks_[0].size());
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    // AllGather 无 root（恒 INVALID_VALUE_RANKID → 代表恒低位 t）；root 与 RunStepDpu 中
    // RhdAlgRankOfVirtual 的 root 入参一致（均为 INVALID，upstream 脚手架契约）
    CHK_RET(CalcRhdPart1Info(scopeRankSize_, myAlgRank, INVALID_VALUE_RANKID, part1Info_));
    mainSteps_.clear();
    // 主体步骤表仅对有效 rank 生成（block 或配对代表；被合并方 PRE 后直等 POST——参与性判据
    // 与基类一致，upstream virtualRank 对被合并方恒有效不能作门控）
    if (!part1Info_.inPart1 || part1Info_.isRepresentative) {
        CHK_RET(GenRhdAllGatherSteps(part1Info_, mainSteps_));
    }
    // AG 族 (SHARE, FULL)：PRE 被合并方发送自身份额段、代表接收镜像；POST 代表发送全量段
    // [0, p)、被合并方接收；2^ 幂（r=0，!inPart1）输出空步骤（peer=INVALID，RunStepDpu 全跳过）
    CHK_RET(GenRhdPrePostSteps(part1Info_, preStep_, postStep_, RhdSegMode::SHARE, RhdSegMode::FULL));

    // WQE 观测汇总（对齐 AICPU 款 InsTempAllGatherRHD 口径）：本 rank 单份发送 WQE 总数。
    // 口径契约：① RPT 不计入——单份口径，只在 rpt==0 份累计（每 rpt 份独立同构，rpt>1 时
    // 打印为单份数）；② skipPrePhase=true 时 PRE 段未执行故计 0（与实际执行面一致）；
    // ③ 空步骤/空 tx 段计 0（空 tx 段不产生 slice）；④ 汇总打印 rankSize 用域内口径
    // scopeRankSize_、stepNum 用 GetRhdStepNum(part1Info_.effectiveRankSize)。状态归属：
    // DPU 侧实例无参构造、每次卸载请求重建域信息、无实例级状态残留（本函数域重建段注释
    // 明示设计原则）——计数用本函数局部变量+RunStepDpu 出参传递，不加实例成员（与 AICPU 款
    // totalWqeNum_ 成员形式的刻意差异，口径一致；且该成员为 private，DPU 子类本就不可达）
    u32 totalWqeNum = 0;
    // 三相位（每 rpt 份独立；与基类 RunAllGatherRHD 镜像）：skipPrePhase 只跳 PRE——POST 恒执行
    // （被合并方须靠 POST 取回全量）；域内一致性契约由调用方保证，本任务两绑定恒 false
    for (u64 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        u32 wqeCnt = 0; // 本步发送 WQE 数（RunStepDpu 出参，每步覆写）
        if (!tempAlgParams_.skipPrePhase) {
            CHK_RET(RunStepDpu(preStep_, rpt, channels, wqeCnt));
            if (rpt == 0) {
                totalWqeNum += wqeCnt; // 单份口径：只在 rpt==0 份累计（RPT 不计入）
            }
        }
        for (const RhdStepInfo& step : mainSteps_) {
            CHK_RET(RunStepDpu(step, rpt, channels, wqeCnt));
            if (rpt == 0) {
                totalWqeNum += wqeCnt;
            }
        }
        CHK_RET(RunStepDpu(postStep_, rpt, channels, wqeCnt));
        if (rpt == 0) {
            totalWqeNum += wqeCnt;
        }
    }
    HCCL_INFO(
        "[InsTempAllGatherRHDDPU][RunAllGatherRHDDpu] rank[%u] rankSize[%u] stepNum[%u] skipPrePhase[%u] "
        "totalWqeNum[%u]",
        myRank_, scopeRankSize_, GetRhdStepNum(part1Info_.effectiveRankSize),
        static_cast<u32>(tempAlgParams_.skipPrePhase), totalWqeNum);
#endif
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRHDDPU::RunStepDpu(
    const RhdStepInfo& step, const u64 rpt, const std::map<u32, std::vector<ChannelInfo>>& channels, u32& wqeCnt)
{
#ifndef AICPU_COMPILE
    // 出参全路径定值：空步骤（peer=INVALID 早退）计 0（口径契约：空段/未执行相位计 0）——
    // 覆写语义下防止调用方累加到上一步的陈旧值（block rank 的 PRE/POST 为空步骤路径）
    wqeCnt = 0;
    u32 peerAlgRank = INVALID_VALUE_RANKID;
    if (step.phase == RhdPhase::MAIN_AG) {
        // 主体对端为虚拟序 v⊕dist → 物理 algRank（physRankSize 取物理规模 p=scopeRankSize_，
        // 非有效规模 P；root 与 CalcRhdPart1Info 一致恒 INVALID——upstream 脚手架契约）
        peerAlgRank = RhdAlgRankOfVirtual(scopeRankSize_, INVALID_VALUE_RANKID, step.peer);
    } else {
        // PRE/POST：物理 algRank（空步骤为 INVALID——2^ 幂 r=0 与 block/无关角色无握手边）
        peerAlgRank = step.peer;
    }
    if (peerAlgRank == INVALID_VALUE_RANKID) {
        return HCCL_SUCCESS; // 空步骤完全跳过（无数据边、无握手边）
    }
    u32 peerRankKey = GetRankFromMap(peerAlgRank);
    // 全互连申请保证域内每个对端都有 channel（CalcRes 建链）；到达即内部错误
    CHK_PRT_RET(
        channels.count(peerRankKey) == 0 || channels.at(peerRankKey).empty(),
        HCCL_ERROR(
            "[InsTempAllGatherRHDDPU][%s] rank[%u] invalid channel access, peerRankKey[%u] channels.size[%zu] "
            "peerChannelSize[%zu]",
            __func__, myRank_, peerRankKey, channels.size(),
            channels.count(peerRankKey) ? channels.at(peerRankKey).size() : 0),
        HCCL_E_INTERNAL);
    // RHD 每步单对端：收发同 channel（收发同偏移直写）；数据面恒取对端 [0] 单通道整 run
    // （NCH=1，每对端 1+ 通道为链路超集，多余通道不参与数据搬运）
    const ChannelInfo& peerChannel = channels.at(peerRankKey)[0];

    DpuTransferCtx ctx;
    ctx.txCh = &peerChannel;
    ctx.rxCh = &peerChannel;
    CHK_RET(BuildStepSlicesDpu(peerChannel, step, rpt, ctx));

    // WQE 观测（对齐基类 RunStepRHD wqeCnt = txSrcSlices.size() 口径，rhd.cc:534-538）：
    // 每条 tx run 恰 1 条 DataSlice→1 条 desc→1 条 WQE（NCH=1 无多通道切分，与基类单
    // channel 口径同粒度）——本步发送 WQE 数 = ctx.txSrcSlices 条数；空 tx 段不产生
    // slice 故计 0（接收侧仅靠 wrapper 握手推进）；RPT 不计入由调用方 rpt==0 门控承担
    wqeCnt = static_cast<u32>(ctx.txSrcSlices.size());

    const char* phaseStr = (step.phase == RhdPhase::PRE) ? "PRE" : (step.phase == RhdPhase::MAIN_AG) ? "MAIN" : "POST";
    HCCL_DEBUG(
        "[InsTempAllGatherRHDDPU][RunStepDpu] rank[%u] phase[%s] step[%u] peer[%u] txSlices[%zu] rxSlices[%zu]",
        myRank_, phaseStr, step.step, peerAlgRank, ctx.txSrcSlices.size(), ctx.rxSrcSlices.size());

    std::vector<DpuTransferCtx> pairs;
    pairs.push_back(ctx);
    // 三阶段批量传输（空 tx 侧接收方仍 Record/NotifyWait、空 rx 侧发送方仍 Wait——wrapper 语义
    // 保证步骤级握手对称；空-空边已在上文 INVALID 拦截）
    CHK_RET(DpuBatchTransfer(pairs));
#endif
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRHDDPU::BuildStepSlicesDpu(
    const ChannelInfo& peerChannel, const RhdStepInfo& step, const u64 rpt, DpuTransferCtx& ctx)
{
#ifndef AICPU_COMPILE
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    void* remoteBuff = peerChannel.remoteCclMem.addr; // remote 分支不可达（AICPU 侧已拒绝）
    void* localBuff = tempAlgParams_.buffInfo.hcclBuff.addr;
    // run 表复用基类 RhdBuildRuns（提取前两处同源 lambda 的合一实现，消除 AICPU/DPU 双份维护
    // 漂移；数据流 spec §5 RO 公式为验收基准）：段（σ unit 区间）→ owner 反查 → 槽位落点 +
    // 字节邻接合并；收发双方同 OSS/ORS/base + 同步骤表 + 同 owner 反查闭式 → run 表逐值一致
    // （收发同偏移直写）。NCH=1 无多通道切分：每 run 恰 1 条 DataSlice → 1 条 WQE
    // （与基类单 channel 口径同粒度）
    // 空 tx 段（代表 PRE / 被合并方 POST 接收侧）不产生 tx slice：接收侧仅靠 wrapper 的
    // Record/NotifyWait 握手推进（写路径仅消费 txSrcSlices，rx 列表为接收等待计数依据）
    std::vector<std::pair<u64, u64>> runs;
    RhdBuildRuns(step.txSegStart, step.txSegEnd, rpt, runs);
    for (const auto& run : runs) {
        if (run.second == 0) {
            continue;
        }
        // count 仅元数据（DPU 写路径按 size_ 字节搬运，HcommWrite 消费 size），语义为元素数：
        // dataType 已由 AICPU 侧 KernelRun 序列化前回填真实 dtype 下传（对齐
        // reduce_scatter_nhr_dpu.cc 先例），run.second / dataTypeSize 即按真实 dtype 折算的元素数
        ctx.txSrcSlices.emplace_back(localBuff, run.first, run.second, run.second / dataTypeSize);
        ctx.txDstSlices.emplace_back(remoteBuff, run.first, run.second, run.second / dataTypeSize);
    }
    // 接收段同构 run 划分（收发同偏移直写：src=对端同偏移、dst=本地 owner 槽位）
    RhdBuildRuns(step.rxSegStart, step.rxSegEnd, rpt, runs);
    for (const auto& run : runs) {
        if (run.second == 0) {
            continue;
        }
        ctx.rxSrcSlices.emplace_back(remoteBuff, run.first, run.second, run.second / dataTypeSize);
        ctx.rxDstSlices.emplace_back(localBuff, run.first, run.second, run.second / dataTypeSize);
    }
#endif
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempAllGatherRHDDPU", InsTempAllGatherRHDDPU);
} // namespace ops_hccl
