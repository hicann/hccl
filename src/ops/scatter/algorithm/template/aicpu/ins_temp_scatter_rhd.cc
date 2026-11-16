/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_scatter_rhd.h"
#include "cost_model.h"

namespace ops_hccl {

// preferred-only：固定占位系数（不可返回空——空系数会使算法在 cost 表注册阶段被跳过，
// 显式配置也无法选中）
std::vector<CostModelParam> InsTempScatterRHD::CalcCostCoeff(CalcCostCoeffParam param)
{
    (void)param;
    std::vector<CostModelParam> params;
    params.push_back({0.0f, 0.0f, 1.0f, 0.0f});
    return params;
}

InsTempScatterRHD::InsTempScatterRHD(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempScatterRHD::~InsTempScatterRHD() {}

u64 InsTempScatterRHD::GetThreadNum() const
{
    u64 threadNum = channelsPerRank_;
    return threadNum;
}

void InsTempScatterRHD::SetRoot(u32 root)
{
    HCCL_INFO("[InsTempScatterRHD][SetRoot] myRank_ [%u], set root_ [%u] ", myRank_, root);
    root_ = root;
}

HcclResult InsTempScatterRHD::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_PTR_NULL(topoInfo);

    // 建链口径对齐 InsTempScatterNHR 的 CalcChannelRequestNhr（±2^k 连接集）；非 2^N 时 RHD
    // 的物理对端经 v→phys 平移会落在 ±2^k 之外（如 p=9、root=0 时 algRank0↔3 距离 3 缺链，
    // 运行时报 E_INTERNAL 且对端挂死），用既有接口 CreateChannelRequestByRankId 补齐缺失
    // 对端；2^N 时连接集与 ±2^k 一致，补齐循环零命中、行为零变化
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level0Channels));
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    std::set<u32> connectRanks;
    CHK_RET(RhdCalcConnectRanks(subCommRanks_[0].size(), myAlgRank, connectRanks));
    for (u32 peerAlgRank : connectRanks) {
        const u32 remoteRank = subCommRanks_[0].at(peerAlgRank);
        bool covered = false;
        for (const auto& desc : level0Channels) {
            if (desc.remoteRank == remoteRank) {
                covered = true;
                break;
            }
        }
        if (covered) {
            continue;
        }
        // CreateChannelRequestByRankId 会 clear 入参向量——经临时向量转接后并入
        std::vector<HcclChannelDesc> peerChannels;
        CHK_RET(CreateChannelRequestByRankId(comm, param, myRank_, remoteRank, peerChannels));
        level0Channels.insert(level0Channels.end(), peerChannels.begin(), peerChannels.end());
        HCCL_INFO(
            "[InsTempScatterRHD][CalcRes] myRank[%u] append channel to remoteRank[%u] (outside NHR +/-2^k set).",
            myRank_, remoteRank);
    }
    resourceRequest.channels.push_back(level0Channels);
    channelsPerRank_ = CalcChannelsPerRank(level0Channels);
    HCCL_DEBUG(" %s channelsPerRank_ is %u ", __func__, channelsPerRank_);
    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterRHD::GetRes(AlgResourceRequest& resourceRequest) const
{
    // 防御：退化层或建链为空时 channelsPerRank_ 可能为 0，threadNum-1 的 u32 下溢会把
    // notifyNumPerThread.assign(0xFFFFFFFF, 1) 变成巨量分配（OOM/length_error），显式置零兜底。
    if (channelsPerRank_ == 0) {
        resourceRequest.slaveThreadNum = 0;
        resourceRequest.notifyNumOnMainThread = 0;
        return HCCL_SUCCESS;
    }
    u32 threadNum = channelsPerRank_;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempScatterRHD::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

HcclResult InsTempScatterRHD::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    u32 rootAlgRank = 0;
    CHK_RET(GetAlgRank(root_, subCommRanks_[0], rootAlgRank));
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    threadNum_ = GetThreadNum();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;

    // 防御校验：非法入参报 HCCL_E_NOT_SUPPORT
    CHK_PRT_RET(
        !RhdScatterParamValid(templateRankSize_, rootAlgRank),
        HCCL_ERROR(
            "[InsTempScatterRHD][KernelRun] invalid params: rankSize[%u] root_[%u] rootAlgRank[%u].", templateRankSize_,
            root_, rootAlgRank),
        HCCL_E_NOT_SUPPORT);
    // 防御校验：非法/保留 dataType 的尺寸表值为 0，下游除法（PreCopy/RunRHD/PostCopy）将除零，
    // 单点前置校验覆盖全部
    CHK_PRT_RET(
        DATATYPE_SIZE_TABLE[dataType_] == 0,
        HCCL_ERROR("[InsTempScatterRHD][KernelRun] dataType[%u] size is 0, invalid.", static_cast<u32>(dataType_)),
        HCCL_E_PARA);
    // 防御校验：tailSize 须为类型整数倍（0 视为无尾；可大于 sliceSize，末位吸收余数）
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(
        !RhdTailSizeValid(tempAlgParams.tailSize, dataTypeSize),
        HCCL_ERROR("[InsTempScatterRHD][KernelRun] invalid tailSize[%llu].", tempAlgParams.tailSize), HCCL_E_PARA);
    // 对称内存/图模式仅支持 2^N（σ=恒等，unit=rank 自然布局一致）；非 2^N 用户 buffer 无 σ 重排语义
    const bool isPowerOfTwo = (templateRankSize_ & (templateRankSize_ - 1)) == 0;
    CHK_PRT_RET(
        enableRemoteMemAccess_ && !isPowerOfTwo,
        HCCL_ERROR(
            "[InsTempScatterRHD][KernelRun] remoteMemAccess only supports power-of-two rankSize[%u].",
            templateRankSize_),
        HCCL_E_NOT_SUPPORT);

    bool isPcieProtocol = IsPcieProtocol(templateResource.channels); // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocol;                                     // 是否使用Read模式
    HCCL_DEBUG("[InsTempScatterRHD] Use Dma Read[%d]", isDmaRead_);

    HCCL_INFO("[InsTempScatterRHD] queNum_ = [%u], threads size = [%zu]", threadNum_, templateResource.threads.size());
    HCCL_INFO("[InsTempScatterRHD] Run Start");
    CHK_PRT_RET(
        templateResource.threads.empty(), HCCL_ERROR("[InsTempScatterRHD][KernelRun] threads is empty"),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        threadNum_ > templateResource.threads.size(),
        HCCL_ERROR("[InsTempScatterRHD] Rank [%u], requiredThread Error.", myRank_), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempAlgParams.buffInfo.hcclBuff.addr);
    if (u32(myRank_) == root_ && tempAlgParams.buffInfo.inBuffType != BufferType::HCCL_BUFFER) {
        CHK_PTR_NULL(tempAlgParams.buffInfo.inputPtr);
    }
    CHK_PTR_NULL(tempAlgParams.buffInfo.outputPtr);
    CHK_RET(PreCopy(tempAlgParams, templateResource.threads));
    CHK_RET(RearrangeToSigma(param, tempAlgParams, templateResource.threads));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        // sub→main 表提前填充（幂等：clear+定值重填）——RunRHD 的多通道步间屏障需复用；
        // 尾部 PostSync 处的再次填充无害
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(RunRHD(templateResource.channels, templateResource.threads, tempAlgParams));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    // 融合直通（fusedWithNextAgPhase，载体 localCopyFlag bit1）：跳过 PostCopy——代表的
    // σ 双份布局留在 ccl 里给相邻 AG 续跑；不跳则被合并方读到未写入的 ccl（脏数据）
    if (!RhdFusedWithNextAg(tempAlgParams)) {
        CHK_RET(PostCopy(tempAlgParams, templateResource.threads));
    }
    HCCL_INFO("[InsTempScatterRHD] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempScatterRHD::PreCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const
{
    if (u32(myRank_) != root_ || tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER
        || enableRemoteMemAccess_) {
        HCCL_INFO("[InsTempScatterRHD][Precopy] skip precopy, myRank = %u, root = %u", myRank_, root_);
        return HCCL_SUCCESS;
    }

    u64 curSliceSize = 0;
    u64 curCount = 0;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        for (u32 algRank = 0; algRank < templateRankSize_; algRank++) {
            u64 srcOffset = r * tempAlgParams.inputRepeatStride + tempAlgParams.inputSliceStride * algRank
                            + tempAlgParams.buffInfo.inBuffBaseOff;
            // 切片落位 σ 布局（RhdUnitStartOfRank；2^N 退化为自然偏移 algRank·sliceSize，与 NHR 一致）
            u64 dstOffset = r * templateRankSize_ * tempAlgParams.sliceSize + tempAlgParams.buffInfo.hcclBuffBaseOff
                            + RhdUnitStartOfRank(templateRankSize_, algRank) * tempAlgParams.sliceSize;

            curSliceSize = tempAlgParams.tailSize != 0 && algRank == templateRankSize_ - 1 ? tempAlgParams.tailSize :
                                                                                             processSize_;
            curCount = curSliceSize / dataTypeSize;
            DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, curSliceSize, curCount);
            DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, dstOffset, curSliceSize, curCount);
            CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), srcSlice, dstSlice)));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterRHD::RearrangeToSigma(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const
{
    // 多级流：上游按自然布局写 ccl（块内 algRank j 在 j·sliceSize），RHD 交换区段连续性要求 σ
    // 布局。仅交换 root（唯一持有数据方）需要置换；2^N 时 σ=恒等，与 NHR 同构零开销。
    const bool isPowerOfTwo = (templateRankSize_ & (templateRankSize_ - 1)) == 0;
    if (isPowerOfTwo || count_ == 0 || u32(myRank_) != root_
        || tempAlgParams.buffInfo.inBuffType != BufferType::HCCL_BUFFER) {
        return HCCL_SUCCESS;
    }

    // σ 环移位暂存借本 rank 接收缓冲尾段（param.outputPtr + [每rank总长−unit, 总长)）：
    // recvbuf 全 rank 有效（sendbuf 仅 root 有效，非全局 root 的层 root 无 input 保证）；
    // 借首份 [0,unit) 在多 loop 下不安全——loop≥1 时它是前序 chunk 已写出的用户输出，
    // 暂存脏写后无人回写（非平凡 σ×多 loop 时行 root 的 chunk0 静默损坏）；尾段由最后一个
    // loop 的终态 post-copy 覆盖，对最终层与非最终层通用，且与 ccl 置换区无重叠。
    CHK_PTR_NULL(param.outputPtr);
    const u64 perRankTotalBytes = static_cast<u64>(param.DataDes.count) * DATATYPE_SIZE_TABLE[dataType_];
    const u64 tmpOffset = perRankTotalBytes > processSize_ ? perRankTotalBytes - processSize_ : 0;
    DataSlice tmpSlice = DataSlice(param.outputPtr, tmpOffset, processSize_, count_);
    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        // done 逐 block 重置：r=0 的环发现会把全部位置标记，带入 r≥1 会使后续 block 一次拷贝
        // 都不做、保持自然布局，RHD 按 σ 区段搬运 → 静默数据错位
        std::vector<bool> done(templateRankSize_, false);
        const u64 blockBase = tempAlgParams.buffInfo.hcclBuffBaseOff + r * templateRankSize_ * processSize_;
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
                tempAlgParams.buffInfo.hcclBuff.addr, blockBase + cyc.back() * processSize_, processSize_, count_);
            DataSlice firstSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr, blockBase + cyc.front() * processSize_, processSize_, count_);
            // 暂存 old[c_last]（其归宿是 c_0，但反向移位会先覆盖它）
            CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), lastSlice, tmpSlice)));
            // 反向移位：c_k ← c_{k-1}（k = m-1..1，读侧恒为未被覆盖的原始数据）
            for (u32 k = cyc.size() - 1; k >= 1; k--) {
                DataSlice srcSlice = DataSlice(
                    tempAlgParams.buffInfo.hcclBuff.addr, blockBase + cyc[k - 1] * processSize_, processSize_, count_);
                DataSlice dstSlice = DataSlice(
                    tempAlgParams.buffInfo.hcclBuff.addr, blockBase + cyc[k] * processSize_, processSize_, count_);
                CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), srcSlice, dstSlice)));
            }
            // c_0 ← 暂存（old[c_last]）
            CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), tmpSlice, firstSlice)));
        }
    }
    // 排空舞步（先例 ins_temp_reduce_scatter_mesh_1D_intra.cc）：保证暂存生命周期内的
    // SDMA 全部完成后才离开本函数（暂存为局部对象，未排空即返回会是悬垂读）
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempScatterRHD][RearrangeToSigma] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }
    // eager 窗口内的失败路径统一走单出口，HcommBatchModeStart 无条件恢复（对齐同 PR DPU 侧
    // 同名函数的修复口径）——否则 algTag 停留在 eager 模式，后续同 tag 算子的 batch 语义被破坏
    HcclResult eagerRet = HCCL_SUCCESS;
    do {
        if (HcommThreadSynchronize(threads.at(0)) != 0) {
            HCCL_ERROR("[InsTempScatterRHD][RearrangeToSigma] HcommThreadSynchronize failed");
            eagerRet = HCCL_E_INTERNAL;
            break;
        }
    } while (false);
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempScatterRHD][RearrangeToSigma] failed set batch mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }
    CHK_RET(eagerRet);
    HCCL_INFO(
        "[InsTempScatterRHD][RearrangeToSigma] rankSize[%u] repeats[%llu] natural layout rearranged to sigma.",
        templateRankSize_, tempAlgParams.repeatNum);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempScatterRHD::PostCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const
{
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 curSliceSize
        = tempAlgParams.tailSize != 0 && myAlgRank == templateRankSize_ - 1 ? tempAlgParams.tailSize : processSize_;
    u64 curCount = curSliceSize / dataTypeSize;
    // 自身份额 unit（σ 布局；2^N 退化为 myAlgRank·sliceSize 与 NHR 一致）；输出侧恒为自然布局
    const u32 ownUnit = RhdUnitStartOfRank(templateRankSize_, myAlgRank);
    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        u64 srcOffset = r * templateRankSize_ * tempAlgParams.sliceSize + tempAlgParams.buffInfo.hcclBuffBaseOff
                        + ownUnit * tempAlgParams.sliceSize;
        u64 dstOffset = tempAlgParams.buffInfo.outBuffBaseOff + myAlgRank * tempAlgParams.outputSliceStride
                        + r * tempAlgParams.sliceSize;
        DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcOffset, curSliceSize, curCount);
        DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr, dstOffset, curSliceSize, curCount);
        if ((tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER && srcOffset == dstOffset)
            || enableRemoteMemAccess_) {
            continue;
        }
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), srcSlice, dstSlice)));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterRHD::RunRHD(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const TemplateDataParams& tempAlgParams)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    u32 rootAlgRank = 0;
    CHK_RET(GetAlgRank(root_, subCommRanks_[0], rootAlgRank));
    std::vector<RhdScatterStep> steps;
    CHK_RET(GenRhdScatterSteps(templateRankSize_, rootAlgRank, myAlgRank, steps, RhdFusedWithNextAg(tempAlgParams)));
    // 退化防御：无步骤或无链路时早退（对空 map 取 begin()->second 是解引用 end() 的 UB）。
    if (steps.empty() || channels.empty()) {
        HCCL_INFO(
            "[InsTempScatterRHD][RunRHD] degenerate layer: steps[%zu] channels[%zu], skip.", steps.size(),
            channels.size());
        return HCCL_SUCCESS;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    // 端口组代表通道表（对齐 NHR PrepareDataSplitForMultiChannel 的取表方式）
    const std::vector<ChannelInfo>& splitChannels = channels.begin()->second;
    HCCL_INFO(
        "[InsTempScatterRHD][RunRHD] rootAlgRank[%u] myAlgRank[%u] rankSize[%u] steps[%zu]", rootAlgRank, myAlgRank,
        templateRankSize_, steps.size());
    // 多通道步间屏障：递归折半的步间数据依赖系统性跨通道——比例切分边界随步长缩半错位
    // （2 通道时 ch1 的 k+1 步发送恰读 ch0 的 k 步接收区段），每通道独立 SDMA 队列无跨线程
    // 序，不同步会读到未落地数据静默错发。逐步 PostSync+PreSync 对构成全屏障（队列 notify
    // 原语）；单通道线程内 FIFO 已保证序，零开销跳过。
    std::vector<ThreadHandle> subThreads;
    if (channelsPerRank_ > 1) {
        CHK_PRT_RET(
            threads.size() < threadNum_,
            HCCL_ERROR(
                "[InsTempScatterRHD][RunRHD] threads size[%zu] < threadNum[%u], step barrier cannot be set up.",
                threads.size(), threadNum_),
            HCCL_E_INTERNAL);
        subThreads.assign(threads.begin() + 1, threads.begin() + threadNum_);
    }
    for (const auto& step : steps) {
        // 合并段字节区间：尾块经 tempAlgParams.tailSize（== sliceSize 时为均匀退化，当前编排层
        // 恒为此值）；尾 algRank 的 σ 槽位恒为末槽（RhdUnitStartOfRankImpl 对 rankSize-1 恒返回
        // 自身，且被合并方永不处于末位——POST 步不携带尾块），tailUnit 取末槽与 σ 布局自洽
        const RhdSegBytes seg = RhdSegToBytesUniform(
            step.segStartUnit, step.segEndUnit, processSize_, tempAlgParams.tailSize, templateRankSize_ - 1);
        // 零字节区段跳过（与 DPU 版同契约）：编排侧 loopTimes=0 已挡 count==0，本守卫为对称
        // 防御——防以空切分结果进入传输循环（0 字节任务或空 vector 越界）
        if (seg.len == 0) {
            HCCL_INFO(
                "[InsTempScatterRHD][RunRHD] segment[%u,%u) covers 0 byte, skip step[%u] phase[%u].", step.segStartUnit,
                step.segEndUnit, step.step, static_cast<u32>(step.phase));
            continue;
        }
        // 多通道：对合并后大段做比例切分（输出每通道仍为单条连续子段）
        std::vector<u64> elemCountOut;
        std::vector<u64> sizeOut;
        std::vector<u64> elemOffset;
        CHK_RET(CalcDataSplitByPortGroup(
            seg.len / dataTypeSize, dataTypeSize, splitChannels, elemCountOut, sizeOut, elemOffset));
        // 观测口径：每步每对端恰一条传输任务
        HCCL_INFO(
            "[RHD-Scatter] step=%u phase=%u peer=%u seg=[%u,%u) bytes=[%llu,%llu)", step.step,
            static_cast<u32>(step.phase), step.peerAlgRank, step.segStartUnit, step.segEndUnit, seg.offset, seg.len);
        for (u32 repeat = 0; repeat < tempAlgParams.repeatNum; repeat++) {
            // 上界用切分表长度（与 CalcDataSplitByPortGroup 的输出同源），而非 channelsPerRank_
            // （各对端最大值）——不对称链路下首对端通道数 < max 时，切分表比 max 短，
            // 按_max_遍历会越界读 elemOffset/sizeOut/elemCountOut（UB）。对通道数更多的
            // 对端只欠利用不丢数据（全段已按首对端通道数比例切分）
            for (u32 channelIdx = 0; channelIdx < splitChannels.size(); channelIdx++) {
                RhdChannelSplit split{&elemCountOut, &sizeOut, &elemOffset, repeat, channelIdx};
                CHK_RET(ScatterStep(step, channels, threads, tempAlgParams, seg, split));
            }
        }
        // 多通道步间全屏障：本步全部通道收发落地（RecvWrite 完成=数据已在本地 ccl）后，任一
        // 线程才进入下一步提交——封死跨通道读未落地的竞态窗口（单通道零开销跳过）
        if (channelsPerRank_ > 1) {
            CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_));
            CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterRHD::ScatterStep(
    const RhdScatterStep& step, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const std::vector<ThreadHandle>& threads, const TemplateDataParams& tempAlgParams, const RhdSegBytes& seg,
    const RhdChannelSplit& split) const
{
    const u32 channelIdx = split.channelIdx;
    const u32 repeat = split.repeat;
    const u32 peerRank = subCommRanks_[0].at(step.peerAlgRank);
    auto peerIt = channels.find(peerRank);
    CHK_PRT_RET(
        peerIt == channels.end() || peerIt->second.empty(),
        HCCL_ERROR("[InsTempScatterRHD][ScatterStep] remoteRank[%u] not found in channels", peerRank), HCCL_E_INTERNAL);
    // 逐对端 bounds：channelIdx 上界来自层级值 channelsPerRank_，安全性依赖建链"每对端条数
    // 一致"的隐式契约——显式校验，把不对称形态下 vector::operator[] 的静默越界变为可诊断的
    // HCCL_E_INTERNAL（threads.at 同步越界会 throw，此处统一失败模式）。
    CHK_PRT_RET(
        channelIdx >= peerIt->second.size(),
        HCCL_ERROR(
            "[InsTempScatterRHD][ScatterStep] remoteRank[%u] channelIdx[%u] exceeds channel num[%zu]", peerRank,
            channelIdx, peerIt->second.size()),
        HCCL_E_INTERNAL);
    const ChannelInfo& link = peerIt->second[channelIdx];
    // 按分支校验实际使用的远端指针（图模式的 remoteOutputGraphMode.addr 与 ccl 的填充条件不同）
    void* remoteBuff = nullptr;
    if (!enableRemoteMemAccess_) {
        CHK_PTR_NULL(link.remoteCclMem.addr);
        remoteBuff = link.remoteCclMem.addr;
    } else {
        CHK_PTR_NULL(link.remoteOutputGraphMode.addr);
        remoteBuff = link.remoteOutputGraphMode.addr;
    }
    // 切分表尺寸防线：正常执行路径下 RunRHD 的循环上界已与切分表同源（splitChannels.size()），
    // 本守卫为函数级自洽防御——防未来调用路径绕过循环上界时静默越界
    CHK_PRT_RET(
        channelIdx >= split.elemOffset->size(),
        HCCL_ERROR(
            "[InsTempScatterRHD][ScatterStep] remoteRank[%u] channelIdx[%u] exceeds split table size[%zu]", peerRank,
            channelIdx, split.elemOffset->size()),
        HCCL_E_INTERNAL);
    void* localBuff
        = (!enableRemoteMemAccess_) ? tempAlgParams.buffInfo.hcclBuff.addr : tempAlgParams.buffInfo.outputPtr;

    // 本步本通道单条连续子段（合并段内按端口组比例切分）；elemOffset 为字节口径，不可再乘 dataTypeSize
    const u64 partialOffset = split.elemOffset->at(channelIdx);
    const u64 partialSize = split.sizeOut->at(channelIdx);
    const u64 partialCount = split.elemCountOut->at(channelIdx);
    u64 srcDstOffset = 0;
    if (!enableRemoteMemAccess_) {
        srcDstOffset = repeat * templateRankSize_ * tempAlgParams.sliceSize + tempAlgParams.buffInfo.hcclBuffBaseOff
                       + seg.offset + partialOffset;
    } else {
        // 图模式直写（KernelRun 已守卫非 2^N 拒绝）：unit=rank 恒等，按 outputSliceStride 定位
        srcDstOffset = tempAlgParams.buffInfo.outBuffBaseOff + step.segStartUnit * tempAlgParams.outputSliceStride
                       + repeat * tempAlgParams.sliceSize + partialOffset;
    }
    DataSlice localSlice = DataSlice(localBuff, srcDstOffset, partialSize, partialCount);
    DataSlice remoteSlice = DataSlice(remoteBuff, srcDstOffset, partialSize, partialCount);
    if (step.isSend) {
        SlicesList txSlicesList({localSlice}, {remoteSlice});
        DataInfo sendData(link, txSlicesList);
        if (isDmaRead_) {
            CHK_PRT_RET(
                static_cast<HcclResult>(SendRead(sendData, threads.at(channelIdx))),
                HCCL_ERROR(
                    "[InsTempScatterRHD][ScatterStep] step[%u] repeat[%u] channelIdx[%u] SendRead failed", step.step,
                    repeat, channelIdx),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                static_cast<HcclResult>(SendBatchWrite(sendData, threads.at(channelIdx))),
                HCCL_ERROR(
                    "[InsTempScatterRHD][ScatterStep] step[%u] repeat[%u] channelIdx[%u] SendWrite failed", step.step,
                    repeat, channelIdx),
                HcclResult::HCCL_E_INTERNAL);
        }
    } else {
        // 接收步：rx 与 tx 同段同偏移（同构布局），RecvWrite 按既有 NHR 模式做计数与同步
        SlicesList rxSlicesList({remoteSlice}, {localSlice});
        DataInfo recvData(link, rxSlicesList);
        if (isDmaRead_) {
            CHK_PRT_RET(
                static_cast<HcclResult>(RecvRead(recvData, threads.at(channelIdx))),
                HCCL_ERROR(
                    "[InsTempScatterRHD][ScatterStep] step[%u] repeat[%u] channelIdx[%u] RecvRead failed", step.step,
                    repeat, channelIdx),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                static_cast<HcclResult>(RecvWrite(recvData, threads.at(channelIdx))),
                HCCL_ERROR(
                    "[InsTempScatterRHD][ScatterStep] step[%u] repeat[%u] channelIdx[%u] RecvWrite failed", step.step,
                    repeat, channelIdx),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempScatterRHD::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempScatterRHD::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}
} // namespace ops_hccl
