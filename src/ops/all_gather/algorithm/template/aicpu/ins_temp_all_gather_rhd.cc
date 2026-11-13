/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_rhd.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {

std::vector<CostModelParam> InsTempAllGatherRHD::CalcCostCoeff(CalcCostCoeffParam param)
{
    (void)param;
    // RHD preferred-only（topo.topoCustomCheck 恒 false，不参与默认竞争）；占位系数非空是硬要求：
    // 空返回会使算法在 cost 表注册阶段被跳过（cost_model.cc:423-427），显式配置时无法命中。
    // 对齐 Dpu 家族先例恒返回 {{0, 0, 1, 0}}，不参与比价（显式配置时为唯一候选）
    std::vector<CostModelParam> params;
    params.push_back({0.0f, 0.0f, 1.0f, 0.0f});
    return params;
}

InsTempAllGatherRHD::InsTempAllGatherRHD(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempAllGatherRHD::~InsTempAllGatherRHD() {}

HcclResult InsTempAllGatherRHD::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> level0Channels;
    CHK_PRT_RET(topoInfo == nullptr, HCCL_ERROR("[InsTempAllGatherRHD][CalcRes] topoInfo is nullptr"), HCCL_E_PARA);
    std::vector<HcclChannelDesc> myChannelDescs;
    // 单层 MESH_1D 全互连申请；过滤 RoCE 协议链路（与既有 AICPU all_gather 模板的过滤方式一致）。
    // RHD 逻辑对端（主体 XOR 伙伴 + part1 配对端）⊆ 域内全体，全互连为安全超集
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, myChannelDescs));
    for (auto channel : myChannelDescs) {
        if (channel.channelProtocol != CommProtocol::COMM_PROTOCOL_ROCE) {
            level0Channels.push_back(channel);
        }
    }
    resourceRequest.channels.push_back(level0Channels);
    channelsPerRank_ = CalcChannelsPerRankMin(level0Channels);
    maxChannelsPerRank_ = CalcChannelsPerRank(level0Channels);
    HCCL_INFO(
        "[InsTempAllGatherRHD][CalcRes] channelsPerRank_ is %u, maxChannelsPerRank_ is %u ", channelsPerRank_,
        maxChannelsPerRank_);
    // 过滤 ROCE 通道后可能为空（全 RoCE 拓扑/本 rank 无远端通道），
    // maxChannelsPerRank_ 为 0 时 GetRes 的 threadNum-1 会 u32 下溢
    CHK_PRT_RET(
        maxChannelsPerRank_ == 0, HCCL_ERROR("[InsTempAllGatherRHD][CalcRes] maxChannelsPerRank_ is 0."),
        HcclResult::HCCL_E_INTERNAL);
    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRHD::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 threadNum = GetThreadNum();
    HCCL_INFO("[InsTempAllGatherRHD][GetRes] threadNum[%u]", threadNum);
    resourceRequest.slaveThreadNum = threadNum - 1;
    // 每个从流 1 个 notify 用于主从流同步（RHD 三相位在每 channel 主循环内串行，无 last-step 并行优化）
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherRHD::GetThreadNum() const
{
    // 线程数等于每 rank 最大 channel 数；MESH_1D 单 channel 时为 1（单线程，无主从同步开销）。
    // 值来自 CalcRes 对过滤后 level0Channels 的 CalcChannelsPerRank 计算（覆盖头文件默认值 1；空输入时
    // 该函数返回初始值 1 而非 0，不构成下溢）；若异常为 0，由 CalcRes 守卫拦截 HCCL_E_INTERNAL，不进入 GetRes。
    return maxChannelsPerRank_;
}

HcclResult InsTempAllGatherRHD::SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    CHK_PRT_RET(channels.empty(), HCCL_ERROR("[SetchannelsPerRank] channels is empty."), HCCL_E_INTERNAL);
    channelsPerRank_ = CalcChannelsPerRankMin(channels);
    maxChannelsPerRank_ = CalcChannelsPerRank(channels);
    HCCL_INFO(
        "[InsTempAllGatherRHD][SetchannelsPerRank] channelsPerRank_[%u], maxChannelsPerRank_[%u]", channelsPerRank_,
        maxChannelsPerRank_);
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherRHD::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // staging 为 rank 槽位网格（每 rank 份一段）：一轮 loop 的 staging 覆盖 = p 个 unit（方案乙
    // owner 槽位与 HD/NHR 同构），倍数与 HD/NHR 相同（不含 RPT；三级 L1 的 RPT=d2 由 executor 外乘）
    u64 scratchMultiple = templateRankSize_;
    return scratchMultiple;
}

HcclResult InsTempAllGatherRHD::PrepareDataSplitForMultiChannel(const TemplateResource& templateResource)
{
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalDataCount = tempAlgParams_.sliceSize / dataTypeSize;
    if (templateResource.channels.empty() || templateResource.channels.begin()->second.empty()) {
        dataSplit_.assign(1, tempAlgParams_.sliceSize);
        dataOffset_.assign(1, 0);
        if (tempAlgParams_.tailSize > 0) {
            dataSplitTail_.assign(1, tempAlgParams_.tailSize);
            dataOffsetTail_.assign(1, 0);
        }
        return HCCL_SUCCESS;
    }
    std::vector<u64> elemCountOut;
    CHK_RET(CalcDataSplitByPortGroup(
        totalDataCount, dataTypeSize, templateResource.channels.begin()->second, elemCountOut, dataSplit_,
        dataOffset_));
    if (tempAlgParams_.tailSize > 0) {
        u64 totalDataCountTail = tempAlgParams_.tailSize / dataTypeSize;
        CHK_RET(CalcDataSplitByPortGroup(
            totalDataCountTail, dataTypeSize, templateResource.channels.begin()->second, elemCountOut, dataSplitTail_,
            dataOffsetTail_));
    }

    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRHD::CheckKernelRunParams(const TemplateResource& templateResource)
{
    // RHD 无幂守卫（非 2^幂层内维度正是前/后处理的目标场景），拓扑门槛仅层级匹配
    // （TopoMatch HD 门控按 AlgoType::HD 精确匹配，RHD 不触发）。按算法作用域 subCommRanks_[0]
    // 校验（基类 templateRankSize_ 在多组绑定时为两段乘积语义）：rankSize==0/1 防御报参数错误
    // （N==1 二级绑定时由 TopoMatchTwoLevel d1==1 拦截兜底）
    const u32 scopeRankSize = subCommRanks_.empty() ? 0 : static_cast<u32>(subCommRanks_[0].size());
    if (scopeRankSize < 2) {
        HCCL_ERROR(
            "[InsTempAllGatherRHD] Rank [%u], rankSize[%u] is less than 2, not supported.", myRank_, scopeRankSize);
        return HcclResult::HCCL_E_PARA;
    }
    threadNum_ = GetThreadNum();
    if (templateResource.threads.size() < threadNum_) {
        HCCL_ERROR(
            "[InsTempAllGatherRHD] Rank [%u], thread num[%zu] is not as expected[%u].", myRank_,
            templateResource.threads.size(), threadNum_);
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRHD::PrepareKernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    // CCL 别名绑定判定（nhr.cc:170 先例）：决定网络落点公式（RhdSlotOffset 双情形）
    isOutputCclBuff_ = (tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER);

    bool isPcieProtocol = IsPcieProtocol(templateResource.channels); // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocol;                                     // 是否使用Read模式
    HCCL_DEBUG("[InsTempAllGatherRHD] Use Dma Read[%d]", isDmaRead_);
    CHK_RET(PrepareDataSplitForMultiChannel(templateResource));

    // RHD 域信息与步骤表（upstream 脚手架；每次 KernelRun 按当前域重算，无实例级状态残留；
    // 二级/三级共用同一模板类，subCommRanks_[0] 动态取域——三级 L1/L2 域与二级 intra 域同一落点公式）
    scopeRankSize_ = static_cast<u32>(subCommRanks_[0].size());
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    // AllGather 无 root（恒 INVALID_VALUE_RANKID → 代表恒低位 t）；root 须与 RunStepRHD 中
    // RhdAlgRankOfVirtual 的 root 入参一致（均为 INVALID，脚手架契约）
    CHK_RET(CalcRhdPart1Info(scopeRankSize_, myAlgRank, INVALID_VALUE_RANKID, part1Info_));
    mainSteps_.clear();
    // 主体步骤表仅对有效 rank 生成。upstream CalcRhdPart1Info 的 virtualRank 对被合并方恒有效
    // （myAlgRank % r，与代表同值），不能沿用旧"virtualRank != INVALID"门控——被合并方若误入主体，
    // 会以代表的虚拟序与对端交互造成协议错配。参与性判据：block（!inPart1）或配对代表
    // （isRepresentative）；被合并方 PRE 后直等 POST
    if (!part1Info_.inPart1 || part1Info_.isRepresentative) {
        CHK_RET(GenRhdAllGatherSteps(part1Info_, mainSteps_));
    }
    // AG 族 (SHARE, FULL)：PRE 被合并方发送自身份额段、代表接收镜像；POST 代表发送全量段
    // [0, p)、被合并方接收；2^ 幂（r=0，!inPart1）输出空步骤（peer=INVALID，RunStepRHD 全跳过）
    CHK_RET(GenRhdPrePostSteps(part1Info_, preStep_, postStep_, RhdSegMode::SHARE, RhdSegMode::FULL));
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRHD::SyncInterThreads(const std::vector<ThreadHandle>& threads, bool isPreSync)
{
    std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.begin() + threadNum_);
    if (isPreSync) {
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
    } else {
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRHD::RunChannelMainLoop(const TemplateResource& templateResource)
{
    if (channelsPerRank_ <= 1) {
        // 单通道路径（Checker 全部证据形态）：逐字保留原"通道-外层"结构，零行为漂移——单通道下
        // appendSlices 每条 run 整段归 channel 0，"通道 c 只搬每槽位第 c 子段"约定天然自洽，
        // 无步间跨流竞争窗口，不引入锁步与屏障开销
        for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
            if (dataSplit_[channelIdx] != 0) {
                // 装载空子段仅跳过本地拷贝边；网络步骤（三相位收发/握手）不跳过——
                // 段级切分与本 rank 份切分相互独立，空握手保证 channel 级 notify 协议对称
                CHK_RET(LocalDataCopy(templateResource.threads, channelIdx)); // input buffer拷贝到scratch buffer上
            }
            if (templateRankSize_ > 1) {
                u32 channelWqeNum = 0;
                CHK_RET(
                    RunAllGatherRHD(templateResource.threads, templateResource.channels, channelIdx, channelWqeNum));
                if (channelIdx == 0) {
                    totalWqeNum_ = channelWqeNum; // WQE 计数取 channel 0 口径（多通道每 channel 独立同构）
                }
            }
            CHK_RET(PostLocalCopy(templateResource.threads[channelIdx], channelIdx));
        }
        return HCCL_SUCCESS;
    }
    // ===== 多通道锁步（步骤-外层 / 通道-内层；PR#3933 检视 CRITICAL 修复，用户 2026-11-09 裁决方案） =====
    // 竞争窗口：RhdBuildRuns 把字节邻接 unit 合并成大 run（2^幂紧凑网格整段合并、三级 L1、真拷贝中转必然
    // 合并），appendSlices 再按通道比例切分——切分点不对齐槽位边界，打破 HD/NHR 维持的"通道 c 只搬
    // 每槽位第 c 子段"逐流自洽约定。步骤 k 的 rx 与步骤 k+1 的合并 tx 切分点不同（如 rx [U,2U) 切
    // 1.5U、tx [0,2U) 切 U），ch1 步骤 k+1 要发的 [U,1.5U) 是 ch0 步骤 k 才写入 scratch 的；原
    // "通道-外层"结构下 KernelRun 仅循环前后各一次主从同步，步间无跨流屏障 → 静默数据错误。
    // 修复：所有 channel 以同一步骤表锁步推进，每步 RHD 后加线程间流同步。
    // 屏障=集中式两相位：到达相 SyncInterThreads(false)=PostSync（各从流 record→主流、主流 wait 全部，
    // 主流自身步工作因队列 FIFO 先于 wait，天然到场）；放行相 SyncInterThreads(true)=PreSync（主流
    // record→各从流、从流 wait）。二者背靠背无环无死锁（依赖链沿步骤号单调递增）。notify 槽位与外层
    // KernelRun Pre/PostSync 同池（主→从 idx 0 / 从→主 idx 0..N-2），底层 wait 为消费式
    // （hrtNotifyWaitWithTimeOut=aclrtWaitAndResetNotify），record→wait 严格成对串行复用，与框架
    // loopTimes 多轮 KernelRun 复用同槽先例同构；屏障对对端不可见，每 channel 对端协议序不变
    // （对端逐 channel 兼容）。屏障全线程参与（threads[0..threadNum_-1]，含 dataSplit_==0 与
    // channelIdx>=channelsPerRank_ 的空闲线程，缺一即死锁）。
    auto barrierInterThreads = [&]() -> HcclResult {
        CHK_RET(SyncInterThreads(templateResource.threads, false)); // 到达相（PostSync 汇合）
        return SyncInterThreads(templateResource.threads, true);    // 放行相（PreSync 放行）
    };
    const u64 rptNum = tempAlgParams_.repeatNum;
    const u64 preBarrierNum = tempAlgParams_.skipPrePhase ? 0 : 1; // skipPrePhase=true 时无 PRE 相位屏障
    const u64 barrierNum = (templateRankSize_ > 1) ? 1 + rptNum * (preBarrierNum + mainSteps_.size() + 1) : 0;
    HCCL_INFO(
        "[InsTempAllGatherRHD][RunChannelMainLoop] rank[%u] multi-channel step-lockstep: channelsPerRank[%u] "
        "threadNum[%u] rptNum[%llu] stepNum[%zu] barrierNum[%llu]",
        myRank_, channelsPerRank_, threadNum_, rptNum, mainSteps_.size(), barrierNum);
    // 1) 装载（通道-内层；空子段仅跳过本地拷贝边，网络步骤与屏障照旧——空握手保证 channel 级 notify
    //    协议对称）。装载后 1 道屏障：步骤 k 网络段的合并 run 子段可覆盖他通道装载字节
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        if (dataSplit_[channelIdx] != 0) {
            CHK_RET(LocalDataCopy(templateResource.threads, channelIdx)); // input buffer拷贝到scratch buffer上
        }
    }
    if (templateRankSize_ > 1) {
        CHK_RET(barrierInterThreads());
        std::vector<u32> channelWqeNums(channelsPerRank_, 0); // 每通道独立计数；totalWqeNum_ 取 channel 0 口径
        // 2) 网络三相位锁步（每 rpt 份独立；RPT 份间 CCL 上不连续，不能合并为单条 desc）：
        //    PRE/每主体步/POST 后各 1 道屏障（步间写读全覆盖）
        for (u64 rpt = 0; rpt < rptNum; ++rpt) {
            // 2a. 前处理：被合并方 t+r 发自身份额 unit → 代表同偏移直写其 owner 槽位（skipPrePhase
            //     =true 时整相位连同屏障跳过，与 RunAllGatherRHD 门控一致）
            if (!tempAlgParams_.skipPrePhase) {
                for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
                    CHK_RET(RunStepRHD(
                        templateResource.threads, templateResource.channels, channelIdx, preStep_, rpt,
                        channelWqeNums[channelIdx]));
                }
                CHK_RET(barrierInterThreads());
            }
            // 2b. 主体：⌊log₂P⌋ 步倍增（仅有效 rank——代表/block；被合并方前处理后直等后处理）
            for (const RhdStepInfo& step : mainSteps_) {
                for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
                    CHK_RET(RunStepRHD(
                        templateResource.threads, templateResource.channels, channelIdx, step, rpt,
                        channelWqeNums[channelIdx]));
                }
                CHK_RET(barrierInterThreads());
            }
            // 2c. 后处理：代表 t 发全量段 → 被合并方同偏移直写覆盖（收完即最终结果）
            for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
                CHK_RET(RunStepRHD(
                    templateResource.threads, templateResource.channels, channelIdx, postStep_, rpt,
                    channelWqeNums[channelIdx]));
            }
            // POST 后屏障兼护下一 rpt 的 PRE 与（末轮）真拷贝卸载：PostLocalCopy ch_c 读各槽位
            // [dataOffset_[c],+dataSplit_[c]) 子段，写方为任意通道
            CHK_RET(barrierInterThreads());
        }
        totalWqeNum_ = channelWqeNums[0]; // WQE 计数取 channel 0 口径（多通道每 channel 独立同构）
    }
    // 3) 卸载（通道-内层）：末轮 POST 后屏障已保证全部通道网络写完成。templateRankSize_<=1 退化时
    //    无网络相位，通道间装载/卸载子段分区不相交且同线程 FIFO，无需屏障（与原结构行为一致）
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(PostLocalCopy(templateResource.threads[channelIdx], channelIdx));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRHD::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllGatherRHD] Run start");
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        HCCL_INFO("[InsTempAllGatherRHD] Rank [%u], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }
    totalWqeNum_ = 0;
    CHK_RET(CheckKernelRunParams(templateResource));
    CHK_RET(PrepareKernelRun(param, tempAlgParams, templateResource));
    if (threadNum_ > 1) {
        CHK_RET(SyncInterThreads(templateResource.threads, true)); // 预同步
    }
    CHK_RET(RunChannelMainLoop(templateResource));
    if (threadNum_ > 1) {
        CHK_RET(SyncInterThreads(templateResource.threads, false)); // 后同步
    }
    // WQE 观测汇总（域内 p 口径 = scopeRankSize_；RPT 不计入——每 rpt 份独立计数，channel 0 口径）。
    // σ 交错 run 口径：2^ 幂（r=0）σ 恒等 → 每步每对端恒 1 条（整段合并，先行 HD 路径形态）；
    // 非幂 part1 覆盖段逐 unit 断 run（被合并方 PRE=1 条、代表 POST ≤ 2r+1 条）、block 覆盖段
    // 紧凑网格 1 条 / 稀疏网格逐 unit；空子段/握手方为 0
    HCCL_INFO(
        "[InsTempAllGatherRHD] rank[%u] rankSize[%u] stepNum[%u] totalWqeNum[%u]", myRank_, scopeRankSize_,
        GetRhdStepNum(part1Info_.effectiveRankSize), totalWqeNum_);

    HCCL_INFO("[InsTempAllGatherRHD] Run End");
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempAllGatherRHD::GetRankFromMap(const u32 algRankIdx) const { return subCommRanks_[0].at(algRankIdx); }

u64 InsTempAllGatherRHD::RhdSlotOffset(const u64 owner, const u64 rpt) const
{
    if (enableRemoteMemAccess_) {
        // remote（symMem/图模式）：OUT 用户缓冲直写（幂/非幂统一——σ 交错由区段 run 的 owner 反查消化）
        return tempAlgParams_.buffInfo.outBuffBaseOff + owner * tempAlgParams_.outputSliceStride
               + rpt * tempAlgParams_.outputRepeatStride;
    }
    if (isOutputCclBuff_) {
        // 情形 A：CCL 别名绑定（OUT≡SCR 同缓冲同网格）网络直写输出网格（nhr.cc:249-252/488-490 同款）；
        // 三绑定点（二级 inter/三级 L1/三级 L2）的 outBuffBaseOff 与 hcclBuffBaseOff 逐一相等
        return tempAlgParams_.buffInfo.hcclBuffBaseOff + owner * tempAlgParams_.outputSliceStride
               + rpt * tempAlgParams_.outputRepeatStride;
    }
    // 情形 B：真拷贝绑定（CCU）紧凑 CCL 中转，每 rpt 一份 N·S 平铺
    const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
    return tempAlgParams_.buffInfo.hcclBuffBaseOff + owner * tempAlgParams_.sliceSize + rpt * scratchRepeatStride;
}

u32 InsTempAllGatherRHD::RhdOwnerOfUnit(const u32 unit) const
{
    // σ⁻¹ 闭式（RhdUnitStartOfRank 反查；upstream 脚手架只提供正向 σ）：
    //   unit ≥ 2r → block：owner = unit；unit < 2r 偶 → 代表 t = unit/2；unit < 2r 奇 → 被合并方 t+r = unit/2 + r
    // 与 σ 互为双射（p 任意 ≥2；r=0/1 时 σ 恒等，owner = unit）
    const u32 r = part1Info_.r;
    if (unit >= 2 * r) {
        return unit;
    }
    return (unit / 2) + ((unit % 2 == 0) ? 0 : r);
}

void InsTempAllGatherRHD::RhdBuildRuns(
    const u32 segStart, const u32 segEnd, const u64 rpt, std::vector<std::pair<u64, u64>>& runs) const
{
    // 段构造核心（σ 感知 owner 槽位 run 划分，适用 tx/rx/PRE/POST 一切段；提取自基类
    // BuildStepSlice 与 DPU 版 BuildStepSlicesDpu 两处同源 lambda，PR#3933 评审合一，零语义变更）：
    //   段（σ unit 区间 [segStart, segEnd)）→ 逐 unit 反查 owner=σ⁻¹(u) → owner 槽位落点
    //   run(u) = [RhdSlotOffset(owner(u), rpt), unitLen(u))；
    //   合并判据 = 字节邻接（owner(u+1) 落点 == 当前 run 端点）——同时覆盖 owner 连续性与网格密度：
    //     part1 区（r≥2）：相邻 unit 分属 rank t 与 t+r，槽位交错不邻接 → 逐 unit 断 run（σ 交错的
    //       根本代价，方案乙裁决点，头文件类注释论证）；
    //     block 区（u≥2r）及 2r−1/2r 边界：owner 连续，紧凑网格（stride==S：二级 inter 正常、
    //       三级 L1 正常、三级 L2 d1==1 退化、真拷贝紧凑 SCR）→ 整段合并 1 条 run；
    //     稀疏网格（stride≠S：三级 L2 正常 OSS=S·d1、remote/CCU OSS=d0·S 或 d0·d1·S）→ 永不邻接
    //       → 逐 unit 1 条 run（NHR 同粒度；单条连续 WQE 会夹带间隙垃圾字节且末 run 端点越出
    //       per-loop CCL 预算——前任务 rev3 F2 教训，逐 run 构造同时消除：末 run 端点
    //       base+owner·OSS+S ≤ 布局上界，恒在预算内）；
    //     r=0（2^ 幂）σ 恒等 → 全段 owner 连续，紧凑网格整段 1 条 run（先行 HD 路径形态逐字节一致）。
    //   收发双方同 OSS/ORS/base（executor 契约对称）+ 同步骤表 + 同 owner 反查闭式 → run 表逐值
    //   一致，切点一致（收发同偏移直写）。
    const u64 sliceSize = tempAlgParams_.sliceSize;
    const u32 tailUnit = scopeRankSize_ - 1; // 尾 unit 恒为布局最后一个（Sequence 绑定恒 tailSize==S 退化）
    // unit 长度取 upstream RhdSegToBytesUniform 的单 unit 区间长度（内含尾块规则、仅一次乘法；
    // 其返回 offset 为 σ 紧凑布局偏移，方案乙 owner 槽位空间不用，仅取 len）
    auto unitLenOf = [&](const u32 unit) -> u64 {
        return RhdSegToBytesUniform(unit, unit + 1, sliceSize, tempAlgParams_.tailSize, tailUnit).len;
    };
    runs.clear();
    if (segEnd <= segStart) {
        return;
    }
    u64 runStart = 0;
    u64 runLen = 0;
    bool hasRun = false;
    for (u32 unit = segStart; unit < segEnd; ++unit) {
        const u64 off = RhdSlotOffset(RhdOwnerOfUnit(unit), rpt);
        const u64 len = unitLenOf(unit);
        if (hasRun && off == runStart + runLen) {
            runLen += len; // 字节邻接：并入当前 run（block 区连续 owner 紧凑网格）
        } else {
            if (hasRun) {
                runs.emplace_back(runStart, runLen);
            }
            runStart = off;
            runLen = len;
            hasRun = true;
        }
    }
    if (hasRun) {
        runs.emplace_back(runStart, runLen);
    }
}

HcclResult InsTempAllGatherRHD::BuildStepSlice(
    const ChannelInfo& peerChannel, const std::map<u32, std::vector<ChannelInfo>>& channels, const RhdStepInfo& step,
    const u64 rpt, const u32& channelIdx, std::vector<DataSlice>& txSrcSlices, std::vector<DataSlice>& txDstSlices,
    std::vector<DataSlice>& rxSrcSlices, std::vector<DataSlice>& rxDstSlices)
{
    txSrcSlices.clear();
    txDstSlices.clear();
    rxSrcSlices.clear();
    rxDstSlices.clear();
    if (step.txSegEnd <= step.txSegStart && step.rxSegEnd <= step.rxSegStart) {
        // 空数据边（握手方）：tx 与 rx 均空——空 slice 列表（RunBatchTransfer 空循环零 WARNING，仅握手
        // 生效）。tx 空而 rx 非空（读模式拉取段——代表 PRE rx=[pair 份额 unit、被合并方 POST
        // rx=[0,p)，upstream GenRhdPrePostSteps 冻结语义）不得早退，须进入 rx 段构造：读路径
        // SendRecvRead 仅消费 rxSlicesList_.srcSlices_（wrapper 取 src）；写路径 SendRecvBatchWrite
        // 仅消费 txSlicesList_（DoSendRecvBatchTx 取 tx）——rx 段对写模式零影响
        return HCCL_SUCCESS;
    }
    // 主体步 nSlices 为步距 dist（upstream 脚手架语义，非区段 unit 数）；区段字段 txSeg*/rxSeg*
    // 一律为 σ 物理单元空间，本函数以 owner 反查落位，不消费 nSlices

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    void* sendRemoteBuff
        = (!enableRemoteMemAccess_) ? peerChannel.remoteCclMem.addr : peerChannel.remoteOutputGraphMode.addr;
    void* recvRemoteBuff = sendRemoteBuff; // RHD 每步单对端：send/recv 同 channel
    void* localBuff
        = (!enableRemoteMemAccess_) ? tempAlgParams_.buffInfo.hcclBuff.addr : tempAlgParams_.buffInfo.outputPtr;

    // 多通道按比例切分（逐 run 独立调 CalcDataSplitByPortGroup；单 channel 时整 run 一条）：
    // 切分点由各 run 长度与 channel 端口组唯一决定，收发双方同 run 表 → 切点一致（紧凑网格
    // 1 run/步时与整段一次切分逐字节一致）；本方 tx 子段拼接 = 本方 tx 段全量（write desc 由
    // 本方发起、落对端同偏移，无跨方切点耦合）；空子段跳过数据边（空 slice 列表，握手协议保持对称）。
    // 切分基准与装载预切分一致（channels.begin()->second，HD 先例口径）
    const std::vector<ChannelInfo>& splitChannels = channels.begin()->second;
    auto appendSlices = [&](const u64 runOff, const u64 runBytes, void* remoteBuff, bool isTx) -> HcclResult {
        if (runBytes == 0) {
            return HCCL_SUCCESS;
        }
        u64 runCount = runBytes / dataTypeSize;
        std::vector<u64> elemCountOut;
        std::vector<u64> dataSplit;
        std::vector<u64> dataOffset;
        CHK_RET(CalcDataSplitByPortGroup(runCount, dataTypeSize, splitChannels, elemCountOut, dataSplit, dataOffset));
        if (channelIdx >= dataSplit.size() || dataSplit[channelIdx] == 0) {
            return HCCL_SUCCESS; // 本 channel 空子段：跳过数据边（发送/拉取方各自独立切分）
        }
        // 字节口径契约（前任务 a6 9/11 FAIL 根因消解，ag-rhd-sequence-20261106 续修口径）：
        // CalcDataSplitByPortGroup 输出——dataSplit(sizeOut) 已是字节（elemCount×dts）、
        // dataOffset(elemOffset) 已是字节偏移累计（ΣsizeOut[<ch]）、elemCountOut 为元素数。
        // 消费侧取原值，不得再乘 dataTypeSize（二次乘使段长膨胀 dts 倍→相邻槽交叠，
        // dts=1 恒等掩盖）；count_ 为元素语义（template_utils.h DataSlice：
        // size_ 字节/count_ 元素，恒满足 count_×dts==size_）。
        const u64 chLen = dataSplit[channelIdx];
        const u64 chOff = dataOffset[channelIdx];
        const u64 sliceOff = runOff + chOff;
        if (isTx) {
            txSrcSlices.emplace_back(localBuff, sliceOff, chLen, elemCountOut[channelIdx]);
            txDstSlices.emplace_back(remoteBuff, sliceOff, chLen, elemCountOut[channelIdx]);
        } else {
            rxSrcSlices.emplace_back(remoteBuff, sliceOff, chLen, elemCountOut[channelIdx]);
            rxDstSlices.emplace_back(localBuff, sliceOff, chLen, elemCountOut[channelIdx]);
        }
        return HCCL_SUCCESS;
    };
    // 每条 run 恰 1 条 DataSlice（本 channel 非空时）→ 1 条 desc → 1 条 WQE：2^ 幂每步每对端
    // 1 条（σ 恒等整段合并）；非幂 part1 覆盖段 = unit 数（POST 代表 ≤ 2r+1 条）、PRE=1 条
    // （p 段 unit 数恒 1）；block 覆盖段紧凑 1 条/稀疏 unit 数
    std::vector<std::pair<u64, u64>> runs;
    RhdBuildRuns(step.txSegStart, step.txSegEnd, rpt, runs);
    for (const auto& run : runs) {
        CHK_RET(appendSlices(run.first, run.second, sendRemoteBuff, true));
    }
    // 接收段同构 run 划分（read 模式拉取段 = 对端发送段镜像；write 模式仅握手/计数）
    RhdBuildRuns(step.rxSegStart, step.rxSegEnd, rpt, runs);
    for (const auto& run : runs) {
        CHK_RET(appendSlices(run.first, run.second, recvRemoteBuff, false));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRHD::RunStepRHD(
    const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const u32& channelIdx, const RhdStepInfo& step, const u64 rpt, u32& wqeCntTotal)
{
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
    CHK_PRT_RET(
        channels.count(peerRankKey) == 0 || channelIdx >= channels.at(peerRankKey).size(),
        HCCL_ERROR(
            "[InsTempAllGatherRHD][%s] rank[%u] invalid channel access, peerRankKey[%u] channelIdx[%u] "
            "channels.size[%zu] peerChannelSize[%zu]",
            __func__, myRank_, peerRankKey, channelIdx, channels.size(),
            channels.count(peerRankKey) ? channels.at(peerRankKey).size() : 0),
        HCCL_E_INTERNAL);
    // RHD 每步单对端：channelSend 与 channelRecv 为同一对端 channel（收发同偏移直写）
    const ChannelInfo& peerChannel = channels.at(peerRankKey)[channelIdx];

    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    CHK_RET(BuildStepSlice(
        peerChannel, channels, step, rpt, channelIdx, txSrcSlices, txDstSlices, rxSrcSlices, rxDstSlices));

    TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
    TxRxChannels sendRecvChannels(peerChannel, peerChannel);
    SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);

    // WQE 观测（owner 槽位 run 口径）：wqeCnt = 本步本 channel 的 tx run 条数（每 run 1 条
    // DataSlice→1 desc→1 WQE）。2^ 幂域恒 ==1（σ 恒等整段合并）；非幂 part1 覆盖段 == tx 段
    // unit 数（POST 步 ≤2r+1 条、PRE 步=1 条）；空子段/握手方为 0。每 rpt 份独立计数
    // （RPT 不计入——只在 rpt==0 份累计）
    const u32 wqeCnt = static_cast<u32>(txSrcSlices.size());
    const char* phaseStr = (step.phase == RhdPhase::PRE) ? "PRE" : (step.phase == RhdPhase::MAIN_AG) ? "MAIN" : "POST";
    HCCL_DEBUG(
        "[InsTempAllGatherRHD][RunStepRHD] rank[%u] phase[%s] step[%u] peer[%u] wqeCnt[%u]", myRank_, phaseStr,
        step.step, peerAlgRank, wqeCnt);
    if (rpt == 0) {
        wqeCntTotal += wqeCnt;
    }

    if (isDmaRead_) {
        // 读模式消费 rx 段从对端拉（拉取段 = 对端发送段镜像，收发同偏移）；对端由 notify 协议配合
        CHK_PRT_RET(
            SendRecvRead(sendRecvInfo, threads[channelIdx]),
            HCCL_ERROR("[InsTempAllGatherRHD] SendRecv read failed (phase[%s] step=%u)", phaseStr, step.step),
            HcclResult::HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(
            SendRecvBatchWrite(sendRecvInfo, threads[channelIdx]),
            HCCL_ERROR("[InsTempAllGatherRHD] SendRecv batch failed (phase[%s] step=%u)", phaseStr, step.step),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRHD::RunAllGatherRHD(
    const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const u32& channelIdx, u32& totalWqeNum)
{
    totalWqeNum = 0;
    // 三相位（每 rpt 份独立；RPT 份间 CCL 上不连续，不能合并为单条 desc——每份每步单 WQE）
    for (u64 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        // 2a. 前处理：被合并方 t+r 发自身份额 unit（σ=[2t+1, 2t+2)）→ 代表同偏移直写其 owner 槽位
        //     （= 被合并方槽位 t+r；代表侧零本地动作）；代表/被合并方双向调用（空 tx 方仅握手，
        //     保证进入主体前写入完成）；block（!inPart1）空步骤全跳过。
        //     skipPrePhase=true 时整相位跳过（AllReduce twoshot 融合复用：被合并方份额已由调用方
        //     置于代表侧 owner 槽位；调用方契约=全通信域一致置位，本任务两绑定恒 false）
        if (!tempAlgParams_.skipPrePhase) {
            CHK_RET(RunStepRHD(threads, channels, channelIdx, preStep_, rpt, totalWqeNum));
        }
        // 2b. 主体：⌊log₂P⌋ 步倍增（仅有效 rank——代表/block；被合并方前处理后直等后处理）
        for (const RhdStepInfo& step : mainSteps_) {
            CHK_RET(RunStepRHD(threads, channels, channelIdx, step, rpt, totalWqeNum));
        }
        // 2c. 后处理：代表 t 发全量段（σ unit 区间 [0, p) → owner 槽位 run 表：part1 逐 unit、
        //     block 合并/逐 unit）→ 被合并方同偏移直写覆盖（收完即最终结果）；block 空步骤全跳过
        CHK_RET(RunStepRHD(threads, channels, channelIdx, postStep_, rpt, totalWqeNum));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRHD::LocalDataCopy(const std::vector<ThreadHandle>& threads, const u32& channelIdx)

{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    u64 partialSliceSize = dataSplit_[channelIdx];
    u64 partialOffset = dataOffset_[channelIdx];
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    // 尾块模式（Sequence 绑定恒置 tailSize=sliceSize，退化为常规块）
    if (tempAlgParams_.tailSize != 0 && myAlgRank == templateRankSize_ - 1) {
        partialSliceSize = dataSplitTail_[channelIdx];
        partialOffset = dataOffsetTail_[channelIdx];
    }
    for (u64 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        // 输入地址需含算法 rank 项（对齐 NHR LocalDataCopy 公式）：二级 intra 绑定 ISS=0 时为加零项、
        // 行为等价；三级 L1（ISS=S、IRS=S·d1）下缺少该项会把装载源错读为上一阶段输出网格的组首位置
        const u64 inOff = tempAlgParams_.inputSliceStride * myAlgRank + inBaseOff + partialOffset;
        u64 sliceCount = partialSliceSize / dataTypeSize;

        if (enableRemoteMemAccess_) {
            // remote 模式（symMem/图模式）：IN 直写 OUT 本 rank 块（幂/非幂统一——owner 槽位=自身槽位）
            const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
            const u64 outOff = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff + partialOffset;
            if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr && inOff == outOff) {
                continue; // inplace 同址跳过
            }
            DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, partialSliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOff, partialSliceSize, sliceCount);
            CHK_RET(LocalCopy(threads[channelIdx], srcSlice, dstSlice));
        } else if (isOutputCclBuff_) {
            // CCL 别名绑定：装载直写输出网格槽位 i（owner 槽位=自身 algRank 槽位，σ 由网络区段
            // owner 反查消化；nhr.cc:488-490 先例）
            const u64 outBaseOff = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
            const u64 scOff = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff + partialOffset;
            if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.hcclBuff.addr && inOff == scOff) {
                continue; // IN 已在 CCL buffer 且与落点同址同长时跳过本边
            }
            DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, partialSliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scOff, partialSliceSize, sliceCount);
            CHK_RET(LocalCopy(threads[channelIdx], srcSlice, dstSlice));
        } else {
            // 真拷贝绑定：紧凑 CCL 中转网格
            const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
            const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
            const u64 scOff = tempAlgParams_.sliceSize * myAlgRank + scratchBase + partialOffset;
            if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.hcclBuff.addr && inOff == scOff) {
                continue; // IN 已在 CCL buffer 且与落点同址同长时跳过本边
            }
            DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, partialSliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scOff, partialSliceSize, sliceCount);
            CHK_RET(LocalCopy(threads[channelIdx], srcSlice, dstSlice));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRHD::PostLocalCopy(const ThreadHandle& thread, const u32& channelIdx)
{
    if (enableRemoteMemAccess_) {
        // remote 模式：装载与收发已直写 OUT，无后拷贝
        HCCL_INFO("[InsTempAllGatherRHD] PostLocalCopy skip because remote memory access enabled");
        return HcclResult::HCCL_SUCCESS;
    }
    if (tempAlgParams_.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        // CCL 别名绑定：三相位直写 OUT 网格即最终布局（σ 已由 owner 反查消化），跳过后拷贝
        // （nhr.cc:513-515 先例：PostLocalCopy skip because output is scratch）
        HCCL_INFO("[InsTempAllGatherRHD] PostLocalCopy skip because output is scratch");
        return HcclResult::HCCL_SUCCESS;
    }
    // 真拷贝绑定（CCU：OUT 为用户缓冲）——恒等回搬：src/dst 槽位一致（owner 槽位与 OUT 槽位同序，
    // 一对一仿射映射任意次序安全；与 HD 先行公式同构）
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    for (u64 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (auto rank : subCommRanks_[0]) {
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
            u64 partialSliceSize = dataSplit_[channelIdx];
            u64 partialOffset = dataOffset_[channelIdx];
            // 尾块模式（Sequence 绑定恒退化）
            if (tempAlgParams_.tailSize != 0 && algRank == templateRankSize_ - 1) {
                partialSliceSize = dataSplitTail_[channelIdx];
                partialOffset = dataOffsetTail_[channelIdx];
            }
            u64 sliceCount = partialSliceSize / dataTypeSize;
            u64 scratchOffset = tempAlgParams_.sliceSize * algRank + scratchBase + partialOffset;
            u64 outOffset = tempAlgParams_.outputSliceStride * algRank + outBaseOff + partialOffset;
            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, partialSliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset, partialSliceSize, sliceCount);
            CHK_RET(LocalCopy(thread, srcSlice, dstSlice));
        }
    }
    return HCCL_SUCCESS;
}

void InsTempAllGatherRHD::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAllGatherRHD::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

} // namespace ops_hccl
