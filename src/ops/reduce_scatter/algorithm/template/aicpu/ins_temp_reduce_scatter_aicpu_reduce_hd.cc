/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_aicpu_reduce_hd.h"

namespace ops_hccl {

std::vector<CostModelParam> InsTempReduceScatterAicpuReduceHD::CalcCostCoeff(CalcCostCoeffParam param)
{
    // 守卫 1（2 幂，前置，host/device 共用）：HD 递归折半按 r ^ 2^k 配对，要求 rank 数为 2 的幂；
    // 非 2 幂（含显式 HCCL_ALGO 配置场景）返回空，cost model 跳过本算法
    // （preferred 同样被 FilterByTopoAndCalibrate 跳过并 WARNING 回退自动选路）。
    if (param.rankSize < 2 || (param.rankSize & (param.rankSize - 1)) != 0) {
        HCCL_DEBUG(
            "[InsTempReduceScatterAicpuReduceHD][CalcCostCoeff] rankSize[%u] is not a power of two, skip.",
            param.rankSize);
        return {};
    }
    // 返回公式型 cost：A/B/C/D 全部经 CostModelManager 既有接口推导（host/device 共用，无
    // AICPU_COMPILE 分裂——设备侧选路管线同样调用 CalcCostCoeff，空返回会触发管线级
    // "uncalibrated" WARNING；对齐同族 NHR-AicpuReduce/AllReduce HD 模板）。
    // preferred-only 门控在控制面：注册处 topoCustomCheck 恒 false 使默认选路（软策略）过滤本算法，
    // 显式 HCCL_ALGO 配置打 preferred 后软策略跳过该检查仍可选（对齐 AicpuAllReduceParallelMeshHD）。
    u32 nSteps = 0;
    for (u32 v = param.rankSize; v > 1; v >>= 1) {
        ++nSteps;
    }
    const int portNum = param.portNum.empty() ? 0 : static_cast<int>(param.portNum[0]);
    float A = 0.0f;
    float B1 = 0.0f;
    float B3 = 0.0f;
    float C = 0.0f;
    float D = 0.0f;
    // A（跨卡传输）：HD 递归折半与 NHR 递归减半同为 halving 系调度（每步交换 dist*SS 逐折半递减、
    // 总传输 (N-1)*SS 同阶），带宽模型同构；单层 MESH_1D 无跨 pod 端口预留语义（halvePodPort=false）
    CostModelManager::Global()->CalcNHRParams(
        param.dataRatio, param.netType, portNum, param.rankSize, A, param.isPod, false);
    // B（本地计算）两段求和，比例按同族 NHR 口径以 dataRatio 缩放（绝对比例会使 Parallel 绑定下
    // B 相对 A 高估约 dataRatio 倍）：镜像保留半区 (N/2)*SS = dataRatio*N/2（write 模式跳过首步
    // 发送区；read 模式全量镜像，取半量近似）+ 终态输出拷贝 SS = dataRatio；步间归约由 DMA
    // WRITE_REDUCE/READ_REDUCE 在传输侧完成，无本地归约项
    CostModelManager::Global()->CalcLocalCopyParams(
        param.dataRatio * static_cast<float>(param.rankSize) / 2.0f, EngineType::AICPU, B1);
    CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B3);
    const float B = B1 + B3;
    // 本地拷贝 task/段按段与 repeat 归一：输入取自 INPUT 的段（Sole/Parallel inter1）镜像+输出随
    // repeat 放大，repeat 数即 repeatednum（inter1 的 RPT=N0）；输入取自 HCCL_BUFFER 的段（inter0：
    // intra0 已把数据写入 scratch，executor 固定 RPT=1）无镜像、不放大
    const u32 repNum = (param.inputBuffer == BufferType::INPUT) ? std::max(1u, param.repeatednum) : 1u;
    // 每 repeat 镜像 N/2 槽（write 模式跳过首步发送区；PCIe read 模式全量 N 槽，取 write 口径与
    // B 项一致，read 模式低估约 2 倍）+ 终态输出 1 次
    const u32 localCopyNum = repNum * (static_cast<u32>(param.rankSize) / 2 + 1);
    // C（时延常数）：log2(N) 步交换段（每步收发 + ACK/DATA_SIGNAL 通知一段，无 barrier/本地归约）
    // + 本地拷贝段；C/D 同步计入本地拷贝、口径一致（镜像/输出段即 localCopyNum，不另加常数项
    // 避免重复计数）（整数算术）
    const int kernelNum = static_cast<int>(nSteps) + static_cast<int>(localCopyNum);
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AICPU, C);
    // D（launch）：每步 1 组传输 task + 每步 ACK/DATA_SIGNAL 两组同步 task + 本地拷贝 task
    // （按段与 repeat 数计入）（整数算术）
    const int taskNum = CostModelManager::CalcTransTaskNum(nSteps) + CostModelManager::CalcSyncTaskNum(nSteps) * 2
                        + static_cast<int>(localCopyNum);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AICPU, D);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    HCCL_DEBUG("[InsTempReduceScatterAicpuReduceHD][CalcCostCoeff] nSteps[%u] A=%f B=%f C=%f D=%f", nSteps, A, B, C, D);
    return params;
}

InsTempReduceScatterAicpuReduceHD::InsTempReduceScatterAicpuReduceHD(
    const OpParam& param, const u32 rankId, // 传通信域的u32，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempReduceScatterAicpuReduceHD::~InsTempReduceScatterAicpuReduceHD() {}

HcclResult InsTempReduceScatterAicpuReduceHD::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // 通道请求对齐 InsTempReduceScatterHD（AllReduce HD 的 RS 阶段）：
    // UBX 拓扑走 MultiJettyUbx（每对 rank 多 jetty 连接），其余拓扑复用 NHR 全连接并过滤 ROCE 通道
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
        channelsPerRank_ == 0, HCCL_ERROR("[InsTempReduceScatterAicpuReduceHD][CalcRes] channelsPerRank_ is 0."),
        HcclResult::HCCL_E_INTERNAL);
    if (isUBX && channelsPerRank_ > MAX_JETTY_NUM) {
        HCCL_ERROR(
            "[InsTempReduceScatterAicpuReduceHD][CalcRes] channelsPerRank_ %u is greater than MAX_JETTY_NUM %u.",
            channelsPerRank_, MAX_JETTY_NUM);
    } else {
        HCCL_DEBUG("[InsTempReduceScatterAicpuReduceHD][CalcRes] channelsPerRank_ is %u.", channelsPerRank_);
    }
    GetRes(resourceRequest);
    HCCL_INFO(
        "[InsTempReduceScatterAicpuReduceHD][CalcRes] channelsPerRank_: [%u], slaveThreadNum: [%u], "
        "notifyNumOnMainThread: [%u].",
        channelsPerRank_, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceHD::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 threadNum = 1 * channelsPerRank_;
    CHK_PRT_RET(
        threadNum == 0,
        HCCL_ERROR("[InsTempReduceScatterAicpuReduceHD][GetRes] threadNum is 0, channelsPerRank_ "
                   "invalid."),
        HcclResult::HCCL_E_INTERNAL);
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    HCCL_INFO(
        "[InsTempReduceScatterAicpuReduceHD][GetRes] channelsPerRank_: [%u], slaveThreadNum: [%u].", channelsPerRank_,
        resourceRequest.slaveThreadNum);
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterAicpuReduceHD::GetThreadNum() const { return 1 * channelsPerRank_; }

HcclResult InsTempReduceScatterAicpuReduceHD::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempReduceScatterAicpuReduceHD] KernelRun start");

    tempAlgParams_ = tempAlgParams;
    channels_ = templateResource.channels;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    bool isPcieProtocol = IsPcieProtocol(channels_); // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocol;                     // 是否使用Read模式

    // 边界守卫：空数据直接成功（executor 保证 count > 0，此处防御）
    if (tempAlgParams_.count == 0) {
        HCCL_INFO("[InsTempReduceScatterAicpuReduceHD][KernelRun] count is 0, nothing to do.");
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        templateResource.threads.empty(), HCCL_ERROR("[InsTempReduceScatterAicpuReduceHD][KernelRun] empty threads"),
        HcclResult::HCCL_E_INTERNAL);
    // 单 rank 防御分支：selector 层 SingleRankProc 已拦截，此处直拷 IN -> OUT（单线程，不走通道切分）
    if (templateRankSize_ == 1) {
        return HandleSingleRank(templateResource.threads);
    }
    // 非 2 的幂：HD 递归折半不适用（选路层已结构性拦截，此处为最后防线）
    if ((templateRankSize_ & (templateRankSize_ - 1)) != 0) {
        HCCL_ERROR(
            "[InsTempReduceScatterAicpuReduceHD][KernelRun] tempRankSize[%u] is not a power of two.",
            templateRankSize_);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        subCommRanks_.empty() || subCommRanks_[0].size() != templateRankSize_,
        HCCL_ERROR(
            "[InsTempReduceScatterAicpuReduceHD][KernelRun] subCommRanks mismatch, tempRankSize[%u].",
            templateRankSize_),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        channels_.empty(), HCCL_ERROR("[InsTempReduceScatterAicpuReduceHD][KernelRun] channels is empty."),
        HcclResult::HCCL_E_INTERNAL);
    for (const auto& channelsByRank : channels_) {
        CHK_PRT_RET(
            channelsByRank.second.empty(),
            HCCL_ERROR(
                "[InsTempReduceScatterAicpuReduceHD][KernelRun] channel list of remote rank[%u] is empty.",
                channelsByRank.first),
            HcclResult::HCCL_E_INTERNAL);
    }

    // 按端口组把每槽数据水平切分到各通道（对齐 InsTempReduceScatterHD：切分基于第一组 rank 的通道表，
    // 通道数按 channelsPerRank_ 展平）
    std::vector<u64> elemCountOut;
    std::vector<u64> sizeOut;
    std::vector<u64> elemOffset;
    const u64 totalDataCount = tempAlgParams_.sliceSize / dataTypeSize_;
    CHK_RET(CalcDataSplitByPortGroup(
        totalDataCount, dataTypeSize_, channels_.begin()->second, elemCountOut, sizeOut, elemOffset));
    sizeOut_ = sizeOut;
    elemOffset_ = elemOffset;
    // 通道切分表与 channelsPerRank_ 一致性校验（对齐同族多通道模板）：channelsPerRank_ 取各远端
    // rank 通道数的最大值，切分表按首个远端 rank 的通道表生成，多 jetty 部分建链等场景两者可能
    // 不一致，校验防止后续按 channelIdx 下标访问越界
    CHK_PRT_RET(
        channelsPerRank_ > sizeOut_.size() || channelsPerRank_ > elemOffset_.size(),
        HCCL_ERROR(
            "[InsTempReduceScatterAicpuReduceHD][KernelRun] split size[%zu]/offset size[%zu] less than "
            "channelsPerRank_[%u].",
            sizeOut_.size(), elemOffset_.size(), channelsPerRank_),
        HcclResult::HCCL_E_INTERNAL);

    threadNum_ = static_cast<u32>(GetThreadNum());
    CHK_PRT_RET(
        threadNum_ > templateResource.threads.size(),
        HCCL_ERROR(
            "[InsTempReduceScatterAicpuReduceHD][KernelRun] thread num[%u] more than threads size[%zu].", threadNum_,
            templateResource.threads.size()),
        HcclResult::HCCL_E_INTERNAL);
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    // 阶段一：本地镜像（按通道切分执行，write 模式跳过首步发送区）
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(LocalDataCopy(templateResource.threads, channelIdx));
    }
    // 阶段二：递归折半（按通道切分执行，步内归约由 DMA 在传输侧完成，无 barrier）
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(RunHD(templateResource.threads, channelIdx));
    }
    // 阶段三：终态输出（按通道切分执行）
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(PostLocalCopy(templateResource.threads, channelIdx));
    }

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[InsTempReduceScatterAicpuReduceHD] KernelRun end");
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterAicpuReduceHD::RptScratchBase(u64 rpt) const
{
    // 每 repeat 块占用 N*SS（N 槽单缓冲）；RPT=1 时退化为 hcclBuffBaseOff
    const u64 rptScratchStride = templateRankSize_ * tempAlgParams_.sliceSize;
    return tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * rptScratchStride;
}

HcclResult InsTempReduceScatterAicpuReduceHD::HandleSingleRank(const std::vector<ThreadHandle>& threads)
{
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        CHK_RET(SingleRankCopy(threads, rpt));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceHD::SingleRankCopy(const std::vector<ThreadHandle>& threads, u64 rpt)
{
    ThreadHandle q = threads[0];
    // N==1 时唯一 algIdx==0：输出按通用公式 OUT[base+rpt*ORS+myAlgIdx*OSS] 补 OSS 项，
    // myAlgIdx==0 下恒为 +0（Sole/Parallel 语境等价；Parallel 语境 N==1 被 TopoMatchTwoLevel 拒绝不可达）
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));
    const u64 inOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
    const u64 outOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride
                       + myAlgIdx * tempAlgParams_.outputSliceStride;
    DataSlice src = DataSlice(tempAlgParams_.buffInfo.inputPtr, inOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
    DataSlice dst
        = DataSlice(tempAlgParams_.buffInfo.outputPtr, outOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
    if (tempAlgParams_.buffInfo.inBuffType != tempAlgParams_.buffInfo.outBuffType || inOff != outOff) {
        CHK_RET(LocalCopy(q, src, dst));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceHD::LocalDataCopy(const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterAicpuReduceHD][LocalDataCopy] empty threads"),
        HcclResult::HCCL_E_INTERNAL);
    ThreadHandle q = threads[channelIdx];
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);

    // write 模式下首步发送区（我的放弃半区 [sentStart0, +N/2)）从不被本地归约写入：
    // 交换时对端 RMW 的只是我的保留半区，放弃区仅被首步 tx 读一次（直接读输入区），
    // 跳过镜像省一半拷贝；read 模式对端直接拉本端 scratch 放弃区，必须全量镜像
    const u32 halfN = templateRankSize_ / 2;
    const u32 step0SentStart = (myAlgIdx ^ halfN) & ~(halfN - 1);
    const bool skipStep0Tx = !isDmaRead_;

    // 输入片按 algIdx 索引（通用五参数契约）：IN[base+rpt*IRS+i*ISS]，i 为算法 rank 索引，不依赖
    // subCommRanks 与 userRank 的映射关系。Sole 语境 localRanks 升序恒等（algIdx==userRank），与按
    // userRank 索引逐字节等价；Parallel inter1 语境 level1 组内 userRank=p+i*d0 != i，必须按 algIdx。
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 scratchBase = RptScratchBase(rpt);
        for (u32 i = 0; i < templateRankSize_; ++i) {
            if (skipStep0Tx && i >= step0SentStart && i < step0SentStart + halfN) {
                HCCL_DEBUG(
                    "[InsTempReduceScatterAicpuReduceHD][LocalDataCopy] rpt[%llu] skip step0 tx slot[%u]", rpt, i);
                continue;
            }
            const u32 sliceUserRank = subCommRanks_[0].at(i); // 仅日志观察 algIdx<->userRank 映射，不参与寻址
            const u64 inOff = inBaseOff + tempAlgParams_.inputSliceStride * i + elemOffset_[channelIdx];
            const u64 scOff = scratchBase + tempAlgParams_.sliceSize * i + elemOffset_[channelIdx];

            DataSlice src = DataSlice(
                tempAlgParams_.buffInfo.inputPtr, inOff, sizeOut_[channelIdx], sizeOut_[channelIdx] / dataTypeSize_);
            DataSlice dst = DataSlice(
                tempAlgParams_.buffInfo.hcclBuff.addr, scOff, sizeOut_[channelIdx],
                sizeOut_[channelIdx] / dataTypeSize_);

            HCCL_DEBUG(
                "[InsTempReduceScatterAicpuReduceHD][LocalDataCopy] rpt[%llu] algIdx[%u] userRank[%u] "
                "inOff[%llu] scOff[%llu] channelIdx[%u]",
                rpt, i, sliceUserRank, inOff, scOff, channelIdx);

            // 如果源地址和目标地址相同，则不需要做拷贝
            if (tempAlgParams_.buffInfo.inBuffType != tempAlgParams_.buffInfo.hcclBuffType || inOff != scOff) {
                CHK_RET(LocalCopy(q, src, dst));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceHD::RunHD(const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterAicpuReduceHD][RunHD] empty threads"),
        HcclResult::HCCL_E_INTERNAL);
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));

    u32 nSteps = 0;
    for (u32 v = templateRankSize_; v > 1; v >>= 1) {
        ++nSteps;
    }
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);

    // 步 k（k=nSteps..1）：dist=2^(k-1)、partner algIdx=r^dist；我发送放弃区 [sentStart,+dist)
    // （对端保留区起点，双端同号槽位）、接收保留区 [keepStart,+dist)（对端放弃区）。
    // 每步一次交换原语（所有 rpt 的槽合并成一个批量）：write 模式我推放弃区 write-reduce 进对端
    // 保留槽（对端对称推我），read 模式我拉对端保留区 read-reduce 进我保留槽（对端对称拉我）；
    // 步内无本地归约（DMA 侧完成），步间无 barrier（同队列任务定序 + ACK 链跨 rank 传递先行步完成关系）
    for (u32 k = nSteps; k > 0; --k) {
        const u32 dist = 1u << (k - 1);
        const u32 partnerAlgIdx = myAlgIdx ^ dist;
        const u32 keepStart = myAlgIdx & ~(dist - 1);
        const u32 sentStart = keepStart ^ dist;
        const u32 partnerUserRank = subCommRanks_[0].at(partnerAlgIdx);
        const bool isFirstStep = (k == nSteps);

        HCCL_INFO(
            "[InsTempReduceScatterAicpuReduceHD][RunHD] step[%u/%u] dist[%u] partnerAlgIdx[%u] "
            "partnerUserRank[%u] keepStart[%u] sentStart[%u] channelIdx[%u]",
            nSteps - k + 1, nSteps, dist, partnerAlgIdx, partnerUserRank, keepStart, sentStart, channelIdx);

        CHK_PRT_RET(
            channels_.count(partnerUserRank) == 0 || channelIdx >= channels_.at(partnerUserRank).size(),
            HCCL_ERROR(
                "[InsTempReduceScatterAicpuReduceHD][RunHD] link missing: partner=%d channelIdx=%u", partnerUserRank,
                channelIdx),
            HcclResult::HCCL_E_INTERNAL);
        const ChannelInfo& channel = channels_.at(partnerUserRank).at(channelIdx);

        void* localCclBuffAddr = tempAlgParams_.buffInfo.hcclBuff.addr;
        void* remoteCclBuffAddr = channel.remoteCclMem.addr;
        // 发送目标始终是对端 scratch 同号槽（sentStart_me == keepStart_partner，双端对称）
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        txSrcSlices.reserve(dist * rptNum);
        txDstSlices.reserve(dist * rptNum);
        rxSrcSlices.reserve(dist * rptNum);
        rxDstSlices.reserve(dist * rptNum);

        for (u64 rpt = 0; rpt < rptNum; ++rpt) {
            const u64 scratchBase = RptScratchBase(rpt);
            const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
            for (u32 i = 0; i < dist; ++i) {
                const u32 txSlot = sentStart + i;
                const u32 rxSlot = keepStart + i;
                const u64 txDstOff = scratchBase + tempAlgParams_.sliceSize * txSlot + elemOffset_[channelIdx];
                const u64 rxDstOff = scratchBase + tempAlgParams_.sliceSize * rxSlot + elemOffset_[channelIdx];

                void* txSrcAddr = localCclBuffAddr;
                u64 txSrcOff = txDstOff;
                if (isFirstStep && !isDmaRead_) {
                    // 首步放弃区未镜像（见 LocalDataCopy），直接从输入区读取
                    txSrcAddr = tempAlgParams_.buffInfo.inputPtr;
                    txSrcOff = inBaseOff + tempAlgParams_.inputSliceStride * txSlot + elemOffset_[channelIdx];
                }

                txSrcSlices.emplace_back(
                    txSrcAddr, txSrcOff, sizeOut_[channelIdx], sizeOut_[channelIdx] / dataTypeSize_);
                txDstSlices.emplace_back(
                    remoteCclBuffAddr, txDstOff, sizeOut_[channelIdx], sizeOut_[channelIdx] / dataTypeSize_);
                rxSrcSlices.emplace_back(
                    remoteCclBuffAddr, rxDstOff, sizeOut_[channelIdx], sizeOut_[channelIdx] / dataTypeSize_);
                rxDstSlices.emplace_back(
                    localCclBuffAddr, rxDstOff, sizeOut_[channelIdx], sizeOut_[channelIdx] / dataTypeSize_);
            }
        }

        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        TxRxChannels sendRecvChannels(channel, channel);
        SendRecvReduceInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_, reduceOp_);

        if (isDmaRead_) {
            // read 模式：本端拉对端保留区（对端的部分和）read-reduce 进本端保留槽；tx 列表为对端拉我的对称描述
            CHK_PRT_RET(
                SendRecvReadReduce(sendRecvInfo, threads[channelIdx]),
                HCCL_ERROR(
                    "[InsTempReduceScatterAicpuReduceHD][RunHD] SendRecvReadReduce failed, partnerUserRank[%u] "
                    "dist[%u] channelIdx[%u]",
                    partnerUserRank, dist, channelIdx),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            // write 模式：本端推放弃区 write-reduce 进对端保留槽（完成通知融合进最后一个数据 WQE，
            // 无独立通知 WQE）；rx 列表为对端推我的对称描述（原语仅使用 tx 列表）
            CHK_PRT_RET(
                SendRecvBatchWriteReduce(sendRecvInfo, threads[channelIdx]),
                HCCL_ERROR(
                    "[InsTempReduceScatterAicpuReduceHD][RunHD] SendRecvBatchWriteReduce failed, "
                    "partnerUserRank[%u] dist[%u] channelIdx[%u]",
                    partnerUserRank, dist, channelIdx),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceHD::PostLocalCopy(const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterAicpuReduceHD][PostLocalCopy] empty threads"),
        HcclResult::HCCL_E_INTERNAL);
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));
    ThreadHandle q = threads[channelIdx];
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);

    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 stateOff = RptScratchBase(rpt) + tempAlgParams_.sliceSize * myAlgIdx + elemOffset_[channelIdx];
        // 输出按通用五参数公式 OUT[base+rpt*ORS+myAlgIdx*OSS]：Sole 语境 OSS=0 退化（与用户输出直连）；
        // Parallel inter1 语境 OSS=SS1——各 rank 把自己保留的片落到自己 scratch 槽位：intra1 以
        // inBuffBaseOff=inter1ScratchOffset+rankIdxLevel1_*SS1、ISS=SS1*N1 读取，恰好命中本 rank
        // 写下的 rpt=i、algIdx=rankIdxLevel1_ 的终态槽（跨 template 交接恒等式：outOff 与 stateOff 同址）
        const u64 outOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride
                           + myAlgIdx * tempAlgParams_.outputSliceStride + elemOffset_[channelIdx];

        DataSlice src = DataSlice(
            tempAlgParams_.buffInfo.hcclBuff.addr, stateOff, sizeOut_[channelIdx],
            sizeOut_[channelIdx] / dataTypeSize_);
        DataSlice dst = DataSlice(
            tempAlgParams_.buffInfo.outputPtr, outOff, sizeOut_[channelIdx], sizeOut_[channelIdx] / dataTypeSize_);

        HCCL_DEBUG(
            "[InsTempReduceScatterAicpuReduceHD][PostLocalCopy] rpt[%llu] myAlgIdx[%u] stateOff[%llu] "
            "outOff[%llu] channelIdx[%u]",
            rpt, myAlgIdx, stateOff, outOff, channelIdx);

        // 拷贝条件按双语境判定：scratch 区与输出区错位（Sole：OUTPUT 与 HCCL_BUFFER 不同型；
        // Parallel inter1：OUT 落 HCCL_BUFFER scratch 区，与终态槽同址）时回搬；inter1 同址自动跳过
        if (tempAlgParams_.buffInfo.hcclBuffType != tempAlgParams_.buffInfo.outBuffType || stateOff != outOff) {
            CHK_RET(LocalCopy(q, src, dst));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterAicpuReduceHD::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // N 槽单缓冲：镜像初始化后槽内即归约态（对端 write-reduce/read-reduce 直接 RMW 保留槽），
    // 无独立接收区；按单次 repeat 报
    const u64 templateScratchMultiplier = templateRankSize_;
    HCCL_INFO(
        "[InsTempReduceScatterAicpuReduceHD][CalcScratchMultiple] templateScratchMultiplier[%llu]",
        templateScratchMultiplier);
    return templateScratchMultiplier;
}

void InsTempReduceScatterAicpuReduceHD::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = 1 * channelsPerRank_;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempReduceScatterAicpuReduceHD::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = 1 * channelsPerRank_;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

} // namespace ops_hccl
