/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_hd_op.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {
constexpr int POD_TASK_NUM_OVERHEAD = 5;

// HD 递归倍增步数 k = log2(rankSize)；实际选路已由 CalcCostCoeff 限定 rankSize 为 2 的幂
static u32 GetHDStepNum(u32 rankSize)
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize; tmp > 1; tmp >>= 1) {
        nSteps++;
    }
    return nSteps;
}

std::vector<CostModelParam> InsTempAllGatherHDOp::CalcCostCoeff(CalcCostCoeffParam param)
{
    // 递归倍增要求 rankSize 为 2 的幂且不小于 2；不适用时返回空，
    // 显式配置 HCCL_ALGO 时回退自动选择并 WARNING，默认链路不参与竞争
    if (param.rankSize < 2 || (param.rankSize & (param.rankSize - 1)) != 0) {
        return {};
    }
    // 沿用 NHR 口径：以 (rankSize-1) 传输量公式建模（与 HD 每 rank 总传输量 S*(N-1) 量级一致），
    // portNum 取 level0 端口数
    param.netType = CommTopo::COMM_TOPO_CLOS;
    int portNum = param.portNum[0];
    int log2R = 0;
    for (u32 r = param.rankSize; r > 1; r >>= 1) {
        log2R++;
    }
    // HD 共 k=log2(N) 步交换加 2 个本地拷贝阶段；kernelNum/taskNum 沿用 NHR 标定形状，系数未针对 HD 重标定
    int kernelNum = 8 * log2R - 7;
    kernelNum = std::max(kernelNum, 10);
    u32 rEff = static_cast<u32>(5 * log2R / 3);
    rEff = std::max(rEff, 2u);
    int taskNum = CostModelManager::CalcTransTaskNum(rEff) + CostModelManager::CalcSyncTaskNum(rEff) * 2;
    taskNum = param.isPod ? taskNum + POD_TASK_NUM_OVERHEAD : taskNum;

    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;
    CostModelManager::Global()->CalcNHRParams(param.dataRatio, param.netType, portNum, param.rankSize, A, param.isPod);
    if (param.inputBuffer != param.scratchBuffer) {
        // IN->SCR 与 SCR->OUT 两段本地拷贝按 NHR 同口径计一项，保守
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B);
    }

    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AICPU, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AICPU, D);
    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

InsTempAllGatherHDOp::InsTempAllGatherHDOp(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempAllGatherHDOp::~InsTempAllGatherHDOp() {}

HcclResult InsTempAllGatherHDOp::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> level0Channels;
    CHK_PRT_RET(topoInfo == nullptr, HCCL_ERROR("[InsTempAllGatherHDOp][CalcRes] topoInfo is nullptr"), HCCL_E_PARA);
    std::vector<HcclChannelDesc> myChannelDescs;
    // 单层 MESH_1D 全互连申请；过滤 RoCE 协议链路（与既有 AICPU all_gather 模板的过滤方式一致）
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
        "[InsTempAllGatherHDOp][CalcRes] channelsPerRank_ is %u, maxChannelsPerRank_ is %u ", channelsPerRank_,
        maxChannelsPerRank_);
    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherHDOp::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 threadNum = GetThreadNum();
    HCCL_INFO("[InsTempAllGatherHDOp][GetRes] threadNum[%u]", threadNum);
    resourceRequest.slaveThreadNum = threadNum - 1;
    // 每个从流 1 个 notify 用于主从流同步（v1 无 last-step 并行优化，无第二 notify）
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherHDOp::GetThreadNum() const
{
    // 线程数等于每 rank 最大 channel 数；MESH_1D 单 channel 时为 1（单线程，无主从同步开销）。
    // N==1 时无远端 channel，maxChannelsPerRank_ 保持默认 1。
    return maxChannelsPerRank_;
}

HcclResult InsTempAllGatherHDOp::SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    CHK_PRT_RET(channels.empty(), HCCL_ERROR("[SetchannelsPerRank] channels is empty."), HCCL_E_INTERNAL);
    channelsPerRank_ = CalcChannelsPerRankMin(channels);
    maxChannelsPerRank_ = CalcChannelsPerRank(channels);
    HCCL_INFO(
        "[InsTempAllGatherHDOp][SetchannelsPerRank] channelsPerRank_[%u], maxChannelsPerRank_[%u]", channelsPerRank_,
        maxChannelsPerRank_);
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherHDOp::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // SCR 每轮 loop 划为 N 段（每 rank 块一段）；remote 模式不使用 SCR，仍按 N 保守口径
    u64 scratchMultiple = templateRankSize_;
    return scratchMultiple;
}

HcclResult InsTempAllGatherHDOp::PrepareDataSplitForMultiChannel(const TemplateResource& templateResource)
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

HcclResult InsTempAllGatherHDOp::CheckKernelRunParams(const TemplateResource& templateResource)
{
    // 幂守卫：按算法作用域 subCommRanks_[0] 校验（基类 templateRankSize_ 在多组绑定时为两段乘积语义，
    // 不能代表 HD 递归倍增作用域，单组绑定时两者相等、行为不变）。递归倍增要求作用域 rankSize 为 2 的
    // 幂且不小于 2。选路期门控分流（upstream 共享 TopoMatch HD 非 2^N 门控，two-level 仅查外层 d1）：
    // 外层非幂形态（如 12_2_4：N0=2 幂、d1=3）与 sole 单段非幂已在选路期被门控拦截回退，此处对外层
    // 非幂形态为防御层；内层非幂形态（如 2x3：N0=3 非幂、d1=2 幂）过门控、幂段系数拼接使 cost 非空、
    // 显式可命中，此处为该形态执行期唯一拦截点：peerRank = R XOR 2^v 越出组边界（subCommRanks_[0]
    // 越界）时返回参数错误而非静默错误数据
    const u32 scopeRankSize = subCommRanks_.empty() ? 0 : static_cast<u32>(subCommRanks_[0].size());
    if (scopeRankSize < 2 || (scopeRankSize & (scopeRankSize - 1)) != 0) {
        HCCL_ERROR(
            "[InsTempAllGatherHDOp] Rank [%u], rankSize[%u] is not a power of two or is less than 2, not supported.",
            myRank_, scopeRankSize);
        return HcclResult::HCCL_E_PARA;
    }
    threadNum_ = GetThreadNum();
    if (templateResource.threads.size() < threadNum_) {
        HCCL_ERROR(
            "[InsTempAllGatherHDOp] Rank [%u], thread num[%zu] is not as expected[%u].", myRank_,
            templateResource.threads.size(), threadNum_);
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherHDOp::PrepareKernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;

    bool isPcieProtocol = IsPcieProtocol(templateResource.channels); // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocol;                                     // 是否使用Read模式
    HCCL_DEBUG("[InsTempAllGatherHDOp] Use Dma Read[%d]", isDmaRead_);
    CHK_RET(PrepareDataSplitForMultiChannel(templateResource));
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherHDOp::SyncInterThreads(const std::vector<ThreadHandle>& threads, bool isPreSync)
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

HcclResult InsTempAllGatherHDOp::RunChannelMainLoop(const TemplateResource& templateResource)
{
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        if (dataSplit_[channelIdx] == 0) {
            HCCL_DEBUG("[InsTempAllGatherHDOp] Rank [%u], channelIdx[%u] split is empty, skip.", myRank_, channelIdx);
            continue; // 空切片跳过该 channel 边
        }
        CHK_RET(LocalDataCopy(templateResource.threads, channelIdx)); // input buffer拷贝到scratch buffer上
        if (templateRankSize_ > 1) {
            CHK_RET(RunAllGatherHD(templateResource.threads, templateResource.channels, channelIdx));
        }
        CHK_RET(PostLocalCopy(templateResource.threads[channelIdx], channelIdx));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherHDOp::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllGatherHDOp] Run start");
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        HCCL_INFO("[InsTempAllGatherHDOp] Rank [%u], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }
    CHK_RET(CheckKernelRunParams(templateResource));
    CHK_RET(PrepareKernelRun(param, tempAlgParams, templateResource));
    if (threadNum_ > 1) {
        CHK_RET(SyncInterThreads(templateResource.threads, true)); // 预同步
    }
    CHK_RET(RunChannelMainLoop(templateResource));
    if (threadNum_ > 1) {
        CHK_RET(SyncInterThreads(templateResource.threads, false)); // 后同步
    }

    HCCL_INFO("[InsTempAllGatherHDOp] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherHDOp::GetStepInfoHD(u32 step, AicpuHDStepInfo& stepInfo)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    stepInfo.step = step;
    stepInfo.myRank = myAlgRank;
    stepInfo.peerRank = myAlgRank ^ (1u << step);
    stepInfo.windowBase = myAlgRank & ~((1u << step) - 1u);
    stepInfo.nSlices = 1u << step;

    HCCL_DEBUG(
        "[InsTempAllGatherHDOp][GetStepInfoHD] step[%u] myRank[%u] peerRank[%u] windowBase[%u] nSlices[%u]", step,
        stepInfo.myRank, stepInfo.peerRank, stepInfo.windowBase, stepInfo.nSlices);
    return HcclResult::HCCL_SUCCESS;
}

InsTempAllGatherHDOp::HDCalcInfo
InsTempAllGatherHDOp::CalcSliceInfoHD(const AicpuHDStepInfo& stepInfo, u32 rpt, u32 i, u32 channelIdx) const
{
    HDCalcInfo info;
    info.txBlk = stepInfo.windowBase + i;
    info.rxBlk = (stepInfo.windowBase ^ (1u << stepInfo.step)) + i;
    info.txPartialOffset = (info.txBlk == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ?
                               dataOffsetTail_[channelIdx] :
                               dataOffset_[channelIdx];
    info.rxPartialOffset = (info.rxBlk == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ?
                               dataOffsetTail_[channelIdx] :
                               dataOffset_[channelIdx];
    const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
    const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
    info.txScratchOff = scratchBase + tempAlgParams_.sliceSize * info.txBlk + info.txPartialOffset;
    info.rxScratchOff = scratchBase + tempAlgParams_.sliceSize * info.rxBlk + info.rxPartialOffset;
    const u64 outBase = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
    info.txOutOff = outBase + tempAlgParams_.outputSliceStride * info.txBlk + info.txPartialOffset;
    info.rxOutOff = outBase + tempAlgParams_.outputSliceStride * info.rxBlk + info.rxPartialOffset;
    info.txSliceSize = (info.txBlk == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ?
                           dataSplitTail_[channelIdx] :
                           dataSplit_[channelIdx];
    info.rxSliceSize = (info.rxBlk == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ?
                           dataSplitTail_[channelIdx] :
                           dataSplit_[channelIdx];
    return info;
}

HcclResult InsTempAllGatherHDOp::BuildStepSlicesHD(
    const ChannelInfo& channelSend, const ChannelInfo& channelRecv, const AicpuHDStepInfo& stepInfo,
    const u32& channelIdx, std::vector<DataSlice>& txSrcSlices, std::vector<DataSlice>& txDstSlices,
    std::vector<DataSlice>& rxSrcSlices, std::vector<DataSlice>& rxDstSlices)
{
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    void* sendCclBuffAddr = channelSend.remoteCclMem.addr;
    void* recvCclBuffAddr = channelRecv.remoteCclMem.addr;
    void* localBuff
        = (!enableRemoteMemAccess_) ? tempAlgParams_.buffInfo.hcclBuff.addr : tempAlgParams_.buffInfo.outputPtr;
    void* sendRemoteBuff = (!enableRemoteMemAccess_) ? sendCclBuffAddr : channelSend.remoteOutputGraphMode.addr;
    void* recvRemoteBuff = (!enableRemoteMemAccess_) ? recvCclBuffAddr : channelRecv.remoteOutputGraphMode.addr;

    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        for (u32 i = 0; i < stepInfo.nSlices; ++i) {
            HDCalcInfo info = CalcSliceInfoHD(stepInfo, rpt, i, channelIdx);
            u64 txOff = (!enableRemoteMemAccess_) ? info.txScratchOff : info.txOutOff;
            u64 rxOff = (!enableRemoteMemAccess_) ? info.rxScratchOff : info.rxOutOff;
            txSrcSlices.emplace_back(localBuff, txOff, info.txSliceSize, info.txSliceSize / dataTypeSize);
            txDstSlices.emplace_back(sendRemoteBuff, txOff, info.txSliceSize, info.txSliceSize / dataTypeSize);
            rxSrcSlices.emplace_back(recvRemoteBuff, rxOff, info.rxSliceSize, info.rxSliceSize / dataTypeSize);
            rxDstSlices.emplace_back(localBuff, rxOff, info.rxSliceSize, info.rxSliceSize / dataTypeSize);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherHDOp::RunStepHD(
    const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const u32& channelIdx, u32 step)
{
    AicpuHDStepInfo stepInfo;
    CHK_RET(GetStepInfoHD(step, stepInfo));
    u32 peerRankKey = GetRankFromMap(stepInfo.peerRank);
    CHK_PRT_RET(
        channels.count(peerRankKey) == 0 || channelIdx >= channels.at(peerRankKey).size(),
        HCCL_ERROR(
            "[InsTempAllGatherHDOp][%s] rank[%u] invalid channel access, peerRankKey[%u] channelIdx[%u] "
            "channels.size[%zu] peerChannelSize[%zu]",
            __func__, myRank_, peerRankKey, channelIdx, channels.size(),
            channels.count(peerRankKey) ? channels.at(peerRankKey).size() : 0),
        HCCL_E_INTERNAL);
    // HD 对端收发同 rank：channelSend 与 channelRecv 为同一对端 channel
    const ChannelInfo& peerChannel = channels.at(peerRankKey)[channelIdx];
    HCCL_DEBUG(
        "[InsTempAllGatherHDOp] rank[%u] rankSize[%u] peer[%u] step[%u] windowBase[%u] nSlices[%u]", myRank_,
        templateRankSize_, stepInfo.peerRank, step, stepInfo.windowBase, stepInfo.nSlices);

    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    CHK_RET(BuildStepSlicesHD(
        peerChannel, peerChannel, stepInfo, channelIdx, txSrcSlices, txDstSlices, rxSrcSlices, rxDstSlices));

    TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
    TxRxChannels sendRecvChannels(peerChannel, peerChannel);
    SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);

    if (isDmaRead_) {
        // 读模式仅消费 rx slices：从对端拉取接收半窗；对端拉我由框架 notify 协议保证
        CHK_PRT_RET(
            SendRecvRead(sendRecvInfo, threads[channelIdx]),
            HCCL_ERROR("[InsTempAllGatherHDOp] SendRecv read failed (step=%u)", step), HcclResult::HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(
            SendRecvBatchWrite(sendRecvInfo, threads[channelIdx]),
            HCCL_ERROR("[InsTempAllGatherHDOp] SendRecv batch failed (step=%u)", step), HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherHDOp::RunAllGatherHD(
    const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const u32& channelIdx)
{
    const u32 nSteps = GetHDStepNum(templateRankSize_);

    for (u32 step = 0; step < nSteps; ++step) {
        CHK_RET(RunStepHD(threads, channels, channelIdx, step));
    }
    return HCCL_SUCCESS;
}

u32 InsTempAllGatherHDOp::GetRankFromMap(const u32 algRankIdx) const { return subCommRanks_[0].at(algRankIdx); }

HcclResult InsTempAllGatherHDOp::LocalDataCopy(const std::vector<ThreadHandle>& threads, const u32& channelIdx)

{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    u64 partialSliceSize = dataSplit_[channelIdx];
    u64 partialOffset = dataOffset_[channelIdx];
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    // 尾块模式（本 executor 恒置 tailSize=sliceSize，退化为常规块）
    if (tempAlgParams_.tailSize != 0 && myAlgRank == templateRankSize_ - 1) {
        partialSliceSize = dataSplitTail_[channelIdx];
        partialOffset = dataOffsetTail_[channelIdx];
    }
    for (u64 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        // 输入地址需含算法 rank 项（对齐 NHR LocalDataCopy 公式）：sole 绑定 ISS=0 时为加零项、行为等价；
        // parallel Inter0（ISS=N0*D、RPT=N0）/Intra1（ISS=D、RPT=N1）下缺少该项会把装载源错读为
        // 上一阶段输出网格的组首位置（server 0 位置）
        const u64 inOff = tempAlgParams_.inputSliceStride * myAlgRank + inBaseOff + partialOffset;
        u64 sliceCount = partialSliceSize / dataTypeSize;

        if (enableRemoteMemAccess_) {
            // remote 模式：IN 直写 OUT 本 rank 块
            const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
            const u64 outOff = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff + partialOffset;
            if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr && inOff == outOff) {
                continue; // inplace R==0 同址跳过
            }
            DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, partialSliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOff, partialSliceSize, sliceCount);
            CHK_RET(LocalCopy(threads[channelIdx], srcSlice, dstSlice));
        } else {
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

HcclResult InsTempAllGatherHDOp::PostLocalCopy(const ThreadHandle& thread, const u32& channelIdx)
{
    if (enableRemoteMemAccess_) {
        HCCL_INFO("[InsTempAllGatherHDOp] PostLocalCopy skip because remote memory access enabled");
        return HcclResult::HCCL_SUCCESS;
    }
    // 绑定的 sole executor（OrchestrateLoop）与 parallel executor 四阶段（GenTemplateAlgParamsIntra0/
    // Inter0/Inter1/Intra1）均恒置 outBuffType=OUTPUT，SCR->OUT 按 OSS*algRank 的回搬布局仅对该落位
    // 成立；其余落位属未验证绑定，报错而不是猜测搬移
    if (tempAlgParams_.buffInfo.outBuffType != BufferType::OUTPUT) {
        HCCL_ERROR(
            "[InsTempAllGatherHDOp] PostLocalCopy unsupported outBuffType[%d], only OUTPUT is verified.",
            static_cast<int>(tempAlgParams_.buffInfo.outBuffType));
        return HcclResult::HCCL_E_INTERNAL;
    }
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
            // 尾块模式
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

void InsTempAllGatherHDOp::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAllGatherHDOp::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

} // namespace ops_hccl
