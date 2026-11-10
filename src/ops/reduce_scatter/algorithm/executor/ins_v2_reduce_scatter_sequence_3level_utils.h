/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_REDUCE_SCATTER_SEQUENCE_3LEVEL_UTILS_H
#define HCCLV2_INS_V2_REDUCE_SCATTER_SEQUENCE_3LEVEL_UTILS_H

#include "executor_common_ops.h"

namespace ops_hccl {
// 三级 Sequence 参数生成所需的成员快照(OmniPipeCostAxes 同思路)
struct RsSeq3LevelCtx {
    u32 myRank = 0;
    u64 dataTypeSize = 0;
    u64 dataSize = 0;
    bool supportSymmetricMemory = false;
    u32 rankSizeLevel0 = 0;
    u32 rankSizeLevel1 = 0;
    u32 rankSizeLevel2 = 0;
    u32 rankIdxLevel0 = 0;
    u32 rankIdxLevel1 = 0;
};

/* 三级 Sequence 公共基类: 设备版(InsV2ReduceScatterSequenceExecutor3Level)与 DPU 版
 * (InsV2ReduceScatterSequenceExecutor3LevelDpu)完全相同的层级/通道/线程成员, 派生类成员访问名
 * 保持不变, 叶子 .cc 可直接访问(参照 InsV2OmniPipeExecutorBase 先例)。 */
class InsV2ReduceScatterSequence3LevelBase : public InsCollAlgBase {
public:
    InsV2ReduceScatterSequence3LevelBase() = default;
    ~InsV2ReduceScatterSequence3LevelBase() override = default;

protected:
    uint32_t rankSizeLevel0_{0};
    uint32_t rankSizeLevel1_{0};
    uint32_t rankSizeLevel2_{0};
    bool skipLevel1_{false};

    uint32_t rankIdxLevel0_{0};
    uint32_t rankIdxLevel1_{0};
    uint32_t rankIdxLevel2_{0};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;

    template <typename T0, typename T1, typename T2>
    friend HcclResult RsSeq3LevelCalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest,
        InsV2ReduceScatterSequence3LevelBase& ctx);

    friend HcclResult RsSeq3LevelRestoreOrchestrateCtx(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, InsV2ReduceScatterSequence3LevelBase& ctx);

    friend AlgNetMeta RsSeq3LevelBuildNetMeta(
        const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        u32 ratioDivisor, const InsV2ReduceScatterSequence3LevelBase& ctx);

    friend RsSeq3LevelCtx RsSeq3LevelMakeCtx(const InsV2ReduceScatterSequence3LevelBase& ctx);
};

// 叶子 OrchestrateLoop 开头聚参一次, 供 RsSeq3LevelGenIntraParams/GenInter1Params 循环内读取
inline RsSeq3LevelCtx RsSeq3LevelMakeCtx(const InsV2ReduceScatterSequence3LevelBase& ctx)
{
    RsSeq3LevelCtx rsCtx;
    rsCtx.myRank = ctx.myRank_;
    rsCtx.dataTypeSize = ctx.dataTypeSize_;
    rsCtx.dataSize = ctx.dataSize_;
    rsCtx.supportSymmetricMemory = ctx.supportSymmetricMemory_;
    rsCtx.rankSizeLevel0 = ctx.rankSizeLevel0_;
    rsCtx.rankSizeLevel1 = ctx.rankSizeLevel1_;
    rsCtx.rankSizeLevel2 = ctx.rankSizeLevel2_;
    rsCtx.rankIdxLevel0 = ctx.rankIdxLevel0_;
    rsCtx.rankIdxLevel1 = ctx.rankIdxLevel1_;
    return rsCtx;
}

// 三级 Sequence 公共 CalcRes, 原 InitCommInfo 的成员初始化改用官方 InitRankAndDataInfo
// + algHierarchyInfo_ 赋值
template <typename T0, typename T1, typename T2>
HcclResult RsSeq3LevelCalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest,
    InsV2ReduceScatterSequence3LevelBase& ctx)
{
    HCCL_DEBUG("[RsSeq3Level][CalcRes] myRank[%u] start", ctx.myRank_);
    ctx.InitRankAndDataInfo(param, topoInfo);
    ctx.algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO(
        "[RsSeq3Level][InitCommInfo] myRank [%u], rankSize [%u], redOp [%u], "
        "dataType [%u] dataTypeSize [%llu]",
        ctx.myRank_, ctx.rankSize_, ctx.reduceOp_, ctx.dataType_, ctx.dataTypeSize_);
    // 外层 3 层且各层中层/最内层均非空方可取 [i][0](MatchTopo 成功路径结构上恒满足, 防御性收口)
    if (algHierarchyInfo.infos.size() != TOPO_LEVEL_NUM_3 || algHierarchyInfo.infos[0].empty()
        || algHierarchyInfo.infos[1].empty() || algHierarchyInfo.infos[2].empty()
        || algHierarchyInfo.infos[0][0].empty() || algHierarchyInfo.infos[1][0].empty()
        || algHierarchyInfo.infos[2][0].empty()) {
        HCCL_ERROR(
            "[RsSeq3Level] myRank[%u] algHierarchyInfo invalid (size[%zu], should be %u with non-empty levels)",
            ctx.myRank_, algHierarchyInfo.infos.size(), TOPO_LEVEL_NUM_3);
        return HCCL_E_INTERNAL;
    }
    ctx.skipLevel1_ = (algHierarchyInfo.infos[1][0].size() == 1);
    std::shared_ptr<T0> tempAlgLevel0 = std::make_shared<T0>(param, ctx.myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<T1> tempAlgLevel1 = std::make_shared<T1>(param, ctx.myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<T2> tempAlgLevel2 = std::make_shared<T2>(param, ctx.myRank_, algHierarchyInfo.infos[2]);

    AlgResourceRequest resReq0;
    AlgResourceRequest resReq1;
    AlgResourceRequest resReq2;
    CHK_RET(tempAlgLevel0->CalcRes(comm, param, topoInfo, resReq0));
    if (ctx.skipLevel1_) {
        HCCL_INFO("[RsSeq3Level][CalcRes] myRank[%u] level1 rankSize is 1, skip level1 CalcRes", ctx.myRank_);
    } else {
        CHK_RET(tempAlgLevel1->CalcRes(comm, param, topoInfo, resReq1));
    }
    CHK_RET(tempAlgLevel2->CalcRes(comm, param, topoInfo, resReq2));

    if (ctx.skipLevel1_) {
        resourceRequest.slaveThreadNum = std::max(resReq0.slaveThreadNum, resReq2.slaveThreadNum);
    } else {
        resourceRequest.slaveThreadNum
            = std::max({resReq0.slaveThreadNum, resReq1.slaveThreadNum, resReq2.slaveThreadNum});
    }
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.resize(resourceRequest.slaveThreadNum);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        if (i < resReq0.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReq0.notifyNumPerThread[i]);
        }
        if (!ctx.skipLevel1_ && i < resReq1.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReq1.notifyNumPerThread[i]);
        }
        if (i < resReq2.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReq2.notifyNumPerThread[i]);
        }
    }
    if (ctx.skipLevel1_) {
        resourceRequest.notifyNumOnMainThread = std::max(resReq0.notifyNumOnMainThread, resReq2.notifyNumOnMainThread);
    } else {
        resourceRequest.notifyNumOnMainThread
            = std::max({resReq0.notifyNumOnMainThread, resReq1.notifyNumOnMainThread, resReq2.notifyNumOnMainThread});
    }
    HCCL_INFO(
        "[RsSeq3Level] myRank[%u] notifyNumOnMainThread is %u", ctx.myRank_, resourceRequest.notifyNumOnMainThread);
    resourceRequest.channels.resize(TOPO_LEVEL_NUM_3);
    if (resReq0.channels.empty() || resReq2.channels.empty()) {
        HCCL_ERROR(
            "[RsSeq3Level] myRank[%u] channels empty, level0[%zu] level2[%zu]", ctx.myRank_, resReq0.channels.size(),
            resReq2.channels.size());
        return HCCL_E_INTERNAL;
    }
    if (!ctx.skipLevel1_ && resReq1.channels.empty()) {
        HCCL_ERROR("[RsSeq3Level] myRank[%u] channels empty, level1[%zu]", ctx.myRank_, resReq1.channels.size());
        return HCCL_E_INTERNAL;
    }
    resourceRequest.channels[0] = resReq0.channels[0];
    if (!ctx.skipLevel1_) {
        resourceRequest.channels[1] = resReq1.channels[0];
    }
    resourceRequest.channels[2] = resReq2.channels[0];
    HCCL_INFO(
        "[RsSeq3Level] myRank[%u] slaveThreadNum is [%u], notifyNumOnMainThread is [%u], "
        "level0 channel size [%zu], level1 channel size [%zu], level2 channel size [%zu]",
        ctx.myRank_, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread,
        resourceRequest.channels[0].size(), resourceRequest.channels[1].size(), resourceRequest.channels[2].size());
    return HCCL_SUCCESS;
}

// 三级 Sequence 公共 Orchestrate ctx 恢复段: 两叶子 Orchestrate 前半段(成员恢复 + 对称内存设置 +
// 层级校验 + rankIdx/rankSize/skipLevel1 + 通道图恢复 + 状态日志)逐字同构, 抽为自由函数
inline HcclResult RsSeq3LevelRestoreOrchestrateCtx(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, InsV2ReduceScatterSequence3LevelBase& ctx)
{
    ctx.myRank_ = resCtx.topoInfo.userRank;
    ctx.rankSize_ = resCtx.topoInfo.userRankSize;
    ctx.algHierarchyInfo_ = resCtx.algHierarchyInfo;
    ctx.threads_ = resCtx.threads;
    ctx.dataCount_ = param.DataDes.count;
    ctx.dataType_ = param.DataDes.dataType;
    ctx.reduceOp_ = param.reduceType;
    ctx.dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    ctx.dataSize_ = ctx.dataCount_ * ctx.dataTypeSize_;
    ctx.supportSymmetricMemory_ = param.supportSymmetricMemory;
    ctx.SetSymmetricMemoryInfo(param);

    // 层级数口径与 RsSeq3LevelCalcRes 统一(恰为 3 层): 超层场景由 CalcRes 先行拦截, 此处同口径防御
    if (ctx.algHierarchyInfo_.infos.size() != TOPO_LEVEL_NUM_3 || ctx.algHierarchyInfo_.infos[0].empty()
        || ctx.algHierarchyInfo_.infos[1].empty() || ctx.algHierarchyInfo_.infos[2].empty()
        || ctx.algHierarchyInfo_.infos[0][0].empty() || ctx.algHierarchyInfo_.infos[1][0].empty()
        || ctx.algHierarchyInfo_.infos[2][0].empty()) {
        HCCL_ERROR("[RsSeq3Level][%s] invalid algHierarchyInfo infos.", __func__);
        return HCCL_E_PARA;
    }
    ctx.rankIdxLevel0_ = ctx.myRank_ % ctx.algHierarchyInfo_.infos[0][0].size();
    ctx.rankIdxLevel1_
        = (ctx.myRank_ / ctx.algHierarchyInfo_.infos[0][0].size()) % ctx.algHierarchyInfo_.infos[1][0].size();
    ctx.rankIdxLevel2_
        = ctx.myRank_ / (ctx.algHierarchyInfo_.infos[0][0].size() * ctx.algHierarchyInfo_.infos[1][0].size());

    ctx.rankSizeLevel0_ = ctx.algHierarchyInfo_.infos[0][0].size();
    ctx.rankSizeLevel1_ = ctx.algHierarchyInfo_.infos[1][0].size();
    ctx.rankSizeLevel2_ = ctx.algHierarchyInfo_.infos[2][0].size();
    ctx.skipLevel1_ = (ctx.rankSizeLevel1_ == 1);
    if (ctx.skipLevel1_) {
        HCCL_INFO("[RsSeq3Level][Orchestrate] myRank[%u] level1 rankSize is 1, skip level1", ctx.myRank_);
    }
    CHK_RET(ctx.RestoreChannelMap(resCtx, ctx.remoteRankToChannelInfo_));
    HCCL_INFO(
        "[RsSeq3Level][Orchestrate] myRank_[%u] rankIdxLevel0_[%u] rankIdxLevel1_[%u] rankIdxLevel2_[%u] "
        "rankSizeLevel0_[%u] rankSizeLevel1_[%u] rankSizeLevel2_[%u]",
        ctx.myRank_, ctx.rankIdxLevel0_, ctx.rankIdxLevel1_, ctx.rankIdxLevel2_, ctx.rankSizeLevel0_,
        ctx.rankSizeLevel1_, ctx.rankSizeLevel2_);
    return HCCL_SUCCESS;
}

// 三级 Sequence 公共 GetAlgNetMeta 骨架: 两叶子的 netTypes/groupSizes/dataRatios/rankSizes 产出
// 同构, 仅 dataRatios 口径不同(设备 ÷userRankSize, DPU 家族口径不除), 统一为
// {d1*d2/div, d2/div, 1/div}, 除数由叶子传(设备叶传 topoInfo->userRankSize, DPU 叶传 1);
// 各叶壳保留 MatchTopoForProbe 探测与各自 algoTypes 段。
inline AlgNetMeta RsSeq3LevelBuildNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo, u32 ratioDivisor,
    const InsV2ReduceScatterSequence3LevelBase& ctx)
{
    // 探测/拓扑降级场景下层级可能不足或某层内层为空, 取 [i][0] 前须两级判空, 否则越界读
    // (与执行器叶子 CalcCostCoeff 的 phyLevel* 守卫同族)。仅需校验实际会被访问的层:
    // 第 0 层无条件访问, 1/2 层仅在外层 size 足够时访问
    auto levelsUsable = [](const auto& levels) {
        if (levels.empty() || levels[0].empty()) {
            return false;
        }
        for (size_t i = 1; i < levels.size(); i++) {
            if (levels[i].empty()) {
                return false;
            }
        }
        return true;
    };
    CHK_PRT_RET(
        !levelsUsable(algHierarchyInfo.infos) || !levelsUsable(algHierarchyInfo.physicalIdxForAlgoLevels),
        HCCL_WARNING(
            "[RsSeq3Level][GetAlgNetMeta] algHierarchyInfo levels insufficient or inner empty "
            "(infos[%zu] physIdx[%zu]).",
            algHierarchyInfo.infos.size(), algHierarchyInfo.physicalIdxForAlgoLevels.size()),
        AlgNetMeta{});
    u32 rankSizeLevel0 = algHierarchyInfo.infos[0][0].size();
    u32 rankSizeLevel1 = (algHierarchyInfo.infos.size() > 1) ? algHierarchyInfo.infos[1][0].size() : 1;
    u32 rankSizeLevel2 = (algHierarchyInfo.infos.size() > 2) ? algHierarchyInfo.infos[2][0].size() : 1;
    u32 physIdxLevel0 = static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]);
    u32 physIdxLevel1 = (algHierarchyInfo.physicalIdxForAlgoLevels.size() > 1) ?
                            static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[1][0]) :
                            physIdxLevel0;
    u32 physIdxLevel2 = (algHierarchyInfo.physicalIdxForAlgoLevels.size() > 2) ?
                            static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[2][0]) :
                            physIdxLevel1;
    CommTopo netTypeLevel0 = ctx.GetPhysicalLevelTopoType(topoInfo, physIdxLevel0);
    CommTopo netTypeLevel1 = ctx.GetPhysicalLevelTopoType(topoInfo, physIdxLevel1);
    CommTopo netTypeLevel2 = ctx.GetPhysicalLevelTopoType(topoInfo, physIdxLevel2);
    AlgNetMeta meta;
    meta.netTypes.push_back(netTypeLevel0);
    meta.netTypes.push_back(netTypeLevel1);
    meta.netTypes.push_back(netTypeLevel2);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1, 1, 1};
    meta.dataRatios
        = {1.0f * rankSizeLevel1 * rankSizeLevel2 / ratioDivisor, 1.0f * rankSizeLevel2 / ratioDivisor,
           1.0f / ratioDivisor};
    meta.rankSizes = {rankSizeLevel0, rankSizeLevel1, rankSizeLevel2};
    return meta;
}

// 三级 Sequence 公共 L0 模板参数生成(Intra)
inline void RsSeq3LevelGenIntraParams(
    const RsSeq3LevelCtx& ctx, TemplateDataParams& tempAlgParamsIntra, const u64 processedDataCount,
    const u64 currDataCount, const u64 loop)
{
    tempAlgParamsIntra.count = currDataCount;
    tempAlgParamsIntra.buffInfo.inBuffBaseOff = processedDataCount * ctx.dataTypeSize;
    tempAlgParamsIntra.buffInfo.outBuffBaseOff = ctx.supportSymmetricMemory ? processedDataCount * ctx.dataTypeSize : 0;
    tempAlgParamsIntra.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsIntra.sliceSize = currDataCount * ctx.dataTypeSize;
    tempAlgParamsIntra.tailSize = tempAlgParamsIntra.sliceSize;

    tempAlgParamsIntra.inputSliceStride = ctx.dataSize;
    tempAlgParamsIntra.outputSliceStride = currDataCount * ctx.dataTypeSize;
    tempAlgParamsIntra.repeatNum = ctx.rankSizeLevel1 * ctx.rankSizeLevel2;
    tempAlgParamsIntra.inputRepeatStride = ctx.rankSizeLevel0 * ctx.dataSize;
    tempAlgParamsIntra.outputRepeatStride = ctx.rankSizeLevel0 * currDataCount * ctx.dataTypeSize;

    HCCL_INFO(
        "[RsSeq3Level] myRank[%u] loop[%llu] Intra inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] "
        "repeatNum[%llu] inputRepeatStride[%llu] outputRepeatStride[%llu]",
        ctx.myRank, loop, tempAlgParamsIntra.inputSliceStride, tempAlgParamsIntra.outputSliceStride,
        tempAlgParamsIntra.sliceSize, tempAlgParamsIntra.buffInfo.inBuffBaseOff,
        tempAlgParamsIntra.buffInfo.outBuffBaseOff, tempAlgParamsIntra.repeatNum, tempAlgParamsIntra.inputRepeatStride,
        tempAlgParamsIntra.outputRepeatStride);
}

// 三级 Sequence 公共 L1 模板参数生成(Inter1): 两叶子逐字相同, 改经 RsSeq3LevelCtx 聚参读取。
inline void RsSeq3LevelGenInter1Params(
    const RsSeq3LevelCtx& ctx, TemplateDataParams& tempAlgParamsInter, const u64 processedDataCount,
    const u64 currDataCount, const u64 loop)
{
    tempAlgParamsInter.count = currDataCount;
    u64 symMemBaseOff = ctx.supportSymmetricMemory ? processedDataCount * ctx.dataTypeSize : 0;
    tempAlgParamsInter.buffInfo.inBuffBaseOff = symMemBaseOff + ctx.rankIdxLevel0 * currDataCount * ctx.dataTypeSize;
    tempAlgParamsInter.buffInfo.outBuffBaseOff
        = symMemBaseOff
          + (ctx.rankIdxLevel0 + (ctx.rankIdxLevel1 * ctx.rankSizeLevel0)) * currDataCount * ctx.dataTypeSize;
    tempAlgParamsInter.buffInfo.hcclBuffBaseOff = ctx.rankIdxLevel0 * currDataCount * ctx.dataTypeSize;

    tempAlgParamsInter.sliceSize = currDataCount * ctx.dataTypeSize;
    tempAlgParamsInter.tailSize = tempAlgParamsInter.sliceSize;

    tempAlgParamsInter.inputSliceStride = ctx.rankSizeLevel0 * currDataCount * ctx.dataTypeSize;
    tempAlgParamsInter.outputSliceStride = 0;
    tempAlgParamsInter.repeatNum = ctx.rankSizeLevel2;
    tempAlgParamsInter.inputRepeatStride = ctx.rankSizeLevel0 * ctx.rankSizeLevel1 * currDataCount * ctx.dataTypeSize;
    tempAlgParamsInter.outputRepeatStride = ctx.rankSizeLevel0 * ctx.rankSizeLevel1 * currDataCount * ctx.dataTypeSize;

    HCCL_INFO(
        "[RsSeq3Level] myRank[%u] loop[%llu] Inter1 inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] "
        "repeatNum[%llu] inputRepeatStride[%llu] outputRepeatStride[%llu]",
        ctx.myRank, loop, tempAlgParamsInter.inputSliceStride, tempAlgParamsInter.outputSliceStride,
        tempAlgParamsInter.sliceSize, tempAlgParamsInter.buffInfo.inBuffBaseOff,
        tempAlgParamsInter.buffInfo.outBuffBaseOff, tempAlgParamsInter.repeatNum, tempAlgParamsInter.inputRepeatStride,
        tempAlgParamsInter.outputRepeatStride);
}

// 三级 Sequence 公共 GenTempResource(取 DPU 超集版): 两版差异仅在 shmem 指针透传与 size 打印
// %u/%zu, 设备模板不读 shmem 字段、无条件透传无害; remoteRankToChannelInfo 由叶子传入,
// myRank 经 resCtx.topoInfo.userRank 读取, 无需友元。
template <typename InsAlgTemplate>
HcclResult RsSeq3LevelGenTempResource(
    const AlgResourceCtxSerializable& resCtx, const u32 channelLevelIdx,
    const std::vector<std::map<u32, std::vector<ChannelInfo>>>& remoteRankToChannelInfo,
    const std::shared_ptr<InsAlgTemplate>& algTemplate, TemplateResource& tempResource)
{
    AlgResourceRequest req;
    CHK_RET(algTemplate->GetRes(req));
    if (channelLevelIdx >= remoteRankToChannelInfo.size()) {
        HCCL_ERROR(
            "[RsSeq3Level][GenTempResource] myRank[%u] channelLevelIdx[%u] should be lower than"
            " remoteRankToChannelInfo.size()[%zu]",
            resCtx.topoInfo.userRank, channelLevelIdx, remoteRankToChannelInfo.size());
        return HCCL_E_INTERNAL;
    }
    tempResource.channels = remoteRankToChannelInfo[channelLevelIdx];
    // threads 切片长度来自模板 GetRes 的运行期状态, 失真(如 threadNum-1 在 0 时回绕成极大值)会使
    // 迭代器越过 end(); 加法先升位 size_t 再比较, 防 u32 回绕绕过校验。本实现为设备/DPU 两叶子共用
    if (resCtx.threads.size() < static_cast<size_t>(1) + req.slaveThreadNum) {
        HCCL_ERROR(
            "[RsSeq3Level][GenTempResource] myRank[%u] threads size[%zu] less than required[%llu], "
            "slaveThreadNum from GetRes may be corrupted",
            resCtx.topoInfo.userRank, resCtx.threads.size(), static_cast<unsigned long long>(1) + req.slaveThreadNum);
        return HCCL_E_INTERNAL;
    }
    tempResource.threads.assign(
        resCtx.threads.begin(), resCtx.threads.begin() + static_cast<size_t>(1) + req.slaveThreadNum);
    // shmem 指针透传(照二级 DPU RS 写法): TemplateResource 中该二字段为裸 void* 无默认初始化, 不赋值
    // 是野值; 无条件透传不等于额外申请(100MB 共享内存仅 host-DPU 场景按通信域申请一次)。三级均设置,
    // 设备模板不读该字段, 无害
    tempResource.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    tempResource.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    tempResource.isDpuThreadEnabled = resCtx.isDpuThreadEnabled;
    tempResource.dpuThread = resCtx.dpuThread;
    tempResource.dpuThreadNotifyIdxOnMain = resCtx.notifyNumOnMainThread + 1;
    tempResource.isHcommChannelDrainSupported = resCtx.isHcommChannelDrainSupported;
    return HCCL_SUCCESS;
}
} // namespace ops_hccl

#endif // HCCLV2_INS_V2_REDUCE_SCATTER_SEQUENCE_3LEVEL_UTILS_H
