/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_GATHER_SEQUENCE_3LEVEL_UTILS_H
#define HCCLV2_INS_V2_ALL_GATHER_SEQUENCE_3LEVEL_UTILS_H

#include "executor_v2_base.h"
#include "executor_common_ops.h"
#include "alg_param.h"
#include "channel.h"
#include "coll_alg_v2_exec_registry.h"

namespace ops_hccl {

// ============================================================================
// 聚参 struct：打包自由函数依赖的成员。叶子不逐字段手抄，统一由
// InsV2AllGatherSequence3LevelBase::MakeCtx() 产出，避免与基类成员两处维护。
// ============================================================================
struct AgSeq3LevelCtx {
    u64 myRank;
    u64 dataTypeSize;
    u64 dataSize;
    u64 rankSizeLevel0;
    u64 rankSizeLevel1;
    u64 rankSizeLevel2;
    u64 rankIdxLevel0;
    u64 rankIdxLevel1;
};

// ============================================================================
// 公共基类：收敛 AICPU / DPU 两实现的公共成员变量（参照 OmniPipe 模式）
// ============================================================================
class InsV2AllGatherSequence3LevelBase : public InsCollAlgBase {
public:
    struct LevelInfo {
        std::map<u32, std::vector<ChannelInfo>> channels;
        std::vector<std::vector<u32>> hierarchyInfo;
        u64 rankSize;
        std::vector<ThreadHandle> threads;
        LevelInfo(std::map<u32, std::vector<ChannelInfo>> channels_, std::vector<std::vector<u32>> hierarchyInfo_)
            : channels(std::move(channels_)),
              hierarchyInfo(std::move(hierarchyInfo_)),
              rankSize(GetRankSize(hierarchyInfo)),
              threads()
        {}
    };

protected:
    u64 rankIdxLevel0_{0};
    u64 rankIdxLevel1_{0};
    bool skipLevel1_{false};

    std::vector<ThreadHandle> threads_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<LevelInfo> levels_;

    // 从 resCtx 恢复 Orchestrate 循环所需的全部状态：两实现逐字相同的初始化样板收敛于此。
    // 语句与原实现逐行一致，仅错误日志的 tag 由 __func__ 换成入参 executorName。
    HcclResult
    InitOrchestrateState(const OpParam& param, const AlgResourceCtxSerializable& resCtx, const char* executorName)
    {
        maxTmpMemSize_ = resCtx.cclMem.size; // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
        myRank_ = resCtx.topoInfo.userRank;
        // 给channels_和threads_赋值
        threads_ = resCtx.threads;
        supportSymmetricMemory_ = param.supportSymmetricMemory;
        if (supportSymmetricMemory_) {
            inputOffset_ = param.inputOffset;
            outputOffset_ = param.outputOffset;
            inputSymWindow_ = param.inputSymWindow;
            outputSymWindow_ = param.outputSymWindow;
        }
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

        dataCount_ = param.DataDes.count;
        dataType_ = param.DataDes.dataType;
        dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
        dataSize_ = dataCount_ * dataTypeSize_;
        if (resCtx.algHierarchyInfo.infos.size() < TOPO_LEVEL_NUM_3) {
            HCCL_ERROR(
                "[%s] algHierarchyInfo.infos size[%zu] < 3.", executorName, resCtx.algHierarchyInfo.infos.size());
            return HCCL_E_PARA;
        }
        // resCtx 为跨端反序列化恢复的状态: 逐级校验内层向量非空, 防止 rankSize 连乘为 0
        // 导致叶子 Orchestrate 的 myRank_ % levels_[0].rankSize 除零; 同时校验通道映射元素数,
        // 防止下方按层索引取值越界。
        for (u32 lvl = 0; lvl < TOPO_LEVEL_NUM_3; lvl++) {
            CHK_PRT_RET(
                resCtx.algHierarchyInfo.infos[lvl].empty() || GetRankSize(resCtx.algHierarchyInfo.infos[lvl]) == 0,
                HCCL_ERROR("[%s] algHierarchyInfo.infos[%u] is empty or rankSize is 0.", executorName, lvl),
                HCCL_E_PARA);
        }
        CHK_PRT_RET(
            remoteRankToChannelInfo_.size() < TOPO_LEVEL_NUM_3,
            HCCL_ERROR("[%s] remoteRankToChannelInfo size[%zu] < 3.", executorName, remoteRankToChannelInfo_.size()),
            HCCL_E_PARA);
        levels_.emplace_back(remoteRankToChannelInfo_[0], resCtx.algHierarchyInfo.infos[0]);
        levels_.emplace_back(remoteRankToChannelInfo_[1], resCtx.algHierarchyInfo.infos[1]);
        levels_.emplace_back(remoteRankToChannelInfo_[2], resCtx.algHierarchyInfo.infos[2]);
        return HCCL_SUCCESS;
    }

    // 基类成员 -> 聚参 struct 的唯一映射点：叶子在 levels_ 填好后调一次，不逐字段手抄。
    // 注：对称标志不经 ctx 传递，GenLevel0/1Params 统一直读 param.supportSymmetricMemory（单一来源）。
    AgSeq3LevelCtx MakeCtx() const
    {
        return AgSeq3LevelCtx{
            myRank_,        dataTypeSize_, dataSize_, levels_[0].rankSize, levels_[1].rankSize, levels_[2].rankSize,
            rankIdxLevel0_, rankIdxLevel1_};
    }

    // GetAlgNetMeta：AICPU / DPU 两版逻辑完全相同（仅日志 tag 不同），收敛为基类成员，
    // 叶子 GetAlgNetMeta 传入各自 executorName 一行转发。
    template <typename AlgTopoMatch>
    AlgNetMeta AgSeq3LevelGetAlgNetMeta(
        const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName,
        const char* executorName) const
    {
        (void)param;
        AlgHierarchyInfoForAllLevel algHierarchyInfo;
        const std::string probeTag = std::string("[") + executorName + "][GetAlgNetMeta]";
        if (!MatchTopoForProbe<AlgTopoMatch>(
                topoInfo, algHierarchyInfo, algName, probeTag.c_str(), TopoProbeScene::PROBE_GET_ALG_NET_META)) {
            return {};
        }
        u32 rankSizeLevel0 = algHierarchyInfo.infos[0][0].size();
        u32 rankSizeLevel1 = (algHierarchyInfo.infos.size() > 1) ? algHierarchyInfo.infos[1][0].size() : 1;
        u32 rankSizeLevel2 = (algHierarchyInfo.infos.size() > 2) ? algHierarchyInfo.infos[2][0].size() : 1;
        CHK_PRT_RET(
            algHierarchyInfo.physicalIdxForAlgoLevels.size() != TOPO_LEVEL_NUM_3,
            HCCL_WARNING(
                "[%s][GetAlgNetMeta] physicalIdxForAlgoLevels size[%zu] != 3.", executorName,
                algHierarchyInfo.physicalIdxForAlgoLevels.size()),
            {});
        CommTopo netTypeLevel0
            = GetPhysicalLevelTopoType(topoInfo, static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]));
        CommTopo netTypeLevel1
            = GetPhysicalLevelTopoType(topoInfo, static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[1][0]));
        CommTopo netTypeLevel2
            = GetPhysicalLevelTopoType(topoInfo, static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[2][0]));
        AlgNetMeta meta;
        meta.netTypes.push_back(netTypeLevel0);
        meta.netTypes.push_back(netTypeLevel1);
        meta.netTypes.push_back(netTypeLevel2);
        meta.intraGroupMode = CostAggMode::SUM;
        meta.groupSizes = {1, 1, 1};
        meta.dataRatios = {1.0f * rankSizeLevel2 * rankSizeLevel1, 1.0f * rankSizeLevel2, 1.0f};
        meta.rankSizes = {rankSizeLevel0, rankSizeLevel1, rankSizeLevel2};
        return meta;
    }

public:
    static u64 GetRankSize(const std::vector<std::vector<u32>>& vTopo)
    {
        u64 count = 1;
        for (const auto& i : vTopo) {
            count *= i.size();
        }
        return count;
    }
};

// ============================================================================
// 自由函数：公共逻辑以自由函数共享，差异逻辑留在各自叶子
// ============================================================================

// GenTempResource：统一用 DPU 版签名。
// 比设备侧版本多 resCtx：用于向 L2 DPU 模板透传共享内存句柄。
// AICPU 侧透传过来无害（设备模板不读该字段，且 AlgResourceCtxSerializable 侧有默认值 nullptr）
inline HcclResult AgSeq3LevelGenTempResource(
    const std::vector<InsV2AllGatherSequence3LevelBase::LevelInfo>& levels, int idx,
    const AlgResourceCtxSerializable& resCtx, TemplateResource& res)
{
    res.channels = levels[idx].channels;
    res.threads = levels[idx].threads;
    res.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    res.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    res.isDpuThreadEnabled = resCtx.isDpuThreadEnabled;
    res.dpuThread = resCtx.dpuThread;
    res.dpuThreadNotifyIdxOnMain = resCtx.notifyNumOnMainThread + 1;
    res.isHcommChannelDrainSupported = resCtx.isHcommChannelDrainSupported;
    return HCCL_SUCCESS;
}

// PrepareResForTemplate：纯线程分配，两版完全相同
template <typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
inline HcclResult AgSeq3LevelPrepareResForTemplate(
    std::vector<InsV2AllGatherSequence3LevelBase::LevelInfo>& levels, const std::vector<ThreadHandle>& threads,
    InsAlgTemplate0& tempAlgLevel0, InsAlgTemplate1& tempAlgLevel1, InsAlgTemplate2& tempAlgLevel2)
{
    AlgResourceRequest Level0TempRequest;
    AlgResourceRequest Level1TempRequest;
    AlgResourceRequest Level2TempRequest;
    tempAlgLevel0.GetRes(Level0TempRequest);
    tempAlgLevel1.GetRes(Level1TempRequest);
    tempAlgLevel2.GetRes(Level2TempRequest);

    levels[0].threads.assign(threads.begin(), threads.begin() + Level0TempRequest.slaveThreadNum + 1);
    levels[1].threads.assign(threads.begin(), threads.begin() + Level1TempRequest.slaveThreadNum + 1);
    levels[2].threads.assign(threads.begin(), threads.begin() + Level2TempRequest.slaveThreadNum + 1);
    return HCCL_SUCCESS;
}

// GenTemplateAlgParamsLevel0：两版逻辑完全相同（仅日志 tag 不同）
inline void AgSeq3LevelGenLevel0Params(
    const AgSeq3LevelCtx& ctx, const OpParam& param, const AlgResourceCtxSerializable& resCtx, const u64 curCount,
    const u64 dataOffset, TemplateDataParams& tempAlgParamsLevel0, const char* executorName)
{
    tempAlgParamsLevel0.enableRemoteMemAccess = param.supportSymmetricMemory;
    if (tempAlgParamsLevel0.enableRemoteMemAccess) {
        tempAlgParamsLevel0.buffInfo.inputPtr = param.outputPtr;
        tempAlgParamsLevel0.buffInfo.inBuffType = BufferType::OUTPUT;
        tempAlgParamsLevel0.buffInfo.inBuffBaseOff = dataOffset;
    } else {
        tempAlgParamsLevel0.buffInfo.inputPtr = resCtx.cclMem.addr;
        tempAlgParamsLevel0.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsLevel0.buffInfo.inBuffBaseOff = 0;
    }
    tempAlgParamsLevel0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsLevel0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsLevel0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsLevel0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel0.buffInfo.inputSize = param.inputSize;
    tempAlgParamsLevel0.buffInfo.outputSize = param.outputSize;

    tempAlgParamsLevel0.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsLevel0.buffInfo.hcclBuffBaseOff = 0;
    tempAlgParamsLevel0.count = curCount;
    tempAlgParamsLevel0.sliceSize = curCount * ctx.dataTypeSize;
    tempAlgParamsLevel0.tailSize = tempAlgParamsLevel0.sliceSize;

    tempAlgParamsLevel0.inputSliceStride = 0;
    tempAlgParamsLevel0.outputSliceStride = ctx.dataSize;
    tempAlgParamsLevel0.repeatNum = ctx.rankSizeLevel1 * ctx.rankSizeLevel2;
    tempAlgParamsLevel0.inputRepeatStride = curCount * ctx.dataTypeSize;
    tempAlgParamsLevel0.outputRepeatStride = ctx.rankSizeLevel0 * ctx.dataSize;
    if (tempAlgParamsLevel0.enableRemoteMemAccess) {
        tempAlgParamsLevel0.inputSliceStride = ctx.dataSize;
        tempAlgParamsLevel0.inputRepeatStride = ctx.rankSizeLevel0 * ctx.dataSize;
    }

    HCCL_DEBUG(
        "[%s][GenTemplateAlgParamsLevel0] rank[%llu] inBuffBaseOff[%llu] "
        "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu] "
        "levels_[0].rankSize[%llu] levels_[1].rankSize[%llu] rankIdxLevel0[%llu] rankIdxLevel1[%llu]",
        executorName, ctx.myRank, tempAlgParamsLevel0.buffInfo.inBuffBaseOff,
        tempAlgParamsLevel0.buffInfo.outBuffBaseOff, tempAlgParamsLevel0.buffInfo.hcclBuffBaseOff,
        tempAlgParamsLevel0.sliceSize, tempAlgParamsLevel0.outputSliceStride, ctx.rankSizeLevel0, ctx.rankSizeLevel1,
        ctx.rankIdxLevel0, ctx.rankIdxLevel1);
}

// GenTemplateAlgParamsLevel1：两版逻辑完全相同（仅日志 tag 不同）
inline void AgSeq3LevelGenLevel1Params(
    const AgSeq3LevelCtx& ctx, const OpParam& param, const AlgResourceCtxSerializable& resCtx, const u64 curCount,
    const u64 dataOffset, TemplateDataParams& tempAlgParamsLevel1, const char* executorName)
{
    (void)dataOffset;
    tempAlgParamsLevel1.enableRemoteMemAccess = param.supportSymmetricMemory;
    if (tempAlgParamsLevel1.enableRemoteMemAccess) {
        tempAlgParamsLevel1.buffInfo.inputPtr = param.outputPtr;
        tempAlgParamsLevel1.buffInfo.outputPtr = param.outputPtr;
        tempAlgParamsLevel1.buffInfo.inBuffType = BufferType::OUTPUT;
        tempAlgParamsLevel1.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParamsLevel1.buffInfo.inBuffBaseOff = ctx.rankIdxLevel0 * ctx.dataSize;
        tempAlgParamsLevel1.buffInfo.outBuffBaseOff = ctx.rankIdxLevel0 * ctx.dataSize;
        tempAlgParamsLevel1.inputSliceStride = ctx.rankSizeLevel0 * ctx.dataSize;
        tempAlgParamsLevel1.outputSliceStride = ctx.rankSizeLevel0 * ctx.dataSize;
        tempAlgParamsLevel1.inputRepeatStride = ctx.rankSizeLevel0 * ctx.rankSizeLevel1 * ctx.dataSize;
        tempAlgParamsLevel1.outputRepeatStride = ctx.rankSizeLevel0 * ctx.rankSizeLevel1 * ctx.dataSize;
    } else {
        tempAlgParamsLevel1.buffInfo.inputPtr = resCtx.cclMem.addr;
        tempAlgParamsLevel1.buffInfo.outputPtr = resCtx.cclMem.addr;
        tempAlgParamsLevel1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsLevel1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsLevel1.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsLevel1.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsLevel1.inputSliceStride = curCount * ctx.dataTypeSize;
        tempAlgParamsLevel1.outputSliceStride = curCount * ctx.dataTypeSize;
        tempAlgParamsLevel1.inputRepeatStride = curCount * ctx.dataTypeSize * ctx.rankSizeLevel1;
        tempAlgParamsLevel1.outputRepeatStride = curCount * ctx.dataTypeSize * ctx.rankSizeLevel1;
    }
    tempAlgParamsLevel1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsLevel1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.inputSize = param.inputSize;
    tempAlgParamsLevel1.buffInfo.outputSize = param.outputSize;
    tempAlgParamsLevel1.buffInfo.hcclBuffBaseOff = 0;
    tempAlgParamsLevel1.count = curCount;
    tempAlgParamsLevel1.sliceSize = curCount * ctx.dataTypeSize;
    tempAlgParamsLevel1.tailSize = tempAlgParamsLevel1.sliceSize;
    tempAlgParamsLevel1.repeatNum = ctx.rankSizeLevel2;

    HCCL_DEBUG(
        "[%s][GenTemplateAlgParamsLevel1] rank[%llu] inBuffBaseOff[%llu] "
        "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu] "
        "outputRepeatStride[%llu]",
        executorName, ctx.myRank, tempAlgParamsLevel1.buffInfo.inBuffBaseOff,
        tempAlgParamsLevel1.buffInfo.outBuffBaseOff, tempAlgParamsLevel1.buffInfo.hcclBuffBaseOff,
        tempAlgParamsLevel1.sliceSize, tempAlgParamsLevel1.outputSliceStride, tempAlgParamsLevel1.outputRepeatStride);
}

// CalcRes：两版逻辑完全相同（仅日志 tag 不同）
template <typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
inline HcclResult AgSeq3LevelCalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest, const char* executorName)
{
    if (algHierarchyInfo.infos.size() != TOPO_LEVEL_NUM_3) {
        HCCL_ERROR(
            "[%s] algHierarchyInfo size %zu should be %u", executorName, algHierarchyInfo.infos.size(),
            TOPO_LEVEL_NUM_3);
        return HCCL_E_INTERNAL;
    }
    // 构建template
    InsAlgTemplate0 Level0TempAlg(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    InsAlgTemplate1 Level1TempAlg(param, topoInfo->userRank, algHierarchyInfo.infos[1]);
    InsAlgTemplate2 Level2TempAlg(param, topoInfo->userRank, algHierarchyInfo.infos[2]);

    // 调用计算资源的函数
    AlgResourceRequest Level0TempRequest, Level1TempRequest, Level2TempRequest;
    CHK_RET(Level0TempAlg.CalcRes(comm, param, topoInfo, Level0TempRequest));
    CHK_RET(Level1TempAlg.CalcRes(comm, param, topoInfo, Level1TempRequest));
    CHK_RET(Level2TempAlg.CalcRes(comm, param, topoInfo, Level2TempRequest));

    resourceRequest.notifyNumOnMainThread = std::max(
        {Level0TempRequest.notifyNumOnMainThread, Level1TempRequest.notifyNumOnMainThread,
         Level2TempRequest.notifyNumOnMainThread});
    resourceRequest.slaveThreadNum = std::max(
        {Level0TempRequest.slaveThreadNum, Level1TempRequest.slaveThreadNum, Level2TempRequest.slaveThreadNum});
    resourceRequest.notifyNumPerThread.insert(
        resourceRequest.notifyNumPerThread.end(), Level0TempRequest.notifyNumPerThread.begin(),
        Level0TempRequest.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.insert(
        resourceRequest.notifyNumPerThread.end(), Level1TempRequest.notifyNumPerThread.begin(),
        Level1TempRequest.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.insert(
        resourceRequest.notifyNumPerThread.end(), Level2TempRequest.notifyNumPerThread.begin(),
        Level2TempRequest.notifyNumPerThread.end());
    CHK_PRT_RET(
        Level0TempRequest.channels.empty() || Level1TempRequest.channels.empty() || Level2TempRequest.channels.empty(),
        HCCL_ERROR(
            "[%s][CalcRes] Level0Template, Level1Template or Level2TempRequest has empty channels.", executorName),
        HcclResult::HCCL_E_INTERNAL);
    resourceRequest.channels.emplace_back(Level0TempRequest.channels[0]);
    resourceRequest.channels.emplace_back(Level1TempRequest.channels[0]);
    resourceRequest.channels.emplace_back(Level2TempRequest.channels[0]);
    HCCL_DEBUG(
        "[%s][CalcRes] notifyNumOnMainThread[%u], slaveThreadNum[%u], channels[%zu]", executorName,
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum, resourceRequest.channels.size());
    for (size_t i = 0; i < resourceRequest.notifyNumPerThread.size(); i++) {
        HCCL_DEBUG(
            "[%s][CalcRes] notifyNumPerThread[%zu]=[%u]", executorName, i, resourceRequest.notifyNumPerThread[i]);
    }

    return HCCL_SUCCESS;
}

} // namespace ops_hccl

#endif // HCCLV2_INS_V2_ALL_GATHER_SEQUENCE_3LEVEL_UTILS_H
