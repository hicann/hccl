/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_EXECUTOR_H
#define OPS_EXECUTOR_H

#include <map>
#include <vector>
#include <algorithm>
#include <climits>
#include <numeric>
#include <hccl/hccl_res.h>
#include "algo_desc.h"
#include "template/base_template.h"
#include "template/template_factory.h"
#include "template/data_ops.h"

namespace ops_hccl {

struct BufferInfo {
    void* ptr = nullptr;
    u64 size = 0;
    BufferType bufferType = BufferType::HCCL_BUFFER;
};

struct ExecDataInfo {
    void* inputPtr = nullptr;
    u64 inputSize = 0;
    void* outputPtr = nullptr;
    u64 outputSize = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HCCL_REDUCE_RESERVED;
};

struct AlgoExecDataDesc {
    u64 dataOffset{0};
    u64 dataStride{0};
    u64 sliceCount{0};
    u64 sliceOffset{0};
    u64 scratchStride{0};
    u64 tailCount{0};
    u32 globalTailRankId{INVALID_VALUE_RANKID};
    std::vector<std::vector<u32>> ranksForInputDataGroup;
    std::vector<std::vector<u32>> ranksForOutputDataGroup;
    BufferType inputBufferType{BufferType::INPUT};
    BufferType outputBufferType{BufferType::OUTPUT};
    BufferType cclBufferType{BufferType::HCCL_BUFFER};
};
struct OmniPipeXYdata {
    u32 steps;
    double scale;
    double bandwidthRatio;
    u32 xEqRankSize;
    u32 yEqRankSize;
};
class OpsExecutor {
public:
    OpsExecutor(HcclAlgorithm& algo, const OpParam& param);
    ~OpsExecutor();
    HcclResult InitAlgHierarchyInfo(
        const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    HcclResult CalcRes(HcclComm comm, AlgResourceRequest& resReq);
    HcclResult Orchestrate(const AlgResourceCtxSerializable& resCtx);

private:
    HcclResult GetRes(AlgResourceRequest& resReq);
    HcclResult CalcChannelResRecursion(
        HcclComm comm, AlgoExecDesc& algoExecDesc, std::vector<std::vector<HcclChannelDesc>>& requestChannels);
    HcclResult GetResRecursion(AlgoExecDesc& algoExecDesc, u32& subCommMask);
    HcclResult CalcTemplateChannelRes(
        HcclComm comm, const TemplateExecDesc& templateExeDes,
        std::vector<std::vector<HcclChannelDesc>>& requestChannels);
    HcclResult GetTemplateRes(const TemplateExecDesc& templateExeDes);
    HcclResult OrchestrateLoop(
        AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc,
        std::vector<AlgoExecDataDesc>* reusableChildren = nullptr);
    void InitChildrenDataDesc(
        AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc,
        std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc);
    HcclResult ExecChildren(
        AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc,
        std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc);
    HcclResult OrchestrateOmniPipeLoop(AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc);
    HcclResult GenTemplateDataParams(
        AlgoExecDataDesc& algoExecDataDesc, DataParams& templateDataParams, u32 overrideRoot = INVALID_VALUE_RANKID);
    void UpdateSubCommMaskMap(AlgoExecDesc& algoExecDesc, const u32 subCommMask);
    HcclResult PreSyncBySubCommMask(const AlgoExecDesc& execDesc);
    HcclResult PostSyncBySubCommMask(const AlgoExecDesc& execDesc);
    void InitAlgoExecDataDesc(
        AlgoExecDataDesc& algoExecDataDesc, u64 dataOffset, u64 dataCount, u64 tailCount, u64 dataStride);
    HcclResult UpdateDataSplitParallel(
        AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc, u32 childrenId,
        std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc);
    void UpdateDataSplitSequence(
        AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc, u32 childrenId,
        std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc);
    HcclResult PreSyncSingleSubComm(u32 subCommIndex);
    HcclResult PostSyncSingleSubComm(u32 subCommIndex);
    HcclResult MergeChildrenOutput(
        const AlgoExecDesc& algoExecDesc, const std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc,
        AlgoExecDataDesc& algoExecDataDesc);
    HcclResult RunTemplateDesc(
        TemplateExecDesc* templateExeDes, AlgoExecDataDesc& algoExecDataDesc, u32 overrideRoot = INVALID_VALUE_RANKID);
    HcclResult InitSubCommRoots();
    HcclResult InitRes(const AlgResourceCtxSerializable& resCtx);
    HcclResult PrepareOrchestrate(
        u64& dataCount, u64& maxProcCntPerLoop, u64& loopTimes, u64& dataStride, u64& lastProcessCount,
        u64& lastTailCount);
    std::vector<std::map<u32, std::vector<ChannelInfo>>> RestoreChannelMap(const AlgResourceCtxSerializable& resCtx);
    u64 GetMaxProcCntPerLoop(u64 dataCount);
    HcclResult OmniPipeUpdateEqBWAndReorder(VariantType& algoExecDesc, u32& eqRankSize, double& eqBw);
    HcclResult OmniPipeReorderChildren(VariantType& algoExecDesc, u32& eqRankSize, double& eqBw);
    HcclResult OmniPipeCalcExecData(
        AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc, OmniPipeXYdata& omniPipeXYdata,
        std::vector<std::vector<AlgoExecDataDesc>>& childrenExecDataDesc);
    HcclResult OmniPipeUpdateDataSlice(
        OmniPipeXYdata& omniPipeXYdata, const AlgoExecDataDesc& algoExecDataDesc,
        std::vector<AlgoExecDataDesc>& xExecDataDesc, std::vector<AlgoExecDataDesc>& yExecDataDesc);
    HcclResult CalcPeerAxisRanksForOutput(
        const AlgoExecDesc& algoExecDesc, u32 peerChildrenId, std::vector<u32> ranksForInput,
        std::vector<u32>& detaRanksForOutput);
    HcclAlgorithm algo_;
    u32 myRank_ = INVALID_VALUE_RANKID;
    u32 rankSize_ = 0;
    u32 root_ = INVALID_VALUE_RANKID;
    std::vector<u32> subCommRoots_;
    ExecDataInfo dataInfo_;
    u64 dataTypeSize_ = 0;
    u32 scratchMultiple_ = 0;
    OpMode opMode_;
    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    BufferInfo cclBufferInfo_;
    ThreadHandle mainThread_ = 0;
    std::vector<ThreadHandle> threads_;
    std::vector<std::vector<ThreadHandle>> subThreads_;
    std::vector<u32> notifyNumOnSubMainThread_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> channelTable_;
    std::vector<u32> maxSlaveThreadNum_;
    std::vector<u32> maxNotifyNumOnMainThread_;
    std::vector<u32> maxNotifyNumPerThread_;
    std::map<const AlgoExecDesc*, u32> execDescSubCommMaskMap_;
    std::map<const AlgoExecDesc*, OmniPipeXYdata> omniPipeXYdataMap_;
};
} // namespace ops_hccl
#endif // OPS_EXECUTOR_H
