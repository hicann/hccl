/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_executor.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {
OpsExecutor::OpsExecutor(HcclAlgorithm& algo, const OpParam& param) : algo_(algo), rankSize_(0), root_(param.root)
{
    opMode_ = param.opMode;
    dataInfo_.inputPtr = param.inputPtr;
    dataInfo_.inputSize = param.inputSize;
    dataInfo_.outputPtr = param.outputPtr;
    dataInfo_.outputSize = param.outputSize;
    dataInfo_.reduceOp = param.reduceType;
    dataInfo_.dataType = param.DataDes.dataType;
    if (param.DataDes.dataType >= HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR(
            "[OpsExecutor] dataType=%d out of range [0, %d)", static_cast<int>(param.DataDes.dataType),
            static_cast<int>(HCCL_DATA_TYPE_RESERVED));
        return;
    }
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
}

OpsExecutor::~OpsExecutor() {}

HcclResult OpsExecutor::InitAlgHierarchyInfo(
    const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::Orchestrate(const AlgResourceCtxSerializable& resCtx) { return HCCL_SUCCESS; }

u64 OpsExecutor::GetMaxProcCntPerLoop(u64 dataCount) { return 0; }

HcclResult OpsExecutor::PrepareOrchestrate(
    u64& dataCount, u64& maxProcCntPerLoop, u64& loopTimes, u64& dataStride, u64& lastProcessCount, u64& lastTailCount)
{
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::InitRes(const AlgResourceCtxSerializable& resCtx) { return HCCL_SUCCESS; }

std::vector<std::map<u32, std::vector<ChannelInfo>>>
OpsExecutor::RestoreChannelMap(const AlgResourceCtxSerializable& resCtx)
{
    return {};
}

HcclResult OpsExecutor::PreSyncBySubCommMask(const AlgoExecDesc& execDesc) { return HCCL_SUCCESS; }

HcclResult OpsExecutor::PostSyncBySubCommMask(const AlgoExecDesc& execDesc) { return HCCL_SUCCESS; }

HcclResult OpsExecutor::CalcChannelResRecursion(
    HcclComm comm, AlgoExecDesc& algoExecDesc, std::vector<std::vector<HcclChannelDesc>>& requestChannels)
{
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::GetResRecursion(AlgoExecDesc& algoExecDesc, u32& subCommMask) { return HCCL_SUCCESS; }

HcclResult OpsExecutor::CalcTemplateChannelRes(
    HcclComm comm, const TemplateExecDesc& templateExeDes, std::vector<std::vector<HcclChannelDesc>>& requestChannels)
{
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::GetTemplateRes(const TemplateExecDesc& templateExeDes) { return HCCL_SUCCESS; }

inline void OpsExecutor::UpdateSubCommMaskMap(AlgoExecDesc& algoExecDesc, const u32 subCommMask) {}

HcclResult OpsExecutor::CalcRes(HcclComm comm, AlgResourceRequest& resourceRequest) { return HCCL_SUCCESS; }

HcclResult OpsExecutor::GetRes(AlgResourceRequest& resourceRequest) { return HCCL_SUCCESS; }

inline void OpsExecutor::InitAlgoExecDataDesc(
    AlgoExecDataDesc& algoExecDataDesc, u64 dataOffset, u64 dataCount, u64 tailCount, u64 dataStride)
{}

inline HcclResult
OpsExecutor::GenTemplateDataParams(AlgoExecDataDesc& algoExecDataDesc, DataParams& templateDataParams, u32 overrideRoot)
{
    return HCCL_SUCCESS;
}

inline HcclResult OpsExecutor::UpdateDataSplitParallel(
    AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc, u32 childrenId,
    std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc)
{
    return HCCL_SUCCESS;
}

inline void OpsExecutor::UpdateDataSplitSequence(
    AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc, u32 childrenId,
    std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc)
{}

HcclResult OpsExecutor::MergeChildrenOutput(
    const AlgoExecDesc& algoExecDesc, const std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc,
    AlgoExecDataDesc& algoExecDataDesc)
{
    return HCCL_SUCCESS;
}

HcclResult
OpsExecutor::RunTemplateDesc(TemplateExecDesc* templateExeDes, AlgoExecDataDesc& algoExecDataDesc, u32 overrideRoot)
{
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::InitSubCommRoots() { return HCCL_SUCCESS; }

HcclResult OpsExecutor::OrchestrateLoop(
    AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc, std::vector<AlgoExecDataDesc>* reusableChildren)
{
    return HCCL_SUCCESS;
}

void OpsExecutor::InitChildrenDataDesc(
    AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc,
    std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc)
{}

HcclResult OpsExecutor::ExecChildren(
    AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc,
    std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc)
{
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::OrchestrateOmniPipeLoop(AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc)
{
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::OmniPipeCalcExecData(
    AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc, OmniPipeXYdata& omniPipeXYdata,
    std::vector<std::vector<AlgoExecDataDesc>>& childrenExecDataDesc)
{
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::OmniPipeUpdateDataSlice(
    OmniPipeXYdata& omniPipeXYdata, const AlgoExecDataDesc& algoExecDataDesc,
    std::vector<AlgoExecDataDesc>& xExecDataDesc, std::vector<AlgoExecDataDesc>& yExecDataDesc)
{
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::CalcPeerAxisRanksForOutput(
    const AlgoExecDesc& algoExecDesc, u32 peerChildrenId, std::vector<u32> ranksForInput,
    std::vector<u32>& detaRanksForOutput)
{
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::OmniPipeUpdateEqBWAndReorder(VariantType& algoExecDesc, u32& eqRankSize, double& eqBw)
{
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::OmniPipeReorderChildren(VariantType& algoExecDesc, u32& eqRankSize, double& eqBw)
{
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::PreSyncSingleSubComm(u32 subCommIndex) { return HCCL_SUCCESS; }

HcclResult OpsExecutor::PostSyncSingleSubComm(u32 subCommIndex) { return HCCL_SUCCESS; }

} // namespace ops_hccl
