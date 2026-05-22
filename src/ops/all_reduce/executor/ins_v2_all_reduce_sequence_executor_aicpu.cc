/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_reduce_sequence_executor_aicpu.h"
#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"

namespace ops_hccl {

constexpr u32 SEQUENCE_EXECUTOR_LEVEL_NUM = 2;

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
InsV2AllReduceSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::InsV2AllReduceSequenceExecutorAicpu()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult InsV2AllReduceSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::InitCommInfo(const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO("[InsV2AllReduceSequenceExecutorAicpu][InitCommInfo] myRank [%u], rankSize [%u], redOp [%u], "
        "dataType [%u] dataTypeSize [%u]", myRank_, rankSize_, devType_, reduceOp_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult InsV2AllReduceSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult InsV2AllReduceSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    // 初始化一些基本成员变量
    InitCommInfo(param, topoInfo, algHierarchyInfo);
    if (algHierarchyInfo.infos.size() != SEQUENCE_EXECUTOR_LEVEL_NUM) {
        HCCL_ERROR("[InsV2AllReduceSequenceExecutorAicpu] algHierarchyInfo size should be %u", SEQUENCE_EXECUTOR_LEVEL_NUM);
        return HCCL_E_INTERNAL;
    }
    rankSizeLevel0_ = algHierarchyInfo.infos[0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[1].size();

    std::shared_ptr<InsAlgTemplate0> reduceScatterMesh1DTempAlg =
        std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> reduceScatterNhrTempAlg =
        std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate2> allGatherNhrTempAlg =
        std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate3> allGatherMesh1DTempAlg =
        std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo.infos[0]);

    AlgResourceRequest resReqReduceScatterMesh1D;
    AlgResourceRequest resReqReduceScatterNhr;
    AlgResourceRequest resReqAllGatherNhr;
    AlgResourceRequest resReqAllGatherMesh1D;
    CHK_RET(reduceScatterMesh1DTempAlg->CalcRes(comm, param, topoInfo, resReqReduceScatterMesh1D));
    CHK_RET(reduceScatterNhrTempAlg->CalcRes(comm, param, topoInfo, resReqReduceScatterNhr));
    CHK_RET(allGatherNhrTempAlg->CalcRes(comm, param, topoInfo, resReqAllGatherNhr));
    CHK_RET(allGatherMesh1DTempAlg->CalcRes(comm, param, topoInfo, resReqAllGatherMesh1D));

    // step1、2、3、4为串行，因此slaveThread和对应notify可以复用
    resourceRequest.slaveThreadNum = std::max({resReqReduceScatterMesh1D.slaveThreadNum,
        resReqReduceScatterNhr.slaveThreadNum, resReqAllGatherNhr.slaveThreadNum, resReqAllGatherMesh1D.slaveThreadNum});
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        if (i < resReqReduceScatterMesh1D.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReqReduceScatterMesh1D.notifyNumPerThread[i]);
        }
        if (i < resReqReduceScatterNhr.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReqReduceScatterNhr.notifyNumPerThread[i]);
        }
        if (i < resReqAllGatherNhr.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReqAllGatherNhr.notifyNumPerThread[i]);
        }
        if (i < resReqAllGatherMesh1D.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReqAllGatherMesh1D.notifyNumPerThread[i]);
        }
    }
    resourceRequest.notifyNumOnMainThread = std::max({resReqReduceScatterMesh1D.notifyNumOnMainThread, resReqReduceScatterNhr.notifyNumOnMainThread,
        resReqAllGatherNhr.notifyNumOnMainThread, resReqAllGatherMesh1D.notifyNumOnMainThread});

    resourceRequest.channels = {resReqReduceScatterMesh1D.channels[0],resReqReduceScatterNhr.channels[0]};
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult InsV2AllReduceSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::Orchestrate(const OpParam &param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceSequenceExecutorAicpu][Orchestrate] Orchestrate Start");
    // 参数填充
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    dataCount_ = param.DataDes.count;
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];
    reduceOp_ = param.reduceType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = myRank_ / algHierarchyInfo_.infos[0][0].size();

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllReduceSequenceExecutorAicpu][Orchestrate]errNo[0x%016llx] AllReduce executor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
void InsV2AllReduceSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::GenBaseTempAlgParams(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    TemplateDataParams &tempAlgParamsStepOne, TemplateDataParams &tempAlgParamsStepTwo,
    TemplateDataParams &tempAlgParamsStepThree, TemplateDataParams &tempAlgParamsStepFour) const
{
    tempAlgParamsStepOne.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsStepOne.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepOne.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepOne.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsStepOne.buffInfo.outputPtr = resCtx.cclMem.addr; 
    tempAlgParamsStepOne.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParamsStepTwo.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepTwo.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepTwo.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepTwo.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsStepTwo.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsStepTwo.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParamsStepThree.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepThree.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepThree.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepThree.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsStepThree.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsStepThree.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParamsStepFour.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepFour.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsStepFour.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepFour.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsStepFour.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsStepFour.buffInfo.hcclBuff = resCtx.cclMem;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
void InsV2AllReduceSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::GenTempAlgParamsStepOne(const u64 loop, const u64 currDataCount, const u64 processedDataCount,
    TemplateDataParams &tempAlgParamsStepOne) const
{
    tempAlgParamsStepOne.count = currDataCount; // 没用到
    tempAlgParamsStepOne.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsStepOne.buffInfo.outBuffBaseOff = outCclBuffOffset_;
    tempAlgParamsStepOne.buffInfo.hcclBuffBaseOff = inCclBuffOffset_;

    tempAlgParamsStepOne.sliceSize = currDataCount / rankSizeLevel0_ * dataTypeSize_;
    tempAlgParamsStepOne.tailSize = (currDataCount / rankSizeLevel0_ + currDataCount % rankSizeLevel0_) * dataTypeSize_; // 最后一个rank的数据量

    tempAlgParamsStepOne.inputSliceStride = tempAlgParamsStepOne.sliceSize;
    tempAlgParamsStepOne.outputSliceStride = 0; // 归约时固定归约到offset0位置

    HCCL_INFO("[InsV2AllReduceSequenceExecutorAicpu] loop [%u] tempAlgParamsStepOne.inputSliceStride [%u], "
        "tempAlgParamsStepOne.outputSliceStride [%u], tempAlgParamsStepOne.sliceSize [%u], tempAlgParamsStepOne.tailSize [%u], "
        "tempAlgParamsStepOne.buffInfo.inBuffBaseOff [%u], tempAlgParamsStepOne.buffInfo.outBuffBaseOff [%u]",
        loop, tempAlgParamsStepOne.inputSliceStride, tempAlgParamsStepOne.outputSliceStride, tempAlgParamsStepOne.sliceSize,
        tempAlgParamsStepOne.tailSize, tempAlgParamsStepOne.buffInfo.inBuffBaseOff, tempAlgParamsStepOne.buffInfo.outBuffBaseOff);
    // 不需要重复
    tempAlgParamsStepOne.repeatNum = 1;
    tempAlgParamsStepOne.inputRepeatStride = 0;
    tempAlgParamsStepOne.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
void InsV2AllReduceSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::GenTempAlgParamsStepTwo(const u64 loop, const u64 currDataCount, const u64 sliceSizeLastStep,
    const u64 tailSizeLastStep, TemplateDataParams &tempAlgParamsStepTwo) const
{
    tempAlgParamsStepTwo.count = currDataCount; // 没用到
    if (rankIdxLevel0_ == rankSizeLevel0_ - 1) {
        // 如果在step1中是尾块，则需要用step1的tailcount为基础计算step2的数据量
        u64 tailCountLastStep = tailSizeLastStep / dataTypeSize_;
        tempAlgParamsStepTwo.sliceSize = tailCountLastStep / rankSizeLevel1_ * dataTypeSize_;
        tempAlgParamsStepTwo.tailSize = tempAlgParamsStepTwo.sliceSize + tailCountLastStep % rankSizeLevel1_ * dataTypeSize_;
    } else {
        u64 sliceCountLastStep = sliceSizeLastStep / dataTypeSize_;
        tempAlgParamsStepTwo.sliceSize = sliceCountLastStep / rankSizeLevel1_ * dataTypeSize_;
        tempAlgParamsStepTwo.tailSize = tempAlgParamsStepTwo.sliceSize + sliceCountLastStep % rankSizeLevel1_ * dataTypeSize_;
    }
    // 上一步会归约到offset0位置，所以这一步offset为0
    tempAlgParamsStepTwo.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsStepTwo.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsStepTwo.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsStepTwo.inputSliceStride = tempAlgParamsStepTwo.sliceSize;
    tempAlgParamsStepTwo.outputSliceStride = tempAlgParamsStepTwo.sliceSize;

    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutorAicpu] loop [%u] tempAlgParamsStepTwo.inputSliceStride [%u], "
        "tempAlgParamsStepTwo.outputSliceStride [%u], tempAlgParamsStepTwo.sliceSize [%u], tempAlgParamsStepTwo.tailSize [%u], "
        "tempAlgParamsStepTwo.buffInfo.inBuffBaseOff [%u], tempAlgParamsStepTwo.buffInfo.outBuffBaseOff [%u]",
        loop, tempAlgParamsStepTwo.inputSliceStride, tempAlgParamsStepTwo.outputSliceStride,
        tempAlgParamsStepTwo.sliceSize, tempAlgParamsStepTwo.tailSize, tempAlgParamsStepTwo.buffInfo.inBuffBaseOff,
        tempAlgParamsStepTwo.buffInfo.outBuffBaseOff);
    // 不需要重复
    tempAlgParamsStepTwo.repeatNum = 1;
    tempAlgParamsStepTwo.inputRepeatStride = 0;
    tempAlgParamsStepTwo.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
void InsV2AllReduceSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::GenTempAlgParamsStepThree(const u64 loop, const u64 currDataCount, const u64 sliceSize,
    const u64 tailSize, TemplateDataParams &tempAlgParamsStepThree) const
{
    tempAlgParamsStepThree.count = currDataCount; // 没用到
    tempAlgParamsStepThree.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsStepThree.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsStepThree.buffInfo.hcclBuffBaseOff = 0;
    // 与上一步框间ReduceScatter数据量一致
    tempAlgParamsStepThree.sliceSize = sliceSize;
    tempAlgParamsStepThree.tailSize = tailSize;

    tempAlgParamsStepThree.inputSliceStride = tempAlgParamsStepThree.sliceSize;
    tempAlgParamsStepThree.outputSliceStride = tempAlgParamsStepThree.sliceSize;

    HCCL_INFO("[InsV2AllReduceSequenceExecutorAicpu] loop [%u] tempAlgParamsStepThree.inputSliceStride [%u],"
        "tempAlgParamsStepThree.outputSliceStride [%u] tempAlgParamsStepThree.sliceSize [%u], tempAlgParamsStepThree.tailSize [%u], "
        "tempAlgParamsStepThree.buffInfo.inBuffBaseOff [%u], tempAlgParamsStepThree.buffInfo.outBuffBaseOff [%u]",
        loop, tempAlgParamsStepThree.inputSliceStride, tempAlgParamsStepThree.outputSliceStride,
        tempAlgParamsStepThree.sliceSize, tempAlgParamsStepThree.tailSize, tempAlgParamsStepThree.buffInfo.inBuffBaseOff,
        tempAlgParamsStepThree.buffInfo.outBuffBaseOff);

    tempAlgParamsStepThree.repeatNum = 1;
    tempAlgParamsStepThree.inputRepeatStride = 0;
    tempAlgParamsStepThree.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
void InsV2AllReduceSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::GenTempAlgParamsStepFour(const u64 loop, const u64 currDataCount, const u64 processedDataCount,
    const u64 sliceSize, const u64 tailSize, TemplateDataParams &tempAlgParamsStepFour) const
{
    tempAlgParamsStepFour.count = currDataCount; // 没用到
    tempAlgParamsStepFour.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsStepFour.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsStepFour.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsStepFour.sliceSize = sliceSize;
    tempAlgParamsStepFour.tailSize = tailSize;

    tempAlgParamsStepFour.inputSliceStride = 0;
    tempAlgParamsStepFour.outputSliceStride = tempAlgParamsStepFour.sliceSize;
    
    HCCL_INFO("[InsV2AllReduceSequenceExecutorAicpu] loop [%u] tempAlgParamsStepFour.inputSliceStride [%u], "
        "tempAlgParamsStepFour.outputSliceStride [%u], tempAlgParamsStepFour.sliceSize [%u], tempAlgParamsStepFour.tailSize [%u], "
        "tempAlgParamsStepFour.buffInfo.inBuffBaseOff [%u], tempAlgParamsStepFour.buffInfo.outBuffBaseOff [%u]",
        loop, tempAlgParamsStepFour.inputSliceStride, tempAlgParamsStepFour.outputSliceStride, tempAlgParamsStepFour.sliceSize,
        tempAlgParamsStepFour.tailSize, tempAlgParamsStepFour.buffInfo.inBuffBaseOff, tempAlgParamsStepFour.buffInfo.outBuffBaseOff);

    tempAlgParamsStepFour.repeatNum = 1;
    tempAlgParamsStepFour.inputRepeatStride = 0;
    tempAlgParamsStepFour.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
template <typename InsAlgTemplate>
HcclResult InsV2AllReduceSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::GenTempResource(const AlgResourceCtxSerializable &resCtx, const u32 channelLevelIdx,
    const std::shared_ptr<InsAlgTemplate> &algTemplate, TemplateResource &tempReousrce) const
{
    AlgResourceRequest req;
    algTemplate->GetRes(req);
    if (channelLevelIdx >= remoteRankToChannelInfo_.size()) {
        HCCL_ERROR("[InsV2AllReduceSequenceExecutorAicpu][GenTempResource] channelLevelIdx[%u] should be lower"
            "than remoteRankToChannelInfo_.size()[%u]", channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    tempReousrce.channels = remoteRankToChannelInfo_[channelLevelIdx];
    tempReousrce.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult InsV2AllReduceSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllReduceSequenceExecutorAicpu][OrchestrateLoop] Start");

    TemplateDataParams tempAlgParamsStepOne; // 框内ReduceScatterMesh1D的模板参数
    TemplateDataParams tempAlgParamsStepTwo; // 框间ReduceScatterNhr的模板参数
    TemplateDataParams tempAlgParamsStepThree; // 框间AllGatherNhr的模板参数
    TemplateDataParams tempAlgParamsStepFour; // 框内AllGatherMesh1D的模板参数
    // 填充buff类型和buff指针参数
    GenBaseTempAlgParams(param, resCtx, tempAlgParamsStepOne, tempAlgParamsStepTwo, tempAlgParamsStepThree, tempAlgParamsStepFour);

    // 构建框内ReduceScatterMesh1D的template
    std::shared_ptr<InsAlgTemplate0> algTemplateStepOne = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);
    algTemplateStepOne->SetchannelsPerRank(remoteRankToChannelInfo_[0]);

    // 构建框间ReduceScatterNhr的template
    std::shared_ptr<InsAlgTemplate1> algTemplateStepTwo = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);
    algTemplateStepTwo->SetchannelsPerRank(remoteRankToChannelInfo_[1]);

    // 构建框间AllGatherNhr的template
    std::shared_ptr<InsAlgTemplate2> algTemplateStepThree = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[1]);
    algTemplateStepThree->SetchannelsPerRank(remoteRankToChannelInfo_[1]);

    // 构建框内AllGatherMesh1D的template
    std::shared_ptr<InsAlgTemplate3> algTemplateStepFour = std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo_.infos[0]);
    algTemplateStepFour->SetchannelsPerRank(remoteRankToChannelInfo_[0]);

    // 构造框内ReduceScatterMesh1D的template资源
    TemplateResource templateResourceStepOne;
    CHK_RET(GenTempResource(resCtx, 0, algTemplateStepOne, templateResourceStepOne));
    // 构造框间ReduceScatterNhr的template资源
    TemplateResource templateResourceStepTwo;
    CHK_RET(GenTempResource(resCtx, 1, algTemplateStepTwo, templateResourceStepTwo));
    // 构造框间AllGatherNhr的template资源
    TemplateResource templateResourceStepThree;
    CHK_RET(GenTempResource(resCtx, 1, algTemplateStepThree, templateResourceStepThree));
    // 构造框内AllGatherMesh1D的template资源
    TemplateResource templateResourceStepFour;
    CHK_RET(GenTempResource(resCtx, 0, algTemplateStepFour, templateResourceStepFour));
    
    // 计算中转内存单次最多能够接受的output count
    // CCL buffer切分为2块，前1块作为ReduceScatter mesh1D归约操作的output，后1块作为ccl buffer接收其他卡的数据
    outCclBuffSize_ = tempAlgParamsStepOne.buffInfo.hcclBuff.size / 2;
    inCclBuffSize_ = tempAlgParamsStepOne.buffInfo.hcclBuff.size - outCclBuffSize_;
    outCclBuffOffset_ = 0;
    inCclBuffOffset_ = outCclBuffSize_;
    // 最大搬运数据量向下对齐到rankSize的倍数，方便数据切分，只用最后一个loop处理尾块
    u32 totalRankAlign = rankSizeLevel0_ * rankSizeLevel1_;
    u64 maxCountPerLoop = inCclBuffOffset_ / HCCL_MIN_SLICE_ALIGN *
                          HCCL_MIN_SLICE_ALIGN / dataTypeSize_ / totalRankAlign * totalRankAlign;
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop; // 判断是最后一轮，就处理尾块长度
        // ----------- Step1:框内ReduceScatter数据搬运 -----------
        // 框内的数据偏移和搬运计算
        GenTempAlgParamsStepOne(loop, currDataCount, processedDataCount, tempAlgParamsStepOne);
        CHK_RET(algTemplateStepOne->KernelRun(param, tempAlgParamsStepOne, templateResourceStepOne));

        // ----------- Step2:框间ReduceScatter数据搬运 -----------
        // 框间的数据偏移和搬运量计算
        GenTempAlgParamsStepTwo(loop, currDataCount, tempAlgParamsStepOne.sliceSize,
            tempAlgParamsStepOne.tailSize, tempAlgParamsStepTwo);
        CHK_RET(algTemplateStepTwo->KernelRun(param, tempAlgParamsStepTwo, templateResourceStepTwo));

        // ----------- Step3:框间AllGather数据搬运 -----------
        // 框间的数据偏移和搬运量计算
        GenTempAlgParamsStepThree(loop, currDataCount, tempAlgParamsStepTwo.sliceSize,
            tempAlgParamsStepTwo.tailSize, tempAlgParamsStepThree);
        CHK_RET(algTemplateStepThree->KernelRun(param, tempAlgParamsStepThree, templateResourceStepThree));

        // ----------- Step4:框内AllGather数据搬运 -----------
        // 框内的数据偏移和搬运计算
        GenTempAlgParamsStepFour(loop, currDataCount, processedDataCount, tempAlgParamsStepOne.sliceSize,
            tempAlgParamsStepOne.tailSize, tempAlgParamsStepFour);
        CHK_RET(algTemplateStepFour->KernelRun(param, tempAlgParamsStepFour, templateResourceStepFour));

        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2AllReduceSequenceExecutorAicpu][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE,
                                InsAllReduceSequenceMesh1DNhr,
                                InsV2AllReduceSequenceExecutorAicpu,
                                TopoMatchMultilevel,
                                InsTempReduceScatterMesh1DZAxisDetour,
                                InsTempReduceScatterNHR,
                                InsTempAllGatherNHR,
                                InsTempAllGatherMesh1D1DZAxisDetour);
}