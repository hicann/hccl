/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_broadcast_sequence_executor.h"
#include "ins_temp_scatter_mesh_1D_intra.h"
#include "ins_temp_scatter_nhr_dpu_inter.h"
#include "ins_temp_allgather_nhr_dpu_inter.h"
#include "ins_temp_allgather_mesh_1D_intra.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
InsV2BroadcastSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::InsV2BroadcastSequenceExecutor() {}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult InsV2BroadcastSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::InitCommInfo(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO("[InsV2BroadcastSequenceExecutor][InitCommInfo] myRank [%u], rankSize [%u], dataTypeSize [%u]",
        myRank_,
        rankSize_,
        dataTypeSize_);
    return HCCL_SUCCESS;
}

// 实例化实际执行以来AutoMatchMeshNhr这个类的实现
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult InsV2BroadcastSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult InsV2BroadcastSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    rankSizeLevel0_ = algHierarchyInfo.infos[0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[1].size();
    HCCL_INFO("[InsV2BroadcastSequenceExecutor][CalcRes] rankSizeLevel0 [%u], rankSizeLevel1 [%u]", rankSizeLevel0_,
        rankSizeLevel1_);

    std::shared_ptr<InsAlgTemplate0> intraScatterTempAlg =
        std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> interScatterTempAlg =
        std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate2> interAllGatherTempAlg =
        std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate3> intraAllGatherTempAlg =
        std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo.infos[0]);

    AlgResourceRequest resReqScatterIntra;
    AlgResourceRequest resReqAllGatherIntra;
    AlgResourceRequest resReqScatterInter;
    AlgResourceRequest resReqAllGatherInter;

    CHK_RET(intraScatterTempAlg->CalcRes(comm, param, topoInfo, resReqScatterIntra));
    CHK_RET(intraAllGatherTempAlg->CalcRes(comm, param, topoInfo, resReqAllGatherIntra));
    CHK_RET(interScatterTempAlg->CalcRes(comm, param, topoInfo, resReqScatterInter));
    CHK_RET(interAllGatherTempAlg->CalcRes(comm, param, topoInfo, resReqAllGatherInter));

    // step1在完成后，完成后同步后展开step2，因此slaveThread和对应notify可以复用
    resourceRequest.slaveThreadNum = std::max({resReqScatterIntra.slaveThreadNum,
        resReqAllGatherIntra.slaveThreadNum,
        resReqScatterInter.slaveThreadNum,
        resReqAllGatherInter.slaveThreadNum});
    resourceRequest.notifyNumPerThread = std::max({resReqScatterIntra.notifyNumPerThread,
        resReqAllGatherIntra.notifyNumPerThread,
        resReqScatterInter.notifyNumPerThread,
        resReqAllGatherInter.notifyNumPerThread});
    resourceRequest.notifyNumOnMainThread = std::max({resReqScatterIntra.notifyNumOnMainThread,
        resReqAllGatherIntra.notifyNumOnMainThread,
        resReqScatterInter.notifyNumOnMainThread,
        resReqAllGatherInter.notifyNumOnMainThread});

    u64 channelsSize = 2;
    resourceRequest.channels.resize(channelsSize);
    resourceRequest.channels[0] = resReqScatterIntra.channels[0];
    resourceRequest.channels[1] = resReqScatterInter.channels[0];

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult InsV2BroadcastSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2BroadcastSequenceExecutor][Orchestrate] Orchestrate Start");
    // 参数填充
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    CHK_RET(InitExecutorInfo(param, resCtx));
    threads_ = resCtx.threads;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2BroadcastSequenceExecutor][Orchestrate]errNo[0x%016llx] Broadcast excutor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult InsV2BroadcastSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::InitExecutorInfo(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = myRank_ / algHierarchyInfo_.infos[0][0].size();

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    root_ = param.root;
    // 计算框内的root同号卡
    intraLocalRoot_ = root_ % rankSizeLevel0_ + rankIdxLevel1_ * rankIdxLevel0_;
    
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    HCCL_INFO("[InsV2BroadcastSequenceExecutor][InitExecutorInfo] myRank [%u], rankSize [%u], dataTypeSize [%u]",
        +myRank_,
        rankSize_,
        dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult InsV2BroadcastSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2BroadcastSequenceExecutor][OrchestrateLoop] Start");

    // 不区分CCL-IN 与 CCL-OUT
    // 声明框内Scatter templateargs
    TemplateDataParams tempAlgParamsScatterIntra;
    tempAlgParamsScatterIntra.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsScatterIntra.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterIntra.buffInfo.hcclBuff = resCtx.cclMem;

    // 构建框内Scatter template
    std::shared_ptr<InsAlgTemplate0> algTemplateScatterIntra =
        std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);

    // 声明框间Scatter templateargs
    TemplateDataParams tempAlgParamsScatterInter;
    tempAlgParamsScatterInter.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterInter.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterInter.buffInfo.hcclBuff = resCtx.cclMem;

    // 构建框间Scatter template
    std::shared_ptr<InsAlgTemplate1> algTemplateScatterInter =
        std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);

    // 声明框间AllGather templateargs
    TemplateDataParams tempAlgParamsAllGatherInter;
    tempAlgParamsAllGatherInter.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherInter.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherInter.buffInfo.hcclBuff = resCtx.cclMem;

    // 构建框间AllGather template
    std::shared_ptr<InsAlgTemplate2> algTemplateAllGatherInter =
        std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[1]);

    // 声明框内AllGather templateargs
    TemplateDataParams tempAlgParamsAllGatherIntra;
    tempAlgParamsAllGatherIntra.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherIntra.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsAllGatherIntra.buffInfo.hcclBuff = resCtx.cclMem;

    // 构建框内AllGather template
    std::shared_ptr<InsAlgTemplate3> algTemplateAllGatherIntra =
        std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo_.infos[0]);

    // 构造框内template资源
    TemplateResource templateResourceIntra;
    templateResourceIntra.channels = remoteRankToChannelInfo_[0];
    templateResourceIntra.threads = resCtx.threads;
    templateResourceIntra.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceIntra.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    // 构造框间template资源
    TemplateResource templateResourceInter;
    templateResourceInter.channels = remoteRankToChannelInfo_[1];
    templateResourceInter.threads = resCtx.threads;
    templateResourceInter.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceInter.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;

    // 中转内存单次最多能够接受的output count，注意是count不是size
    u64 dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    u64 dataCount_ = param.DataDes.count;
    u64 maxCountPerLoop = tempAlgParamsScatterIntra.buffInfo.hcclBuff.size / HCCL_MIN_SLICE_ALIGN *
                          HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;

    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        // ----------- 框内Scatter数据搬运 -----------
        // 框内Scatter的数据偏移和搬运量计算
        tempAlgParamsScatterIntra.count = currDataCount;
        tempAlgParamsScatterIntra.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamsScatterIntra.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsScatterIntra.buffInfo.hcclBuffBaseOff = 0;

        CHK_RET(SplitData(
            currDataCount, rankSizeLevel0_, tempAlgParamsScatterIntra));  // 计算每个卡对应位置的offset,count,size
        CHK_PRT_RET(tempAlgParamsScatterIntra.allRankSliceSize.size() != rankSizeLevel0_,
            HCCL_ERROR("[InsV2BroadcastSequenceExecutor][tempAlgParamsScatterIntra] slice num[%u] is not equal to rank "
                       "size[%u].",
                tempAlgParamsScatterIntra.allRankSliceSize.size(),
                rankSizeLevel0_),
            HcclResult::HCCL_E_INTERNAL);

        tempAlgParamsScatterIntra.sliceSize = 0;
        tempAlgParamsScatterIntra.tailSize = 0;
        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParamsScatterIntra.inputSliceStride = 0;
        tempAlgParamsScatterIntra.outputSliceStride = 0;

        // 不需要重复
        tempAlgParamsScatterIntra.repeatNum = 1;
        tempAlgParamsScatterIntra.inputRepeatStride = 0;
        tempAlgParamsScatterIntra.outputRepeatStride = 0;
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplateScatterIntra->KernelRun(param, tempAlgParamsScatterIntra, templateResourceIntra));

        // ----------- 框间Scatter数据搬运 -----------
        // 框间的数据偏移和搬运计算
        tempAlgParamsScatterInter.count = tempAlgParamsScatterIntra.allRankProcessedDataCount.at(rankIdxLevel0_);
        tempAlgParamsScatterInter.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsScatterInter.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsScatterInter.buffInfo.hcclBuffBaseOff = tempAlgParamsScatterIntra.allRankDispls.at(rankIdxLevel0_); // 将框内的切片偏移传到框间
        tempAlgParamsScatterInter.root = (param.root / rankSizeLevel0_) * rankSizeLevel0_ + (myRank_ % rankSizeLevel0_);

        tempAlgParamsScatterInter.sliceSize = 0;
        tempAlgParamsScatterInter.tailSize = 0;

        CHK_RET(SplitData(tempAlgParamsScatterInter.count, rankSizeLevel1_, tempAlgParamsScatterInter));
        CHK_PRT_RET(tempAlgParamsScatterInter.allRankSliceSize.size() != rankSizeLevel1_,
            HCCL_ERROR("[InsV2BroadcastSequenceExecutor][tempAlgParamsScatterInter] slice num[%u] is not equal to rank "
                       "size[%u].",
                tempAlgParamsScatterInter.allRankSliceSize.size(),
                rankSizeLevel1_),
            HcclResult::HCCL_E_INTERNAL);
        HCCL_INFO("[InsV2BroadcastSequenceExecutor][SplitData][ScatterInter] count[%u] slicenum[%u]",
            tempAlgParamsScatterInter.count, rankSizeLevel1_);

        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParamsScatterInter.inputSliceStride = 0;
        tempAlgParamsScatterInter.outputSliceStride = 0;

        HCCL_DEBUG("[InsV2BroadcastSequenceExecutor] loop [%u] tempAlgParamsScatterInter.inputSliceStride [%u],"
                  "tempAlgParamsScatterInter.outputSliceStride [%u] tempAlgParamsScatterInter.sliceSize [%u]",
            loop,
            tempAlgParamsScatterInter.inputSliceStride,
            tempAlgParamsScatterInter.outputSliceStride,
            tempAlgParamsScatterInter.sliceSize);
        HCCL_DEBUG("[InsV2BroadcastSequenceExecutor] loop [%u] tempAlgParamsScatterInter.buffInfo.inBuffBaseOff [%u],"
                  "tempAlgParamsScatterInter.buffInfo.outBuffBaseOff [%u]",
            loop,
            tempAlgParamsScatterInter.buffInfo.inBuffBaseOff,
            tempAlgParamsScatterInter.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsScatterInter.repeatNum = 1;
        tempAlgParamsScatterInter.inputRepeatStride = 0;
        tempAlgParamsScatterInter.outputRepeatStride = 0;
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        if (tempAlgParamsScatterInter.count != 0) {  // 如果卡里没有数据，不需要参与框间
            CHK_RET(algTemplateScatterInter->KernelRun(param, tempAlgParamsScatterInter, templateResourceInter));
        }

        // ----------- 框间AllGather数据搬运 -----------
        // 框间的数据偏移和搬运计算
        tempAlgParamsAllGatherInter.count = tempAlgParamsScatterIntra.allRankProcessedDataCount.at(rankIdxLevel0_);  // 沿用框间Scatter的切片结果
        tempAlgParamsAllGatherInter.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsAllGatherInter.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsAllGatherInter.buffInfo.hcclBuffBaseOff = tempAlgParamsScatterIntra.allRankDispls.at(rankIdxLevel0_); // 将框内的切片偏移传到框间
        tempAlgParamsScatterInter.root = (param.root / rankSizeLevel0_) * rankSizeLevel0_ + (myRank_ % rankSizeLevel0_);

        tempAlgParamsAllGatherInter.allRankDispls = tempAlgParamsScatterInter.allRankDispls;
        tempAlgParamsAllGatherInter.allRankSliceSize = tempAlgParamsScatterInter.allRankSliceSize;
        tempAlgParamsAllGatherInter.allRankProcessedDataCount = tempAlgParamsScatterInter.allRankProcessedDataCount;

        tempAlgParamsAllGatherInter.sliceSize = 0;
        tempAlgParamsAllGatherInter.tailSize = 0;
        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParamsAllGatherInter.inputSliceStride = 0;  // 如果是输入，偏移是算子的output datasize
        tempAlgParamsAllGatherInter.outputSliceStride = 0;  // 如果是scratchbuffer，偏移是单次循环处理的最大数据量

        HCCL_DEBUG("[InsV2BroadcastSequenceExecutor] loop [%u] tempAlgParamsAllGatherInter.inputSliceStride [%u],"
                  "tempAlgParamsAllGatherInter.outputSliceStride [%u] tempAlgParamsAllGatherInter.sliceSize [%u]",
            loop,
            tempAlgParamsAllGatherInter.inputSliceStride,
            tempAlgParamsAllGatherInter.outputSliceStride,
            tempAlgParamsAllGatherInter.sliceSize);
        HCCL_DEBUG("[InsV2BroadcastSequenceExecutor] loop [%u] tempAlgParamsAllGatherInter.buffInfo.inBuffBaseOff [%u],"
                  "tempAlgParamsAllGatherInter.buffInfo.outBuffBaseOff [%u]",
            loop,
            tempAlgParamsAllGatherInter.buffInfo.inBuffBaseOff,
            tempAlgParamsAllGatherInter.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsAllGatherInter.repeatNum = 1;
        tempAlgParamsAllGatherInter.inputRepeatStride = 0;
        tempAlgParamsAllGatherInter.outputRepeatStride = 0;
        HCCL_DEBUG(
            "[InsV2BroadcastSequenceExecutor] loop [%u] tempAlgParamsAllGatherInter.repeatNum [%u],"
            "tempAlgParamsAllGatherInter.inputRepeatStride [%u], tempAlgParamsAllGatherInter.outputRepeatStride [%u]",
            loop,
            tempAlgParamsAllGatherInter.repeatNum,
            tempAlgParamsAllGatherInter.inputRepeatStride,
            tempAlgParamsAllGatherInter.outputRepeatStride);
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplateAllGatherInter->KernelRun(param, tempAlgParamsAllGatherInter, templateResourceInter));

        // ----------- 框内AllGather数据搬运 -----------
        // 框内的数据偏移和搬运计算
        tempAlgParamsAllGatherIntra.count = currDataCount;
        tempAlgParamsAllGatherIntra.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsAllGatherIntra.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamsAllGatherIntra.buffInfo.hcclBuffBaseOff = 0;

        tempAlgParamsAllGatherIntra.allRankDispls = tempAlgParamsScatterIntra.allRankDispls;  // 沿用框内Scatter的切片结果
        tempAlgParamsAllGatherIntra.allRankSliceSize = tempAlgParamsScatterIntra.allRankSliceSize;
        tempAlgParamsAllGatherIntra.allRankProcessedDataCount = tempAlgParamsScatterIntra.allRankProcessedDataCount;

        tempAlgParamsAllGatherIntra.sliceSize = 0;
        tempAlgParamsAllGatherIntra.tailSize = 0;

        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParamsAllGatherIntra.inputSliceStride = 0;  // 如果是输入，偏移是算子的output datasize
        tempAlgParamsAllGatherIntra.outputSliceStride = 0;  // 如果是scratchbuffer，偏移是单次循环处理的最大数据量

        HCCL_DEBUG("[InsV2BroadcastSequenceExecutor] loop [%u] tempAlgParamsAllGatherIntra.inputSliceStride [%u],"
                  "tempAlgParamsAllGatherIntra.outputSliceStride [%u] tempAlgParamsAllGatherIntra.sliceSize [%u]",
            loop,
            tempAlgParamsAllGatherIntra.inputSliceStride,
            tempAlgParamsAllGatherIntra.outputSliceStride,
            tempAlgParamsAllGatherIntra.sliceSize);
        HCCL_DEBUG("[InsV2BroadcastSequenceExecutor] loop [%u] tempAlgParamsAllGatherIntra.buffInfo.inBuffBaseOff [%u],"
                  "tempAlgParamsAllGatherIntra.buffInfo.outBuffBaseOff [%u]",
            loop,
            tempAlgParamsAllGatherIntra.buffInfo.inBuffBaseOff,
            tempAlgParamsAllGatherIntra.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsAllGatherIntra.repeatNum = 1;
        tempAlgParamsAllGatherIntra.inputRepeatStride = 0;
        tempAlgParamsAllGatherIntra.outputRepeatStride = 0;
        HCCL_DEBUG(
            "[InsV2BroadcastSequenceExecutor] loop [%u] tempAlgParamsAllGatherIntra.repeatNum [%u],"
            "tempAlgParamsAllGatherIntra.inputRepeatStride [%u], tempAlgParamsAllGatherIntra.outputRepeatStride [%u]",
            loop,
            tempAlgParamsAllGatherIntra.repeatNum,
            tempAlgParamsAllGatherIntra.inputRepeatStride,
            tempAlgParamsAllGatherIntra.outputRepeatStride);
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplateAllGatherIntra->KernelRun(param, tempAlgParamsAllGatherIntra, templateResourceIntra));

        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2BroadcastSequenceExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult InsV2BroadcastSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::SplitData(const u64 &dataCount, const uint64_t &rankSize, TemplateDataParams &tempAlgParams)
{
    u32 sliceNum = rankSize;
    u64 offsetCount = 0;
    u64 offsetSize = 0;
    tempAlgParams.allRankSliceSize.clear();
    tempAlgParams.allRankDispls.clear();
    tempAlgParams.allRankProcessedDataCount.clear();
    tempAlgParams.allRankSliceSize.reserve(sliceNum);
    tempAlgParams.allRankDispls.reserve(sliceNum);
    tempAlgParams.allRankProcessedDataCount.reserve(sliceNum);

    u64 sliceCount = RoundUp(dataCount, sliceNum);
    u64 sliceSize = sliceCount * dataTypeSize_;

    for (u32 sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
        if (dataCount - offsetCount >= sliceCount) {
            tempAlgParams.allRankDispls.emplace_back(offsetSize);
            tempAlgParams.allRankSliceSize.emplace_back(sliceSize);
            tempAlgParams.allRankProcessedDataCount.emplace_back(sliceCount);
            offsetCount += sliceCount;
            offsetSize = offsetCount * dataTypeSize_;
        } else {
            u64 curSliceCount = dataCount - offsetCount;
            u64 curSliceSize = curSliceCount * dataTypeSize_;
            tempAlgParams.allRankDispls.emplace_back(offsetSize);
            tempAlgParams.allRankSliceSize.emplace_back(curSliceSize);
            tempAlgParams.allRankProcessedDataCount.emplace_back(curSliceCount);
            offsetCount = dataCount;
            offsetSize = offsetCount * dataTypeSize_;
        }
    }

    for (u32 i = 0; i < tempAlgParams.allRankSliceSize.size(); ++i) {
        HCCL_DEBUG("[InsV2BroadcastSequenceExecutor] SliceInfo: offset[%u] size[%u] count[%u]",
            tempAlgParams.allRankDispls.at(i),
            tempAlgParams.allRankSliceSize.at(i),
            tempAlgParams.allRankProcessedDataCount.at(i));
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
u64 InsV2BroadcastSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::RoundUp(const u64 dividend, const u64 divisor)
{
    if (divisor == 0) {
        HCCL_WARNING("[InsV2BroadcastSequenceExecutor][RoundUp] divisor is 0.");
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_BROADCAST, InsBroadcastSequenceMeshNhrDPU, InsV2BroadcastSequenceExecutor,
    TopoMatchMultilevel, InsTempScatterMesh1DIntra, InsTempScatterNHRDPUInter, InsTempAllGatherNHRDPUInter,
    InsTempAllGatherMesh1DIntra);
}  // namespace ops_hccl