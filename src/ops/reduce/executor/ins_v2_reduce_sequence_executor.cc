/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_sequence_executor.h"
#include "ins_temp_reduce_mesh_1D.h"
#include "ins_temp_reduce_nhr_dpu.h"

namespace ops_hccl {
// 序列执行器需要的层级数
constexpr u32 SEQUENCE_EXECUTOR_LEVEL_NUM = 2;

// ! 已经编码完成
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2ReduceSequenceExecutor()
{}

// ! 已编码完成
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitCommInfo(
    const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO("[InsV2ReduceSequenceExecutor][InitCommInfo] myRank [%u], rankSize [%u], devType [%u], redOp [%u], "
              "dataType [%u] dataTypeSize [%u]",
        myRank_,
        rankSize_,
        devType_,
        reduceOp_,
        dataType_,
        dataTypeSize_);
    return HCCL_SUCCESS;
}

// ! 已编码完成，实例化实际执行以来AutoMatchMeshNhr这个类的实现
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

// ! 已编码完成
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(HcclComm comm,
    const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
    AlgResourceRequest &resourceRequest)
{
    // 初始化一些基本成员变量
    InitCommInfo(param, topoInfo, algHierarchyInfo);

    rankSizeLevel0_ = algHierarchyInfo.infos[0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[1].size();

    std::shared_ptr<InsAlgTemplate0> interTempAlg =
        std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> intraTempAlg =
        std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);

    AlgResourceRequest resReqInter;
    AlgResourceRequest resReqIntra;
    interTempAlg->CalcRes(comm, param, topoInfo, resReqInter);
    intraTempAlg->CalcRes(comm, param, topoInfo, resReqIntra);

    // step1在完成后，完成后同步后展开step2，因此slaveThread和对应notify可以复用
    resourceRequest.slaveThreadNum = std::max(resReqInter.slaveThreadNum, resReqIntra.slaveThreadNum);
    resourceRequest.notifyNumPerThread = std::max(resReqInter.notifyNumPerThread, resReqIntra.notifyNumPerThread);
    resourceRequest.notifyNumOnMainThread =
        std::max(resReqInter.notifyNumOnMainThread, resReqIntra.notifyNumOnMainThread);

    resourceRequest.channels.resize(SEQUENCE_EXECUTOR_LEVEL_NUM);
    resourceRequest.channels[0] = resReqInter.channels[0];
    resourceRequest.channels[1] = resReqIntra.channels[0];
    return HCCL_SUCCESS;
}

// ! 已编码完成
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2ReduceSequenceExecutor][Orchestrate] Orchestrate Start");
    // 参数填充
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = myRank_ / algHierarchyInfo_.infos[0][0].size();

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();

    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2ReduceSequenceExecutor][Orchestrate]errNo[0x%016llx] Reduce excutor kernel "
                   "run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

// ! 已编码完成
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2ReduceSequenceExecutor][OrchestrateLoop] Start");
    // 将ccl-buffer分成ccl-in和ccl-out 2部分区分使用
    void *cclInAddr = resCtx.cclMem.addr;
    HcclMem cclInMem = {resCtx.cclMem.type, cclInAddr, resCtx.cclMem.size / 2};
    void *cclOutAddr = static_cast<void*>(static_cast<s8 *>(resCtx.cclMem.addr) + resCtx.cclMem.size / 2);
    HcclMem cclOutMem = {resCtx.cclMem.type , cclOutAddr, resCtx.cclMem.size / 2};

    // 声明框内ReduceScatterMesh1D的templateargs
    TemplateDataParams tempAlgParamsReduceScatterMesh1D;
    tempAlgParamsReduceScatterMesh1D.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsReduceScatterMesh1D.buffInfo.outputPtr = cclOutMem.addr; 
    tempAlgParamsReduceScatterMesh1D.buffInfo.inputSize = param.inputSize;
    tempAlgParamsReduceScatterMesh1D.buffInfo.outputSize = param.outputSize;
    tempAlgParamsReduceScatterMesh1D.buffInfo.hcclBuff = cclInMem;
    tempAlgParamsReduceScatterMesh1D.root = param.root;

    // 构建框内ReduceScatterMesh1D的template
    std::shared_ptr<InsAlgTemplate0> algTemplateReduceScatterMesh1D =
        std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);

    // 声明框间ReduceScatterMesh1dDpu的templateargs
    TemplateDataParams tempAlgParamsReduceScatterMesh1dDpu;
    tempAlgParamsReduceScatterMesh1dDpu.buffInfo.inputPtr = cclOutMem.addr;
    tempAlgParamsReduceScatterMesh1dDpu.buffInfo.outputPtr = cclOutMem.addr;
    tempAlgParamsReduceScatterMesh1dDpu.buffInfo.inputSize = param.inputSize;
    tempAlgParamsReduceScatterMesh1dDpu.buffInfo.outputSize = param.outputSize;
    tempAlgParamsReduceScatterMesh1dDpu.buffInfo.hcclBuff = cclInMem;
    tempAlgParamsReduceScatterMesh1dDpu.root = param.root;

    // 构建框间ReduceScatterMesh1dDpu的template
    std::shared_ptr<InsAlgTemplate1> algTemplateReduceScatterMesh1dDpu =
        std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);

    // 声明框间GatherDpu的templateargs，ccl-out搬运到ccl-out
    TemplateDataParams tempAlgParamsGatherDpu;
    tempAlgParamsGatherDpu.buffInfo.inputPtr = cclOutMem.addr;
    tempAlgParamsGatherDpu.buffInfo.outputPtr = cclOutMem.addr;
    tempAlgParamsGatherDpu.buffInfo.inputSize = param.inputSize;
    tempAlgParamsGatherDpu.buffInfo.outputSize = param.outputSize;
    tempAlgParamsGatherDpu.buffInfo.hcclBuff = cclInMem;
    tempAlgParamsGatherDpu.root = param.root;

    // 构建框间GatherDpu的template
    std::shared_ptr<InsAlgTemplate2> algTemplateGatherDpu =
        std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[1]);

    // 声明框内GatherMesh1D的templateargs，ccl-out搬运到user-out
    TemplateDataParams tempAlgParamsGatherMesh1D;
    tempAlgParamsGatherMesh1D.buffInfo.inputPtr = cclOutMem.addr;
    tempAlgParamsGatherMesh1D.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsGatherMesh1D.buffInfo.inputSize = param.inputSize;
    tempAlgParamsGatherMesh1D.buffInfo.outputSize = param.outputSize;
    tempAlgParamsGatherMesh1D.buffInfo.hcclBuff = cclInMem;
    tempAlgParamsGatherMesh1D.root = param.root;

    // 构建框内GatherMesh1D的template
    std::shared_ptr<InsAlgTemplate3> algTemplateGatherMesh1D =
        std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo_.infos[0]);

    // 构造框内ReduceScatterMesh1D的template资源
    TemplateResource templateResourceReduceScatterMesh1D;
    templateResourceReduceScatterMesh1D.channels = remoteRankToChannelInfo_[0];
    templateResourceReduceScatterMesh1D.threads = resCtx.threads;
    templateResourceReduceScatterMesh1D.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceReduceScatterMesh1D.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    // 构造框间ReduceScatterMesh1dDpu的template资源
    TemplateResource templateResourceReduceScatterMesh1dDpu;
    templateResourceReduceScatterMesh1dDpu.channels = remoteRankToChannelInfo_[1];
    templateResourceReduceScatterMesh1dDpu.threads = resCtx.threads;
    templateResourceReduceScatterMesh1dDpu.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceReduceScatterMesh1dDpu.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    // 构造框间GatherDpu的template资源
    TemplateResource templateResourceGatherDpu;
    templateResourceGatherDpu.channels = remoteRankToChannelInfo_[1];
    templateResourceGatherDpu.threads = resCtx.threads;
    templateResourceGatherDpu.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceGatherDpu.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    // 构造框内GatherMesh1D的template资源
    TemplateResource templateResourceGatherMesh1D;
    templateResourceGatherMesh1D.channels = remoteRankToChannelInfo_[0];
    templateResourceGatherMesh1D.threads = resCtx.threads;
    templateResourceGatherMesh1D.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceGatherMesh1D.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    
    // 中转内存单次最多能够接受的output count，注意是count不是size
    u64 maxCountPerLoop = tempAlgParamsInter.buffInfo.hcclBuff.size / 2 / templateScratchMultiplier /
                          HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        // ----------- 框内数据搬运 -----------
        // 框内的数据偏移和搬运计算
        tempAlgParamsInter.count = currDataCount;
        tempAlgParamsInter.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamsInter.buffInfo.outBuffBaseOff = 0;  // 从user-in搬运到ccl-in，最终输出到ccl-in上面
        tempAlgParamsInter.buffInfo.hcclBuffBaseOff = 0;

        tempAlgParamsInter.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParamsInter.tailSize = tempAlgParamsInter.sliceSize;
        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParamsInter.inputSliceStride =
            maxCountPerLoop * dataTypeSize_;  // ccl-in按照rank偏移量，每次偏移是单次循环最大数据量
        tempAlgParamsInter.outputSliceStride =
            maxCountPerLoop * dataTypeSize_;  // 如果是scratchbuffer，偏移是单次循环处理的最大数据量

        HCCL_INFO("[InsV2ReduceSequenceExecutor] loop [%u] tempAlgParamsInter.inputSliceStride [%u],"
                  "tempAlgParamsInter.outputSliceStride [%u] tempAlgParamsInter.sliceSize [%u]",
            loop,
            tempAlgParamsInter.inputSliceStride,
            tempAlgParamsInter.outputSliceStride,
            tempAlgParamsInter.sliceSize);
        HCCL_INFO("[InsV2ReduceSequenceExecutor] loop [%u] tempAlgParamsInter.buffInfo.inBuffBaseOff [%u],"
                  "tempAlgParamsInter.buffInfo.outBuffBaseOff [%u]",
            loop,
            tempAlgParamsInter.buffInfo.inBuffBaseOff,
            tempAlgParamsInter.buffInfo.outBuffBaseOff);
        // m*n组网框内需要做n次重复
        tempAlgParamsInter.repeatNum = algHierarchyInfo_.infos[0][0].size();
        tempAlgParamsInter.inputRepeatStride = templateScratchMultiplierInter * dataCount_ * dataTypeSize_;
        tempAlgParamsInter.outputRepeatStride = templateScratchMultiplierInter * processedDataCount * dataTypeSize_;
        HCCL_INFO("[InsV2ReduceSequenceExecutor] loop [%u] tempAlgParamsInter.repeatNum [%u],"
                  "tempAlgParamsInter.inputRepeatStride [%u], tempAlgParamsInter.outputRepeatStride [%u]",
            loop,
            tempAlgParamsInter.repeatNum,
            tempAlgParamsInter.inputRepeatStride,
            tempAlgParamsInter.outputRepeatStride);
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplateInter->KernelRun(param, tempAlgParamsInter, templateResourceInter));

        // ----------- 框间数据搬运 -----------
        // 框间的数据偏移和搬运量计算
        tempAlgParamsIntra.count = currDataCount;
        tempAlgParamsIntra.buffInfo.inBuffBaseOff = 0;  // ccl-out偏移量，每次更新，所以是0
        tempAlgParamsIntra.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamsIntra.buffInfo.hcclBuffBaseOff = 0;

        tempAlgParamsIntra.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParamsIntra.tailSize = tempAlgParamsIntra.sliceSize;
        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParamsIntra.inputSliceStride = maxCountPerLoop * dataTypeSize_;  // 框间从ccl-in拿数据，
        tempAlgParamsIntra.outputSliceStride =
            maxCountPerLoop * dataTypeSize_;  // 如果是scratchbuffer，偏移是单次循环处理的最大数据量

        HCCL_INFO("[InsV2ReduceSequenceExecutor] loop [%u] tempAlgParamsIntra.inputSliceStride [%u],"
                  "tempAlgParamsIntra.outputSliceStride [%u] tempAlgParamsIntra.sliceSize [%u]",
            loop,
            tempAlgParamsIntra.inputSliceStride,
            tempAlgParamsIntra.outputSliceStride,
            tempAlgParamsIntra.sliceSize);
        HCCL_INFO("[InsV2ReduceSequenceExecutor] loop [%u] tempAlgParamsIntra.buffInfo.inBuffBaseOff [%u],"
                  "tempAlgParamsIntra.buffInfo.outBuffBaseOff [%u]",
            loop,
            tempAlgParamsIntra.buffInfo.inBuffBaseOff,
            tempAlgParamsIntra.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsIntra.repeatNum = 1;
        tempAlgParamsIntra.inputRepeatStride = 0;
        tempAlgParamsIntra.outputRepeatStride = 0;
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplateIntra->KernelRun(param, tempAlgParamsIntra, templateResourceIntra));
        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2ReduceSequenceExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_REDUCE, InsV2ReduceSequenceMeshNhr, InsV2ReduceSequenceExecutor,
    TopoMatchMultilevel, InsTempReduceMesh1D, InsTempReduceNhrDpu);
}  // namespace ops_hccl