/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_broadcast_sole_executor.h"
#include "ins_temp_broadcast_mesh_1D_two_shot.h"
#include "ins_temp_broadcast_nhr.h"
#include "aiv_temp_broadcast_mesh_1D.h"
namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2BroadcastSoleExecutor()
{
    
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfo *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfo* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate = std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);

    // 调用计算资源的函数
    algTemplate->CalcRes(comm, param, topoInfo, resourceRequest);
    // 在comb_exector或是parallel_exector中合并两个template的资源
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2BroadcastSoleExecutor][Orchestrate] Orchestrate Start");
    maxTmpMemSize_ = resCtx.cclMem.size; // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    HCCL_INFO("[InsV2BroadcastSoleExecutor][Orchestrate] threads_size[%d]", threads_.size());
    if (param.engine != CommEngine::COMM_ENGINE_AIV) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ =  DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2BroadcastSoleExecutor][Orchestrate]errNo[0x%016llx] Reduce scatter excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2BroadcastSoleExecutor][OrchestrateLoop] Start");
    // 准备资源
    TemplateResource templateAlgRes;
    if (param.engine != CommEngine::COMM_ENGINE_AIV) {
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    templateAlgRes.threads = resCtx.threads;
    templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    // 准备数据
    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.inputPtr;
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    tempAlgParams.buffInfo.hcclBuffBaseOff = 0;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
        // 不需要重复
    tempAlgParams.repeatNum = 1;
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;


    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate = std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);

    // 根据CCL Buffer大小和UB_MAX_DATA_SIZE，计算出一轮中最多能输出多少数据
    u32 templateScratchMultiplier = algTemplate->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType,
                                                                     tempAlgParams.buffInfo.outBuffType);
    u64 maxDataSizePerLoop = 0;
    maxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE; // algTemplate->CalcLoopMaxCount();
    HCCL_INFO("[InsV2BroadcastSoleExecutor]maxTmpMemSize_ [%u]", maxTmpMemSize_);
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;

    u64 dataSize = dataCount_ * dataTypeSize_;

    u64 maxLoopOutputSize = maxDataCountPerLoop  * dataTypeSize_;

    HCCL_INFO(
        "[InsV2BroadcastSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
        "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    CHK_PRT_RET(maxDataCountPerLoop == 0,
        HCCL_ERROR("[InsV2BroadcastSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop is 0"), HCCL_E_INTERNAL);

    u64 loopTimes = dataSize / maxLoopOutputSize + static_cast<u64>(dataSize % maxLoopOutputSize != 0);

    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currloopOffset = loop * maxLoopOutputSize;
        u64 currSize = (loop == (loopTimes - 1)) ?  dataSize - currloopOffset : maxLoopOutputSize;
        tempAlgParams.count = currSize/dataTypeSize_;
        // 当前搬运的数据片
        tempAlgParams.buffInfo.inBuffBaseOff = currloopOffset;
        tempAlgParams.buffInfo.outBuffBaseOff = currloopOffset;

        tempAlgParams.sliceSize = currSize;
        tempAlgParams.tailSize = tempAlgParams.sliceSize;

        CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes));
        HCCL_DEBUG("[InsV2BroadcastSoleExecutor] Rank[%d], done generating instruction queues, currSize[%llu], currOffset[%llu].",
                   myRank_, currSize, currloopOffset);
    }
    HCCL_INFO("[InsV2BroadcastSoleExecutor][OrchestrateLoop] End.");

    return HCCL_SUCCESS;
}


REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_BROADCAST, InsBroadcastMesh1DTwoShot, InsV2BroadcastSoleExecutor, TopoMatch1D, 
                 InsTempBroadcastMesh1DTwoShot);
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_BROADCAST, InsBroadcastNHR, InsV2BroadcastSoleExecutor, TopoMatch1D,
                InsTempBroadcastNHR);

#ifndef AICPU_COMPILE
    REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_BROADCAST, AivBroadcastMesh1D, InsV2BroadcastSoleExecutor, TopoMatch1D,
                     AivTempBroadcastMesh1D);
#endif
}
