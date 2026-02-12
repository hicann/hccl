/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_v_sole_executor.h"
#include "topo_match_1d.h"
#include "ins_temp_all_gather_v_mesh_1D.h"
namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AllGatherVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2AllGatherVSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfo *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfo *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    // 调用计算资源的函数
    algTemplate->CalcRes(comm, param, topoInfo, resourceRequest);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherVSoleExecutor][Orchestrate] Orchestrate Start");
    maxTmpMemSize_ = resCtx.cclMem.size;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.vDataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllGatherVSoleExecutor][Orchestrate]errNo[0x%016llx] All Gather V excutor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherVSoleExecutor][OrchestrateLoop] Start");
    u64 rankSize_ = resCtx.algHierarchyInfo.infos[0][0].size();
    myRank_ = resCtx.topoInfo.userRank;
    // 准备资源
    TemplateResource templateAlgRes;
    templateAlgRes.channels = remoteRankToChannelInfo_[0];
    templateAlgRes.threads = resCtx.threads;
    templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    // 准备数据
    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    // 不需要重复
    tempAlgParams.repeatNum = 1;
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;

    // 从零数组中获取counts和displs
    // 强制类型转换
    const u64 *varData = reinterpret_cast<const u64 *>(param.varData);
    std::vector<u64> counts;
    counts.assign(varData, varData + rankSize_);
    tempAlgParams.allRankDispls.assign(varData + rankSize_, varData + rankSize_ * 2);
    HCCL_DEBUG("[InsV2AllGatherVSoleExecutor][OrchestrateLoop] Rank[%u], inputPtr[%#llx] outputPtr[%#llx], "
               "cclAddr[%#llx], cclSize[%u]",
        myRank_,
        param.inputPtr,
        param.outputPtr,
        resCtx.cclMem.addr,
        resCtx.cclMem.size);
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);

    u32 templateScratchMultiplier =
        algTemplate->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.outBuffType);
    maxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size;
    // 中转内存单次最多能够接受的output count，注意是count不是size
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 maxDataSizePerLoop = 0;
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize =
            maxTmpMemSize_ / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    u64 maxCountPerLoop = maxDataSizePerLoop / dataTypeSize_;

    // 计算loopTimes
    u64 maxSendDataCount = 0;
    for (u64 j = 0; j < rankSize_; j++) {
        maxSendDataCount = std::max(maxSendDataCount, counts[j]);
    }
    u64 loopTimes = 1 + ((maxSendDataCount - 1) / maxCountPerLoop);
    // 带V算子统计所有Rank的processedDataCount
    std::vector<u64> allRankProcessedDataCount(rankSize_, 0);
    tempAlgParams.sliceSize = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        tempAlgParams.allRankSliceSize = {};
        for (u64 i = 0; i < rankSize_; i++) {
            tempAlgParams.allRankSliceSize.push_back(
                ((allRankProcessedDataCount[i] < counts[i])
                        ? std::min(maxCountPerLoop, counts[i] - allRankProcessedDataCount[i])
                        : 0) *
                dataTypeSize_);
            if (loop == 0) {
                tempAlgParams.sliceSize = std::max(tempAlgParams.sliceSize, tempAlgParams.allRankSliceSize[i]);
            }
        }
        u64 processedDataCount = allRankProcessedDataCount[myRank_];
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.allRankProcessedDataCount = allRankProcessedDataCount;
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;
        tempAlgParams.tailSize = tempAlgParams.sliceSize;
        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParams.inputSliceStride = dataSize_;  // 如果是输入，偏移是算子的output datasize
        tempAlgParams.outputSliceStride =
            maxCountPerLoop * dataTypeSize_;  // 如果是scratchbuffer，偏移是单次循环处理的最大数据量

        HCCL_INFO("[InsV2AllGatherVSoleExecutor] loop [%u] tempAlgParams.inputSliceStride [%u],"
                  "tempAlgParams.outputSliceStride [%u] tempAlgParams.sliceSize [%u]",
            loop,
            tempAlgParams.inputSliceStride,
            tempAlgParams.outputSliceStride,
            tempAlgParams.sliceSize);
        HCCL_INFO("[InsV2AllGatherVSoleExecutor] loop [%u] tempAlgParams.buffInfo.inBuffBaseOff [%u],"
                  "tempAlgParams.buffInfo.outBuffBaseOff [%u]",
            loop,
            tempAlgParams.buffInfo.inBuffBaseOff,
            tempAlgParams.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParams.repeatNum = 1;
        tempAlgParams.inputRepeatStride = 0;
        tempAlgParams.outputRepeatStride = 0;
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes));
        for (u64 i = 0; i < rankSize_; i++) {
            allRankProcessedDataCount[i] += tempAlgParams.allRankSliceSize[i] / dataTypeSize_;
        }
    }
    HCCL_INFO("[InsV2AllGatherVSoleExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER_V, InsAllGatherVMesh1D, InsV2AllGatherVSoleExecutor, TopoMatch1D,
    InsTempAllGatherVMesh1D);
}  // namespace ops_hccl