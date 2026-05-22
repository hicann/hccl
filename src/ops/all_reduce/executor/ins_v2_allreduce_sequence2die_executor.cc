/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "ins_v2_allreduce_sequence2die_executor.h"
#include "alg_data_trans_wrapper.h"
#ifndef AICPU_COMPILE
#if !defined(HCCL_CANN_COMPAT_850)
#include "ccu_temp_all_gather_2dies_mesh_1d_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh_1D_2die_mem2mem.h"
#include "ccu_temp_all_gather_2dies_mesh_1D.h"
#include "ccu_temp_reduce_scatter_mesh2die.h"
#endif /* !HCCL_CANN_COMPAT_850 */
#endif

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2AllReduceSequence2DieExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2AllReduceSequence2DieExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2AllReduceSequence2DieExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::~InsV2AllReduceSequence2DieExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceSequence2DieExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(HcclComm comm,
    TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo){
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));

    return HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceSequence2DieExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    std::shared_ptr<InsAlgTemplate0> reduceTempAlg = std::make_shared<InsAlgTemplate0>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> gatherTempAlg = std::make_shared<InsAlgTemplate1>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    
    AlgResourceRequest resReqStepReduce;
    AlgResourceRequest resReqStepGather;
    CHK_RET(reduceTempAlg->CalcRes(comm, param, topoInfo, resReqStepReduce));
    CHK_RET(gatherTempAlg->CalcRes(comm, param, topoInfo, resReqStepGather));
 
    // step1在完成后，完成后同步后展开step2，因此slaveThread和对应notify可以复用
    resourceRequest.slaveThreadNum = std::max(resReqStepReduce.slaveThreadNum, resReqStepGather.slaveThreadNum);
    resourceRequest.notifyNumOnMainThread = std::max(resReqStepReduce.notifyNumOnMainThread, resReqStepGather.notifyNumOnMainThread);
    if (resourceRequest.slaveThreadNum != 0) {
        resourceRequest.notifyNumPerThread.assign(resReqStepReduce.slaveThreadNum, 1);
    }
    
    if (param.engine == COMM_ENGINE_CCU) {
        HCCL_INFO("[InsV2AllReduceSequence2DieExecutor][CalcRes] intraTemplate has [%d] kernels.", resReqStepReduce.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            resReqStepReduce.ccuKernelInfos.begin(),
                                            resReqStepReduce.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(resReqStepReduce.ccuKernelNum[0]);
        HCCL_INFO("[InsV2AllReduceSequence2DieExecutor][CalcRes] interTemplate has [%d] kernels.", resReqStepGather.ccuKernelNum[0]);
        std::for_each(resourceRequest.ccuKernelInfos.begin(), resourceRequest.ccuKernelInfos.end(), [](CcuKernelInfo &info) {
            info.resGroup = 0;
        });
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            resReqStepGather.ccuKernelInfos.begin(),
                                            resReqStepGather.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(resReqStepGather.ccuKernelNum[0]);
        std::for_each(resourceRequest.ccuKernelInfos.begin() + resReqStepReduce.ccuKernelNum[0], resourceRequest.ccuKernelInfos.end(), [](CcuKernelInfo &info) {
            info.resGroup = 1;
        });
        HCCL_INFO("[InsV2AllReduceSequence2DieExecutor][CalcRes] all has [%d] kernels.", resourceRequest.ccuKernelInfos.size());
    }
    return HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceSequence2DieExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(const OpParam &param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceSequence2DieExecutor][Orchestrate] Orchestrate Start");
    // 参数填充
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
 
    dataCount_ = param.DataDes.count;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    dataSize_ = dataCount_ * dataTypeSize_;
    maxTmpMemSize_ = resCtx.cclMem.size;

    threads_ = resCtx.threads; // 包含主流

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllReduceSequence2DieExecutor][Orchestrate]errNo[0x%016llx] Reduce scatter excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);
    return HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceSequence2DieExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcSliceInfoAllReduce(u64 dataCount)
{
    u32 sliceNum = rankSize_;
    sliceInfoList_.clear();
    sliceInfoList_.reserve(sliceNum);
 
    u64 sliceCount = RoundUp(dataCount, sliceNum);
    u64 sliceSize = sliceCount * dataTypeSize_;
 
    u64 offsetCount = 0;
    u64 offsetSize = 0;
    for (u32 sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
        if (dataCount - offsetCount > sliceCount) {
            sliceInfoList_.emplace_back(offsetSize, sliceSize, sliceCount);
            offsetCount += sliceCount;
            offsetSize = offsetCount * dataTypeSize_;
        } else {
            u64 curSliceCount = dataCount - offsetCount;
            u64 curSliceSize = curSliceCount * dataTypeSize_;
            sliceInfoList_.emplace_back(offsetSize, curSliceSize, curSliceCount);
            offsetCount = dataCount;
            offsetSize = offsetCount * dataTypeSize_;
        }
    }
 
    return HcclResult::HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
u64  InsV2AllReduceSequence2DieExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::RoundUp(const u64 dividend, const u64 divisor) const
{
    if (divisor == 0) {
        HCCL_WARNING("[InsTempAllReduceMesh1DTwoShot][RoundUp] divisor is 0.");
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}
 
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceSequence2DieExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceSequence2DieExecutor][OrchestrateLoop] Start");
 
    // 构建template:RS
    std::shared_ptr<InsAlgTemplate0> algTemplate0 =
        std::make_shared<InsAlgTemplate0>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
 
    // 构建template AG
    std::shared_ptr<InsAlgTemplate1> algTemplate1 =
        std::make_shared<InsAlgTemplate1>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
    BufferType inBuffType = BufferType::INPUT;
    BufferType outBuffType = BufferType::OUTPUT;
    u32 templateScratchMultiplier = algTemplate0->CalcScratchMultiple(inBuffType, outBuffType);
    TemplateResource templateAlgRes0;
    // 构造reducescatter template资源
    templateAlgRes0.threads = resCtx.threads;
    // 构造allgather template资源
    TemplateResource templateAlgRes1;
    templateAlgRes1.threads = resCtx.threads;

    if (param.engine == COMM_ENGINE_CCU) {
        templateAlgRes0.ccuKernels.insert(templateAlgRes0.ccuKernels.end(),
                                        resCtx.ccuKernels.begin(),
                                        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
        templateAlgRes1.ccuKernels.insert(templateAlgRes1.ccuKernels.end(),
                                        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
                                        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);
    }
    u64 maxDataSizePerLoop = 0;
 
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
 
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    // 单次循环处理的数据量大小
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    HCCL_INFO(
        "[InsV2AllReduceSequence2DieExecutor][OrchestrateOpbase] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
        "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    CHK_PRT_RET(maxDataCountPerLoop == 0,
        HCCL_ERROR("[InsV2AllReduceSequence2DieExecutor][OrchestrateOpbase] maxDataCountPerLoop is 0"), HCCL_E_INTERNAL);
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxDataCountPerLoop + static_cast<u64>(dataCount_ % maxDataCountPerLoop != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxDataCountPerLoop;

        CHK_RET(CalcSliceInfoAllReduce(currDataCount));
       
        TemplateDataParams tempAlgParams0;
        
        // ReduceScatter
        tempAlgParams0.buffInfo.inputPtr = param.inputPtr;
        tempAlgParams0.buffInfo.outputPtr = resCtx.cclMem.addr;
        tempAlgParams0.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParams0.buffInfo.inputSize = param.inputSize;
        tempAlgParams0.buffInfo.outputSize = resCtx.cclMem.size;
        tempAlgParams0.buffInfo.hcclBuffSize = resCtx.cclMem.size;
        tempAlgParams0.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParams0.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
        tempAlgParams0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParams0.count = sliceInfoList_.at(myRank_).count;
        tempAlgParams0.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams0.buffInfo.outBuffBaseOff = 0;
        tempAlgParams0.buffInfo.hcclBuffBaseOff = 0;
        tempAlgParams0.sliceSize = sliceInfoList_.at(myRank_).size;
        tempAlgParams0.tailSize = tempAlgParams0.sliceSize;
        // 如果是输入，偏移是算子的output datasize
        tempAlgParams0.inputSliceStride = (myRank_ == (rankSize_ - 1)) ? sliceInfoList_.at(myRank_-1).size : sliceInfoList_.at(myRank_).size; 
        tempAlgParams0.outputSliceStride = 0; // 如果是scratchbuffer，偏移是单次循环处理的最大数据量
        tempAlgParams0.repeatNum = 1;
        CHK_RET(algTemplate0->KernelRun(param, tempAlgParams0, templateAlgRes0));
        // Allgather
        TemplateDataParams tempAlgParams1;
        tempAlgParams1.buffInfo.inputPtr = resCtx.cclMem.addr;
        tempAlgParams1.buffInfo.outputPtr = param.outputPtr;
        tempAlgParams1.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParams1.buffInfo.inputSize = resCtx.cclMem.size;
        tempAlgParams1.buffInfo.outputSize = param.outputSize;
        tempAlgParams1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        tempAlgParams1.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParams1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParams1.count = sliceInfoList_.at(myRank_).count;
        tempAlgParams1.buffInfo.inBuffBaseOff = 0;
        tempAlgParams1.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams1.buffInfo.hcclBuffBaseOff = 0;
        tempAlgParams1.sliceSize = sliceInfoList_.at(myRank_).size;
        tempAlgParams1.tailSize = tempAlgParams1.sliceSize;
        tempAlgParams1.inputSliceStride = 0; // 如果是输入，偏移是算子的output datasize
        // 如果是scratchbuffer，偏移是单次循环处理的最大数据量
        tempAlgParams1.outputSliceStride = (myRank_ == (rankSize_ - 1)) ? sliceInfoList_.at(myRank_-1).size : sliceInfoList_.at(myRank_).size; 
        tempAlgParams1.repeatNum = 1;
        CHK_RET(algTemplate1->KernelRun(param, tempAlgParams1, templateAlgRes1));
 
        processedDataCount += currDataCount;
        HCCL_DEBUG("[InsV2AllReduceSequence2DieExecutor] testargs fortemplate0  tempAlgParams0.buffInfo.inBuffBaseOff[%u], tempAlgParams0.buffInfo.outBuffBaseOff[%u], tempAlgParams0.sliceSize[%u],"
        "tempAlgParams0.tailSize[%u], tempAlgParams0.inputSliceStride[%u], tempAlgParams0.outputSliceStride[%u].", tempAlgParams0.buffInfo.inBuffBaseOff, tempAlgParams0.buffInfo.outBuffBaseOff
        , tempAlgParams0.sliceSize,tempAlgParams0.tailSize, tempAlgParams0.inputSliceStride, tempAlgParams0.outputSliceStride);

        HCCL_DEBUG("[InsV2AllReduceSequence2DieExecutor] testargs fortemplate1  tempAlgParams1.buffInfo.inBuffBaseOff[%u], tempAlgParams0.buffInfo.outBuffBaseOff[%u], tempAlgParams0.sliceSize[%u],"
        "tempAlgParams1.tailSize[%u], tempAlgParams1.inputSliceStride[%u], tempAlgParams1.outputSliceStride[%u].", tempAlgParams1.buffInfo.inBuffBaseOff, tempAlgParams1.buffInfo.outBuffBaseOff
        , tempAlgParams1.sliceSize, tempAlgParams1.tailSize, tempAlgParams1.inputSliceStride, tempAlgParams1.outputSliceStride);
    }

    HCCL_INFO("[InsV2AllReduceSequence2DieExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}
#ifndef AICPU_COMPILE

#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE, CcuAllreduceMesh2DieBigMs,
    InsV2AllReduceSequence2DieExecutor, TopoMatch1D, CcuTempReduceScatterMesh2Die, CcuTempAllGather2DiesMesh1D);
#endif /* !HCCL_CANN_COMPAT_850 */
#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE, CcuAllreduceMesh2DieBigSche,
    InsV2AllReduceSequence2DieExecutor, TopoMatch1D, CcuTempReduceScatterMeshMem2Mem1D2Die, CcuTempAllGather2DiesMeshMem2Mem1D);
#endif /* !HCCL_CANN_COMPAT_850 */
#endif
}
