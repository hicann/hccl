/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_scatter_order_preserved_executor.h"
#include "ins_temp_reduce_scatter_order_preserved_level1.h"
#include "ins_temp_reduce_scatter_order_preserved_group.h"
#include "alg_env_config.h"
#include "order_preserved_common.h"
#include <cmath>
#include <algorithm>

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2ReduceScatterOrderPreservedExecutor()
{
    // 初始化严格确定性模式标志为true（保序模式默认启用）
    deterministicStrict_ = true;
}

// 计算算法层级信息：确定算法在通信层级中的位置和子通信域
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

// 计算资源请求：确定执行所需的线程、通知和通道资源
// algHierarchyInfo: 算法层级信息
// resourceRequest: 输出参数，填充资源请求信息
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    // 调用模板的CalcRes函数计算所需资源
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][CalcRes] slaveThreadNum[%u], notifyNumOnMainThread[%u]",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    return HCCL_SUCCESS;
}

// 编排执行：协调ReduceScatter操作的执行流程
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][Orchestrate] Start");

    OrderPreservedBaseParams baseParams = InitOrderPreservedBaseParams(param, resCtx);
    SetOrderPreservedBaseParams(baseParams);
    
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    InitExecutorInfo(param);

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2ReduceScatterOrderPreservedExecutor][Orchestrate] kernel run failed, err[0x%016llx]",
            HCCL_ERROR_CODE(ret)), ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][OrchestrateLoop] Start, deterministicStrict[%d]",
        deterministicStrict_);

    // 准备模板资源
    TemplateResource templateAlgRes;
    if (remoteRankToChannelInfo_.size() > 0) {
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    templateAlgRes.threads = resCtx.threads;
    templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    // 准备模板数据
    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, myRank_, resCtx.algHierarchyInfo.infos[0]);

    // 计算临时缓冲区倍数，实际上就是ranksize
    u32 templateScratchMultiplier = algTemplate->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType,
                                                                     tempAlgParams.buffInfo.outBuffType);
    u64 maxDataSizePerLoop = 0;
    maxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor]maxTmpMemSize_ [%llu]", maxTmpMemSize_);
    if (templateScratchMultiplier != 0) {
        // 计算基于缓冲区的数据量边界：缓冲区大小/倍数，按对齐向下取整
        u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN_ORDER_PRESERVED
            * HCCL_MIN_SLICE_ALIGN_ORDER_PRESERVED;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    // 单次循环最大数据元素个数
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    HCCL_INFO(
        "[InsV2ReduceScatterOrderPreservedExecutor][OrchestrateLoop] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
        "transportBoundDataSize[%llu], templateScratchMultiplier[%u]",
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    CHK_PRT_RET(maxDataCountPerLoop == 0,
        HCCL_ERROR("[InsV2ReduceScatterOrderPreservedExecutor][OrchestrateLoop] maxDataCountPerLoop is 0"), HCCL_E_INTERNAL);

    // 计算循环次数：总数据量 / 单次最大数据量，向上取整
    u64 loopTimes = dataCount_ / maxDataCountPerLoop + static_cast<u64>(dataCount_ % maxDataCountPerLoop != 0);
    tempAlgParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
    // 初始化已处理的数据元素个数
    u64 processedDataCount = 0;

    for (u64 loop = 0; loop < loopTimes; loop++) {
        // 计算当前循环处理的数据元素个数
        // 如果是最后一次循环，处理剩余数据；否则处理单次最大数据量
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxDataCountPerLoop;
        tempAlgParams.count = currDataCount;
        // 计算输入缓冲区基址偏移（已处理数据的偏移）
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        // 计算输出缓冲区基址偏移（已处理数据的偏移）
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        // 设置临时缓冲区基址偏移为0（每次循环重新使用临时缓冲区）
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;

        u64 currSizePerBlock = currDataCount * dataTypeSize_;

        // 创建内存块信息结构体，用于存储每个rank的数据大小和偏移
        MemBlockInfo memBlockInfo;
        memBlockInfo.size.clear();
        memBlockInfo.userInputOffsets.clear();
        memBlockInfo.inputOffsets.clear();
        memBlockInfo.outputOffsets.clear();
        u64 unitSize = dataTypeSize_;
        // 遍历每个数据块，计算其偏移量
        for (u32 dataId = 0; dataId < rankSize_; dataId++) {
            // 计算当前数据块在临时缓冲区中的偏移
            u64 offset = dataId * currSizePerBlock;
            // 计算当前数据块在用户输入缓冲区中的偏移
            // 基址偏移 + 数据块ID * 当前循环数据量
            u64 userMemInOffset = processedDataCount * unitSize + dataId * dataSize_;
            memBlockInfo.size.push_back(currSizePerBlock);
            // 添加用户输入偏移
            memBlockInfo.userInputOffsets.push_back(userMemInOffset);
            // 添加输入偏移（临时缓冲区）
            memBlockInfo.inputOffsets.push_back(offset);
            // 添加输出偏移（临时缓冲区）
            memBlockInfo.outputOffsets.push_back(offset);
        }

        tempAlgParams.sliceSize = currSizePerBlock;
        tempAlgParams.tailSize = tempAlgParams.sliceSize;
        tempAlgParams.inputSliceStride = dataSize_;
        tempAlgParams.outputSliceStride = 0;
        tempAlgParams.repeatNum = 1;
        tempAlgParams.inputRepeatStride = 0;
        tempAlgParams.outputRepeatStride = 0;

        HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor] loop[%llu] sliceSize[%llu], "
            "inBuffBaseOff[%llu], outBuffBaseOff[%llu]",
            loop, tempAlgParams.sliceSize, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.outBuffBaseOff);

        // 将内存块信息设置到算法模板
        algTemplate->SetMemBlockInfo(memBlockInfo);
        // 调用InsTempReduceScatterOrderPreservedLevel1模板算法
        CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes));
        processedDataCount += currDataCount;
    }

    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][OrchestrateLoop] End");
    return HCCL_SUCCESS;
}

// 初始化执行器信息：检查和设置严格确定性模式
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::InitExecutorInfo(const OpParam &param)
{
    // 规约保序判断已在 selector 中完成，executor 直接启用保序模式
    deterministicStrict_ = true;
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][InitExecutorInfo] deterministicStrict[%d]",
        deterministicStrict_);
    return HCCL_SUCCESS;
}

// 向上对齐到除数的倍数
template <typename AlgTopoMatch, typename InsAlgTemplate>
u64 InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::RoundUpWithDivisor(
    u64 value, u64 divisor) const
{
    if (value == 0 || divisor == 0) {
        return divisor;
    }
    return ((value + (divisor - 1)) / divisor) * divisor;
}

// 注册保序ReduceScatter执行器（32卡及以内场景）
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, ReduceScatterOrderPreserved,
    InsV2ReduceScatterOrderPreservedExecutor, TopoMatch1D, InsTempReduceScatterOrderPreservedLevel1);

// 注册分组 all2all 版保序 ReduceScatter 执行器（大于32卡场景）
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, ReduceScatterOrderPreservedGroup,
    InsV2ReduceScatterOrderPreservedExecutor, TopoMatch1D, InsTempReduceScatterOrderPreservedGroup);

}