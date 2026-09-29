/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_COMMON_OPS_H
#define EXECUTOR_COMMON_OPS_H

#include <string>

#include "alg_param.h"
#include "alg_v2_template_base.h"
#include "channel.h"
#include "coll_alg_v2_exec_registry.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "log.h"
#include "sal.h"
#include "topo_host.h"
#include "utils.h"

namespace ops_hccl {
/*
 * 统一填充template数据参数的buffInfo: 序列/两发等执行器GenBaseTempAlgParams中
 * "三段buffer类型+输入输出指针+hcclBuff"的6行赋值样板, hcclBuffType恒为HCCL_BUFFER。
 */
inline void SetTemplateBuffInfo(
    TemplateDataParams& tempAlgParams, BufferType inBuffType, BufferType outBuffType, void* inputPtr, void* outputPtr,
    const HcclMem& hcclBuff)
{
    tempAlgParams.buffInfo.inBuffType = inBuffType;
    tempAlgParams.buffInfo.outBuffType = outBuffType;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.inputPtr = inputPtr;
    tempAlgParams.buffInfo.outputPtr = outputPtr;
    tempAlgParams.buffInfo.hcclBuff = hcclBuff;
}

// 非CCU模板仅设置执行配置，保留调用方原有engine。
template <bool IsCcu>
inline void SetCostModelExecuteConfig(OpParam& param, const char* algName)
{
    if (IsCcu) {
        param.engine = CommEngine::COMM_ENGINE_CCU;
        param.opExecuteConfig = (std::string(algName).find("CcuMS") != std::string::npos) ? OpExecuteConfig::CCU_MS :
                                                                                            OpExecuteConfig::CCU_SCHED;
    } else {
        param.opExecuteConfig = OpExecuteConfig::AICPU_TS;
    }
}

// 只填写通道、线程和共享内存指针，其他模板资源由调用方管理。
inline void SetTemplateCommResource(
    TemplateResource& resource, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const AlgResourceCtxSerializable& resCtx)
{
    resource.channels = channels;
    resource.threads = resCtx.threads;
    resource.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    resource.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
}

// 按原顺序追加两份CCU资源：先追加两份计数，再追加两份kernel信息。
inline void AppendCcuTemplateResources(
    AlgResourceRequest& resourceRequest, const AlgResourceRequest& temp0ResReq, const AlgResourceRequest& temp1ResReq)
{
    resourceRequest.ccuKernelNum.insert(
        resourceRequest.ccuKernelNum.end(), temp0ResReq.ccuKernelNum.begin(), temp0ResReq.ccuKernelNum.end());
    resourceRequest.ccuKernelNum.insert(
        resourceRequest.ccuKernelNum.end(), temp1ResReq.ccuKernelNum.begin(), temp1ResReq.ccuKernelNum.end());
    // 将两个合并
    resourceRequest.ccuKernelInfos.insert(
        resourceRequest.ccuKernelInfos.end(), temp0ResReq.ccuKernelInfos.begin(), temp0ResReq.ccuKernelInfos.end());
    resourceRequest.ccuKernelInfos.insert(
        resourceRequest.ccuKernelInfos.end(), temp1ResReq.ccuKernelInfos.begin(), temp1ResReq.ccuKernelInfos.end());
}

// 填写等长AllGather切片；调用方保留原rank和元素大小的计算。
inline HcclResult
FillAllGatherEqualSlices(u64 dataCount, u64 rankSize, u64 dataTypeSize, TemplateDataParams& tempAlgParams)
{
    u32 sliceNum = rankSize;
    tempAlgParams.allRankSliceSize.clear();
    tempAlgParams.allRankDispls.clear();
    tempAlgParams.allRankProcessedDataCount.clear();
    tempAlgParams.allRankSliceSize.reserve(sliceNum);
    tempAlgParams.allRankDispls.reserve(sliceNum);
    tempAlgParams.allRankProcessedDataCount.reserve(sliceNum);

    u64 sliceSize = dataCount * dataTypeSize;
    for (u32 i = 0; i < sliceNum; i++) {
        tempAlgParams.allRankDispls.emplace_back(i * sliceSize);
        tempAlgParams.allRankSliceSize.emplace_back(sliceSize);
        tempAlgParams.allRankProcessedDataCount.emplace_back(dataCount);
    }
    return HCCL_SUCCESS;
}

// 等长slice stride，单次执行；不修改slice大小、count和buffer偏移。
inline void SetSingleRepeatSliceStrides(TemplateDataParams& params)
{
    params.inputSliceStride = params.sliceSize;
    params.outputSliceStride = params.sliceSize;

    params.repeatNum = 1;
    params.inputRepeatStride = 0;
    params.outputRepeatStride = 0;
}

} // namespace ops_hccl

#endif
