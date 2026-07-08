/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ORDER_PRESERVED_COMMON_H
#define HCCL_ORDER_PRESERVED_COMMON_H

#include "alg_param.h"
#include "alg_env_config.h"

namespace ops_hccl {

constexpr u32 MIN_STRICT_RANK_NUM_ORDER_PRESERVED = 2;
constexpr u32 MAX_RANK_NUM_FOR_ORDER_PRESERVED = 32;

struct OrderPreservedBaseParams {
    u32 myRank;
    u32 rankSize;
    DevType devType;
    u64 dataCount;
    u32 dataTypeSize;
    u64 dataSize;
    HcclDataType dataType;
    HcclReduceOp reduceOp;
    u64 maxTmpMemSize;
};

inline OrderPreservedBaseParams InitOrderPreservedBaseParams(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    OrderPreservedBaseParams params;
    params.myRank = resCtx.topoInfo.userRank;
    params.rankSize = resCtx.topoInfo.userRankSize;
    params.devType = resCtx.topoInfo.deviceType;
    params.dataCount = param.DataDes.count;
    params.dataTypeSize = SIZE_TABLE[param.DataDes.dataType];
    params.dataSize = params.dataCount * params.dataTypeSize;
    params.dataType = param.DataDes.dataType;
    params.reduceOp = param.reduceType;
    params.maxTmpMemSize = resCtx.cclMem.size;
    return params;
}

inline bool IsNeedStrictModeForOrderPreserved(const OpParam& opParam, u32 rankSize)
{
    u8 deterministicLevel = GetExternalInputHcclDeterministic();
    HcclDataType dataType = opParam.DataDes.dataType;
    HcclReduceOp reduceType = opParam.reduceType;
    return (deterministicLevel == static_cast<u8>(DeterministicEnableLevel::DETERMINISTIC_STRICT))	 
         && (dataType == HcclDataType::HCCL_DATA_TYPE_FP16 ||	 
             dataType == HcclDataType::HCCL_DATA_TYPE_FP32 ||	 
             dataType == HcclDataType::HCCL_DATA_TYPE_BFP16 ||	 
             dataType == HcclDataType::HCCL_DATA_TYPE_FP64)	 
         && (reduceType == HcclReduceOp::HCCL_REDUCE_SUM ||	 
             reduceType == HcclReduceOp::HCCL_REDUCE_PROD)	 
         && rankSize > MIN_STRICT_RANK_NUM_ORDER_PRESERVED;
}

} // namespace ops_hccl

#endif // HCCL_ORDER_PRESERVED_COMMON_H