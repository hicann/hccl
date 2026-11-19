/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sym_mem_gate.h"

namespace ops_hccl {

bool IsReduceOpSymMemAllowed(const OpParam& param)
{
    // RSV 为 V 类算子，数据类型取 vDataDes.dataType（DataDes 对 V 类恒为 RESERVED）
    HcclDataType dataType
        = param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V ? param.vDataDes.dataType : param.DataDes.dataType;
    // AIV 对称路径是 UB 本地归约、不走 Reduce64 的 int64 原子加模拟，int64 放行；其余排除与 AICPU 一致。
    if (param.engine == CommEngine::COMM_ENGINE_AIV) {
        return param.reduceType != HcclReduceOp::HCCL_REDUCE_PROD && dataType != HcclDataType::HCCL_DATA_TYPE_FP64
               && dataType != HcclDataType::HCCL_DATA_TYPE_UINT64;
    }
    return dataType != HcclDataType::HCCL_DATA_TYPE_INT64 && dataType != HcclDataType::HCCL_DATA_TYPE_UINT64
           && dataType != HcclDataType::HCCL_DATA_TYPE_FP64 && param.reduceType != HcclReduceOp::HCCL_REDUCE_PROD;
}

bool IsReduceOpSymMemDeterministic(
    const std::string& algName, const TopoInfoWithNetLayerDetails* topoInfo, HcclCMDType opType)
{
    if (topoInfo == nullptr) {
        return false;
    }
    switch (opType) {
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V:
            if (algName == "AicpuAllReduceSoleNHR" || algName == "AicpuReduceScatterSoleNHR") {
                return true;
            }
            if (topoInfo->topoLevelNums != TOPO_LEVEL_NUM_1) {
                // 多级否决仅针对分层算法
                return false;
            }
            return algName == "AicpuReduceScatterPipeLineMeshNHR" || algName == "AicpuAllReducePipeLineMeshNHR"
                   || algName == "AicpuAllReduceSoleMeshChunkTwoShot" || algName == "AicpuReduceScatterSoleMeshChunk"
                   || algName == "AicpuReduceScatterConcurMeshNHR" || algName == "AicpuAllReduceConcurMeshTwoShotNHR"
                   || algName == "AicpuReduceScatterVSoleMeshChunk" || algName == "AivAllReduceSoleMeshTwoShot"
                   || algName == "AivReduceScatterSoleMesh";
        case HcclCMDType::HCCL_CMD_REDUCE:
            if (topoInfo->topoLevelNums == TOPO_LEVEL_NUM_1) {
                // 单级 MESH_1D 选到 SoleMesh/TwoShot，不在白名单，无零拷贝收益
                return false;
            }
            return algName == "AicpuReduceSoleNHR" || algName == "AicpuReduceParallelMeshNHR"
                   || algName == "AicpuReduceParallelNHRNHR";
        default:
            return false;
    }
}

// 仅 Scatter 与 Reduce 跳过 input 侧探测：Scatter 非 root sendBuf 可空/虚值（API 结构原因）；
// Reduce 只写 root 的 output 窗口、不应额外要求注册 sendBuf。其余算子（含 RSV）双侧照常探测。
bool NeedSymMemInputSide(HcclCMDType opType)
{
    return opType != HcclCMDType::HCCL_CMD_SCATTER && opType != HcclCMDType::HCCL_CMD_REDUCE;
}

bool IsSymMemWindowValid(const OpParam& param)
{
    bool inputValid = !NeedSymMemInputSide(param.opType) || param.inputPtr == nullptr || param.inputSize == 0
                      || param.inputSymWindow != nullptr;
    bool outputValid = param.outputPtr == nullptr || param.outputSize == 0 || param.outputSymWindow != nullptr;
    return inputValid && outputValid;
}

} // namespace ops_hccl
