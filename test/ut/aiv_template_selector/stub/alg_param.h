/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_TEMPLATE_SELECTOR_TEST_ALG_PARAM_H
#define AIV_TEMPLATE_SELECTOR_TEST_ALG_PARAM_H

#include <cstdint>
#include <vector>

// UT 桩环境不引入 dlsym_common.h，此处补齐 CANN 版本宏供源码中的 #if CANN_VERSION_NUM 守卫解析。
#ifndef CANN_VERSION_NUM
#define CANN_VERSION_VAL(M, m, p) ((M) * 10000000 + (m) * 100000 + (p) * 1000)
#define CANN_VERSION(...) CANN_VERSION_VAL(9, 2, 0)
#define CANN_VERSION_NUM CANN_VERSION(9, 2, 0)
#endif

namespace ops_hccl {

using u32 = uint32_t;
using u64 = uint64_t;
using HcclComm = void*;

enum HcclResult { HCCL_SUCCESS = 0, HCCL_E_INTERNAL = 1, HCCL_E_PARA = 2, HCCL_E_PTR = 3, HCCL_E_NOT_SUPPORT = 12 };
enum class OpMode { OPBASE, OFFLOAD };
enum class CommEngine { COMM_ENGINE_AIV, COMM_ENGINE_AICPU_TS, COMM_ENGINE_CCU };
enum class OpExecuteConfig { AIV, AIV_URMA };
enum HcclReduceOp { HCCL_REDUCE_SUM = 0, HCCL_REDUCE_MAX = 2, HCCL_REDUCE_PROD = 3, HCCL_REDUCE_MIN = 4 };
constexpr HcclReduceOp HCCL_REDUCE_RESERVED = HCCL_REDUCE_SUM;
enum class HcclCMDType {
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_ALLTOALL,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_ALLGATHER
};
enum HcclDataType { HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_HIF8 };
enum class AlgoType { UNKNOWN, MESH, MESH_TWOSHOT };
enum class BufferType { INPUT, OUTPUT, HCCL_BUFFER };
enum class Level0Shape { MESH_1D, MESH_1D_CLOS };
enum class CommTopo { COMM_TOPO_1DMESH, COMM_TOPO_CLOS };
enum class EngineType { AIV, AICPU };

inline constexpr u32 DATATYPE_SIZE_TABLE[] = {4, 2, 1};
inline constexpr u32 HCCL_SIZE_TABLE[] = {4, 2, 1};

struct TemplateProp {
    AlgoType algoType = AlgoType::UNKNOWN;
};

struct DataDescription {
    u64 count = 0;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
};

struct OpParam {
    OpMode opMode = OpMode::OPBASE;
    CommEngine engine = CommEngine::COMM_ENGINE_AIV;
    OpExecuteConfig opExecuteConfig = OpExecuteConfig::AIV;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    HcclComm hcclComm = nullptr;
    bool supportSymmetricMemory = false;
    DataDescription DataDes;
    struct {
        HcclDataType sendType = HCCL_DATA_TYPE_FP32;
        void* sendCounts = nullptr;
    } all2AllVDataDes;
    u32 numBlocksLimit = 48;
    u32 reduceType = 0;
    void* stream = nullptr;
};

struct TopoInfoWithNetLayerDetails {
    bool level0PcieMix = false;
    bool level0AllMesh = false;
    u32 userRank = 0;
    u32 userRankSize = 1;
    u32 deviceNumPerModule = 1;
    Level0Shape level0Topo = Level0Shape::MESH_1D;
    bool level0Symmetric = true;
    bool level1Symmetric = true;
};

struct TemplateDataParams {
    struct {
        void* inputPtr = nullptr;
        void* outputPtr = nullptr;
        u64 inBuffBaseOff = 0;
        u64 outBuffBaseOff = 0;
        struct {
            u64 size = 0;
        } hcclBuff;
    } buffInfo;
    u64 sliceSize = 0;
    u64 inputSliceStride = 0;
    u64 outputSliceStride = 0;
    std::vector<u64> sendCounts;
    bool isWrite = false;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    u64 repeatNum = 1;
    u64 inputRepeatStride = 0;
    u64 outputRepeatStride = 0;
    u32 subCommRankLevel = 0;
    u32 level0ChannelNumPerRank = 0;
};

struct TemplateResource {
    void* aivCommInfoPtr = nullptr;
};

enum CommProtocol {
    COMM_PROTOCOL_UB_MEM = 0,
    COMM_PROTOCOL_UB_CTP = 1,
    COMM_PROTOCOL_UBOE = 7,
    COMM_PROTOCOL_UB_RTP = 9
};
struct HcclChannelDesc {
    CommProtocol channelProtocol = COMM_PROTOCOL_UB_MEM;
};

inline void AppendChannelsByProtocol(
    const std::vector<HcclChannelDesc>& source, CommProtocol protocol, std::vector<HcclChannelDesc>& target)
{
    for (const auto& channel : source) {
        if (channel.channelProtocol == protocol) {
            target.push_back(channel);
        }
    }
}

struct AlgResourceRequest {
    u32 slaveThreadNum = 0;
    u32 notifyNumOnMainThread = 0;
    std::vector<u32> notifyNumPerThread;
    std::vector<std::vector<HcclChannelDesc>> channels;
};

struct CostModelParam {
    float A;
    float B;
    float C;
    float D;
};

struct CalcCostCoeffParam {
    std::vector<u32> portNum = {1};
    u32 rankSize = 1;
    bool isPod = false;
    TopoInfoWithNetLayerDetails* topoInfo = nullptr;
    float dataRatio = 1.0f;
    CommTopo netType = CommTopo::COMM_TOPO_1DMESH;
    BufferType inputBuffer = BufferType::INPUT;
    BufferType scratchBuffer = BufferType::HCCL_BUFFER;
};

} // namespace ops_hccl

#endif // AIV_TEMPLATE_SELECTOR_TEST_ALG_PARAM_H
