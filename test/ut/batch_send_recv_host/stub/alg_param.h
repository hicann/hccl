/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSR_HOST_TEST_ALG_PARAM_H
#define BSR_HOST_TEST_ALG_PARAM_H

// 生产 alg_param.h 依赖 CANN 全量头，UT 只保留 host 侧被测代码引用到的字段。
// 字段名与生产保持一致：被测逻辑读的是字段名，改名会让断言与被测行为脱节。

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "hccl_types.h"

namespace ops_hccl {

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using s32 = int32_t;
using s64 = int64_t;

constexpr u32 INVALID_VALUE_RANKID = 0xFFFFFFFF;
// UB 协议一次传输的最大 size，与生产 alg_param.h 一致：切片时用它 clamp 中转容量。
constexpr u64 UB_MAX_DATA_SIZE = 256ULL * 1024 * 1024;
constexpr u32 TOPO_LEVEL_NUM_1 = 1;
constexpr u32 TOPO_LEVEL_NUM_2 = 2;
constexpr u32 TOPO_LEVEL_NUM_3 = 3;

// 按 HcclDataType 枚举值索引；本表只服务被测代码里的 count*size / sliceSize 换算。
inline constexpr u32 DATATYPE_SIZE_TABLE[] = {1, 2, 4, 2, 4, 8, 2, 8, 1, 2, 4, 8, 1};
inline constexpr u32 HCCL_SIZE_TABLE[] = {1, 2, 4, 2, 4, 8, 2, 8, 1, 2, 4, 8, 1};

enum class OpMode { OPBASE, OFFLOAD };
enum class CommEngine { COMM_ENGINE_AIV, COMM_ENGINE_AICPU_TS, COMM_ENGINE_CCU };
enum class BufferType { DEFAULT = 0, INPUT = 1, OUTPUT = 2, HCCL_BUFFER = 3 };
enum class Level0Shape { CLOS = 0, MESH_1D = 1, MESH_1D_CLOS = 2 };
enum class CommTopo { COMM_TOPO_1DMESH = 0, COMM_TOPO_CLOS = 1 };
enum class AlgoType : uint8_t { UNKNOWN = 0, MESH = 1, MESH_TWOSHOT = 2 };
enum class OpExecuteConfig { DEFAULT = 0, AICPU_TS = 1, CCU_MS = 2, CCU_SCHED = 3 };
enum class CostAggMode { SUM = 0, MAX = 1 };

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
    HcclCMDType opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    HcclComm hcclComm = nullptr;
    void* stream = nullptr;
    u32 numBlocksLimit = 0;
    u32 reduceType = 0;
    u32 root = 0;
    bool enableDetour = false;
    DataDescription DataDes;
    struct {
        HcclSendRecvItem* sendRecvItemsPtr = nullptr;
        u32 itemNum = 0;
    } batchSendRecvDataDes;
};

struct TopoInfoWithNetLayerDetails {
    u32 userRank = 0;
    u32 userRankSize = 1;
    Level0Shape level0Topo = Level0Shape::MESH_1D;
    bool level0PcieMix = false;
    // 下面两项只被 AivBatchSendRecvSoleMesh 注册的 topoCustomCheck 读取
    bool level2UbRtp = false;
    bool topLevelUboe = false;
};

struct TopoAttrs {
    uint8_t minTopoLevelNum = 1;
    uint8_t maxTopoLevelNum = 3;
    uint8_t supportLevel0Topos = 0;
    bool isSupportLevel0PcieMix = false;
    u32 maxSupportRankSize = 0;
    std::function<bool(const TopoInfoWithNetLayerDetails*)> topoCustomCheck = nullptr;
};

struct OpAttrs {
    bool isSupportProd = true;
    bool isSupportInplace = true;
};

struct AlgAttrs {
    std::string name;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    OpExecuteConfig engine = OpExecuteConfig::DEFAULT;
    std::vector<AlgoType> algoTypes;
    TopoAttrs topo;
    OpAttrs op;
};

struct AlgHierarchyInfoForAllLevel {
    // 生产按 [算法层][该层子组][子组内 rank] 组织，被测代码只取 infos[0] 与 infos.size()
    std::vector<std::vector<std::vector<u32>>> infos;
};

constexpr u32 COMM_PROTOCOL_UB_MEM = 0;
constexpr u32 COMM_PROTOCOL_OTHER = 1;

struct HcclChannelDesc {
    u32 channelProtocol = COMM_PROTOCOL_UB_MEM;
};

struct AlgResourceRequest {
    u32 slaveThreadNum = 0;
    u32 notifyNumOnMainThread = 0;
    std::vector<std::vector<HcclChannelDesc>> channels;
};

struct CostModelParam {
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;
};

struct AlgNetMeta {
    std::vector<CommTopo> netTypes;
    CostAggMode intraGroupMode = CostAggMode::SUM;
    std::vector<u32> groupSizes;
    std::vector<float> dataRatios;
    std::vector<u32> rankSizes;
};

struct AlgResourceCtxSerializable {
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    HcclMem cclMem;
    void* aivCommInfoPtr = nullptr;
    std::vector<ThreadHandle> threads;
    TopoInfoWithNetLayerDetails topoInfo;
};

} // namespace ops_hccl

#endif // BSR_HOST_TEST_ALG_PARAM_H
