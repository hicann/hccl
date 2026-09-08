/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_COMMON_ALG_PARSE
#define OPS_HCCL_SRC_COMMON_ALG_PARSE

#include <string>
#include <vector>
#include <map>
#include <set>
#include <cstdint>
#include <hccl/hccl_types.h>

namespace ops_hccl {

enum class AlgoType : uint8_t {
    MESH,
    MESH_2DIE,
    MESH_ONESHOT,
    MESH_TWOSHOT,
    MESH_CONCUR,
    MESH_MULTILINK,
    MESH_CHUNK,
    MESH_CHUNK_TWOSHOT,
    NHR,
    NHR_MULTILINK,
    NHR_AICPU_REDUCE,
    MESH_SINGLE_CHANNEL,
    MESH_CONCURRENT,
    NHR_MULTIJETTY,
    MESH_MULTIJETTY,
    UNKNOWN,
};

const std::set<AlgoType> MESH_ALGO_TYPES
    = {AlgoType::MESH,
       AlgoType::MESH_2DIE,
       AlgoType::MESH_ONESHOT,
       AlgoType::MESH_TWOSHOT,
       AlgoType::MESH_CONCUR,
       AlgoType::MESH_MULTILINK,
       AlgoType::MESH_CHUNK,
       AlgoType::MESH_CHUNK_TWOSHOT,
       AlgoType::MESH_SINGLE_CHANNEL,
       AlgoType::MESH_CONCURRENT,
       AlgoType::MESH_MULTIJETTY};
const std::set<AlgoType> NHR_ALGO_TYPES
    = {AlgoType::NHR, AlgoType::NHR_MULTILINK, AlgoType::NHR_AICPU_REDUCE, AlgoType::NHR_MULTIJETTY};
const std::set<AlgoType> MESH_CONCUR_ALGO_TYPES = {AlgoType::MESH_CONCUR, AlgoType::MESH_CONCURRENT};

} // namespace ops_hccl

#endif // OPS_HCCL_SRC_COMMON_ALG_PARSE
