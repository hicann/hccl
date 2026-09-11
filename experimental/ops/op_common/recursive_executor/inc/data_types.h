/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <vector>
#include "alg_param.h"
#include "template_utils.h"

namespace ops_hccl {
enum class TransferDirection : uint8_t {
    WRITE = 0,
    READ = 1,
};

struct DataSlicesList {
    SlicesList txSlicesList_;
    SlicesList rxSlicesList_;
    u32 srcRankId_ = INVALID_VALUE_RANKID;
    u32 dstRankId_ = INVALID_VALUE_RANKID;

    DataSlicesList() : txSlicesList_({}, {}), rxSlicesList_({}, {}) {}

    DataSlicesList(
        const SlicesList& txSlicesList, const SlicesList& rxSlicesList, u32 srcRankId = INVALID_VALUE_RANKID,
        u32 dstRankId = INVALID_VALUE_RANKID)
        : txSlicesList_(txSlicesList),
          rxSlicesList_(rxSlicesList),
          srcRankId_(srcRankId),
          dstRankId_(dstRankId)
    {}
};

struct DataParams {
    void* inputBufferPtr = nullptr;
    void* outputBufferPtr = nullptr;
    void* cclBufferPtr = nullptr;
    BufferType inputBufferType = BufferType::INPUT;
    BufferType outputBufferType = BufferType::OUTPUT;
    BufferType cclBufferType = BufferType::HCCL_BUFFER;
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    u64 dataOffset{0};
    u64 sliceCount{0};
    u64 sliceOffset{0};
    u64 tailCount{0};
    u32 globalTailRankId{INVALID_VALUE_RANKID};
    u64 dataStride{0};
    u64 scratchStride{0};
    HcclReduceOp reduceOp{HCCL_REDUCE_RESERVED};
    u32 root{INVALID_VALUE_RANKID};

    bool enableRemoteMemAccess{false};

    std::vector<u32> ranksForInputData;

    char algTag[ALG_TAG_LENGTH] = "";
};

} // namespace ops_hccl

#endif // DATA_TYPES_H
