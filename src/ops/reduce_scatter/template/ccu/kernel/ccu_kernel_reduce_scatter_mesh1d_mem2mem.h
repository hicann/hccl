/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_MEM2MEM
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_MEM2MEM

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr uint64_t REDUCE_MS_CNT = 8;
constexpr uint16_t REDUCE_SCATTER_GROUP_REDUCE_MAX_PIECE_CNT = 8;
constexpr uint16_t REDUCE_SCATTER_LOOP_COUNT = 16;
constexpr uint32_t RS_UNROLL_NUM = 16; // 最多支持8 * 16 = 128个rank

struct CcuKernelArgReduceScatterMesh1DMem2Mem: CcuKernelArgBase {
    uint64_t                                rankSize;
    uint32_t                                rankId;
    HcclReduceOp                            reduceOp;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
};

struct ReduceScatterMesh1DMem2MemContext: CcuKernelCtxBase {
    const CcuKernelArgReduceScatterMesh1DMem2Mem *arg;

    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;

    std::vector<ccu::Variable> input;
    std::vector<ccu::Variable> scratch;
    std::vector<ccu::Variable> token;
    ccu::Variable output;
    ccu::Variable currentRankSliceInputOffset;
    ccu::Variable currentRankSliceOutputOffset;
    ccu::Variable normalSliceSize;
    ccu::Variable lastSliceSize;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable scratchRepeatStride;
    ccu::Variable repeatNum;
    ccu::Variable flag;
    GroupOpSizeVars goSize;

    std::vector<ccu::Event> events;

    ccu::LocalAddr myInput;
    std::vector<ccu::RemoteAddr> remoteInput;
    std::vector<ccu::LocalAddr> scratchMem;

    // 3阶段设计新增变量
    ccu::Variable constVar1;
    ccu::Variable sliceSize;
    ccu::Variable readRepeatNum;
    ccu::Variable waitRepeatNum;

    // Loop机制相关变量
    std::array<std::vector<ccu::LocalAddr>, 2> loopScratch;
    ccu::LocalAddr loopSrc[2];  // 本rank的输入地址
    ccu::LocalAddr loopDst[2];
    ccu::Variable  loopLen[2];
    ccu::Variable  loopLenExp[2];
};

CcuResult CcuReduceScatterMesh1DMem2MemKernel(CcuKernelArg arg);

}// namespace ops_hccl
#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_MEM2MEM
