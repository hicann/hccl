/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_BROADCAST_NHR_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_BROADCAST_NHR_1D_MEM2MEM_H

#include <memory>
#include <map>
#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

#ifndef NHR_STEP_INFO_DEFINED
#define NHR_STEP_INFO_DEFINED
using NHRStepInfo = struct NHRStepInfoDef {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    NHRStepInfoDef() : nSlices(0)
    {
    }
};
#endif

struct CcuKernelArgBroadcastNhr1DMem2Mem : CcuKernelArgBase {
    uint32_t                                rankId;
    uint32_t                                axisId;
    uint32_t                                axisSize;
    uint64_t                                dimSize;
    std::vector<NHRStepInfo>                stepInfoVector;
    std::map<u32, u32>                      rank2ChannelIdx;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
};

struct BroadcastNhr1DMem2MemContext : CcuKernelCtxBase {
    const CcuKernelArgBroadcastNhr1DMem2Mem *arg;

    uint64_t localSize;
    uint64_t myRankIdx;
    ccu::Variable input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable die0Size;
    ccu::Variable die1Size;
    ccu::Variable die0SliceSize;
    ccu::Variable die1SliceSize;
    ccu::Variable die0LastSliceSize;
    ccu::Variable die1LastSliceSize;
    std::vector<ccu::Variable> sliceOffset;

    ccu::LocalAddr localSrc;
    ccu::LocalAddr localDst;
    ccu::RemoteAddr remoteDst;
    ccu::Event event;
};

CcuResult CcuBroadcastNhr1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_BROADCAST_NHR_1D_MEM2MEM_H
