/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_SCATTER_NHR_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_SCATTER_NHR_1D_MEM2MEM_H

#include <vector>
#include <ios>
#include <map>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "template_utils.h"

namespace ops_hccl {

#ifndef NHR_STEP_INFO_DEFINED
#define NHR_STEP_INFO_DEFINED
using NHRStepInfo = struct NHRStepInfo {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    NHRStepInfo() : nSlices(0)
    {
    }
};
#endif

struct CcuKernelArgScatterNHRMem2Mem1D : CcuKernelArgBase {
    uint64_t rankSize;
    uint32_t rankId;
    uint32_t rootId;
    uint32_t axisId;
    uint32_t axisSize;
    std::vector<NHRStepInfo> stepInfoVector;
    std::map<u32, u32> rank2ChannelIdx;
    OpParam opParam;
    std::vector<std::vector<uint32_t>> subCommRanks;
};

struct ScatterNHR1DContext {
    const CcuKernelArgScatterNHRMem2Mem1D *arg;

    uint64_t rankSize{0};
    uint32_t rankId{0};
    uint32_t rootId{0};
    uint32_t axisId{0};
    uint32_t axisSize{0};
    uint32_t localSize{0};
    uint32_t myRankIdx{0};
    uint32_t signalNum{0};
    HcclDataType dataType{HcclDataType::HCCL_DATA_TYPE_RESERVED};
    std::vector<NHRStepInfo> stepInfoVector;
    std::map<u32, u32> rank2ChannelIdx;

    // args
    ccu::Variable input;
    ccu::Variable output;
    std::vector<ccu::Variable> scratch;
    std::vector<ccu::Variable> token;
    ccu::Variable die0Size;
    ccu::Variable die1Size;
    ccu::Variable inputSliceStride;
    ccu::Variable outputSliceStride;
    ccu::Variable curScratchStride;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable repeatNumVar;
    ccu::Variable isOutputScratch;
    ccu::Variable isInputOutputEqual;
    ccu::Variable die0TailSize;
    ccu::Variable die1TailSize;
    ccu::Variable isSliceSizeZero;

    // temps
    ccu::Variable repeatNumVarTemp;
    ccu::Variable repeatTimeFlag;
    std::vector<ccu::Variable> inputOffset;
    std::vector<ccu::Variable> scratchOffset;
    ccu::Variable curInputOffset;
    ccu::Variable curScratchOffset;
    ccu::Variable curSliceSize;
    ccu::LocalAddr srcMem;
    ccu::LocalAddr dstMem;
    ccu::RemoteAddr dstRemoteMem;
    ccu::Event event;
};

CcuResult CcuScatterNHR1DMem2MemKernel(CcuKernelArg arg);

}  // namespace ops_hccl
#endif  // HCCL_CCU_KERNEL_SCATTER_NHR_1D_MEM2MEM_H
