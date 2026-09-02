/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_SCATTER_OMNIPIPE_NHR_1D_MEM2MEM_NEW
#define HCCL_CCU_KERNEL_SCATTER_OMNIPIPE_NHR_1D_MEM2MEM_NEW

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

    NHRStepInfo() : nSlices(0) {}
};
#endif

struct CcuKernelArgScatterOmniPipeNHR1DMem2Mem : CcuKernelArgBase {
    uint64_t rankSize;
    uint32_t rankId;
    uint32_t rootId;
    OpParam opParam;
    std::vector<std::vector<uint32_t>> subCommRanks;
    bool ifRealRoot;
    uint32_t myrealrank;
    std::vector<NHRStepInfo> stepInfoVector;
    std::map<u32, u32> rank2ChannelIdx;
};

struct ScatterOmniPipeNHR1DMem2MemContext {
    const CcuKernelArgScatterOmniPipeNHR1DMem2Mem* arg;

    uint64_t rankSize{0};
    uint32_t rankId{0};
    uint32_t rootId{0};
    bool ifRealRoot{false};
    uint32_t myrealrank{0};
    uint32_t myRankIdx{0};
    uint32_t localSize{0};
    HcclDataType dataType{HcclDataType::HCCL_DATA_TYPE_RESERVED};
    std::vector<NHRStepInfo> stepInfoVector;
    std::map<u32, u32> rank2ChannelIdx;

    ccu::Variable input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable sliceSize;
    ccu::Variable ifNewRoot;
    ccu::Variable isStepOne;
    ccu::Variable isLastStep;
    std::vector<ccu::Variable> outputOmniSliceStrideVec;
    std::vector<ccu::Variable> inputOmniSliceStrideVec;
    std::vector<ccu::Variable> inputOmniSliceSizeVec;

    ccu::Event event;
};

CcuResult CcuScatterOmniPipeNHR1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl
#endif // HCCL_CCU_KERNEL_SCATTER_OMNIPIPE_NHR_1D_MEM2MEM_NEW
