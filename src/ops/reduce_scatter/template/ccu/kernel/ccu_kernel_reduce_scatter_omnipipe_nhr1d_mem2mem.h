/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_OMNIPIPE_NHR1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_OMNIPIPE_NHR1D_MEM2MEM_H

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using NHRStepInfoRS = struct NHRStepInfoDefRS {
    uint32_t step = 0;
    uint32_t myRank = 0;
    uint32_t nSlices;
    uint32_t toRank = 0;
    uint32_t fromRank = 0;
    std::vector<uint32_t> txSliceIdxs;
    std::vector<uint32_t> rxSliceIdxs;

    NHRStepInfoDefRS() : nSlices(0) {}
};

struct CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem : public CcuKernelArgBase {
    uint64_t rankSize{0};
    uint32_t rankId{0};
    OpParam opParam;
    std::vector<NHRStepInfoRS> stepInfoVector;
    std::map<uint32_t, uint32_t> rank2ChannelIdx;
    std::vector<std::vector<uint32_t>> subCommRanks;
};

struct ReduceScatterOmniPipeNHR1DMem2MemContext : public CcuKernelCtxBase {
    const CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem* arg;

    uint64_t rankSize{0};
    uint32_t userRank{0};
    uint32_t rankId{0};
    uint32_t myRankIdx{0};
    uint32_t localSize{0}; // 本rank所在行或列的总rank数
    std::vector<NHRStepInfoRS> stepInfoVector;
    std::map<uint32_t, uint32_t> rank2ChannelIdx;
    std::vector<std::vector<uint32_t>> subCommRanks;

    HcclReduceOp reduceOp;
    HcclDataType dataType;
    HcclDataType outputDataType;

    ccu::Variable sliceStride;
    ccu::Variable localCopyFlag;
    std::vector<ccu::Variable> input;
    ccu::Variable output;
    std::vector<ccu::Variable> token;
    ccu::Variable sliceSize;
    ccu::Variable inputOmniPipeSliceStride;
    std::vector<ccu::Variable> inputOmniSliceStrideVec;
    std::vector<ccu::Variable> inputOmniSliceSizeVec;
    ccu::Variable inputSliceStride;
    ccu::Event event;
};

CcuResult CcuReduceScatterOmniPipeNHR1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl
#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_OMNIPIPE_NHR1D_MEM2MEM_H
