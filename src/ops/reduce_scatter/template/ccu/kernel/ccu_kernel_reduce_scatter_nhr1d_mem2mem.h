/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_1D_MEM2MEM
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_1D_MEM2MEM

#include <vector>
#include <ios>
#include <map>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "ins_temp_all_reduce_nhr.h"

namespace ops_hccl {

struct CcuKernelArgReduceScatterNHR1D: CcuKernelArgBase {
    uint64_t                                dimSize;
    uint32_t                                rankId;
    uint32_t                                mySubCommRankId;
    uint32_t                                axisId;
    uint32_t                                axisSize;
    std::vector<NHRStepInfo>                stepInfoVector;
    std::map<u32, u32>                      rank2ChannelIdx;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
};

struct ReduceScatterNHR1DMem2MemContext: CcuKernelCtxBase {
    const CcuKernelArgReduceScatterNHR1D *arg;

    uint64_t dimSize;
    uint32_t mySubCommRankId;
    uint32_t axisId;
    uint32_t localSize;
    uint32_t myRankIdx;
    uint32_t axisSize;
    HcclReduceOp reduceOp;
    HcclDataType dataType;
    HcclDataType outputDataType;

    std::vector<ccu::Variable> input;
    ccu::Variable output;
    std::vector<ccu::Variable> token;
    ccu::Variable die0Size;
    ccu::Variable die1Size;
    ccu::Variable die0LastSliceSize;
    ccu::Variable die1LastSliceSize;
    ccu::Variable inputSliceStride;
    ccu::Variable currentRankSliceOutputOffset;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable repeatNumVar;
    ccu::Variable repeatNumVarTemp;
    ccu::Variable isInputOutputEqual;
    ccu::Variable sliceSize;

    ccu::Event event;

    ccu::LocalAddr localSrc;
    ccu::LocalAddr localDst;
    ccu::RemoteAddr remoteDst;
    ccu::Variable isRepeatIter;
    std::vector<ccu::Variable> sliceOffset;
};

CcuResult CcuReduceScatterNHR1DMem2MemKernel(CcuKernelArg arg);

}// namespace ops_hccl
#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_1D_MEM2MEM
