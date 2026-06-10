/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALLREDUCE_NHR_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_ALLREDUCE_NHR_1D_MEM2MEM_H
#include <vector>
#include <ios>
#include <map>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "ins_temp_all_reduce_nhr.h"

namespace ops_hccl {

struct CcuKernelArgAllReduceNHR1D : CcuKernelArgBase {
    uint64_t                         rankSize;
    uint32_t                         rankId;
    uint32_t                         axisId;
    uint32_t                         axisSize;
    std::vector<NHRStepInfo>         stepInfoVector;
    std::map<u32, u32>               indexMap;
    OpParam                          opParam;
    std::vector<std::vector<uint32_t>> tempVTopo;
};

struct AllReduceNHR1DContext: CcuKernelCtxBase {
    const CcuKernelArgAllReduceNHR1D *arg;

    // 构造函数中
    uint32_t rankId{0};
    uint64_t rankSize{0};
    uint32_t axisId{0};
    uint32_t axisSize{0};
    uint32_t localSize{0};  // 本rank所在行或列的总rank数
    uint32_t myRankIdx{0};
    uint32_t repeatNum{0};
    HcclDataType dataType;
    HcclReduceOp reduceOp;
    std::vector<NHRStepInfo> stepInfoVector;
    std::map<u32, u32> indexMap;

    // load进来参数
    ccu::Variable input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable isInputOutputEqual;
    ccu::Variable die0Size;
    ccu::Variable die1Size;
    ccu::Variable sliceSize;
    ccu::Variable die0SliceSize;
    ccu::Variable die1SliceSize;
    ccu::Variable die0LastSliceSize;
    ccu::Variable die1LastSliceSize;

    ccu::Event             localEvent;
    std::vector<ccu::Variable> sliceOffset;

    ccu::LocalAddr srcMem;
    ccu::RemoteAddr rmtDstMem;
    ccu::LocalAddr locDstMem;
};

CcuResult CcuAllReduceNHR1DKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_REDUCE_NHR_1D_H_