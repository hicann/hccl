/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_NHR_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_REDUCE_NHR_1D_MEM2MEM_H

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "ins_temp_all_reduce_nhr.h"

namespace ops_hccl {

struct CcuKernelArgReduceNHR1DMem2Mem: CcuKernelArgBase{
    uint64_t                                rankSize;
    uint32_t                                rankId;
    uint32_t                                rootId;
    uint32_t                                axisId;
    std::vector<NHRStepInfo>                stepInfoVector;
    std::map<u32, u32>                      rank2ChannelIdx;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
    uint32_t                                axisSize;
};

struct ReduceNHR1DMem2MemContext: CcuKernelCtxBase {
    const CcuKernelArgReduceNHR1DMem2Mem *arg;

    uint32_t localSize{0};
    uint32_t myRankIdx{0};
    HcclDataType dataType;
    HcclReduceOp reduceOp;
    ccu::Variable input;
    // 远端rank的output/token：用reserve+push_back(GetResByChannel)避免resize默认构造浪费寄存器
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    // 本rank的output/token：单独存储，用LoadArg加载
    ccu::Variable myOutput;
    ccu::Variable myToken;
    ccu::Variable isInputOutputEqual;
    ccu::Variable die0Size;
    ccu::Variable die1Size;
    ccu::Variable die0SliceSize;
    ccu::Variable die1SliceSize;
    ccu::Variable die0LastSliceSize;
    ccu::Variable die1LastSliceSize;
    ccu::Event event;
    std::vector<ccu::Variable> sliceOffset;

    ccu::LocalAddr localSrc;
    ccu::LocalAddr localDst;
    ccu::RemoteAddr remoteDst;
};

CcuResult CcuReduceNHR1DMem2MemKernel(CcuKernelArg arg);

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_REDUCE_NHR_1D_MEM2MEM_H
