/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_MUTILJETTY_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_MUTILJETTY_1D_MEM2MEM_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
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

namespace ops_hccl {

struct CcuKernelArgReduceScatterNhrMutilJettyMem2Mem1D : public CcuKernelArgBase {
    uint64_t dimSize{0};
    uint32_t rankId{0};
    uint16_t portNum{0};
    OpParam opParam;
    std::vector<NHRStepInfo> stepInfoVector;
    std::map<u32, u32> rank2ChannelIdx;
    std::vector<std::vector<uint32_t>> subCommRanks;
};

struct ReduceScatterNhrMem2Mem1DMultiJettyContext : public CcuKernelCtxBase {
    const CcuKernelArgReduceScatterNhrMutilJettyMem2Mem1D *arg;

    uint64_t dimSize{0};
    uint32_t rankId{0};
    uint32_t localSize{0};
    uint32_t myRankIdx{0};
    uint32_t portNum{0};
    HcclReduceOp reduceOp;
    HcclDataType dataType;
    HcclDataType outputDataType;
    std::vector<NHRStepInfo> stepInfoVector;
    std::map<u32, u32> rank2ChannelIdx;

    std::vector<ccu::Variable> input;
    ccu::Variable output;
    std::vector<ccu::Variable> token;
    ccu::Variable sliceSize;
    ccu::Variable inputSliceStride;
    ccu::Variable outputSliceStride;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable sliceOneJettySize;
    ccu::Variable sliceLastJettySize;
    ccu::Variable repeatNumVar;
    ccu::Variable repeatNumVarTemp;

    std::vector<ccu::Event> jettyEvent;
    ccu::Event event;
    ccu::LocalAddr localSrc;
    ccu::LocalAddr localDst;
    ccu::RemoteAddr remoteDst;
    ccu::Variable flag;
};

CcuResult CcuReduceScatterNhrMem2Mem1DMultiJettyKernel(CcuKernelArg arg);

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_REDUCE_SCATTER_NHR_1D_MUTIL_JETTY_MEM2MEM_H