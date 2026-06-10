/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_REDUCE_NHR_1D_MEM2MEM_MULTI_JETTY_H
#define HCCL_CCU_KERNEL_ALL_REDUCE_NHR_1D_MEM2MEM_MULTI_JETTY_H

#include <vector>
#include <ios>
#include <map>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "ins_temp_all_reduce_nhr.h"

namespace ops_hccl {

struct CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty : CcuKernelArgBase {
    uint32_t rankSize{0};
    uint32_t rankId{0};
    uint32_t portNum{0};
    OpParam opParam;
    std::vector<NHRStepInfo> algStepInfoList;
    std::map<u32, u32> channelIdxMap;
    std::vector<std::vector<uint32_t>> subCommRanks;
};

struct AllReduceNhrMem2Mem1DMultiJettyContext : CcuKernelCtxBase {
    const CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty *arg;

    uint32_t rankSize{0};
    uint32_t rankId{0};
    uint32_t portNum{0};
    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;
    std::vector<NHRStepInfo> algStepInfoList;
    std::map<u32, u32> channelIdxMap;

    ccu::Variable inputAddr;
    std::vector<ccu::Variable> outputAddrs;
    std::vector<ccu::Variable> outputTokens;
    ccu::Variable isInplace;
    ccu::Variable dataSize;
    ccu::Variable dataSizePerRank;
    ccu::Variable dataSizePerPort;
    ccu::Variable lastRankSliceSize;
    ccu::Variable lastPortSliceSize;
    std::vector<ccu::Variable> sliceOffset;
    GroupOpSizeVars localCopyGoSize;
    GroupOpSizeVars localCopyGoSizeLastSlice;

    ccu::LocalAddr localInput;
    ccu::LocalAddr localOutput;
    ccu::RemoteAddr remoteOutput;

    std::vector<ccu::Event> events;
};

CcuResult CcuAllReduceNhrMem2Mem1DMultiJettyKernel(CcuKernelArg arg);

} // namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_MEM2MEM_H