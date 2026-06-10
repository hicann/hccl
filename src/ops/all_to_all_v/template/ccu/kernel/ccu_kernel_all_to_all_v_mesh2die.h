/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_2DIE_H
#define HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_2DIE_H

#include <vector>
#include <ios>
#include "utils.h"
#include "template_utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using RankId = u32;

struct CcuKernelArgAllToAllVMesh2Die : CcuKernelArgBase {
    uint32_t rankId;
    OpParam opParam;
    std::vector<std::vector<RankId>> subCommRanks;
    bool withMyRank;
    std::vector<RankId> rankGroup;
};

struct A2AVSingleSendRecvInfoCtx {
    ccu::Variable sendOffset;
    ccu::Variable recvOffset;
    ccu::Variable sendTailSize;
    GroupOpSizeVars sendTailGoSize;
    ccu::Variable sendLoopNum;
};

struct AllToAllVMesh2DieContext : CcuKernelCtxBase {
    const CcuKernelArgAllToAllVMesh2Die *arg;

    const uint32_t RANK_EVEN = 2;

    const uint32_t GO_ADDR_OFFSET_IDX = 0;
    const uint32_t GO_LOOP_PARAM_IDX = 1;
    const uint32_t GO_PARALLEL_PARAM_IDX = 2;
    const uint32_t GO_RESIDUAL_IDX = 3;

    const uint64_t MAX_TRANSPORT_SIZE = UB_MAX_TRANS_SIZE;

    uint32_t localSize{0};
    uint32_t localId{0};
    uint32_t peerSize{0};
    uint32_t logicId{0};

    ccu::Variable input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;

    ccu::LocalAddr localSrc;
    ccu::LocalAddr localDst;

    std::vector<ccu::LocalAddr> src;
    std::vector<ccu::RemoteAddr> dst;

    ccu::Variable xnConst1;
    ccu::Variable completedRankCount;
    ccu::Variable xnMaxTransportSize;
    GroupOpSizeVars xnMaxTransportGoSize;
    ccu::Variable curSendTailSize;
    GroupOpSizeVars curSendTailGoSize;
    std::vector<A2AVSingleSendRecvInfoCtx> sendRecvInfo;

    uint16_t selfBit{0};
    uint16_t allBit{0};

    ccu::Event event;
};

CcuResult CcuAllToAllVMesh2DieKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_2DIE_H
