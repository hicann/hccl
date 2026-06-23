/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH1D_2DIE_H
#define HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH1D_2DIE_H

#include <vector>
#include "utils.h"
#include "template_utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using RankId = u32;

struct CcuKernelArgAllToAllVMesh1D2Die : CcuKernelArgBase {
    uint32_t rankId;
    OpParam opParam;
    std::vector<std::vector<RankId>> subCommRanks;
    bool withMyRank;
    std::vector<RankId> rankGroup;
};

struct A2AVSingleSendRecvInfoCtx1D2Die {
    ccu::Variable sendOffset;
    ccu::Variable recvOffset;
    ccu::Variable sendTailSize;
    GroupOpSizeVars sendTailGoSize;
    ccu::Variable sendLoopNum;
};

struct AllToAllVMesh1D2DieContext : CcuKernelCtxBase {
    const CcuKernelArgAllToAllVMesh1D2Die *arg;

    const uint64_t MAX_TRANSPORT_SIZE = UB_MAX_TRANS_SIZE;

    uint32_t localId{0};
    uint32_t peerSize{0};

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
    std::vector<A2AVSingleSendRecvInfoCtx1D2Die> sendRecvInfo;

    std::vector<ccu::Event> events;
};

CcuResult CcuAlltoAllVMesh1D2DieKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH1D_2DIE_H
