/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_1D_MULTI_JETTY_H_
#define HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_1D_MULTI_JETTY_H_

#include <vector>
#include <ios>
#include "utils.h"
#include "alg_param.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "template_utils.h"

namespace ops_hccl {

struct CcuKernelArgAllToAllVMesh1DMultiJetty : CcuKernelArgBase {
    uint64_t rankSize{0};
    uint32_t rankId{0};
    OpParam opParam;
    std::vector<std::vector<uint32_t>> subCommRanks;
    std::vector<uint32_t> jettyNums;
};

struct AllToAllVMesh1DMultiJettyContext : public CcuKernelCtxBase {
    const CcuKernelArgAllToAllVMesh1DMultiJetty *arg;

    // a2a 对每个对端的发送接收信息
    struct A2ASingleSendRecvInfo {
        ccu::Variable sliceSize;               // 数据块切片大小
        ccu::Variable tailSliceSize;           // 数据块切片尾片大小
        ccu::Variable lastSliceSize;           // 尾数据块切片大小
        ccu::Variable lastTailSliceSize;       // 尾数据块切片尾片大小
        ccu::Variable loopNum;
        ccu::Variable sendOffset;
        ccu::Variable recvOffset;
        GroupOpSizeVars tailGoSize;
    };

    std::vector<ccu::Variable> input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable srcOffset;
    ccu::Variable dstOffset;
    std::vector<A2ASingleSendRecvInfo> sendRecvInfo;

    ccu::Variable completedRankCount;
    ccu::Variable xnMaxTransportSize;
    GroupOpSizeVars xnMaxTransportGoSize; // 本地ccu的尾块
    ccu::Variable xnConst1;

    std::vector<ccu::LocalAddr> src;
    ccu::LocalAddr myDst;
    std::vector<ccu::RemoteAddr> remoteDst;
    std::vector<ccu::Event> eventList;
};

CcuResult CcuAllToAllVMesh1DMultiJettyKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_1D_MULTI_JETTY_H_