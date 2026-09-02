/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_BROADCAST_MESH_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_BROADCAST_MESH_1D_MEM2MEM_H

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgBroadcastMesh1DMem2Mem : CcuKernelArgBase {
    uint64_t rankSize;
    uint32_t rankId;
    uint32_t rootId;
    OpParam opParam;
    std::vector<std::vector<uint32_t>> subCommRanks;
};

struct BroadcastMesh1DMem2MemContext : CcuKernelCtxBase {
    const CcuKernelArgBroadcastMesh1DMem2Mem* arg;

    // 本 rank 资源（用 LoadArg 加载）：单独存储避免放入 vector 末尾造成 resize+1 浪费
    ccu::Variable myInput;
    ccu::Variable myOutput;
    ccu::Variable myToken;
    // 远端 rank 的 output/token：用 reserve+push_back(GetResByChannel) 避免 resize 默认构造浪费寄存器
    // vector 大小 = rankSize - 1，按 peerId 升序跳过 rankId，访问时用 vecIdx = (peerId < rankId) ? peerId : peerId - 1
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable currentRankSliceInputOffset;
    ccu::Variable currentRankSliceOutputOffset;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable normalSliceSize;
    ccu::Variable lastSliceSize;
    ccu::Variable allgatherOffset;
    ccu::Variable repeatNumVar;
    ccu::Variable flag;

    CcuEventGroup eventGroup;
    ccu::LocalAddr myScatterDst;
    ccu::LocalAddr allgatherSrc;
    std::vector<ccu::LocalAddr> scattersrcMem;
    // scatterdstMem 与 allgatherdstMem 复用同一寄存器集：两者都是 vector<RemoteAddr>，且时间上不重叠
    //   - DoRepeaScatterMem2Mem 构造后立即在 DoScatter 中使用，DoScatter 完成后不再访问
    //   - DoRepeatAllGatherMem2Mem 重新构造后立即在 DoAllGather 中使用，覆盖上一阶段的值
    // 64p 场景节省 1 个 vector<RemoteAddr> = 192 个寄存器
    std::vector<ccu::RemoteAddr> remoteDstMem;
};

CcuResult CcuBroadcastMesh1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_BROADCAST_MESH_1D_MEM2MEM_H
