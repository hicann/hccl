/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_KERNEL_ALL_TO_ALL_HIER_FIRST_STAGE_H_
#define HCCLV2_CCU_KERNEL_ALL_TO_ALL_HIER_FIRST_STAGE_H_

#include <vector>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using RankId = u32;

/*
 * 二级序列 AllToAll 首 Stage（框间代表组）kernel 参数。
 * 布局：INPUT 块号 = 目的 rank，代表组成员 p 承载块 [p*peerSliceNum, (p+1)*peerSliceNum)；
 * cclBuff 槽位号 = 板内序号*groupSize + srcAlgRank，列跨度 = groupSize*slotStride；channels
 * 与代表组升序对齐（不含自身）。
 */
struct CcuKernelArgAllToAllHierFirstStage : CcuKernelArgBase {
    uint32_t rankId;               // 全局 rank
    uint32_t myAlgRank;            // 本 rank 在代表组内的序号
    uint32_t groupSize;            // 代表组大小（板数 d1）
    uint32_t peerSliceNum;         // 每对端承载的块数（板内 rank 数 d0）
    std::vector<RankId> rankGroup; // 预留、当前 kernel 未读取
};

struct AllToAllHierFirstStageContext : CcuKernelCtxBase {
    const CcuKernelArgAllToAllHierFirstStage* arg;

    ccu::Variable input;        // INPUT 基址
    ccu::Variable localCclBuff; // 本地 cclBuff 基址（PreSync 广播）
    ccu::Variable myToken;
    std::vector<ccu::Variable> remoteCclBuff; // 对端 cclBuff 基址（PreSync 交换）
    std::vector<ccu::Variable> peerToken;     // 对端 token
    ccu::Variable segSize;                    // 远端分段大小（2+6 时 MAJOR/MINOR 按带宽比切分）
    ccu::Variable sliceOffset;                // 分段在本轮 slice 内的偏移
    ccu::Variable inBuffBaseOff;              // INPUT 块内本轮窗口偏移
    ccu::Variable hcclBuffBaseOff;            // cclBuff 槽内本轮窗口偏移
    ccu::Variable blockStride;                // INPUT 块步长 = totalCount * dataTypeSize
    ccu::Variable slotStride;                 // cclBuff 单槽步长
    GroupOpSizeVars groupOpSize;              // 本地份 GroupCopy 流水参数
    ccu::Variable offScratch;                 // 块/槽偏移暂存
    ccu::LocalAddr tmpSrc;
    ccu::LocalAddr tmpDst;

    CcuEventGroup eventGroup;
};

CcuResult CcuAllToAllHierFirstStageKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCLV2_CCU_KERNEL_ALL_TO_ALL_HIER_FIRST_STAGE_H_
