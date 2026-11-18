/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_KERNEL_ALL_TO_ALL_HIER_LAST_STAGE_H_
#define HCCLV2_CCU_KERNEL_ALL_TO_ALL_HIER_LAST_STAGE_H_

#include <vector>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using RankId = u32;

/*
 * 二级序列 AllToAll 末 Stage（框内/板内组）kernel 参数。
 * 布局：cclBuff 槽位号 = 板内序号*srcSliceNum + 源板序号，对端 q 取槽位列 [q*srcSliceNum, (q+1)*srcSliceNum)
 * 直写对端 OUTPUT 对应源 rank 块；channels 与板内组升序对齐（不含自身）。
 */
struct CcuKernelArgAllToAllHierLastStage : CcuKernelArgBase {
    uint32_t rankId;               // 全局 rank
    uint32_t myAlgRank;            // 本 rank 在板内组内的序号 s
    uint32_t groupSize;            // 板内组大小（板内 rank 数 d0）
    uint32_t srcSliceNum;          // 每对端承载的槽位数（板数 d1）
    std::vector<RankId> rankGroup; // 预留、当前 kernel 未读取
};

struct AllToAllHierLastStageContext : CcuKernelCtxBase {
    const CcuKernelArgAllToAllHierLastStage* arg;

    ccu::Variable localCclBuff; // 本地 cclBuff 基址
    ccu::Variable output;       // 本地 OUTPUT 基址（PreSync 广播）
    ccu::Variable myToken;
    std::vector<ccu::Variable> remoteOutput; // 对端 OUTPUT 基址（PreSync 交换）
    std::vector<ccu::Variable> peerToken;
    ccu::Variable sliceSize;       // 单槽本轮搬运大小
    ccu::Variable hcclBuffBaseOff; // cclBuff 槽内本轮窗口偏移
    ccu::Variable outBuffBaseOff;  // OUTPUT 块内本轮窗口偏移
    ccu::Variable blockStride;     // OUTPUT 块步长 = totalCount * dataTypeSize
    ccu::Variable slotStride;      // cclBuff 槽位步长
    GroupOpSizeVars groupOpSize;   // 本地份 GroupCopy 流水参数
    ccu::Variable offScratch;
    ccu::LocalAddr tmpSrc;
    ccu::LocalAddr tmpDst;

    CcuEventGroup eventGroup;
};

CcuResult CcuAllToAllHierLastStageKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCLV2_CCU_KERNEL_ALL_TO_ALL_HIER_LAST_STAGE_H_
