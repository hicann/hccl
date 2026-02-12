/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_RING_EXECUTOR_H
#define SCATTER_RING_EXECUTOR_H
#include "scatter_executor_base.h"
#include "coll_alg_exec_registry.h"

namespace ops_hccl {
class ScatterRingExecutor : public ScatterExecutorBase {
public:
    explicit ScatterRingExecutor();
    ~ScatterRingExecutor() override = default;

    HcclResult CalcResRequest(HcclComm comm, const OpParam& param, TopoInfo* topoInfo,
        AlgHierarchyInfo& algHierarchyInfo, AlgResourceRequest& resourceRequest, AlgType& algType) override;

private:
    /* *************** 算法编排 *************** */
    HcclResult KernelRunLevel2(const OpParam &param, ExecMem &execMem);
    HcclResult KernelRunLevel1(const OpParam &param, ExecMem &execMem);
    HcclResult KernelRunLevel0(const OpParam &param, ExecMem &execMem);
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;

    // 多ring相关
    HcclResult MultiRingScatter(HcclMem inputMem, HcclMem outputMem, const u64 count,
        const HcclDataType dataType, const std::vector<Slice> &dataSegsSlice, u32 root, const HcomCollOpInfo *opInfo,
        const u64 baseOffset);
    HcclResult MutliSegSlicePrepare(const std::vector<Slice> &dataSegsSlice, u32 ringNum,
        std::vector<std::vector<Slice>> &mutliSegsSlices);
    HcclResult PrepareMultiRingSlice(const std::vector<Slice> &dataSegsSlice, u32 &ringNum,
        std::vector<std::vector<Slice>> &mutliRingsSlices, std::vector<std::vector<u32>> &rankOrders);

    /* *************** 算法参数 *************** */
    u32 subRoot_ = 0;
    u32 commIndex_ = 0;
    u32 perDataSize_ = 0;
    u64 level1SliceOffset_ = 0;
    u64 level0SliceOffset_ = 0;
    u32 subUserRankRootSupperPod_ = 0;
    SubCommInfo level0CommInfo_;
    SubCommInfo level1CommInfo_;
    SubCommInfo level2CommInfo_;
};

}

#endif