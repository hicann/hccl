/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_EXECUTOR_H
#define SCATTER_EXECUTOR_H

#include "alg_param.h"
#include "executor_base.h"
#include "config_log.h"

namespace ops_hccl {
class ScatterExecutorBase : public ExecutorBase {
public:
    explicit ScatterExecutorBase();
    ~ScatterExecutorBase() override = default;

    HcclResult Orchestrate(const OpParam &param, AlgResourceCtx* resCtx) override;

    /* *************** 资源计算 *************** */
    HcclResult CalcResRequest(HcclComm comm, const OpParam& param, TopoInfo* topoInfo,
        AlgHierarchyInfo& algHierarchyInfo, AlgResourceRequest& resourceRequest, AlgType& algType) override;

protected:
    /* *************** 算法编排 *************** */
    // 按Level0、Level1、Level2可继续进行拆分。
    virtual HcclResult KernelRunLevel1(HcclMem &inputMem, u64 count, HcclDataType dataType, u32 &commIndex,
        u32 root, u32 &subRoot, CommPlane commLevel, ThreadHandle thread);
    // 用于需要Loop的Executor
    virtual HcclResult RunLoop(const OpParam &param);

    /* *************** 通用工具 *************** */
    virtual HcclResult PrepareDataSlice(u64 dataCount, u32 unitSize, u32 sliceNum,
        std::vector<Slice> &dataSlice);
    bool IsHugeData(u64 curSize);

    bool DMAReduceFlag_{false}; // 是否DMA消减的标志

    ThreadHandle thread_ = 0;
    std::vector<ThreadHandle> slaveThreads_;
    std::vector<std::vector<ChannelInfo>> channels_;
    OpParam param_;
    u32 unitSize_;
    u32 root_ = INVALID_VALUE_RANKID;
};

void GetSubRootRank(u32 root, u32 curLevel, AlgHierarchyInfo& algHierarchyInfo, u32 &subRootRank);

}

#endif
