/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_EXECUTOR_H
#define REDUCE_SCATTER_EXECUTOR_H

#include "alg_param.h"
#include "executor_base.h"
#include "config_log.h"
#include "alg_template_base_experimental.h"

namespace ops_hccl_experimental {
using ops_hccl::AlgHierarchyInfo;
using ops_hccl::AlgResourceCtx;
using ops_hccl::AlgResourceRequest;
using ops_hccl::AlgType;
using ops_hccl::ChannelInfo;
using ops_hccl::ExecutorBase;
using ops_hccl::OpParam;
using ops_hccl::Slice;
using ops_hccl::TopoInfo;

class ReduceScatterExecutorBase : public ExecutorBase {
public:
    explicit ReduceScatterExecutorBase();
    ~ReduceScatterExecutorBase() override = default;

    HcclResult Orchestrate(const OpParam& param, AlgResourceCtx* resCtx) override;

    /* *************** 资源计算 *************** */
    HcclResult CalcResRequest(
        HcclComm comm, const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
        AlgResourceRequest& resourceRequest, AlgType& algType) override;

protected:
    // 用于需要Loop的Executor
    virtual HcclResult RunLoop(const OpParam& param);

    /* *************** 通用工具 *************** */
    virtual HcclResult PrepareDataSlice(u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice>& dataSlice);
    bool IsHugeData(u64 curSize) const;

    ThreadHandle thread_ = 0;
    std::vector<ThreadHandle> slaveThreads_;
    std::vector<std::vector<ChannelInfo>> channels_;
    OpParam param_;
    u32 unitSize_;
    u32 root_ = INVALID_VALUE_RANKID;
};

} // namespace ops_hccl_experimental

#endif
