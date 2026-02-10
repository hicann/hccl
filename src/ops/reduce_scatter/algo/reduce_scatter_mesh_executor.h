/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_EXECUTOR_H
#define REDUCE_SCATTER_EXECUTOR_H

#include "alg_param.h"
#include "executor_base.h"
#include "coll_alg_exec_registry.h"

namespace ops_hccl {
class ReduceScatterMeshExecutor : public ExecutorBase {
public:
    explicit ReduceScatterMeshExecutor();
    ~ReduceScatterMeshExecutor() override {}

    HcclResult Orchestrate(const OpParam &param, AlgResourceCtx* resCtx) override;
    HcclResult CalcResRequest(HcclComm comm, const OpParam& param, TopoInfo* topoInfo,
        AlgHierarchyInfo& algHierarchyInfo, AlgResourceRequest& resourceRequest, AlgType& algType) override;
    
protected:
    HcclResult GetSubCommInfoA5(const CommPlane levelIndex, SubCommInfo &info);
    HcclResult RunLoop(const OpParam &param);

    ThreadHandle thread_ = 0;
    std::vector<ThreadHandle> slaveThreads_;
    std::vector<std::vector<ChannelInfo>> channels_;
    u32 unitSize_;
};
}
#endif