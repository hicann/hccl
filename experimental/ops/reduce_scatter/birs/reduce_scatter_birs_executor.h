/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_BIRS_EXECUTOR_H
#define REDUCE_SCATTER_BIRS_EXECUTOR_H
#include "reduce_scatter_executor_base.h"
#include "alg_template_base_experimental.h"
#include "coll_alg_exec_registry.h"

namespace ops_hccl_experimental {
using ops_hccl::AlgHierarchyInfo;
using ops_hccl::AlgResourceCtx;
using ops_hccl::AlgResourceRequest;
using ops_hccl::AlgTemplateBase;
using ops_hccl::AlgTemplateRegistry;
using ops_hccl::AlgType;
using ops_hccl::AlgTypeLevel0;
using ops_hccl::AlgTypeLevel1;
using ops_hccl::AlgTypeLevel2;
using ops_hccl::ChannelInfo;
using ops_hccl::CollAlgExecRegistry;
using ops_hccl::COMM_LEVEL0;
using ops_hccl::DefaultExecCreator;
using ops_hccl::ExecMem;
using ops_hccl::ExecutorBase;
using ops_hccl::HCCL_ALG;
using ops_hccl::OpParam;
using ops_hccl::Slice;
using ops_hccl::SubCommInfo;
using ops_hccl::TemplateType;
using ops_hccl::TopoInfo;

class ReduceScatterBIRSExecutor : public ReduceScatterExecutorBase {
public:
    explicit ReduceScatterBIRSExecutor();
    ~ReduceScatterBIRSExecutor() override = default;

    HcclResult CalcResRequest(
        HcclComm comm, const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
        AlgResourceRequest& resourceRequest, AlgType& algType) override;

private:
    HcclResult SelectAndPrepareBirsTemplate(
        bool isSingleServer, u32 localRank, u32 localRankSize, std::unique_ptr<AlgTemplateBase>& templatePtr);
    HcclResult KernelRunLevel0(const OpParam& param, ExecMem& execMem);
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
    /* *************** 算法参数 *************** */
    u32 subRoot_ = 0;
    u32 commIndex_ = 0;
    u32 perDataSize_ = 0;
    u64 level0SliceOffset_ = 0;
    u32 subUserRankRootSupperPod_ = 0;
    SubCommInfo level0CommInfo_;
};

} // namespace ops_hccl_experimental

#endif
