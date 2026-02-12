/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INS_SEND_EXECUTOR_H
#define HCCL_INS_SEND_EXECUTOR_H

#include "alg_param.h"
#include "topo_host.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"

namespace ops_hccl {
    class InsSendExecutor : public InsCollAlgBase {
    public:
        std::string Describe() const override;

        // 算法编排
        HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

        HcclResult CalcAlgHierarchyInfo(
            HcclComm comm, TopoInfo *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

        // 资源计算
        HcclResult CalcRes(
            HcclComm comm, const OpParam &param, const TopoInfo *topoInfo,
            const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest) override;

    protected:
        HcclResult InitCommInfo(
            HcclComm comm, const OpParam &param, const TopoInfo *topoInfo,
            const AlgHierarchyInfoForAllLevel &algHierarchyInfo);

        // 单算子还是图模式
        OpMode opMode_;
        u32 remoteRank_;
        std::vector<ThreadHandle> threads_;
        // 一次搬运最大数据量
        u64 maxLoopTransSize_;
        // 一次搬运最大数据个数
        u64 maxLoopTransCount_;
    };
} // namespace ops_hccl

#endif  // #ifndef HCCL_INS_SEND_EXECUTOR_H
