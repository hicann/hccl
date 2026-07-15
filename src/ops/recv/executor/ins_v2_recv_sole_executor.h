/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_RECV_EXECUTOR_SOLE_H
#define HCCLV2_INS_V2_RECV_EXECUTOR_SOLE_H

#include "alg_param.h"
#include "topo_host.h"
#include "channel.h"
#include "sal.h"
#include "utils.h"
#include "log.h"
#include "workflow.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "alg_v2_template_base.h"

namespace ops_hccl {
template <typename InsAlgTemplate>
class InsV2RecvSoleExecutor : public InsCollAlgBase {
public:
    explicit InsV2RecvSoleExecutor();
    ~InsV2RecvSoleExecutor() = default;
    std::string Describe() const override;
    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;
    HcclResult OrchestrateWithThread(
            const OpParam &param, const AlgResourceCtxSerializable &resCtx, ThreadHandle sendRecvThread) override;

    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

protected:
    u32 sendRank_ = 0;
    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    ChannelInfo recvChannel_;
    ThreadHandle thread_;  // 只涉及一个thread
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
};
}  // namespace ops_hccl

#endif