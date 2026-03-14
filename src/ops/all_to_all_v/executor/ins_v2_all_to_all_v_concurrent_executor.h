/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_TO_ALL_V_CONCURRENT_EXECUTOR_H
#define HCCLV2_INS_V2_ALL_TO_ALL_V_CONCURRENT_EXECUTOR_H

#include "executor_common_ops.h"
#include "topo_match_base.h"
#include "topo_match_ubx.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsV2AllToAllVConcurrentExecutor : public InsCollAlgBase {
public:
    explicit InsV2AllToAllVConcurrentExecutor();
    ~InsV2AllToAllVConcurrentExecutor() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult CalcRes(HcclComm comm, const OpParam& param,
                       const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
                       AlgResourceRequest& resourceRequest) override;
    
    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                    AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

protected:
    HcclResult InitCommInfo(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo);
    HcclResult SetAlltoAllLocalSendRecvInfo(const OpParam &param);
    HcclResult SplitA2ASendRecvInfo(A2ASendRecvInfo &sendRecvInfoFirst, A2ASendRecvInfo &sendRecvInfoLast);
    HcclResult SetJettyNums(std::vector<uint32_t>& jettyNums, const bool multijetty) const;
    A2ASendRecvInfo localSendRecvInfo_;
};
}

#endif // HCCLV2_INS_V2_ALL_TO_ALL_V_CONCURRENT_EXECUTOR_H