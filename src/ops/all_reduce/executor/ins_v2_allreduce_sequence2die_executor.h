/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef HCCLV2_INS_V2_ALLREDUCE_SEQUENCE2DIE_EXECUTOR_H
#define HCCLV2_INS_V2_ALLREDUCE_SEQUENCE2DIE_EXECUTOR_H
 
#include "alg_param.h"
#include "topo_host.h"
#include "channel.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "log.h"
#include "workflow.h"
#include "sal.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "topo_match_base.h"
#include "topo_match_1d.h"


namespace ops_hccl {
struct SplitSliceInfo {
    u64 offset{0};
    u64 size{0};
    u64 count{0};
 
    SplitSliceInfo(const u64 offset, const u64 size, const u64 count) 
    : offset(offset), size(size), count(count) {}
};

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsV2AllReduceSequence2DieExecutor : public InsCollAlgBase {
public:
    explicit InsV2AllReduceSequence2DieExecutor();
    ~InsV2AllReduceSequence2DieExecutor() override;
 
    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;
 
    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult CalcRes(HcclComm comm, const OpParam& param,
                       const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
                       AlgResourceRequest& resourceRequest) override;    
    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                    AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;
 
protected:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable& resCtx);
    HcclResult CalcSliceInfoAllReduce(u64 dataCount);
    u64 RoundUp(const u64 dividend, const u64 divisor);

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;
    std::vector<SplitSliceInfo> sliceInfoList_;
};
}

#endif