/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_REDUCE_SCATTER_ORDER_PRESERVED_EXECUTOR_H
#define HCCLV2_INS_V2_REDUCE_SCATTER_ORDER_PRESERVED_EXECUTOR_H

#include "executor_common_ops.h"
#include "topo_match_1d.h"
#include "topo_match_base.h"
#include "alg_param.h"
#include "order_preserved_common.h"
#include "topo_host.h"
#include "channel.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "log.h"
#include "sal.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include <cstdlib>
#include <algorithm>
#include <string>

namespace ops_hccl {

struct MemBlockInfo {
    std::vector<u64> size;
    std::vector<u64> userInputOffsets;
    std::vector<u64> inputOffsets;
    std::vector<u64> outputOffsets;
};

constexpr u64 HCCL_MIN_SLICE_ALIGN_ORDER_PRESERVED = 128;

struct OrderPreservedReduceScatterMemInfo {
    u64 sizePerBlock{0};
    std::vector<u64> groupSize;
    bool scratchMemFlag;
    u64 totalSize{0};
};

template <typename AlgTopoMatch, typename InsAlgTemplate>
class InsV2ReduceScatterOrderPreservedExecutor : public InsCollAlgBase {
public:
    explicit InsV2ReduceScatterOrderPreservedExecutor();
    ~InsV2ReduceScatterOrderPreservedExecutor() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    HcclResult CalcRes(HcclComm comm, const OpParam& param,
                       const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
                       AlgResourceRequest& resourceRequest) override;
    
    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                    AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

protected:
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx);
    HcclResult InitExecutorInfo(const OpParam &param);
    HcclResult CalcSizePerBlock(const OpParam &param);
    HcclResult CalcGroupSlices(const OpParam &param);
    u64 RoundUpWithDivisor(u64 value, u64 divisor) const;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;
    OrderPreservedReduceScatterMemInfo memInfo_;
    bool deterministicStrict_{false};
    bool aicpuUnfoldMode_{false};
};

}

#endif