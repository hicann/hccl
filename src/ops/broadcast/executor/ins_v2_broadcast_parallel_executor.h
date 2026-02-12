/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_BROADCAST_SOLE_EXECUTOR_H
#define HCCLV2_INS_V2_BROADCAST_SOLE_EXECUTOR_H

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

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsBroadcastParallelExecutor : public InsCollAlgBase {
public:
    explicit InsBroadcastParallelExecutor();
    ~InsBroadcastParallelExecutor() = default;

    std::string Describe() const override
    {
        return "Instruction based Broadcast Parallel Executor.";
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
                       AlgResourceRequest& resourceRequest) override;


    // AICPU 接口
    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;
    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfo* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

private:
    void GetParallelDataSplit(std::vector<float> &splitDataSize) const;
    uint64_t GetRankSize(const std::vector<std::vector<u32>> &vTopo);
    HcclResult CalcLocalRoot();

    // Aicpu
    HcclResult PrepareResForTemplate(const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter);
    void GenDataParams(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 sliceCount,
                       const u64 scratchOffsetCount, TemplateDataParams &dataParams) const;
        
    HcclResult GenInsQues(const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra0, InsAlgTemplate1 &tempAlgInter0);

    u32 intraLocalRankSize_{0};  // server内算法rankSize
    u32 interLocalRankSize_{0};  // server间算法rankSize
    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};
    uint64_t interlocalroot{0};
    uint64_t intralocalroot{0};

    u32 intraLocalRoot_{0};  // server内算法root
    u32 interLocalRoot_{0};  // server间算法root


    std::vector<std::vector<std::vector<u32>>> vTopo_;
    std::vector<u32>              virtRanks_;
    std::map<u32, u32>            virtRankMap_; // 全局RankID:虚拟RankId

    std::vector<ThreadHandle> intraThreads_;
    std::vector<ThreadHandle> interThreads_;

    ThreadHandle mainThread_;
    std::vector<ThreadHandle> templateMainThreads_;
    std::vector<u32> syncNotifyOnTemplates_;
    std::vector<u32> syncNotifyOnMain_;

    std::map<u32, std::vector<ChannelInfo>> intraLinks_;
    std::map<u32, std::vector<ChannelInfo>> interLinks_;

    std::vector<std::vector<u32>> AlgHierarchyInfoExector;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;

};

} // namespace Hccl

#endif // HCCLV2_INS_BROADCAST_PARALLEL_EXECUTOR_H
