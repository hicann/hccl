/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef INS_ALL_REDUCE_PARALLEL_EXECUTOR
#define INS_ALL_REDUCE_PARALLEL_EXECUTOR
 
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
#include "topo_match_multilevel.h"
 
namespace ops_hccl {
 
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsAllReduceParallelExecutor : public InsCollAlgBase {
public:
    explicit InsAllReduceParallelExecutor();
    ~InsAllReduceParallelExecutor();
 
    std::string Describe() const override
    {
        return "Instruction based AllReduce Parallel Executor.";
    }

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfo* topoInfo,
        AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;
    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;
 
private:
    void GetParallelDataSplit(std::vector<float> &splitDataSize) const;
    HcclResult PrepareResForTemplate(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
        InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter);
    
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
        InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter);
    HcclResult CalcSendDataSize(u64 &memBlockSize, float &SplitRate, u32 &multipleIntra, u32 &multipleInter);
    
    void GenAlgParamsStage0(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
        const u64 sliceCount, const u64 scratchOffsetCount, TemplateDataParams &dataParams) const;
    void GenAlgParamsStage1(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
        const u64 sliceCount, const u64 scratchOffsetCount, TemplateDataParams &dataParams) const;
    
    HcclResult PreSync(const std::vector<ThreadHandle> &threads);
    HcclResult PostSync(const std::vector<ThreadHandle> &threads);

    std::vector<std::vector<u32>> AlgHierarchyInfoExector;

    std::vector<ThreadHandle> threads_;
    std::vector<ThreadHandle> intraThreads_;
    std::vector<ThreadHandle> interThreads_;
    std::vector<ThreadHandle> syncThreads_;

    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::map<u32, std::vector<ChannelInfo>> intraChannelInfo_;
    std::map<u32, std::vector<ChannelInfo>> interChannelInfo_;

    u64 parallelHcclBuffOffsetStage0_{0};
    u64 parallelHcclBuffOffsetStage1_{0};
    u64 intraHcclBuffSizeStage0_{0};
    u64 interHcclBuffSizeStage0_{0};
    u64 intraHcclBuffSizeStage1_{0};
    u64 interHcclBuffSizeStage1_{0};

    std::vector<u32> preSyncNotifyList_;
    std::vector<u32> postSyncNotifyList_;
};
 
}  // namespace ops_hccl
 
#endif  // INS_ALL_REDUCE_PARALLEL_EXECUTOR