/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_REDUCE_SCATTER_OMNIPIPE_2D_EXECUTOR_H
#define HCCLV2_INS_V2_REDUCE_SCATTER_OMNIPIPE_2D_EXECUTOR_H

#include <cmath>
#include "topo_host.h"
#include "alg_param.h"
#include "channel.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "log.h"
#include "workflow.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "sal.h"
#include "topo_match_base.h"
#include "topo_match_ubx.h"
#include "alg_data_trans_wrapper.h"
#include "omnipipe_data_slice_calc.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
class InsV2ReduceScatterOmniPipe2dExecutor : public InsCollAlgBase {
public:
    explicit InsV2ReduceScatterOmniPipe2dExecutor();
    ~InsV2ReduceScatterOmniPipe2dExecutor() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
                       AlgResourceRequest &resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                    AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

protected:
    /* *************** 算法编排 *************** */
    /// 算法编排通用接口
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx);
    HcclResult InitCommInfo(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                            const AlgHierarchyInfoForAllLevel& algHierarchyInfo);

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};

    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;

    /// 对角算法专用
private:
    ThreadHandle              controlThread_;
    std::vector<ThreadHandle> templateMainThreads_;
    std::vector<ThreadHandle> templateLocalCopyThreads_;
    std::vector<u32>          notifyIdxControlToTemplates_;
    std::vector<u32>          notifyIdxTemplatesToControl_;
    std::vector<ThreadHandle> level0Threads_;
    std::vector<ThreadHandle> level1Threads_;

    HcclResult GenTemplateAlgParamsByDimData(
        TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount);

    HcclResult PrepareResForTemplate(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
        InsAlgTempLevel0 &algTempLevel0, InsAlgTempLevel1 &algTempLevel1);
    HcclResult RestoreChannelMap(const AlgResourceCtxSerializable &resCtx,
        std::vector<std::map<u32, std::vector<ChannelInfo>>> &rankIdToChannelInfo);
};
}

#endif // HCCLV2_INS_V2_REDUCE_SCATTER_OMNIPIPE_2D_EXECUTOR_H