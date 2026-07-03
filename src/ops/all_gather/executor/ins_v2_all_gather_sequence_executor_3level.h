/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_GATHER_SEQUENCE_3_LEVEL_EXECUTOR_H
#define HCCLV2_INS_V2_ALL_GATHER_SEQUENCE_3_LEVEL_EXECUTOR_H

#include "executor_common_ops.h"
namespace ops_hccl {
constexpr u32 SEQUENCE_EXECUTOR_3_LEVEL_NUM = 3;

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
class InsV2AllGatherSequenceExecutor3Level : public InsCollAlgBase {
public:
    explicit InsV2AllGatherSequenceExecutor3Level();
    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    /* *************** 资源计算 *************** */
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
                       AlgResourceRequest &resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
                                    AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;


protected:
    HcclResult CalcLocalRankSize();
    HcclResult InitExectorInfo(const OpParam &param);
    HcclResult GenTempResource(int idx, TemplateResource &res) const;
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgLevel0,
                                    InsAlgTemplate1 &tempAlgLevel1, InsAlgTemplate2 &tempAlgLevel2);
    void GenTemplateAlgParamsLevel2(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 curCount, 
                                    const u64 dataOffset, TemplateDataParams &tempAlgParamsLevel2) const;
    void GenTemplateAlgParamsLevel1(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 curCount, 
                                    const u64 dataOffset, TemplateDataParams &tempAlgParamsLevel1) const;
    void GenTemplateAlgParamsLevel0(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 curCount, 
                                    const u64 dataOffset, TemplateDataParams &tempAlgParamsLevel0) const;
    HcclResult PrepareResForTemplate(InsAlgTemplate0 &tempAlgLevel0, InsAlgTemplate1 &tempAlgLevel1, InsAlgTemplate2 &tempAlgLevel2);
    static uint64_t GetRankSize(const std::vector<std::vector<u32>> &vTopo);

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};

    std::vector<ThreadHandle> threads_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;

    struct LevelInfo {
        std::remove_reference_t<decltype(remoteRankToChannelInfo_[0])> channels;
        std::vector<std::vector<u32>> hierarchyInfo;
        uint64_t rankSize;
        std::vector<ThreadHandle> threads;
        LevelInfo(decltype(remoteRankToChannelInfo_[0]) channels_,
                  std::vector<std::vector<u32>> hierarchyInfo_)
            : channels(channels_)
            , hierarchyInfo(hierarchyInfo_)
            , rankSize(GetRankSize(hierarchyInfo))
            , threads() {}
    };

    std::vector<LevelInfo> levels_;
};
}

#endif