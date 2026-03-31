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

#include "executor_common_ops.h"
#include "topo_match_1d.h"
#include "topo_match_base.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
class InsBroadcastParallelExecutor : public InsCollAlgBase {
public:
    explicit InsBroadcastParallelExecutor();
    ~InsBroadcastParallelExecutor() final = default;

    std::string Describe() const override
    {
        return "Instruction based Broadcast Parallel Executor.";
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
                       AlgResourceRequest& resourceRequest) override;
    // AICPU 接口
    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;
    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

private:
    void GetParallelDataSplit(std::vector<float> &splitDataSize) const;
    uint64_t GetRankSize(const std::vector<std::vector<u32>> &vTopo) const;
    HcclResult CalcLocalRoot();
    // Aicpu
    HcclResult PrepareResForTemplate(const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra,
                                     InsAlgTemplate1 &tempAlgInter, InsAlgTemplate2 &tempAlgIntra1);
    HcclResult PrepareResForTemplate23(const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra,
                                       InsAlgTemplate2 &tempAlgIntra1, InsAlgTemplate3 &tempAlgInter1);
    HcclResult PrepareResForTemplateResource(const OpParam &param, const AlgResourceCtxSerializable &resCtx, TemplateResource &intraTempAlgRes,
                                             TemplateResource &interTempAlgRes, bool isScatter);
    void GenDataParamsBufferType(const BufferType inBuffType, const BufferType outBuffType, const BufferType hcclBuffType,
                                 TemplateDataParams &dataParams) const;
    void GenDataParamstempAlg(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 sliceCount,
                              const u64 scratchOffsetCount, TemplateDataParams &dataParams, const u32 LocalRankSize) const;
    void PrePareDataParamstempAlgInter(const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount);
    void PrePareDataParamstempAlgIntra(const u64 dataOffset, const u64 currCountPart, const u64 scratchOffsetCount);
    void GenDataParamsAllRank(const u64 sliceCount, const u32 LocalRankSize, TemplateDataParams &dataParams) const;
    HcclResult RunTemplateIntra0(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
                                  const u64 currCountPart, const u64 scratchOffsetCount, TemplateDataParams &dataParams,
                                  const TemplateResource& templateResource, InsAlgTemplate0 &tempAlgInter);
    HcclResult RunTemplateInter1(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
                                  const u64 currCountPart, const u64 scratchOffsetCount, TemplateDataParams &dataParams,
                                  const TemplateResource& templateResource, InsAlgTemplate1 &tempAlgInter);
    HcclResult RunTemplateInter0(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
                                  const u64 currCountPart, const u64 scratchOffsetCount, TemplateDataParams &dataParams,
                                  const TemplateResource& templateResource, InsAlgTemplate1 &tempAlgInter);
    HcclResult RunTemplateIntra1(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
                                  const u64 currCountPart, const u64 scratchOffsetCount, TemplateDataParams &dataParams,
                                  const TemplateResource& templateResource, InsAlgTemplate0 &tempAlgInter);
    HcclResult RunTemplateInter01(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
                                  const u64 currCountPart, const u64 scratchOffsetCount, TemplateDataParams &dataParams,
                                  const TemplateResource& templateResource, InsAlgTemplate3 &tempAlgInter1);
    HcclResult RunTemplateIntra11(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
                                  const u64 currCountPart, const u64 scratchOffsetCount, TemplateDataParams &dataParams,
                                  const TemplateResource& templateResource, InsAlgTemplate2 &tempAlgInter1);
    HcclResult RunTemplateIntra01(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
                                        const u64 currCountPart, const u64 scratchOffsetCount, TemplateDataParams &dataParams,
                                        const TemplateResource& templateResource, InsAlgTemplate2 &tempAlgIntra1);
    HcclResult RunTemplateInter11(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
                                        const u64 currCountPart, const u64 scratchOffsetCount, TemplateDataParams &dataParams,
                                        const TemplateResource& templateResource, InsAlgTemplate3 &tempAlgInter1);
    HcclResult GenInsQues(const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra0,
                          InsAlgTemplate1 &tempAlgInter0, InsAlgTemplate2 &tempAlgIntra1, InsAlgTemplate3 &tempAlgInter1);
    // rounddown func for uint
    inline u64 RoundDown(u64 dividend, u64 divisor) const
    {
        if (divisor == 0) {
            HCCL_WARNING("[InsBroadcastParallelExecutor][RoundDown] divisor is 0!");
            return dividend;
        }
        return dividend / divisor;
    }
    std::map<u32, u32> tempVirtRankMapInter_;
    std::map<u32, u32> tempVirtRankMapIntra_;
    std::vector<u64> allRankSliceSizeInter_;
    std::vector<u64> allRankDisplsInter_;
    std::vector<u64> allRankSliceSizeIntra_;
    std::vector<u64> allRankDisplsIntra_;
    u32 intraLocalRankSize_{0};  // server内算法rankSize
    u32 interLocalRankSize_{0};  // server间算法rankSize
    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};
    uint64_t interlocalroot{0};
    uint64_t intralocalroot{0};

    u32 intraLocalRoot_{0};  // server内算法root
    u32 interLocalRoot_{0};  // server间算法root

    u64 dataOffset0Inter_;
    u64 currCountPart0_;
    u64 scratchOffsetCountInterStage1_;

    u64 dataOffset0Intra_;
    u64 currCountPart1_;
    u64 scratchOffsetCountIntraStage1_;

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

    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;

};

} // namespace Hccl

#endif // HCCLV2_INS_BROADCAST_PARALLEL_EXECUTOR_H
