/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_REDUCE_CONCURRENT_MESH_NHR_H
#define HCCL_CCU_TEMP_ALL_REDUCE_CONCURRENT_MESH_NHR_H

#include "ccu_alg_template_base.h"
#include "kernel/ccu_kernel_all_reduce_mesh1d.h"
#include "ccu_kernel_all_reduce_nhr1d_mem2mem.h"

namespace ops_hccl {

class CcuTempAllReduceConcurrentMeshNHR : public CcuAlgTemplateBase {
public:
    CcuTempAllReduceConcurrentMeshNHR() = default;
    explicit CcuTempAllReduceConcurrentMeshNHR(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~CcuTempAllReduceConcurrentMeshNHR() override;

    std::string Describe() const override { return "Template of AllReduce ccu concurrent(Mesh+NHR)"; }

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& templateDataParams,
        TemplateResource& templateResource) override;
    HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 GetThreadNum() const override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

private:
    HcclResult CalcDataSplit(
        const TemplateDataParams& templateDataParams, TemplateDataParams& meshParams, TemplateDataParams& nhrParams,
        u64& meshCount, u64& nhrCount) const;
    HcclResult LaunchMeshKernel(
        const TemplateDataParams& meshParams, TemplateResource& templateResource, u64 baseInputAddr, u64 baseOutputAddr,
        uint64_t token, const LoopGroupConfig& config);
    HcclResult LaunchNhrKernel(
        const TemplateDataParams& nhrParams, TemplateResource& templateResource, u64 baseInputAddr, u64 baseOutputAddr,
        uint64_t token);
    HcclResult CalcMeshRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        CcuKernelInfo& meshKernelInfo);
    HcclResult CalcNhrRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, CcuKernelInfo& nhrKernelInfo);

    // NHR 算法逻辑复刻
    HcclResult CalcSlice(u64 dataSize, RankSliceInfo& sliceInfoVec) const;
    HcclResult GetStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo) const;
    HcclResult GetReduceScatterStepInfo(u32 step, NHRStepInfo& stepInfo) const;
    HcclResult GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo) const;
    HcclResult ProcessNHRStepInfo(
        HcclComm comm, u32 enableDieNum, u32 enableDieId, std::vector<NHRStepInfo>& stepInfoVector,
        std::map<u32, u32>& rank2ChannelIdx, std::vector<std::vector<HcclChannelDesc>>& channelsPerDie);

    u32 myMeshRank_{0};
    u32 myNhrRank_{0};
    std::vector<u32> meshGroup_;
    std::vector<u32> nhrGroup_;
    u32 rankSize_{0};
    u64 dataTypeSize_{0};
    std::map<u32, std::vector<HcclChannelDesc>> nhrRankIdToChannelDesc_;
    AlgResourceRequest mergedReq_;
};

} // namespace ops_hccl
#endif // HCCL_CCU_TEMP_ALL_REDUCE_CONCURRENT_MESH_NHR_H
