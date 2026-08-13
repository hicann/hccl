/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_REDUCE_SCATTER_CONCURRENT_MESH_NHR_H
#define HCCL_CCU_TEMP_REDUCE_SCATTER_CONCURRENT_MESH_NHR_H

#include "ccu_alg_template_base.h"
#include "ccu_kernel_reduce_scatter_mesh1d.h"
#include "ccu_kernel_reduce_scatter_nhr1d_mem2mem.h"

namespace ops_hccl {

class CcuTempReduceScatterConcurrentMeshNHR : public CcuAlgTemplateBase {
public:
    CcuTempReduceScatterConcurrentMeshNHR() = default;
    explicit CcuTempReduceScatterConcurrentMeshNHR(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~CcuTempReduceScatterConcurrentMeshNHR() override;

    std::string Describe() const override { return "Template of ReduceScatter ccu concurrent(Mesh+NHR)"; }

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
        const OpParam& param, const TemplateDataParams& templateDataParams, TemplateDataParams& meshParams,
        TemplateDataParams& nhrParams, u64& meshCount, u64& nhrCount) const;
    HcclResult LaunchMeshKernel(
        const TemplateDataParams& meshParams, TemplateResource& templateResource, u64 baseInputAddr, u64 baseOutputAddr,
        uint64_t token, const LoopGroupConfig& config);
    HcclResult LaunchNhrKernel(
        const TemplateDataParams& nhrParams, TemplateResource& templateResource, u64 baseInputAddr, u64 baseOutputAddr,
        uint64_t token);
    HcclResult
    FastLaunchMeshKernel(const CcuKernelSubmitInfo& submitInfo, ThreadHandle meshMain, const BuffInfo& buffInfo);
    HcclResult
    FastLaunchNhrKernel(const CcuKernelSubmitInfo& submitInfo, ThreadHandle nhrMain, const BuffInfo& buffInfo);
    HcclResult CalcMeshRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        CcuKernelInfo& meshKernelInfo);
    HcclResult CalcNhrRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, CcuKernelInfo& nhrKernelInfo);
    HcclResult GetNHRStepInfo(u32 step, NHRStepInfo& stepInfo);
    HcclResult ProcessNHRStepInfo(
        HcclComm comm, u32 enableDieNum, u32 enableDieId, std::vector<NHRStepInfo>& stepInfoVector,
        std::map<u32, u32>& rank2ChannelIdx, std::vector<std::vector<HcclChannelDesc>>& channelsPerDie);

    u32 rankSize_{0};
    u32 myMeshRank_{0};
    u32 myNhrRank_{0};
    u64 dataTypeSize_{0};
    std::vector<u32> meshGroup_;
    std::vector<u32> nhrGroup_;
    std::map<u32, std::vector<HcclChannelDesc>> nhrRankIdToChannelDesc_;
    AlgResourceRequest mergedReq_;
};

} // namespace ops_hccl
#endif // HCCL_CCU_TEMP_REDUCE_SCATTER_CONCURRENT_MESH_NHR_H
