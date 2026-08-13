/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_TO_ALL_CONCURRENT_MESH_NHR_H
#define HCCL_CCU_TEMP_ALL_TO_ALL_CONCURRENT_MESH_NHR_H

#include "utils.h"
#include "ccu_alg_template_base.h"
#include "ccu_temp_all_to_all_mesh_1D.h"

namespace ops_hccl {

// mesh kernel (CcuAlltoAllMesh1DKernel) arg layout: 11 fixed args
struct CcuAlltoAllMesh1DArgLayout {
    static constexpr uint32_t INPUT = 0;
    static constexpr uint32_t OUTPUT = 1;
    static constexpr uint32_t TOKEN = 2;
    static constexpr uint32_t SLICE_SIZE = 3;
    static constexpr uint32_t SRC_STRIDE = 4;
    static constexpr uint32_t SRC_OFFSET = 5;
    static constexpr uint32_t DST_OFFSET = 6;
    static constexpr uint32_t GO_SIZE_0 = 7;
    static constexpr uint32_t GO_SIZE_1 = 8;
    static constexpr uint32_t GO_SIZE_2 = 9;
    static constexpr uint32_t GO_SIZE_3 = 10;
    static constexpr uint32_t ARG_SIZE = 11;
    static constexpr uint32_t IN_BUFF_BASE_OFF = 11;
    static constexpr uint32_t OUT_BUFF_BASE_OFF = 12;
};

// Concurrent: mesh(die1) + clos(die0), both use CcuAlltoAllMesh1DKernel, data split by bw ratio (rankSize-1):8.
// threads[0] -> mesh main, threads[1] -> clos main (executor slave).
// ccuKernels[0] -> mesh, ccuKernels[1] -> clos.
class CcuTempAllToAllConcurrentMeshNHR : public CcuAlgTemplateBase {
public:
    CcuTempAllToAllConcurrentMeshNHR() = default;
    explicit CcuTempAllToAllConcurrentMeshNHR(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~CcuTempAllToAllConcurrentMeshNHR() override = default;

    std::string Describe() const override
    {
        return StringFormat(
            "Template of AlltoAll ccu mesh1d concurrent (mesh die1 + clos die0) with tempRankSize [%u].",
            templateRankSize_);
    }

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& templateDataParams,
        TemplateResource& templateResource) override;
    HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) override;
    u64 GetThreadNum() const override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

private:
    void CalcDataSplit(u64 totalSize, u64 dataTypeSize, u64& meshSize, u64& closSize) const;
    HcclResult BuildMeshTaskArgs(
        const TemplateDataParams& templateDataParams, u64 meshSliceSize, uint64_t token,
        std::vector<uint64_t>& meshTaskArgs);
    HcclResult BuildClosTaskArgs(
        const TemplateDataParams& templateDataParams, u64 meshSliceSize, u64 closSliceSize, uint64_t token,
        std::vector<uint64_t>& closTaskArgs);
    HcclResult LaunchConcurrentKernels(
        TemplateResource& templateResource, bool hasMesh, bool hasClos, const std::vector<uint64_t>& meshTaskArgs,
        const std::vector<uint64_t>& closTaskArgs);
    HcclResult SaveSubmitInfos(
        TemplateResource& templateResource, const std::vector<uint64_t>& meshTaskArgs,
        const std::vector<uint64_t>& closTaskArgs, u64 meshSliceSize, bool hasMesh, bool hasClos, const BuffInfo& buff);
    HcclResult PatchMeshArgs(const TemplateFastLaunchCtx& ctx);
    HcclResult PatchClosArgs(const TemplateFastLaunchCtx& ctx, u32 meshKernelNum);

    uint32_t mySubCommRank_ = 0;
};

} // namespace ops_hccl

#endif // HCCL_CCU_TEMP_ALL_TO_ALL_CONCURRENT_MESH_NHR_H
