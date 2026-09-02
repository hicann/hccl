/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_SCATTER_OMNIPIPE_MESH_1D_MEM2MEM_H
#define HCCL_CCU_TEMP_SCATTER_OMNIPIPE_MESH_1D_MEM2MEM_H

#include "utils.h"
#include "ccu_alg_template_base.h"

namespace ops_hccl {

class CcuTempScatterOmniPipeMesh1DMem2Mem : public CcuAlgTemplateBase {
public:
    CcuTempScatterOmniPipeMesh1DMem2Mem() = default;
    CcuTempScatterOmniPipeMesh1DMem2Mem(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~CcuTempScatterOmniPipeMesh1DMem2Mem() override;

    std::string Describe() const override
    {
        return StringFormat(
            "Template of Scatter ccu OmniPipe mesh 1D Mem2Mem with tempRankSize[%u]", templateRankSize_);
    }

    u64 GetThreadNum() const override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    HcclResult RunScatterMeshDispatch(
        const TemplateDataParams& templateDataParams, TemplateResource& templateResource, uint64_t inputAddr,
        uint64_t outputAddrBase, uint64_t outBuffBaseOff, uint64_t token);
    HcclResult RunLocalCopy(
        const TemplateDataParams& templateDataParams, TemplateResource& templateResource, uint64_t inputAddrBase,
        uint64_t outputAddrBase);
    HcclResult LaunchOneRepeat(
        const StepSliceInfo& stepSliceInfo, TemplateResource& templateResource, uint32_t rpt, uint64_t repeatNum,
        bool ifNewRoot, uint64_t inputAddr, uint64_t outputAddr, uint64_t token);
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& templateDataParams,
        TemplateResource& templateResource) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    void SetRoot(u32 root);
    void UnsetRoot(u32 rank);
    void BuildSliceStrideVec(
        const StepSliceInfo& stepSliceInfo, uint32_t rpt, uint64_t repeatNum, bool ifNewRoot, uint64_t& sliceSize,
        std::vector<uint64_t>& inputOmniSliceSizeVec, std::vector<uint64_t>& inputOmniSliceStrideVec,
        std::vector<uint64_t>& outputOmniSliceStrideVec);

    uint32_t mySubCommRank_ = 0;
    uint32_t subCommRootId_ = UINT32_MAX;
    bool ifRealRoot_ = false;
    bool isStepOne_ = false;
    bool isLastStep_ = false;
};

} // namespace ops_hccl

#endif // HCCL_CCU_TEMP_SCATTER_OMNIPIPE_MESH_1D_MEM2MEM_H
