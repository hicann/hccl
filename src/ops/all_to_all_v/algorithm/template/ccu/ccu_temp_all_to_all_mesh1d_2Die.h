/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TEMP_ALL_TO_ALL_MESH_1D_2DIE_H_
#define HCCLV2_CCU_TEMP_ALL_TO_ALL_MESH_1D_2DIE_H_
#include <array>
#include "utils.h"
#include "ccu_alg_template_base.h"
#include "ccu_kernel_alg_base.h"
#include "cost_model.h"

namespace ops_hccl {

using RankId = u32;
using RankGroup = std::vector<RankId>;

class CcuTempAllToAllMesh1D2Die : public CcuAlgTemplateBase {
public:
    CcuTempAllToAllMesh1D2Die() = default;
    explicit CcuTempAllToAllMesh1D2Die(
        const OpParam& param, RankId rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~CcuTempAllToAllMesh1D2Die() override;

    std::string Describe() const override
    {
        return StringFormat("Template of alltoall ccu mesh 2Die with rankSize[%u]", templateRankSize_);
    }

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& templateDataParams,
        TemplateResource& templateResource) override;
    HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) override;

private:
    HcclResult PartitionChannels(HcclComm comm, std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc);
    HcclResult CalcFillArgsInfo(uint32_t kernelIdx, uint64_t& sliceSize, uint64_t& sliceOffset);
    HcclResult LaunchKernels(
        uint32_t kernelCount, uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t sliceStride,
        const LoopGroupConfig& config, const TemplateDataParams& templateDataParams,
        TemplateResource& templateResource);

    bool is2Plus6_ = false;
    uint32_t kernelCount_ = 2;
    uint32_t fullmeshDieId_ = 0;
    double dieSplitRatio_ = 1.0;
    std::array<bool, MAX_KERNEL_NUM_2DIE> kernelWithMyRank_ = {true, false, false};
    std::array<std::vector<HcclChannelDesc>, MAX_KERNEL_NUM_2DIE> kernelChannels_;
    std::array<std::vector<RankId>, MAX_KERNEL_NUM_2DIE> kernelRankGroup_;
    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc_;
};

} // namespace ops_hccl
#endif // HCCLV2_CCU_TEMP_ALL_TO_ALL_MESH_1D_2DIE_H_
