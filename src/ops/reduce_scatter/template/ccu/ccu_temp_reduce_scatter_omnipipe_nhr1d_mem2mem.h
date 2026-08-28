/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_REDUCE_SCATTER_OMNIPIPE_NHR_1D_MEM2MEM_H
#define HCCL_CCU_TEMP_REDUCE_SCATTER_OMNIPIPE_NHR_1D_MEM2MEM_H

#include "ccu_alg_template_base.h"
#include "utils.h"
#include "ccu_kernel_reduce_scatter_omnipipe_nhr1d_mem2mem.h"

namespace ops_hccl {

class CcuTempReduceScatterOmniPipeNHR1DMem2Mem : public CcuAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::NHR};
    CcuTempReduceScatterOmniPipeNHR1DMem2Mem() = default;
    explicit CcuTempReduceScatterOmniPipeNHR1DMem2Mem(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);

    ~CcuTempReduceScatterOmniPipeNHR1DMem2Mem() override;

    std::string Describe() const override
    {
        return StringFormat(
            "Template of Reduce Scatter CCU OmniPipe NHR 1D Mem2Mem with tempRankSize [%u].", subCommRanks_[0].size());
    }

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;

    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& templateDataParams,
        TemplateResource& templateResource) override;

    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;

    u64 GetThreadNum() const override;

protected:
    HcclResult CalcNHRInfo(std::vector<NHRStepInfoRS>& stepInfoVector) const;
    u32 GetNHRStepNum(u32 rankSize) const;
    HcclResult GetStepInfo(u32 step, NHRStepInfoRS& stepInfo) const;
    uint32_t RemoteRankId2RankId(const uint32_t remoteRankId) const;

    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc_;

private:
    uint32_t mySubCommRank_ = 0;
};

} // namespace ops_hccl

#endif // HCCL_CCU_TEMP_REDUCE_SCATTER_OMNIPIPE_NHR_1D_MEM2MEM_H
