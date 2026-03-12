/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_REDUCE_NHR_MEM2MEM_1D_MULTY_JETTY_H
#define HCCL_CCU_TEMP_ALL_REDUCE_NHR_MEM2MEM_1D_MULTY_JETTY_H

#include "ccu_alg_template_base.h"
#include "utils.h"
#include "ccu_kernel_all_reduce_nhr_mem2mem_1D_multi_jetty.h"

namespace ops_hccl {

class CcuTempAllReduceNhrMem2Mem1DMultiJetty : public CcuAlgTemplateBase {
public:
    explicit  CcuTempAllReduceNhrMem2Mem1DMultiJetty(const OpParam& param,
        const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks);

    ~CcuTempAllReduceNhrMem2Mem1DMultiJetty() override;

    std::string Describe() const override
    {
        return StringFormat("Template of All reduce ccu nhr 1D Mem2Mem multi-jetty with tempRankSize [%u].",
                            templateRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;

    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         const TemplateResource& templateResource) override;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override; // 此template需要将buffer分几块用
private:
    HcclResult GetReduceScatterStepInfo(u32 step, NHRStepInfo &stepInfo);
    HcclResult GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo);
    HcclResult GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo);
    uint32_t GetNHRStepNum(const uint32_t rankSize) const;
    uint32_t localRank2UserRank(const uint32_t localRank) const;
    HcclResult ProcessNHRStepInfo(std::vector<NHRStepInfo> &stepInfoVector);
    HcclDataType dataType_;
    uint32_t localRank_{INVALID_VALUE_RANKID}; // 所在子通信域的rank id
    uint32_t portNum_{0}; // 端口数量
    std::map<u32, u32> subCommRankMap_; // 全局rank号映射到自通信域rank号
};
} // namespace ops_hccl
#endif// HCCL_CCU_TEMP_ALL_REDUCE_NHR_MEM2MEM_1D_MULTY_JETTY_H