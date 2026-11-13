/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_NHR_DPU
#define INS_TEMP_REDUCE_SCATTER_NHR_DPU

#pragma once

#include "alg_v2_template_base.h"

namespace ops_hccl {

class InsTempReduceScatterNhrDpu : public InsAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::NHR};
    InsTempReduceScatterNhrDpu() {}
    InsTempReduceScatterNhrDpu(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);

    ~InsTempReduceScatterNhrDpu() override {}

    std::string Describe() const override
    {
        std::string info = "Template of inline-reduce ReduceScatter NHR-step DPU with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType) override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult DPUKernelRun(
        const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const u32 myRank, const std::vector<std::vector<uint32_t>>& subCommRanks) override;

private:
    HcclResult GetStepInfo(uint32_t step, uint32_t nSteps, AicpuNHRStepInfo& stepInfo);
    u32 GetRankFromMap(const uint32_t rankIdx);
    // stride 约定（reduce_scatter 框间步）归一化为 SplitData 表：槽位 j = j * outputSliceStride，
    // DPU 侧与 wrapper 即可走 SplitData 约定同一条路径。strideLayout 由 KernelRun 判定一次后下传；
    // 非 stride 时只校验表长度一致性，避免后续 .at() 越界。
    HcclResult
    NormalizeStrideLayout(const TemplateDataParams& rawParams, TemplateDataParams& workParams, bool strideLayout);
    // host 预拷贝：本端对每个分片的贡献 input → scratch 对应槽位（write-reduce 的槽位初值）
    HcclResult LocalDataCopy(
        const TemplateDataParams& rawParams, const TemplateDataParams& workParams, bool strideLayout,
        const TemplateResource& templateResource);
    // NHR-step 循环：每步调用 DpuBatchWriteReduceTransferNHR 做 write-reduce 交换
    HcclResult RunNHR(
        const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
        ThreadHandle thread = 0, bool isDrainSupported = false);
    // host 收尾：自己分片的槽位（已是全量规约结果）→ output
    HcclResult PostLocalCopy(
        const TemplateDataParams& rawParams, const TemplateDataParams& workParams,
        const TemplateResource& templateResource);
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override {}
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override {}
};
} // namespace ops_hccl

#endif
