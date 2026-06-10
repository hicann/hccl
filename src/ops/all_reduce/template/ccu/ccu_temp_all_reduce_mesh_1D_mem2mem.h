/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALLREDUCE_MESH_1D_MEM2MEM_H
#define HCCL_CCU_TEMP_ALLREDUCE_MESH_1D_MEM2MEM_H
#include "ccu_alg_template_base.h"
#include "utils.h"

namespace ops_hccl {
class CcuTempAllReduceMeshMem2Mem1D : public CcuAlgTemplateBase {
public:
    CcuTempAllReduceMeshMem2Mem1D() = default;
    explicit CcuTempAllReduceMeshMem2Mem1D(const OpParam& param, 
                                                const u32 rankId, // 传通信域的rankId，userRank
                                                const std::vector<std::vector<u32>> &subCommRanks);
    ~CcuTempAllReduceMeshMem2Mem1D() override;

    std::string Describe() const override
    {
        return StringFormat("Template of All Reduce ccu mesh 1D mem2mem, tempRankSize [%u].",
                           subCommRanks_[0].size());
    }
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;

    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         TemplateResource& templateResource) override;
    HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
private:
    HcclResult CalcSlice(const u64 dataSize, RankSliceInfo &sliceInfoVec);
    uint64_t RoundUp(uint64_t dividend, uint64_t divisor) const;
    void BuildTaskArgs(const uint64_t inputAddr, const uint64_t outputAddr, const uint64_t token,
                       const uint64_t scratchAddr, const uint64_t currentRankSliceInputOffset,
                       const uint64_t currentRankSliceOutputOffset, const uint64_t normalSliceSize,
                       const uint64_t lastSliceSize, const uint64_t mySliceSize, const uint64_t sliceOffset,
                       const uint64_t isInputOutputEqual, const std::vector<uint64_t>& goSize,
                       std::vector<uint64_t>& taskArgs) const;
    void SaveSubmitInfo(const uint64_t inputAddr, const uint64_t outputAddr, const uint64_t token,
                        const uint64_t scratchAddr, const uint64_t currentRankSliceInputOffset,
                        const uint64_t currentRankSliceOutputOffset, const uint64_t normalSliceSize,
                        const uint64_t lastSliceSize, const uint64_t mySliceSize, const uint64_t sliceOffset,
                        const uint64_t isInputOutputEqual, const std::vector<uint64_t>& goSize,
                        TemplateResource& templateResource) const;
    uint32_t mySubCommRank_ = 0;
};

} // namespace ops_hccl

#endif // HCCLV2_CCU_TEMP_ALL_REDUCE_MESH_1D_MEM2MEM_H_