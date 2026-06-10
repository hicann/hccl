/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_TO_ALL_MESH_1D_MULTI_JETTY_H
#define HCCL_CCU_TEMP_ALL_TO_ALL_MESH_1D_MULTI_JETTY_H

#include "utils.h"
#include "ccu_alg_template_base.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuTempAllToAllMesh1dMultiJetty : public CcuAlgTemplateBase {
public:
    CcuTempAllToAllMesh1dMultiJetty() = default;
    explicit  CcuTempAllToAllMesh1dMultiJetty(const OpParam& param, 
                                                const u32 rankId, // 传通信域的rankId，userRank
                                                const std::vector<std::vector<u32>> &subCommRanks);

    ~CcuTempAllToAllMesh1dMultiJetty() override;

    std::string Describe() const override
    {
        return StringFormat("Template of AlltoAll ccu mesh 1D multi jetty with tempRankSize [%u].",
                            templateRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         TemplateResource& templateResource) override;
    HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;
private:
    std::vector<u32> jettyNums_;
    std::vector<u64> sendCounts_;
    std::vector<u64> recvCounts_;
    std::vector<u64> sdispls_;
    std::vector<u64> rdispls_;

    void CalcJettySlices(uint64_t sliceSize, std::vector<uint64_t>& jettySlice,
                         std::vector<uint64_t>& jettySliceTail);
    std::vector<uint64_t> BuildTaskArgs(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                         uint64_t sliceSize, uint64_t srcStride, uint64_t srcOffset, uint64_t dstOffset,
                         const std::vector<uint64_t>& goSize, const std::vector<uint64_t>& jettySlice,
                         const std::vector<uint64_t>& jettySliceTail);
    void BuildSubmitInfo(TemplateResource& templateResource, const std::vector<uint64_t>& taskArgs,
                         uint64_t argSize);
};

}// namespace ops_hccl

#endif// HCCL_CCU_TEMP_ALL_TO_ALL_MESH_1D_MULTI_JETTY_H