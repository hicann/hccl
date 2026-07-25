/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_GATHER_OMNIPIPE_MESH_1D_MEM2MEM_H
#define HCCL_CCU_TEMP_GATHER_OMNIPIPE_MESH_1D_MEM2MEM_H

#include "utils.h"
#include "ccu_alg_template_base.h"

namespace ops_hccl {

class CcuTempGatherOmniPipeMesh1DMem2Mem : public CcuAlgTemplateBase {
public:
    CcuTempGatherOmniPipeMesh1DMem2Mem() = default;
    explicit CcuTempGatherOmniPipeMesh1DMem2Mem(const OpParam& param,
                                                 const u32 rankId,
                                                 const std::vector<std::vector<u32>>& subCommRanks);

    ~CcuTempGatherOmniPipeMesh1DMem2Mem() override;

    std::string Describe() const override
    {
        return StringFormat("Template of Gather ccu omnipipe mesh 1D mem2mem with tempRankSize [%u].",
                            subCommRanks_[0].size());
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    HcclResult KernelRun(const OpParam& param,
                          const TemplateDataParams& templateDataParams,
                          TemplateResource& templateResource) override;
    u64 GetThreadNum() const override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    uint32_t RemoteRankId2RankId(const uint32_t remoteRankId) const;
    void SetRoot(u32 root);
    void UnsetRoot(u32 rank);

    HcclResult RunGatherComm(const StepSliceInfo& stepSliceInfo, uint64_t inputAddr, uint64_t outputAddr,
        uint64_t token, uint64_t localCopyFlag, TemplateResource& templateResource);
    HcclResult LaunchGatherKernel(TemplateResource& templateResource, uint64_t inputAddr, uint64_t outputAddr,
        uint64_t token, uint64_t localCopyFlag, uint64_t sliceSize, bool isFirstPiece, bool isLastPiece,
        bool ifNewRoot, const std::vector<uint64_t>& sliceSizeVec, const std::vector<uint64_t>& inputVec,
        const std::vector<uint64_t>& outputVec);
    HcclResult RunLocalCopy(const TemplateDataParams& templateDataParams, TemplateResource& templateResource);
    
    uint32_t mySubCommRank_ = 0;
    uint32_t subCommRootId_ = UINT32_MAX;
    uint32_t rankId_ = 0;
    bool ifRealRoot_ = false;

    u64 localCopyFlag = 0;
    bool isSameXAxis = false;
    bool isSameYAxis = false;
};

} // namespace ops_hccl

#endif // HCCL_CCU_TEMP_GATHER_OMNIPIPE_MESH_1D_MEM2MEM_H