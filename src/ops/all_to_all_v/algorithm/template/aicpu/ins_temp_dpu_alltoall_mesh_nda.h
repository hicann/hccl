/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_DPU_ALLTOALL_MESH_NDA_H
#define INS_TEMP_DPU_ALLTOALL_MESH_NDA_H

#include <map>
#include <vector>
#include <string>
#include "alg_v2_template_base.h"

namespace ops_hccl {
class InsTempDpuAlltoAllMeshNda : public InsAlgTemplateBase {
public:
    explicit InsTempDpuAlltoAllMeshNda();
    explicit InsTempDpuAlltoAllMeshNda(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);

    ~InsTempDpuAlltoAllMeshNda() override;

    std::string Describe() const override
    {
        std::string info = "Template of alltoall Mesh NDA with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

private:
    static constexpr u64 CCLBUF_SPLIT_PARTS = 2;

    HcclResult LocalCopyforMyRank(
        const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams, std::vector<ThreadHandle>& threads);

    HcclResult PreCopyDataToCclInBuf(
        const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams,
        const std::map<u32, std::vector<ChannelInfo>>& channels, std::vector<ThreadHandle>& threads);

    HcclResult SendRecvData(
        const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams,
        const TemplateResource& templateResource);

    HcclResult PostCopyDataToRecvBuf(
        const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams, std::vector<ThreadHandle>& threads);

    u32 GetIntraRankNumFromPhysicalLevels(const TopoInfoWithNetLayerDetails* topoInfo);

    u64 halfMaxTmpMemSize_{0};
    u64 hcclbuffBlockMemSize_{0};
    u64 dataTypeSize_{0};
    std::vector<ThreadHandle> threads_;
};

} // namespace ops_hccl

#endif // INS_TEMP_DPU_ALLTOALL_MESH_NDA_H
