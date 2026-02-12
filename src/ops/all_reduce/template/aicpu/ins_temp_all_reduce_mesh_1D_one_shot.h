/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_ALL_REDUCE_MESH_1D_ONE_SHOT
#define HCCLV2_INS_TEMP_ALL_REDUCE_MESH_1D_ONE_SHOT

#include <cstring>
#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class InsTempAllReduceMesh1DOneShot : public InsAlgTemplateBase {
public:
    explicit InsTempAllReduceMesh1DOneShot(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAllReduceMesh1DOneShot() override;

    std::string Describe() const override
    {
        std::string info = "Template of all resduce (one-shot) 1D Mesh with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    // 现在的RunAsync就是之前的GenExtIns
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& tempAlgParams,
                         const TemplateResource& templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                        AlgResourceRequest& resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
        
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

private:
    HcclResult CalcSlice(const u64 dataSize, RankSliceInfo &sliceInfoVec);
    HcclResult RunAllReduce(const std::map<u32, std::vector<ChannelInfo>> &channels,
                            const std::vector<ThreadHandle> &threads,
                            const TemplateDataParams &tempAlgParam,
                            const RankSliceInfo &sliceInfoVec);
    u64 processSize_{0};
    u64 count_{0};
};

} // namespace ops_hccl

#endif // HCCLV2_INS_TEMP_ALL_REDUCE_MESH_1D_ONE_SHOT