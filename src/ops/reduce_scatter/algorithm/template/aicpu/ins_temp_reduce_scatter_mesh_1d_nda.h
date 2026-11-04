/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_1D_NDA
#define INS_TEMP_REDUCE_SCATTER_1D_NDA

#include "alg_v2_template_base.h"
#include "alg_v2_template_register.h"
#include "alg_param.h"
#include "executor_base.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class InsTempReduceScatterMesh1DNDA : public InsAlgTemplateBase {
public:
    explicit InsTempReduceScatterMesh1DNDA();
    explicit InsTempReduceScatterMesh1DNDA(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempReduceScatterMesh1DNDA() override;

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter Mesh NDA with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType) override;
    HcclResult PostLocalReduce(
        const OpParam& param, const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads);

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub) override {}
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override {}

private:
    u64 count_{0};
    u64 processSize_{0};
};

} // namespace ops_hccl
#endif
