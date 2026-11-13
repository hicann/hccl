/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_BARRIER_RHD_AICPU_H
#define INS_TEMP_BARRIER_RHD_AICPU_H

#include "alg_v2_template_base.h"

namespace ops_hccl {
class InsTempBarrierRhdAicpu : public InsAlgTemplateBase {
public:
    InsTempBarrierRhdAicpu() = default;
    InsTempBarrierRhdAicpu(const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    std::string Describe() const override { return "Barrier RHD AICPU"; }
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    u64 CalcScratchMultiple(BufferType, BufferType) override { return 0; }
    u64 GetThreadNum() const override { return 1; }
    HcclResult
    KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& resource) override;
    void GetNotifyIdxMainToSub(std::vector<u32>& indices) override { indices.clear(); }
    void GetNotifyIdxSubToMain(std::vector<u32>& indices) override { indices.clear(); }
};
} // namespace ops_hccl

#endif // INS_TEMP_BARRIER_RHD_AICPU_H
