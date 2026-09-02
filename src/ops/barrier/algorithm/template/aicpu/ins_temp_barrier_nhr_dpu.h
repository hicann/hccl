/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_BARRIER_NHR_DPU_H
#define INS_TEMP_BARRIER_NHR_DPU_H

#pragma once

#include "alg_v2_template_base.h"

namespace ops_hccl {

// Barrier 框间 DPU 模板：按 NHR 步数完成全员同步，不搬运数据。
// 复用 AllGather NHR DPU 的 step 计算与 DPU 主流握手协议。
class InsTempBarrierNHRDPU : public InsAlgTemplateBase {
public:
    InsTempBarrierNHRDPU() = default;
    InsTempBarrierNHRDPU(const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempBarrierNHRDPU() override = default;

    std::string Describe() const override
    {
        std::string info = "Template of Barrier NHR with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType) override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult DPUKernelRun(
        const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const u32 myRank, const std::vector<std::vector<uint32_t>>& subCommRanks) override;

protected:
    u32 GetRankFromMap(const uint32_t rankIdx) const;
    HcclResult RunNHRBarrier(const std::map<u32, std::vector<ChannelInfo>>& channels) const;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override {}
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override {}
};

} // namespace ops_hccl

#endif // INS_TEMP_BARRIER_NHR_DPU_H
