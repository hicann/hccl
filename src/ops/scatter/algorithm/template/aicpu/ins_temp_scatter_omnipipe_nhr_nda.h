/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_SCATTER_OMNIPIPE_NHR_NDA_H
#define INS_TEMP_SCATTER_OMNIPIPE_NHR_NDA_H

#include <atomic>
#include "alg_v2_template_base.h"

namespace ops_hccl {

class InsTempScatterOmniPipeNHRNDA : public InsAlgTemplateBase {
public:
    InsTempScatterOmniPipeNHRNDA() = default;
    InsTempScatterOmniPipeNHRNDA(const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempScatterOmniPipeNHRNDA() override = default;

    std::string Describe() const override { return "Scatter OmniPipe NHR NDA"; }
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 GetThreadNum() const override { return 1; }
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override { notifyIdxMainToSub.clear(); }
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override { notifyIdxSubToMain.clear(); }
    void SetRoot(u32 root) { root_ = root; }
    void SetDoTask(bool doTask) { doTask_.store(doTask, std::memory_order_relaxed); }
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;

private:
    HcclResult PreLocalCopy(const TemplateDataParams& tempAlgParams, const ThreadHandle& thread) const;
    HcclResult GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo) const;
    HcclResult RunNHR(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource) const;
    HcclResult BuildLocalSlices(
        const TemplateDataParams& tempAlgParams, const std::vector<u32>& sliceIdxs,
        std::vector<DataSlice>& localSlices) const;
    std::atomic<bool> doTask_{false};
};
} // namespace ops_hccl

#endif // INS_TEMP_SCATTER_OMNIPIPE_NHR_NDA_H
