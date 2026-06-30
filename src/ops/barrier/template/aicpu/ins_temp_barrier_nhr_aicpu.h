/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_BARRIER_NHR_AICPU_H
#define INS_TEMP_BARRIER_NHR_AICPU_H

#pragma once

#include "alg_v2_template_base.h"

namespace ops_hccl {

// Barrier AICPU 模板：按 NHR 步数完成全员同步，不搬运数据。
// 与 DPU 版的区别：直接在 AICPU 上执行 RunNHRBarrier，不通过 DPU 请求-响应封装。
// AICPU 通信原语（alg_data_trans_wrapper.h）需要传 thread 参数。
class InsTempBarrierNhrAicpu : public InsAlgTemplateBase {
public:
    InsTempBarrierNhrAicpu() = default;
    InsTempBarrierNhrAicpu(const OpParam &param, const u32 rankId,
                           const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempBarrierNhrAicpu() override = default;

    std::string Describe() const override
    {
        return "Template of Barrier NHR AICPU with tempRankSize " + std::to_string(templateRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam &param,
                       const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType) override;
    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource) override;
    u64 GetThreadNum() const override { return 1; }  // 仅需主线程，NHR 串行执行

protected:
    u32 GetRankFromMap(const uint32_t rankIdx) const;
    HcclResult RunNHRBarrier(const std::map<u32, std::vector<ChannelInfo>> &channels,
                             const ThreadHandle &thread);
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override {}  // 单线程，无需线程间同步
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override {}  // 单线程，无需线程间同步
};

}  // namespace ops_hccl

#endif  // INS_TEMP_BARRIER_NHR_AICPU_H
