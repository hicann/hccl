/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_SCATTER_OMNIPIPE_NHR_DPU_H
#define INS_TEMP_SCATTER_OMNIPIPE_NHR_DPU_H

#include <atomic>
#include "alg_v2_template_base.h"
#include "alg_v2_template_register.h"
#include "alg_param.h"
#include "executor_v2_base.h"
#include "template_utils.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class InsTempScatterOmniPipeNHRDpu : public InsAlgTemplateBase {
public:
    explicit InsTempScatterOmniPipeNHRDpu();
    explicit InsTempScatterOmniPipeNHRDpu(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempScatterOmniPipeNHRDpu() override;

    std::string Describe() const override
    {
        std::string info = "Template of scatter NHR DPU with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;
    void SetRoot(u32 root);
    void SetDoTask(bool doTask);
    // executor 提前调用 PreLocalCopy 后置 true，KernelRun 内跳过避免重复
    void SetPreLocalCopyDone(bool done) { preLocalCopyDone_ = done; }
    // 供 executor 在 xy 执行期间异步提交 root 数据预拷贝，利用 Z 线程与 level01 并行
    HcclResult PreLocalCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads);

    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    HcclResult DPUKernelRun(
        const TemplateDataParams& tempAlgParam, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const u32 myRank, const std::vector<std::vector<uint32_t>>& subCommRanks) override;
    u64 GetThreadNum() const override;

private:
    HcclResult GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo, u32 rootAlgRank, u32 myAlgRank);
    HcclResult RunNHR(const std::map<u32, std::vector<ChannelInfo>>& channels, const TemplateDataParams& tempAlgParam);
    HcclResult ExecuteNhrStep(
        const std::map<u32, std::vector<ChannelInfo>>& channels, const TemplateDataParams& tempAlgParam, u64 repeatNum,
        u32 rootAlgRank, u32 myAlgRank, u32 step, u32 nSteps);
    HcclResult BuildTxBatchSlices(
        const std::map<u32, std::vector<ChannelInfo>>& channels, const AicpuNHRStepInfo& stepInfo,
        const StepSliceInfo& stepSliceInfo, u32 dim0Idx, u64 repeatNum, u64 outBuffBaseOff, void* localCclBuffAddr,
        u32 dataTypeSize, u32 rootAlgRank, const ChannelInfo*& txCh, std::vector<DataSlice>& txSrcSlices,
        std::vector<DataSlice>& txDstSlices);
    HcclResult BuildRxBatchSlices(
        const std::map<u32, std::vector<ChannelInfo>>& channels, const AicpuNHRStepInfo& stepInfo,
        const StepSliceInfo& stepSliceInfo, u32 dim0Idx, u64 repeatNum, u64 outBuffBaseOff, void* localCclBuffAddr,
        u32 dataTypeSize, u32 rootAlgRank, const ChannelInfo*& rxCh, std::vector<DataSlice>& rxSrcSlices,
        std::vector<DataSlice>& rxDstSlices);
    HcclResult ExecuteDpuCommPrimitive(
        bool hasTx, bool hasRx, const ChannelInfo* txCh, const ChannelInfo* rxCh,
        const std::vector<DataSlice>& txSrcSlices, const std::vector<DataSlice>& txDstSlices,
        const std::vector<DataSlice>& rxSrcSlices, const std::vector<DataSlice>& rxDstSlices, u32 step);
    // PreLocalCopy 已提至 public，供 executor 提前调用
    // DPU数据交换：eager模式切换、DPURunInfo序列化、SendRequest/WaitResponse、msgId校验
    HcclResult RunDpuDataExchange(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource);
    HcclResult PrepareDpuExecution(const OpParam& param, const ThreadHandle& thread);
    HcclResult SendDpuRequest(
        const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource,
        u32& sendMsgId);
    HcclResult WaitDpuResponse(const OpParam& param, const TemplateResource& templateResource, u32 sendMsgId);
    u64 count_{0};
    std::atomic<bool> doTask_{false};
    std::atomic<bool> preLocalCopyDone_{false}; // executor 提前做预拷贝后置 true，KernelRun 内跳过
};

} // namespace ops_hccl

#endif // INS_TEMP_SCATTER_OMNIPIPE_NHR_DPU_H
