/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_SCATTER_RHD_DPU_INTER_NODE_H
#define INS_TEMP_SCATTER_RHD_DPU_INTER_NODE_H

#include <cstring>
#include "alg_v2_template_base.h"
#include "alg_v2_template_register.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "dpu_alg_nhr_opt_wrapper.h"
#include "rhd_scatter_step.h"

namespace ops_hccl {

// Scatter RHD DPU 模板（host RoCE 老路径：单次 DPURunInfo 共享内存请求 + DpuBatchTransfer
// 合并段；PR 3461 DPU Thread 新路径合入前先行）。数据面循环收敛在 RunRhdSteps，
// 届时只改控制面分支。参数契约 allRank*（allRankSliceSize/allRankDispls/
// allRankProcessedDataCount；均匀分片为退化情形，不消费 stride/sliceSize 计算位移）。
// 步骤生成消费 rhd_scatter_step 骨架（GenRhdScatterSteps），区段字节换算消费公共函数
// template_utils（RhdSegToBytesAllRank / RhdBuildUnitSizes / RhdUnitStartOfRank）。
class InsTempScatterRhdDpuInterNode : public InsAlgTemplateBase {
public:
    explicit InsTempScatterRhdDpuInterNode(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);
    InsTempScatterRhdDpuInterNode() {}

    ~InsTempScatterRhdDpuInterNode() override;

    std::string Describe() const override
    {
        std::string info = "Template of scatter RHD DPU with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult DPUKernelRun(
        const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const u32 myRank, const std::vector<std::vector<uint32_t>>& subCommRanks) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    u64 GetThreadNum() const override;
    void SetRoot(u32 root);
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override {};
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override {};

private:
    // 非 2^N 多级流输入桥：交换 root（唯一持有数据方）的 ccl 由上游自然布局原地置换为 σ 布局
    // （跨层契约经 RhdUnitStartOfRank 换算；环循环节约一个元素暂存）。置换入队于 eager 舞步
    // 之前，由 KernelRun 的 BatchModeEnd + HcommThreadSynchronize 排空。暂存借接收缓冲尾段
    // （见 RearrangeToSigma），无堆生命周期问题。2^N（σ=恒等）零开销跳过。
    HcclResult RearrangeToSigma(
        const OpParam& param, const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const;
    HcclResult PostLocalCopy(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource);
    HcclResult
    RunRhdSteps(const std::map<u32, std::vector<ChannelInfo>>& channels, const TemplateDataParams& tempAlgParams);

    u64 count_{0};
    u64 dataTypeSize_{0};
};

} // namespace ops_hccl

#endif // INS_TEMP_SCATTER_RHD_DPU_INTER_NODE_H
