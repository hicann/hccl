/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_SCATTER_OMNIPIPE_NHR_H
#define INS_TEMP_SCATTER_OMNIPIPE_NHR_H

#include <atomic>
#include "alg_v2_template_base.h"
#include "executor_v2_base.h"
#include "alg_data_trans_wrapper.h"
#include "ins_temp_scatter_nhr.h"

namespace ops_hccl {

class InsTempScatterOmniPipeNHR : public InsTempScatterNHR {
public:
    explicit InsTempScatterOmniPipeNHR(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempScatterOmniPipeNHR() override;

    std::string Describe() const override
    {
        std::string info = "Template of scatter omnipipe NHR with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult DoLocalCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads);
    void SetRoot(u32 root);
    void SetDoTask(bool doTask);

private:
    HcclResult RunNHR(
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
        const TemplateDataParams& tempAlgParams);

    // 计算单步Tx/Rx的DataSlice（用stepSliceInfo偏移数组驱动，不做reduce）
    HcclResult GetNHRDataSize(
        const AicpuNHRStepInfo& st, const u32 channelIdx, void* sendCclBuffAddr, void* recvCclBuffAddr,
        const u32 dataTypeSize, const u64 rptNum, std::vector<DataSlice>& txSrcSlices,
        std::vector<DataSlice>& txDstSlices, std::vector<DataSlice>& rxSrcSlices, std::vector<DataSlice>& rxDstSlices);
    bool AppendTxSlice(
        u32 txIdx, u32 channelIdx, u64 rpt, void* sendCclBuffAddr, u32 dataTypeSize,
        std::vector<DataSlice>& txSrcSlices, std::vector<DataSlice>& txDstSlices);
    void AppendRxSlice(
        u32 rxIdx, u32 channelIdx, u64 rpt, void* recvCclBuffAddr, u32 dataTypeSize,
        std::vector<DataSlice>& rxSrcSlices, std::vector<DataSlice>& rxDstSlices);

    // 参考ccu版BuildSliceInfoVec：按rank组织stepSliceInfo（跳过root），并按channel拆分
    HcclResult
    PrepareScatterDataSplit(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource);
    void LogSubCommRanks() const;

    // 处理 PrepareScatterDataSplit 中单个 rank 的数据拆分（root填0，非root按channel拆分）
    HcclResult FillOneRankDataSplit(
        u32 ridx, u32 rootAlgRank, u32& originIndex, u32 dim0Idx, u32 dataTypeSize, const StepSliceInfo& stepSliceInfo,
        const TemplateResource& templateResource);
    // RunNHR 中仅有 Tx 分支（root分发场景）
    HcclResult ExecuteTxOnlyStep(
        const AicpuNHRStepInfo& stepInfo, u32 channelIdx, u32 dataTypeSize, u64 rptNum, bool isPcieProtocal,
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads, u32 step);
    // RunNHR 中仅有 Rx 分支（非root接收场景）
    HcclResult ExecuteRxOnlyStep(
        const AicpuNHRStepInfo& stepInfo, u32 channelIdx, u32 dataTypeSize, u64 rptNum, bool isPcieProtocal,
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads, u32 step);
    // RunNHR 中既有 Tx 又有 Rx 分支（中间步 SendRecv）
    HcclResult ExecuteTxRxStep(
        const AicpuNHRStepInfo& stepInfo, u32 channelIdx, u32 dataTypeSize, u64 rptNum, bool isPcieProtocal,
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads, u32 step);

    TemplateDataParams tempAlgParams_;
    std::map<u32, std::vector<ChannelInfo>> channels_;
    std::vector<std::vector<std::vector<u64>>> dataSplitVec_;  // [rank][rpt][channel]
    std::vector<std::vector<std::vector<u64>>> dataOffsetVec_; // [rank][rpt][channel]
    std::vector<std::vector<u64>> inputOmniSliceStrideVec_;    // [rank][rpt] 预计算的stride
    std::vector<std::vector<u64>> outputOmniSliceStrideVec_;   // [rank][rpt]
    u64 repeatNum_{0};
    std::atomic<bool> doTask_{false};
};

} // namespace ops_hccl

#endif // INS_TEMP_SCATTER_OMNIPIPE_NHR_H
