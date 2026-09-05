/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_SCATTER_OMNIPIPE_MESH1D_H
#define INS_TEMP_SCATTER_OMNIPIPE_MESH1D_H

#include <atomic>
#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class InsTempScatterOmniPipeMesh1D : public InsAlgTemplateBase {
public:
    explicit InsTempScatterOmniPipeMesh1D(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);

    ~InsTempScatterOmniPipeMesh1D() override;

    std::string Describe() const override
    {
        std::string info = "Template of scatter Mesh with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    // 现在的KernelRun就是之前的GenExtIns
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 CalcScratchSlice(u64 dataSize) const;
    void SetRoot(u32 root);

    HcclResult DoLocalCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads);

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 GetThreadNum() const override;
    void SetDoTask(bool doTask);
    u64 xyTotalRankSize_{0};

private:
    HcclResult RunScatter(
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
        const TemplateDataParams& tempAlgParam);
    // root分支：遍历子通信域所有非己rank，按CCU映射只发该rank对应的那组piece
    HcclResult RunRootScatter(
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
        const TemplateDataParams& tempAlgParam, u32 myAlgRank);
    HcclResult SendRootDataToRank(
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
        const TemplateDataParams& tempAlgParam, u32 algRank, u64 repeatNum, u32& originIndex, u32& threadIdx);
    // 非root分支：按CCU映射只收属于自己的那组piece
    HcclResult RunNonRootScatter(
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
        const TemplateDataParams& tempAlgParam, u32 myAlgRank);
    u32 CalcNonRootOriginIndex(u32 myAlgRank, u32 rootAlgRank) const;
    // root分支：为某个非己rank构建tx批数据切片（for rpt循环体）
    HcclResult BuildRootTxBatchSlices(
        const StepSliceInfo& stepSliceInfo, u32 rowIdx, u32 remoteRank, void* srcPtr, u64 srcBaseOff,
        u64 outBuffBaseOff, void* remoteCclBuffAddr, u64 repeatNum, u32 originIndex,
        std::vector<DataSlice>& txSrcSlices, std::vector<DataSlice>& txDstSlices);
    // 非root分支：构建rx批数据切片（for rpt循环体）
    HcclResult BuildNonRootRxBatchSlices(
        const StepSliceInfo& stepSliceInfo, u32 rowIdx, u32 rootRank, void* localCclBuffAddr, void* remoteCclBuffAddr,
        u64 inBuffBaseOff, u64 outBuffBaseOff, u64 repeatNum, u32 myOriginIndex, std::vector<DataSlice>& rxSrcSlices,
        std::vector<DataSlice>& rxDstSlices);
    std::atomic<bool> doTask_{false};
};

} // namespace ops_hccl

#endif // INS_TEMP_SCATTER_OMNIPIPE_MESH1D_H
