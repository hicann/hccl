/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_REDUCE_MESH_1D_TWO_SHOT_H
#define AICPU_REDUCE_MESH_1D_TWO_SHOT_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

struct SplitSliceInfo {
    u64 offset{0};
    u64 size{0};
    u64 count{0};

    SplitSliceInfo(const u64 offset, const u64 size, const u64 count) : offset(offset), size(size), count(count) {}
};

class ReduceMesh1DTwoShot : public InsAlgTemplateBase {
public:
    ReduceMesh1DTwoShot() = default;
    explicit ReduceMesh1DTwoShot(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);

    ~ReduceMesh1DTwoShot() override;

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter Mesh Two Shot with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    // 现在的Kernel就是之前的GenExtIns
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    void SetRoot(u32 root) const;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    u64 GetThreadNum() const override;

private:
    HcclResult CalcSlice();
    HcclResult RunReduceScatter(
        const TemplateDataParams& tempAlgParam, const OpParam& param,
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads);
    HcclResult RunGatherToRoot(
        const TemplateDataParams& tempAlgParam, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const std::vector<ThreadHandle>& threads);

    HcclResult GatherLocalData(const TemplateDataParams& tempAlgParam, const std::vector<ThreadHandle>& threads) const;
    HcclResult GatherRemoteData(
        const TemplateDataParams& tempAlgParam, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const std::vector<ThreadHandle>& threads);
    HcclResult SendToRoot(
        const TemplateDataParams& tempAlgParam, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const std::vector<ThreadHandle>& threads);

    HcclResult SendRecvDataToPeers(
        const TemplateDataParams& tempAlgParam, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const std::vector<ThreadHandle>& threads);
    HcclResult DoLocalReduce(
        const TemplateDataParams& tempAlgParam, const OpParam& param, const std::vector<ThreadHandle>& threads);

    struct LocalSliceInfo {
        u64 sliceSize;
        u64 sliceCount;
        u64 sliceOffset;
        u64 inBuffBaseOffset;
        u64 outBuffBaseOffset;
        u64 hcclBuffBaseOffset;
        void* localInBuffPtr;
        void* localOutBuffPtr;
        void* localHcclBuffPtr;
    };
    LocalSliceInfo GetLocalSliceInfo(const TemplateDataParams& tempAlgParam) const;

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

    u64 processSize_{0};
    u64 count_{0};
    u32 myIdx_ = UINT32_MAX; // 本rank在通信域内的索引
    u32 myRankIdx_{0};
    u32 threadNum_{0};
    std::vector<u32> notifyIdxMainToSub_;
    std::vector<u32> notifyIdxSubToMain_;

    std::vector<SplitSliceInfo> sliceInfoList_;
    std::vector<u32> rankList_;
};

} // namespace ops_hccl

#endif // OPEN_HCCL_INS_TEMP_REDUCE_MESH_H
