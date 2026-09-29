/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_REDUCE_1D_MESH_TWO_SHOT
#define INS_TEMP_ALL_REDUCE_1D_MESH_TWO_SHOT

#include "alg_v2_template_base.h"
#include "template_utils.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class InsTempAllReduceMesh1DTwoShot : public InsAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::MESH_TWOSHOT};
    InsTempAllReduceMesh1DTwoShot() = default;
    explicit InsTempAllReduceMesh1DTwoShot(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempAllReduceMesh1DTwoShot() override;

    std::string Describe() const override
    {
        std::string info = "Template of all reduce mesh 1D two shot with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    u64 GetThreadNum() const override;

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

private:
    HcclResult SplitData();

    HcclResult RunReduceScatter(
        const OpParam& param, const TemplateDataParams& tempAlgParams,
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads);
    HcclResult ScatterData(
        const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const std::vector<ThreadHandle>& threads);
    HcclResult ReduceData(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads);

    HcclResult RunAllGather(
        const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const std::vector<ThreadHandle>& threads);
    HcclResult GatherData(
        const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const std::vector<ThreadHandle>& threads);
    HcclResult ScatterSymmetricReadReduce(
        u64 sendSize, void* localInBuffPtr, u64 inBuffBaseOffset, u64 recvOffset, u64 recvSize, u64 recvCount,
        const ChannelInfo& sendRecvChannel, const std::vector<ThreadHandle>& threads, u32 remoteIdx);
    HcclResult GatherSymmetricRead(
        u64 sendSize, void* localOutBuffPtr, u64 outBuffBaseOffset, u64 sendOffset, u64 sendCount, u64 recvOffset,
        u64 recvSize, u64 recvCount, const ChannelInfo& sendRecvChannel, const std::vector<ThreadHandle>& threads,
        u32 remoteIdx);

    HcclResult PreSync(const std::vector<ThreadHandle>& threads);
    HcclResult PostSync(const std::vector<ThreadHandle>& threads);
    HcclResult CopyRsResultToOutput(const TemplateDataParams& tempAlgParams, const ThreadHandle& thread);

    bool needAicpuReduce_{false};
    u32 dataTypeSize_{0};
    u64 count_{0};
    u64 processSize_{0};

    u32 myRankIdx_{0};
    std::vector<SplitSliceInfo> sliceInfoList_;
    std::vector<u32> rankList_;
    bool supportSymmetricMemAccess_{false};
};

} // namespace ops_hccl

#endif // INS_TEMP_ALL_REDUCE_1D_MESH_TWO_SHOT
