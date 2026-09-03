/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_TO_ALL_V_MESH_1D_H
#define INS_TEMP_ALL_TO_ALL_V_MESH_1D_H

#include "alg_v2_template_base.h"
#include "cost_model.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

const uint32_t ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE = 16; // fullmesh最大的并发数量

class InsTempAlltoAllVMesh1D : public InsAlgTemplateBase {
public:
    InsTempAlltoAllVMesh1D() = default;
    explicit InsTempAlltoAllVMesh1D(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);

    ~InsTempAlltoAllVMesh1D() override;

    std::string Describe() const override
    {
        std::string info = "Template of alltoallv Mesh with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    // 现在的RunAsync就是之前的GenExtIns
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

private:
    HcclResult RunALLtoALL(
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
        const TemplateDataParams& tempAlgParams, const u32 myAlgRank);
    HcclResult PreCopy(
        const TemplateDataParams& tempAlgParams, const ThreadHandle& thread, const u32 myRankCclBuffIdx,
        const u32 remoteRank, const u64& sendSize, const u64& sendCount, const u64& sendOffset) const;
    HcclResult PreCopyByLoop(
        const std::vector<u32>& commRanks, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const std::vector<ThreadHandle>& threads, const TemplateDataParams& tempAlgParams);
    HcclResult PostCopy(
        const TemplateDataParams& tempAlgParams, const ThreadHandle& thread, const u32 myRankCclBuffIdx,
        const u32 remoteRank, const u64& recvSize, const u64& recvCount, const u64& recvOffset) const;
    HcclResult LocalCopyForMyRank(
        const TemplateDataParams& tempAlgParams, const ThreadHandle& thread, const u32 myAlgRank,
        const u32 queIdx) const;
    void CalcCommRankSetForOneLoop(const u32 roundIdx, const u32 remainRankSize, std::vector<u32>& commRanks) const;
    u32 CalcCommLoops() const;
    void CalcCclBuffIdx(u32 remoteRank, u32& myRankCclBuffIdx, u32& remoteCclBuffIdx) const;
    HcclResult RunSendRecvByLoop(
        const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams,
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
        const u32 roundIdx, const u32 commLoops);
    HcclResult RunSendRecvByChannel(
        const TemplateDataParams& tempAlgParams, const u32 roundIdx, const u32 curValidChannelsSize,
        const std::vector<ChannelInfo>& curChannels, const u32 remoteRank, const std::vector<ThreadHandle>& threads,
        const u32 commLoops) const;
    HcclResult RunSendRecv(
        const SendRecvInfo& sendRecvInfo, const DataInfo& sendInfo, const DataInfo& recvInfo,
        const ThreadHandle& thread, const u32 channelId) const;
    HcclResult PreSyncInterThreadsPerRank(
        const ThreadHandle& mainThreadCurRank, const std::vector<ThreadHandle>& subThreadsCurRank) const;
    HcclResult PostSyncInterThreadsPerRank(
        const ThreadHandle& mainThreadCurRank, const std::vector<ThreadHandle>& subThreadsCurRank) const;

    u64 dataTypeSize_{0};
    bool isDmaRead_{false};
    u32 concurrentSendRecvNum_{1};
    std::vector<u64> sendCountsSplit_;
    std::vector<u64> sendSizeSplit_;
    std::vector<u64> sendOffsetSplit_;
    std::vector<u64> recvCountsSplit_;
    std::vector<u64> recvSizeSplit_;
    std::vector<u64> recvOffsetSplit_;
};

} // namespace ops_hccl

#endif // INS_TEMP_ALL_TO_ALL_V_MESH_1D_H
