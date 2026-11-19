/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_MESH_1D_MESH_CHUNK_H
#define INS_TEMP_REDUCE_SCATTER_MESH_1D_MESH_CHUNK_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class InsTempReduceScatterMesh1DMeshChunk : public InsAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::MESH_CHUNK};
    InsTempReduceScatterMesh1DMeshChunk() = default;
    explicit InsTempReduceScatterMesh1DMeshChunk(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempReduceScatterMesh1DMeshChunk() override;

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter Mesh Chunk with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    HcclResult PreCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const;
    HcclResult LocalDataCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads);
    HcclResult PostCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads);
    HcclResult CalcSliceInfoVec(const u64& dataSize, RankSliceInfo& sliceInfoVec);

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

    u64 GetThreadNum() const override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 ScratchSlotBaseOff(const TemplateDataParams& p, u32 repeatIdx, u32 algRank) const;
    HcclResult CalcDataSplitByPortGroup(
        const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
        std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset) override;

protected:
    // 注意: 本成员刻意遮蔽基类 CommonAlgTemplateBase::channelsPerRank_(同名不同体):
    // - sole 路径执行器调用的是基类 SetchannelsPerRank(写基类成员), 本成员保持默认 1,
    //   DoMeshChunk 的 min(peerChs.size(), channelsPerRank_) 因此恒取 1, 维持上游 sole
    //   单通道语义 —— 删除本遮蔽会让 sole 在多通道拓扑读到真实通道数, 复发多通道切分回归;
    // - Concur 路径由子类 override 的 SetchannelsPerRank 写/读本成员, 读写一致。
    // 勿"顺手清理"本遮蔽。
    u32 channelsPerRank_{1};
    std::vector<u64> elemCountOut_;
    std::vector<u64> sizeOut_;
    std::vector<u64> elemOffset_;

private:
    HcclResult RunReduceScatter(
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
        const TemplateDataParams& tempAlgParams, RankSliceInfo& sliceInfoVec);
    HcclResult DoMeshChunk(
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
        const TemplateDataParams& tempAlgParams, const std::vector<uint64_t>& sliceSize, const u32& repeatNum,
        const u32& myAlgRank, uint64_t& sliceSendOffset_, uint64_t& sliceRecvOffset_,
        const uint64_t& sliceRecvBaseOffset);
    void NotifyIdxMainToSubInMeshChunk(std::vector<u32>& notifyIdxMainToSub);
    void NotifyIdxSubToMainInMeshChunk(std::vector<u32>& notifyIdxSubToMain);
    u64 processSize_{0};
    u32 rankIdx_{0};
    u64 count_{0};
    u64 dataTypeSize_{0};
    bool supportSymmetricMemAccess_{false};
};

} // namespace ops_hccl

#endif // OPEN_HCCL_INS_TEMP_REDUCE_SCATTER_MESH_1D_MESH_CHUNK_H
