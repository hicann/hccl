/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_TO_ALL_HIER_MESH_STAGE_H
#define INS_TEMP_ALL_TO_ALL_HIER_MESH_STAGE_H

#include "ins_temp_all_to_all_hier_stage_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

struct StageParams {
    u32 otherDimRankSize{0};
    u32 stageRankSize{0};
    u32 srcGroupSize{0};
    u32 targetGroupSize{0};
    u32 subSlotCount{0};
};

class InsTempAlltoAllHierMeshStage : public InsTempAlltoAllHierStageBase {
public:
    InsTempAlltoAllHierMeshStage(
        const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks, u32 stageIndex);

    ~InsTempAlltoAllHierMeshStage() override = default;

    std::string Describe() const override
    {
        return "Hierarchical AllToAll Mesh Stage, stageIndex=" + std::to_string(stageIndex_)
               + ", rankSize=" + std::to_string(templateRankSize_);
    }

    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;
    std::string GetAlgoType() const override { return "MeshStage"; }

private:
    static constexpr u32 INVALID_ALG_RANK = 0xFFFFFFFF;

    u32 FindMyAlgRank() const;
    void InitStageParams(const TemplateDataParams& tempAlgParams, StageParams& params);
    HcclResult LocalCopyForMyGroup(
        const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource, u32 myAlgRank,
        u32 targetGroupSize, u32 subSlotCount);
    HcclResult LocalCopyForMyRank(
        const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource, u32 myAlgRank,
        u32 srcGroupSize, u32 subSlotCount);
    void BuildSendSlices(
        const TemplateDataParams& tempAlgParams, u32 myAlgRank, u32 peerAlgRank, void* remoteCclBuffAddr,
        u32 targetGroupSize, u32 subSlotCount, std::vector<DataSlice>& txSrcSlices,
        std::vector<DataSlice>& txDstSlices);
    void BuildRecvSlices(
        const TemplateDataParams& tempAlgParams, u32 myAlgRank, u32 peerAlgRank, void* remoteCclBuffAddr,
        const StageParams& params, std::vector<DataSlice>& rxSrcSlices, std::vector<DataSlice>& rxDstSlices);
    HcclResult ExecuteSendRecv(
        const ChannelInfo& linkSend, const ChannelInfo& linkRecv, const ThreadHandle& thread,
        const std::vector<DataSlice>& txSrcSlices, const std::vector<DataSlice>& txDstSlices,
        const std::vector<DataSlice>& rxSrcSlices, const std::vector<DataSlice>& rxDstSlices, u32 remoteRank);
    HcclResult RunSendRecv(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource,
        u32 myAlgRank, const StageParams& params);
    HcclResult PreSyncThreads(const TemplateResource& templateResource);
    HcclResult PostSyncThreads(const TemplateResource& templateResource);
    HcclResult ValidateAndInit(const TemplateDataParams& tempAlgParams, u32& myAlgRank, StageParams& params);

    u64 dataTypeSize_{0};
    HcclDataType dataType_{HCCL_DATA_TYPE_INT8};
    u64 cclBufferCountPerRank_{0};
    u64 totalCount_{0};
};

} // namespace ops_hccl

#endif // !INS_TEMP_ALL_TO_ALL_HIER_MESH_STAGE_H
