/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_V_MESH_1D_MESH_CHUNK_H
#define INS_TEMP_REDUCE_SCATTER_V_MESH_1D_MESH_CHUNK_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

// ReduceScatterV MeshChunk：每片切 N-1 段，共 N-1 步、每步向全部对端各发 1 段（全带宽）；
// 段错开保证每段同时单写者（WriteReduce 串行归约、确定性）。对称：直写对端 userOut（零拷贝）；
// 非对称：落对端 CCL 槽，收尾 PostCopy。
class InsTempReduceScatterVMesh1DMeshChunk : public InsAlgTemplateBase {
public:
    explicit InsTempReduceScatterVMesh1DMeshChunk(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempReduceScatterVMesh1DMeshChunk() override;

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter V Mesh Chunk with tempRankSize ";
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

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

private:
    HcclResult LocalDataCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const;
    HcclResult RunReduceScatterV(
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
        const TemplateDataParams& tempAlgParams);
    HcclResult DoMeshChunk(
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
        const TemplateDataParams& tempAlgParams);
    HcclResult CalcPeerSegList();
    HcclResult PostCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const;
    void NotifyIdxMainToSubInMeshChunk(std::vector<u32>& notifyIdxMainToSub);
    void NotifyIdxSubToMainInMeshChunk(std::vector<u32>& notifyIdxSubToMain);
    u32 rankIdx_{0};
    u64 dataTypeSize_{0};
    bool supportSymmetricMemAccess_{false};
    std::vector<u64> allRankSliceSize_;
    std::vector<u64> allRankDispls_;
    std::vector<u64> allRankCounts_;
    std::vector<std::vector<u64>> peerSegSize_; // 每片（对端 j）的 N-1 段大小，全 rank 切分一致
    std::vector<std::vector<u64>> peerSegOff_;  // 每片段起始偏移（片内），与 peerSegSize_ 对应
};

} // namespace ops_hccl

#endif // INS_TEMP_REDUCE_SCATTER_V_MESH_1D_MESH_CHUNK_H
