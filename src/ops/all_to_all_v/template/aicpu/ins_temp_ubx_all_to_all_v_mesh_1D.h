/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_UBX_ALL_TO_ALL_V_MESH_1D_H
#define INS_TEMP_UBX_ALL_TO_ALL_V_MESH_1D_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class InsTempUBXAllToAllVMesh1D : public InsAlgTemplateBase {
public:
    InsTempUBXAllToAllVMesh1D() = default;
    explicit InsTempUBXAllToAllVMesh1D(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);

    ~InsTempUBXAllToAllVMesh1D() override;

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

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

private:
    void GetNotifyIdxMainToClos(std::vector<u32>& notifyIdxMianToSub);
    void GetNotifyIdxClosToMain(std::vector<u32>& notifyIdxSubToMain);
    void GetNotifyIdxMainToFullMesh(std::vector<u32>& notifyIdxMianToSub);
    void GetNotifyIdxFullMeshToMain(std::vector<u32>& notifyIdxSubToMain);
    HcclResult
    InitParam(const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource);
    HcclResult GetBoardSendRecvMatrix(u32 n, std::vector<std::vector<u32>>& sendRecvMatrix);
    HcclResult GetRankSendRecvMatrix(u32 board1, u32 board2, std::vector<std::vector<u32>>& rankSendRecvMatrix);
    HcclResult GetRankNumPerBoard(TemplateResource& templateResource);
    HcclResult CheckPathNum(TemplateResource& templateResource);
    HcclResult RunFullMesh(const TemplateDataParams& tempAlgParams, TemplateResource& templateResource);
    HcclResult
    RunPairwise(const TemplateDataParams& tempAlgParams, TemplateResource& templateResource, u32 targetBoard);

    u64 dataTypeSize_{0};
    bool isDmaRead_{false};
    u32 concurrentSendRecvNum_{1};
    std::vector<u64> sendCountsSplit_;
    std::vector<u64> sendSizeSplit_;
    std::vector<u64> sendOffsetSplit_;
    std::vector<u64> recvCountsSplit_;
    std::vector<u64> recvSizeSplit_;
    std::vector<u64> recvOffsetSplit_;

    bool needDealWithFullMeshInfo_{false};
    u32 myAlgRank_{0};
    u32 rankNumPerBoard_{0};
    u32 maxPathNum_{4};         // 跨框最多4jetty
    u32 maxRankNumPerBoard_{4}; // ubx机型fullmesh内最多4P
    u32 currBoard_{0};
    u32 currRankIndex_{0};
    u32 boardNum_{0};
    u32 algBoardNum_{0};
    u64 dataCountPerRank_{0};
    u64 dataSizePerRank_{0};
    u64 scratchBufferSizePerRank_{0};
    u64 dataStridePerRank_{0};
    u64 curProcessedDataCount_{0};
    u64 curDataCount_{0};
    u64 curDataSize_{0};
    std::vector<std::vector<DataSlice>> localCopyInfo_;
    std::vector<std::vector<DataSlice>> localCopyInfoFullMesh_;
    std::vector<ThreadHandle> subThreadsBoard_;
    std::vector<ThreadHandle> subThreadsFullMesh_;
};

} // namespace ops_hccl

#endif // INS_TEMP_UBX_ALL_TO_ALL_V_MESH_1D_H
