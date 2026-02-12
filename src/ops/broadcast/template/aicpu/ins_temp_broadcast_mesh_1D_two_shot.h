/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_BROADCAST_MESH_1D_TWO_SHOT_H
#define INS_TEMP_BROADCAST_MESH_1D_TWO_SHOT_H

#include <cstring>
#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"


namespace ops_hccl {


class InsTempBroadcastMesh1DTwoShot : public InsAlgTemplateBase {
public:
    explicit InsTempBroadcastMesh1DTwoShot(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempBroadcastMesh1DTwoShot() override;

    std::string Describe() const override
    {
        std::string info = "Template of broadcast Mesh1D two shot with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams &tempAlgParams,
                         const TemplateResource& templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo, 
                       AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest &resourceReques) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() override;
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;
    void SetRoot(u32 root);

private:
    HcclResult PostCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads);
    HcclResult CalcCommRankSetforScatter(const u32 groupRankSize, std::vector<u32> &commRanks) const;
    HcclResult CalcCommRankSetforAllGather(const u32 groupRankSize, std::vector<u32> &commRanks) const;
    HcclResult RunScatter(const std::vector<u32> &commRanks, const TemplateDataParams &tempAlgParams,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads, const RankSliceInfo &sliceInfoVec);
    HcclResult RunAllGather(const std::vector<u32> &commRanks, const TemplateDataParams &tempAlgParams,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads, const RankSliceInfo &sliceInfoVec);
    HcclResult RootSendData(const u64 memOffset, const u32 remoteRank, const TemplateDataParams &tempAlgParams,
            const std::vector<ThreadHandle> &threads, const u32 id, const std::map<u32, std::vector<ChannelInfo>> &channels, const RankSliceInfo &sliceInfoVec) const;
    HcclResult RankRecvData(const u64 memOffset, const u32 remoteRank, const TemplateDataParams &tempAlgParams,
            const std::vector<ThreadHandle> &threads, const u32 id, const std::map<u32, std::vector<ChannelInfo>> &channels, const RankSliceInfo &sliceInfoVec) const;
    
    HcclResult CalcDataSliceInfo(const u64 dataSize, RankSliceInfo &sliceInfoVec);

    u64 dataTypeSize_{0};
    std::map<u32, u32> tempVirtRankMap_;
    
};

} // namespace Hccl

#endif // INS_TEMP_BROADCAST_MESH_1D_TWO_SHOT_H