/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_ORDER_PRESERVED_GROUP_H
#define INS_TEMP_REDUCE_SCATTER_ORDER_PRESERVED_GROUP_H

#include <cstring>
#include <cmath>
#include <algorithm>
#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"
#include "ins_v2_reduce_scatter_order_preserved_executor.h"

namespace ops_hccl {

// 保序 ReduceScatter "分组 all2all" template
//   - RunAllToAll 用"按环上距离 distance 对称分批"，保证 A→B 与 B→A 落在同一批次，
//     notify 不跨批等待，从根本上避免大卡数场景的 notify 死锁
//   - 适用于大卡数场景（rankSize > MAX_RANK_NUM_FOR_ORDER_PRESERVED）
class InsTempReduceScatterOrderPreservedGroup : public InsAlgTemplateBase {
public:
    InsTempReduceScatterOrderPreservedGroup() = default;
    explicit InsTempReduceScatterOrderPreservedGroup(const OpParam& param, const u32 rankId,
        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempReduceScatterOrderPreservedGroup() override;

    std::string Describe() const override
    {
        std::string info = "Template of Order Preserved ReduceScatter Group with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams,
        TemplateResource& templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

    void SetMemBlockInfo(const MemBlockInfo &info) { memBlockInfo_ = info; }

protected:
    u32 CalcOutputIndex(const u32 round, const u32 localRank);
    bool IsLastBlockData(const u32 outputIndex);
    bool IsLastRank(const u32 rankId);

    HcclResult PreLocalCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads);
    HcclResult SendSingleRankSlice(u32 dstRank, u32 myAlgRank, const std::map<u32, std::vector<ChannelInfo>> &channels,
        const TemplateDataParams &tempAlgParams, const ThreadHandle &thread);
    HcclResult RunAllToAll(const std::map<u32, std::vector<ChannelInfo>> &channels,
        const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams);
    HcclResult RunLocalReduce(const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams);
    HcclResult PostCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads);

    u64 processSize_{0};
    u64 count_{0};
    bool deterministicStrict_{false};
    MemBlockInfo memBlockInfo_;
    std::vector<u64> elemCountOut_;
    std::vector<u64> sizeOut_;
    std::vector<u64> elemOffset_;
    u32 channelsPerRank_{1};
};

} // namespace ops_hccl

#endif
