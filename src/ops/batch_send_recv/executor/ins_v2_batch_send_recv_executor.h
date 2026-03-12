/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCLV2_INS_V2_BATCH_SEND_RECV_EXECUTOR_H
#define HCCLV2_INS_V2_BATCH_SEND_RECV_EXECUTOR_H

#include <unordered_set>
#include <algorithm>

#include "alg_param.h"
#include "channel.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "template_utils.h"

namespace ops_hccl {
class InsV2BatchSendRecvExecutor : public InsCollAlgBase {
public:
    explicit InsV2BatchSendRecvExecutor();
    ~InsV2BatchSendRecvExecutor() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    HcclResult CalcRes(HcclComm comm, const OpParam& param,
        const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
        AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

protected:

    struct SendRecvSlice {
        void* addr_;
        u64 size_;
        u32 remoteRank_;
        SendRecvSlice(void* addr, u64 size, u32 remoteRank) :
            addr_(addr), size_(size), remoteRank_(remoteRank) {}
    }; // 切片任务信息

    std::deque<SendRecvSlice> sendDataSilces_;
    std::deque<SendRecvSlice> recvDataSilces_;
    std::vector<std::map<u32 ,std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;

private:
    // 排序
    bool SortSendItems(const HcclSendRecvItem* a, const HcclSendRecvItem* b) const;
    bool SortRecvItems(const HcclSendRecvItem* a, const HcclSendRecvItem* b) const;
    bool SortSelfItems(const HcclSendRecvItem* a, const HcclSendRecvItem* b) const;
    HcclResult GetPairWiseList(const HcclSendRecvItem *sendRecvInfo, u32 itemNum);
    
    // 实现自发自收
    HcclResult ProcessSelfSendRecvTasks(ThreadHandle& thread);
    
    // 收发任务切片
    HcclResult CalcSendSlices();
    HcclResult CalcRecvSlices();

    // 获取数据发送&接收channel
    HcclResult GetSendChannel(u32 remoteRank, ChannelInfo& sendChannel) const;
    HcclResult GetRecvChannel(u32 remoteRank, ChannelInfo& recvChannel) const;

    // 实现数据发送&接收
    HcclResult ProcessSendDataSlice(SendRecvSlice& sendSlice, ThreadHandle& thread) const;
    HcclResult ProcessRecvDataSlice(SendRecvSlice& recvSlice, ThreadHandle& thread) const;
    HcclResult RunLoopSendRecv();

    static constexpr u32 channelNumPerRankPair_ = 2;
    const HcclSendRecvItem* itemPtr_ = nullptr;
    HcclMem cclMem_{HCCL_MEM_TYPE_DEVICE, nullptr, 0};
    u32 itemNum_ = 0;
    u64 maxRoundTransferSize_ = 0; // 单轮最多能够传输的size
    std::set<u32> commTargetUserRankSet_; // 所有需要通信的remoteRank ID集合
    std::deque<const HcclSendRecvItem*> sendToSelfDeque_;
    std::deque<const HcclSendRecvItem*> recvFromSelfDeque_;
    std::deque<const HcclSendRecvItem*> sendDeque_;
    std::deque<const HcclSendRecvItem*> recvDeque_;
};
} // namespace ops_hccl
#endif