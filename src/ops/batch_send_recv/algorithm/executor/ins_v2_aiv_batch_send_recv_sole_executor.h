/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCLV2_INS_V2_AIV_BATCH_SEND_RECV_SOLE_EXECUTOR_H
#define HCCLV2_INS_V2_AIV_BATCH_SEND_RECV_SOLE_EXECUTOR_H

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "executor_v2_base.h"
#include "hccl_aiv_utils.h"
#include "template_utils.h"
#include "topo_match_one_level.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
class InsV2AivBatchSendRecvSoleExecutor : public InsCollAlgBase {
public:
    InsV2AivBatchSendRecvSoleExecutor() = default;
    ~InsV2AivBatchSendRecvSoleExecutor() override = default;

    HcclResult CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& algAttrs) override;

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    std::string Describe() const override { return "Aiv BatchSendRecv Sole Executor."; }

    std::vector<CostModelParam> CalcCostCoeff(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param) override;

    AlgNetMeta GetAlgNetMeta(
        const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const override;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

private:
    // 清理上一次调用残留的任务队列，executor 对象可能被复用
    void ResetTaskState();

    // 按 (remoteRank 升序, count 降序, dataType 降序) 排，收发共用：保证同 remoteRank 连续、
    // 同 pair 组内顺序一致（两侧才能切出相同的 sliceIdx）。AIV 收发全并发，不需要 pair-wise 相位错开。
    bool SortTasks(const HcclSendRecvItem* a, const HcclSendRecvItem* b) const;

    // 把任务按 send/recv 分到两个队列并排序。函数名不再带 pair-wise：这里做的是纯分组排序。
    HcclResult BuildTaskDeques(const HcclSendRecvItem* sendRecvInfo, u32 itemNum);
    // 自收发不再由 host 拷贝：这里只做配对校验，并产出 "self send item -> 配对 recv 的 buf" 映射，
    // 供切片时填进 SELF_SEND 的 dstAddr。
    HcclResult CheckSelfSendRecvPairs();

    HcclResult CalcSlicesAndRounds(u64 cclCapacity);

    HcclResult DispatchOneRound(
        u64 roundIdx, const OpParam& param, const TemplateResource& templateAlgRes,
        std::shared_ptr<InsAlgTemplate>& algTemplate);

    u32 itemNum_{0};
    const HcclSendRecvItem* itemPtr_{nullptr};
    std::vector<ThreadHandle> threads_;
    HcclMem cclMem_{};
    AlgHierarchyInfoForAllLevel algHierarchyInfo_{};

    std::deque<const HcclSendRecvItem*> sendDeque_;
    std::deque<const HcclSendRecvItem*> recvDeque_;
    // 自收发 send item -> 配对 recv 的目的地址：自收发是直拷，发方向必须手里有目的地址。
    std::map<const HcclSendRecvItem*, u64> selfDst_;
    std::vector<std::vector<AivBsrItemArgs>> roundItems_;
    u64 flagBlock_{0}; // 每对端 flag 信箱格数（= itemCapPerPair），随每轮下发传给 device 做动态分块编址
};

} // namespace ops_hccl

#endif // HCCLV2_INS_V2_AIV_BATCH_SEND_RECV_SOLE_EXECUTOR_H
