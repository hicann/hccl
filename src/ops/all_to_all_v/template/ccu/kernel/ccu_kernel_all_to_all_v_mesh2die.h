/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_2DIE_H
#define HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_2DIE_H

#include <vector>
#include <ios>
#include "utils.h"
#include "template_utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using RankId = u32;

class CcuKernelArgAllToAllVMesh2Die : public hcomm::CcuKernelArg {
public:
    CcuKernelArgAllToAllVMesh2Die(uint32_t rankId, const OpParam &opParam,
        const std::vector<std::vector<RankId>> &subCommRanks, bool withMyRank, const std::vector<RankId> &rankGroup)
        : rankId_(rankId), opParam_(opParam), subCommRanks_(subCommRanks), withMyRank_(withMyRank),
          rankGroup_(rankGroup) {}

    ~CcuKernelArgAllToAllVMesh2Die() override {}

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllToAllVMesh2Die", opParam_, subCommRanks_);
        return signature;
    }

    uint32_t rankId_;
    OpParam opParam_;
    std::vector<std::vector<RankId>> subCommRanks_;
    bool withMyRank_;
    std::vector<RankId> rankGroup_;
};

class CcuTaskArgAllToAllVMesh2Die : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllToAllVMesh2Die(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
        const A2ASendRecvInfo& localSendRecvInfo)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), localSendRecvInfo_(localSendRecvInfo) {}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    A2ASendRecvInfo localSendRecvInfo_;
};

class CcuKernelAllToAllVMesh2Die : public CcuKernelAlgBase {
public:
    CcuKernelAllToAllVMesh2Die(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllToAllVMesh2Die() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

protected:
    // a2av 对每个对端的发送接收信息
    struct A2AVSingleSendRecvInfo {
        CcuRep::Variable sendOffset;
        CcuRep::Variable recvOffset;
        CcuRep::Variable sendTailSize;  // 本rank给其他rank要发的尾块数据
        GroupOpSize      sendTailGoSize;
        CcuRep::Variable sendLoopNum;   // 本rank给其他rank要发的轮数
    };

private:
    HcclResult InitResources();
    void LoadArgs();
    void ExchangeInfoSync();
    void PostSync();

    void DoAll2AllVMultiLoop();
    void LoopStep();
    void CalcGroupSrcDst();
    void GroupCopyToDstOutput(uint32_t peerId);
    void WriteToDstOutput(uint32_t peerId);

    const uint32_t RANK_EVEN = 2;   // 只支持矩形rank分布

    const uint32_t GO_ADDR_OFFSET_IDX = 0;
    const uint32_t GO_LOOP_PARAM_IDX = 1;
    const uint32_t GO_PARALLEL_PARAM_IDX = 2;
    const uint32_t GO_RESIDUAL_IDX = 3;

    const uint64_t MAX_TRANSPORT_SIZE = UB_MAX_TRANS_SIZE;

    uint32_t rankId_{0};        // 全局rankId
    bool withMyRank_{false};
    uint32_t localSize_{0};     // 本rank所在DIE的总rank数
    uint32_t localId_{0};       // 本rank所在DIE的编号，固定放在末尾
    uint32_t peerSize_{0};      // 本次kernel调用需要数据搬移的对端数量，包含LocalCopy
    uint32_t logicId_{0};
    std::vector<RankId> rankGroup_;

    std::vector<ChannelHandle> channels_;

    hcomm::CcuRep::Variable input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;

    hcomm::CcuRep::LocalAddr localSrc_; // for LocalCopy
    hcomm::CcuRep::LocalAddr localDst_;

    std::vector<hcomm::CcuRep::LocalAddr> src_;
    std::vector<hcomm::CcuRep::RemoteAddr> dst_;

    hcomm::CcuRep::Variable xnConst1_;
    hcomm::CcuRep::Variable completedRankCount_;
    hcomm::CcuRep::Variable xnMaxTransportSize_;
    GroupOpSize xnMaxTransportGoSize_;
    hcomm::CcuRep::Variable curSendTailSize_;
    GroupOpSize curSendTailGoSize_;
    std::vector<A2AVSingleSendRecvInfo> sendRecvInfo_;

    uint16_t selfBit_{0};
    uint16_t allBit_{0};

    // 在本地的搬运完成标记
    hcomm::CcuRep::CompletedEvent event_;
};
} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_2DIE_H
