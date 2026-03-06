/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_1D_MULTI_JETTY_H_
#define HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_1D_MULTI_JETTY_H_

#include <vector>
#include <ios>
#include "utils.h"
#include "alg_param.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "template_utils.h"

namespace ops_hccl {

class CcuKernelArgAllToAllVMesh1DMultiJetty : public CcuKernelArg {
public:
    explicit CcuKernelArgAllToAllVMesh1DMultiJetty(uint64_t dimSize, uint32_t rankId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks,
                                                    const std::vector<uint32_t>& jettyNums)
        : dimSize_(dimSize),
          rankId_(rankId),
          opParam_(opParam),
          subCommRanks_(subCommRanks),
          jettyNums_(jettyNums)
    {
        HCCL_DEBUG("[CcuKernelArgAllToAllVMesh1DMultiJetty] dimSize: %llu, rankId: %u",
                   dimSize_, rankId_);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllToAllVMesh1DMultiJetty", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
    std::vector<uint32_t>                   jettyNums_;
};

class CcuTaskArgAllToAllVMesh1DMultiJetty : public CcuTaskArg {
public:
    explicit CcuTaskArgAllToAllVMesh1DMultiJetty(uint64_t inputAddr, uint64_t outputAddr,
        uint64_t token, uint64_t srcOffset, uint64_t dstOffset,
        const A2ASendRecvInfo& localSendRecvInfo) :
        inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), srcOffset_(srcOffset), dstOffset_(dstOffset),
        localSendRecvInfo_(localSendRecvInfo)
    {
        HCCL_DEBUG("[CcuTaskArgAllToAllVMesh1DMultiJetty] inputAddr: %lu, outputAddr: %lu, token: %lu, srcOffset: %lu, "
                   "dstOffset: %lu",
                   inputAddr_, outputAddr_, token_, srcOffset_, dstOffset_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t srcOffset_;
    uint64_t dstOffset_;
    A2ASendRecvInfo localSendRecvInfo_;
};

class CcuKernelAllToAllVMesh1DMultiJetty : public CcuKernelAlgBase {
public:
    CcuKernelAllToAllVMesh1DMultiJetty(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllToAllVMesh1DMultiJetty() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;

private:
    // a2a 对每个对端的发送接收信息
    struct A2AsingleSendRecvInfo {
        hcomm::CcuRep::Variable sliceSize;               // 数据块切片大小
        hcomm::CcuRep::Variable tailSliceSize;           // 数据块切片尾片大小
        hcomm::CcuRep::Variable lastSliceSize;           // 尾数据块切片大小
        hcomm::CcuRep::Variable lastTailSliceSize;       // 尾数据块切片尾片大小
        hcomm::CcuRep::Variable loopNum;
        hcomm::CcuRep::Variable sendOffset;
        hcomm::CcuRep::Variable recvOffset;
        GroupOpSize      tailGoSize;
    };
    HcclResult InitResource();
    HcclResult LoadAll2allSendRecvInfo(A2AsingleSendRecvInfo &sendRecvInfo);
    HcclResult LoadArgs();
    HcclResult PreSync();
    HcclResult PostSync();
    HcclResult DoAll2AllVMultiLoop();
    HcclResult DoAll2AllVLastBlock(u32 rankIdx, u32 channelIdx);
    HcclResult DoAll2AllVBlock(u32 rankIdx, u32 channelIdx);
    HcclResult CalcGroupSrcDst();

    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    std::vector<ChannelHandle> channels_;
    std::vector<hcomm::CcuRep::Variable> input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable srcOffset_;
    hcomm::CcuRep::Variable dstOffset_;
    std::vector<A2AsingleSendRecvInfo> sendRecvInfo_;
    std::vector<uint32_t> jettyNums_;

    hcomm::CcuRep::Variable completedRankCount_;
    hcomm::CcuRep::Variable xnMaxTransportSize_;
    GroupOpSize xnMaxTransportGoSize_; // 本地ccu的尾块
    hcomm::CcuRep::Variable xnConst1_;

    std::vector<hcomm::CcuRep::LocalAddr>      src_;
    hcomm::CcuRep::LocalAddr                   myDst_;
    std::vector<hcomm::CcuRep::RemoteAddr>     remoteDst_;
    std::vector<hcomm::CcuRep::CompletedEvent> eventList_;
};
} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_1D_MULTI_JETTY_H_