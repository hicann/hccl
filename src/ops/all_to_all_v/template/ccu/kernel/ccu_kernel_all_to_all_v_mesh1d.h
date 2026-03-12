/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_1D_H
#define HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_1D_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "../ccu_temp_all_to_all_v_mesh_1D.h"

namespace ops_hccl {
using namespace hcomm;

class CcuKernelArgAlltoAllVMesh1D : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAlltoAllVMesh1D(uint64_t dimSize, uint32_t rankId, bool loadFromMem, const OpParam& opParam,
                                            const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          loadFromMem_(loadFromMem),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgAlltoAllVMesh1D] dimSize: %lu, rankId: %u",
                   dimSize_, rankId_);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAlltoAllVMesh1D", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t dimSize_;
    uint32_t rankId_;
    OpParam opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
    bool loadFromMem_;
};

class CcuTaskArgAlltoAllVMesh1D : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAlltoAllVMesh1D(uint64_t inputAddr, uint64_t outputAddr, std::vector<uint64_t> sliceSize,
        uint64_t token, uint64_t srcOffset, uint64_t dstOffset,const A2ASendRecvInfo& localSendRecvInfo) :
        inputAddr_(inputAddr), outputAddr_(outputAddr), sliceSize_(sliceSize), token_(token), 
        srcOffset_(srcOffset), dstOffset_(dstOffset), localSendRecvInfo_(localSendRecvInfo)
    {
        HCCL_DEBUG("[CcuTaskArgAlltoAllVMesh1D] inputAddr: %lu, outputAddr: %lu, sliceSize: %lu, "
                   "srcOffset: %lu, dstOffset: %lu",
                   inputAddr_, outputAddr_, sliceSize_, srcOffset_, dstOffset_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    std::vector<uint64_t> sliceSize_;
    uint64_t token_;
    uint64_t srcOffset_;
    uint64_t dstOffset_;
    A2ASendRecvInfo localSendRecvInfo_;
};

class CcuKernelAlltoAllVMesh1D : public CcuKernelAlgBase {
public:
    CcuKernelAlltoAllVMesh1D(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAlltoAllVMesh1D() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    // a2a 对每个对端的发送接收信息
    struct A2AsingleSendRecvInfo {
        CcuRep::Variable tailSize;
        CcuRep::Variable loopNum;
        CcuRep::Variable sendOffset;
        CcuRep::Variable recvOffset;
        GroupOpSize      tailGoSize;
    };

    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    hcomm::CcuRep::Variable repeatNum_;
    std::vector<ChannelHandle> channels_;
    std::vector<hcomm::CcuRep::Variable> input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable sliceSize_;
    hcomm::CcuRep::Variable srcStride_;
    hcomm::CcuRep::Variable srcOffset_;
    hcomm::CcuRep::Variable dstOffset_;
    hcomm::CcuRep::Variable groupOpSize_;
    hcomm::CcuRep::Variable a2avXnAddr_;
    hcomm::CcuRep::Variable xnLength_;
    hcomm::CcuRep::Variable completedRankCount_;
    hcomm::CcuRep::Variable xnMaxTransportSize_;
    GroupOpSize xnMaxTransportGoSize_;
    std::vector<A2AsingleSendRecvInfo> sendRecvInfo_;
    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    uint16_t allOtherBit_{0};
    hcomm::CcuRep::LocalAddr                   myDst_;
    std::vector<hcomm::CcuRep::RemoteAddr>     dst_;
    std::vector<hcomm::CcuRep::LocalAddr>      src_;
    hcomm::CcuRep::CompletedEvent event_;
    hcomm::CcuRep::Variable flag_; // 用以判断是否是第一次重复
    hcomm::CcuRep::Variable xnConst1_;
    bool loadFromMem_ = false;
    
    HcclResult InitResource();
    void CreateVariables();
    void LoadAll2allSendRecvInfo(A2AsingleSendRecvInfo &sendRecvInfo);
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoAll2AllVMultiLoop();
    void CalcGroupSrcDst();
};

}// namespace ops_hccl
#endif // HCCL_CCU_KERNEL_ALL_TO_ALL_MESH_1D_H
