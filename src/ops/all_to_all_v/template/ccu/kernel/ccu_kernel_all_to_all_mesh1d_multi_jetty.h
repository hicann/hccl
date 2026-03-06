/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_TO_ALL_MESH1D_MULTI_JETTY_H
#define HCCL_CCU_KERNEL_ALL_TO_ALL_MESH1D_MULTI_JETTY_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgAllToAllMesh1DMultiJetty : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllToAllMesh1DMultiJetty(uint64_t dimSize, uint32_t rankId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks,
                                                    const std::vector<uint32_t>& jettyNums)
        : rankSize_(dimSize),
          rankId_(rankId),
          opParam_(opParam),
          subCommRanks_(subCommRanks),
          jettyNums_(jettyNums)
    {
        HCCL_DEBUG("[CcuKernelArgAllToAllMesh1DMultiJetty] rankSize: %u, rankId: %u", rankSize_, rankId_);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllToAllMesh1DMultiJetty", opParam_, subCommRanks_);
        return signature;
    }
    uint32_t                                rankId_;
    uint64_t                                rankSize_;
    std::vector<uint32_t>                   jettyNums_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgAllToAllMesh1DMultiJetty : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllToAllMesh1DMultiJetty(uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize,
        std::vector<uint64_t> jettySlice, std::vector<uint64_t> jettySliceTail,
        uint64_t token, uint64_t srcOffset, uint64_t dstOffset, uint64_t srcStride) :
        inputAddr_(inputAddr), outputAddr_(outputAddr), sliceSize_(sliceSize), jettySlice_(jettySlice),
        jettySliceTail_(jettySliceTail), token_(token), srcOffset_(srcOffset),
        dstOffset_(dstOffset), srcStride_(srcStride)
    {
        HCCL_DEBUG("[CcuTaskArgAllToAllMesh1DMultiJetty] inputAddr: %lu, outputAddr: %lu, token: %lu, sliceSize: %lu, "
                   "srcOffset: %lu, dstOffset: %lu, srcStride: %lu",
                   inputAddr_, outputAddr_, token_, sliceSize_, srcOffset_, dstOffset_,
                   srcStride_);
        for (uint32_t i = 0; i < jettySlice.size(); i++) {
            HCCL_DEBUG("[CcuTaskArgAllToAllMesh1DMultiJetty] jettySlice: %llu, jettySliceTail: %llu", 
                        jettySlice[i], jettySliceTail[i]);
        }
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t sliceSize_;
    std::vector<uint64_t> jettySlice_;
    std::vector<uint64_t> jettySliceTail_;
    uint64_t token_;
    uint64_t srcOffset_;
    uint64_t dstOffset_;
    uint64_t srcStride_;
};

class CcuKernelAllToAllMesh1DMultiJetty : public CcuKernelAlgBase {
public:
    CcuKernelAllToAllMesh1DMultiJetty(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllToAllMesh1DMultiJetty() override {}

    HcclResult Algorithm();
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    HcclResult LoadArgs();
    HcclResult PreSync();
    HcclResult CalcAddrs();
    HcclResult DoAllToAll();
    HcclResult PostSync();
    HcclResult SendData(ChannelHandle channel, hcomm::CcuRep::RemoteAddr remoteDst, hcomm::CcuRep::LocalAddr remoteSrc,
                        hcomm::CcuRep::Variable sliceLength, hcomm::CcuRep::CompletedEvent event);

    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    std::vector<ChannelHandle> channels_;
    std::vector<uint32_t> jettyNums_;
    std::vector<hcomm::CcuRep::Variable> input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    std::vector<hcomm::CcuRep::Variable> jettySlice_;
    std::vector<hcomm::CcuRep::Variable> jettySliceTail_;
    hcomm::CcuRep::Variable sliceSize_;
    hcomm::CcuRep::Variable srcStride_;
    hcomm::CcuRep::Variable srcOffset_;
    hcomm::CcuRep::Variable dstOffset_;
    GroupOpSize groupOpSize_;
    std::vector<hcomm::CcuRep::LocalAddr>       remoteSrc_;
    std::vector<hcomm::CcuRep::RemoteAddr>      remoteDst_;
    hcomm::CcuRep::LocalAddr                    localSrc_;
    hcomm::CcuRep::LocalAddr                    localDst_;
    std::vector<hcomm::CcuRep::CompletedEvent>  eventList_;
    bool loadFromMem_ = false;
};

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_MEM2MEM_H
