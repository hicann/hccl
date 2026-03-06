/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_NHR_1D_MULTIJETTY_MEM2MEM_H
#define HCCL_CCU_KERNEL_ALL_GATHER_NHR_1D_MULTIJETTY_MEM2MEM_H

#include <vector>
#include <ios>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using NHRStepInfo = struct NHRStepInfoDef {
    uint32_t step = 0;
    uint32_t myRank = 0;
    uint32_t nSlices;
    uint32_t toRank = 0;
    uint32_t fromRank = 0;
    std::vector<uint32_t> txSliceIdxs;
    std::vector<uint32_t> rxSliceIdxs;

    NHRStepInfoDef() : nSlices(0)
    {
    }
};

class CcuKernelArgAllGatherNHR1DMultiJettyMem2Mem : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllGatherNHR1DMultiJettyMem2Mem(uint64_t rankSize_, uint32_t rankId,
                                                const OpParam& opParam, uint32_t jettyNum,
                                                const std::vector<NHRStepInfo> stepInfoVector,
                                                const std::map<uint32_t, uint32_t> rank2ChannelIdx,
                                                const std::vector<std::vector<uint32_t>>& subCommRanks)
        : rankSize_(rankSize_),
          rankId_(rankId),
          opParam_(opParam),
          jettyNum_(jettyNum),
          stepInfoVector_(stepInfoVector),
          rank2ChannelIdx_(rank2ChannelIdx),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] kernelArg: rankId_[%u], rankSize_[%u]",
            rankId_, rankSize_);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllGatherNHR1DMultiJettyMem2Mem", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                rankSize_;
    uint32_t                                rankId_;
    uint32_t                                jettyNum_;
    OpParam                                 opParam_;
    std::vector<NHRStepInfo>                stepInfoVector_;
    std::map<uint32_t, uint32_t>            rank2ChannelIdx_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgAllGatherNHR1DMultiJettyMem2Mem : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllGatherNHR1DMultiJettyMem2Mem(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                uint64_t sliceSize, uint64_t sliceSizePerJetty, uint64_t lastSliceSizePerJetty,
                                                uint64_t repeatNumInv, uint64_t inputSliceStride,
                                                uint64_t outputSliceStride, uint64_t inputRepeatStride, uint64_t outputRepeatStride,
                                                uint64_t isInputOutputEqual)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), sliceSize_(sliceSize),     sliceSizePerJetty_(sliceSizePerJetty),
          lastSliceSizePerJetty_(lastSliceSizePerJetty), repeatNumInv_(repeatNumInv), inputSliceStride_(inputSliceStride), outputSliceStride_(outputSliceStride), inputRepeatStride_(inputRepeatStride), outputRepeatStride_(outputRepeatStride), isInputOutputEqual_(isInputOutputEqual)
    {
        HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
        "token[%llu], sliceSize[%llu], sliceSizePerJetty[%llu], lastSliceSizePerJetty[%llu], repeatNumInv[%llu]," 
        "inputSliceStride[%llu], outputSliceStride[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu], isInputOutputEqual[%llu]",
        inputAddr, outputAddr, token, sliceSize, sliceSizePerJetty, lastSliceSizePerJetty, repeatNumInv, inputSliceStride, 
        outputSliceStride, inputRepeatStride, outputRepeatStride, isInputOutputEqual);
    }
    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t sliceSize_;
    uint64_t sliceSizePerJetty_;
    uint64_t lastSliceSizePerJetty_;
    uint64_t repeatNumInv_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;
    uint64_t isInputOutputEqual_;
};

class CcuKernelAllGatherNHR1DMultiJettyMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelAllGatherNHR1DMultiJettyMem2Mem(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllGatherNHR1DMultiJettyMem2Mem() override {}

    HcclResult Algorithm();
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResources();
    HcclResult LoadArgs();
    HcclResult PreSync();
    HcclResult PostSync();
    HcclResult DoRepeatAllGatherNHR();
    HcclResult DoRepeatAllGatherNHRSingleStep(const NHRStepInfo &nhrStepInfo);
    HcclResult DoSendRecvSlices(const uint32_t &toRank, const CcuRep::LocalAddr &srcMem, const CcuRep::RemoteAddr &dstMem);

    // 构造函数中
    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    uint32_t localSize_{0};  // 本rank所在行或列的总rank数
    uint32_t myRankIdx_{0};
    uint32_t signalNum_{0};
    uint32_t jettyNum_{0};
    std::vector<NHRStepInfo> stepInfoVector_;
    std::map<uint32_t, uint32_t> rank2ChannelIdx_;
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    std::vector<ChannelHandle> channels_;

    // load进来参数
    hcomm::CcuRep::Variable input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> scratch_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable sliceSize_;
    hcomm::CcuRep::Variable sliceSizePerJetty_;
    hcomm::CcuRep::Variable lastSliceSizePerJetty_;
    hcomm::CcuRep::Variable myrankInputSliceOffset_;
    hcomm::CcuRep::Variable isInputOutputEqual_;
    hcomm::CcuRep::Variable repeatNumInv_;
    hcomm::CcuRep::Variable inputRepeatStride_;
    hcomm::CcuRep::Variable outputRepeatStride_;
    hcomm::CcuRep::Variable inputSliceStride_;
    hcomm::CcuRep::Variable outputSliceStride_;
    hcomm::CcuRep::Variable tmpSliceOffset_;

    std::vector<hcomm::CcuRep::Variable> outputSliceOffset_;
    hcomm::CcuRep::Variable tmpCopyRepeatNumInv_;
    hcomm::CcuRep::Variable constVar1_;
    hcomm::CcuRep::Variable repeatTimeflag_;

    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    hcomm::CcuRep::LocalAddr                   srcMem_;
    hcomm::CcuRep::LocalAddr                   myDstMem_;
    hcomm::CcuRep::RemoteAddr                  dstMem_;
    hcomm::CcuRep::LocalAddr                   srcMemTmp_;
    hcomm::CcuRep::RemoteAddr                  dstMemTmp_;
    std::vector<hcomm::CcuRep::CompletedEvent> events_;
    hcomm::CcuRep::CompletedEvent              event_;
    GroupOpSize groupOpSize_;
};

} // namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_ALL_GATHER_NHR_1D_MULTIJETTY_MEM2MEM_H