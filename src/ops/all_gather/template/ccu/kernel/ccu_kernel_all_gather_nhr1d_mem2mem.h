/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_MESH_2D_H
#define HCCL_CCU_KERNEL_ALL_GATHER_MESH_2D_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using NHRStepInfo = struct NHRStepInfoDef {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    NHRStepInfoDef() : nSlices(0)
    {
    }
};

class CcuKernelArgAllGatherNHR1D : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllGatherNHR1D(uint64_t dimSize, uint32_t mySubCommRankId, uint32_t axisId,
                                            const std::vector<NHRStepInfo> stepInfoVector,
                                            const std::map<u32, u32> rank2ChannelIdx, const OpParam& opParam,
                                            const std::vector<std::vector<uint32_t>>& subCommRanks, uint32_t axisSize)
        : dimSize_(dimSize),
          mySubCommRankId_(mySubCommRankId),
          axisId_(axisId),
          stepInfoVector_(stepInfoVector),
          rank2ChannelIdx_(rank2ChannelIdx),
          opParam_(opParam),
          subCommRanks_(subCommRanks),
          axisSize_(axisSize)
    {
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllGatherNHR1D", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint64_t                                mySubCommRankId_;
    uint64_t                                axisId_;            // 记录自己在哪个轴
    std::vector<NHRStepInfo>                stepInfoVector_;    // nhr每一步的信息（发送/接受给谁，发/收哪片数据）
    std::map<u32, u32>                      rank2ChannelIdx_; 
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
    uint32_t                                axisSize_;          // 同时支持单die和双die
};

class CcuTaskArgAllGatherNHR1D : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllGatherNHR1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t die0Size,
                                        uint64_t die1Size, uint64_t repeatNum, uint64_t inputSliceStride,
                                        uint64_t outputSliceStride, uint64_t inputRepeatStride,
                                        uint64_t outputRepeatStride, uint64_t isInputOutputEqual)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), die0Size_(die0Size), die1Size_(die1Size),
          repeatNum_(repeatNum), inputSliceStride_(inputSliceStride), outputSliceStride_(outputSliceStride),
          inputRepeatStride_(inputRepeatStride), outputRepeatStride_(outputRepeatStride),isInputOutputEqual_(isInputOutputEqual)
    {
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t die0Size_;
    uint64_t die1Size_;
    uint64_t repeatNum_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;
    uint64_t isInputOutputEqual_;
};

class CcuKernelAllGatherNHR1DMem2Mem : public hcomm::CcuKernel {
public:
    CcuKernelAllGatherNHR1DMem2Mem(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllGatherNHR1DMem2Mem() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    void LoadArgs();
    HcclResult InitResource();
    void PreSync();
    void PostSync();
    void AxisSync(uint32_t eventIndex);
    void DoRepeatAllGatherNHR();
    void DoRepeatAllGatherNHRSingleStep(const NHRStepInfo &nhrStepInfo);
    void DoRepeatSendRecvSlices(const u32 &toRank, hcomm::CcuRep::LocalAddr &src, hcomm::CcuRep::RemoteAddr &dst,
                                                      u32 signalIndex);

    // 构造函数中
    uint32_t mySubCommRankId_{0};
    uint64_t dimSize_{0};
    uint32_t axisId_{0};
    uint32_t axisSize_{0};
    uint32_t localSize_{0}; // 本rank所在行或列的总rank数
    uint32_t myRankIdx_{0};
    uint32_t eventNum_{0}; // 需要使用的event数量

    std::vector<NHRStepInfo> stepInfoVector_; // nhr算法执行过程中的参数
    std::map<u32, u32>       rank2ChannelIdx_;

    // load进来参数
    CcuRep::Variable              input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    std::vector<ChannelHandle> channels_;
    CcuRep::Variable              die0Size_;
    CcuRep::Variable              die1Size_;
    CcuRep::Variable              inputSliceStride_;
    CcuRep::Variable              outputSliceStride_;
    CcuRep::Variable              inputRepeatStride_;
    CcuRep::Variable              outputRepeatStride_;
    CcuRep::Variable              repeatNum_;
    CcuRep::Variable              isInputOutputEqual_;
    CcuRep::Variable              myrankInputSliceOffset_;
    CcuRep::Variable              tmpSliceOffset_;
    std::vector<CcuRep::Variable> outputSliceOffset_;
    CcuRep::Variable              tmpRepeatNum_;
    CcuRep::Variable              tmpCopyRepeatNum_;
    CcuRep::Variable              constVar1_;
    CcuRep::Variable              ckeBitCounter_;
    CcuRep::Variable              repeatTimeflag_;

    // 跨轴同步信号
    std::string        localAxisEventName_;
    std::string        anotherAxisEventName_;
    CcuRep::CompletedEvent localAxisEvent_;
    CcuRep::CompletedEvent anotherAxisEvent_;
    CcuRep::CompletedEvent localEvent_;

    hcomm::CcuRep::LocalAddr srcMem_;
    hcomm::CcuRep::RemoteAddr dstMem_;
    hcomm::CcuRep::LocalAddr localDst_;
};

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_ALL_GATHER_MESH_1D_H
