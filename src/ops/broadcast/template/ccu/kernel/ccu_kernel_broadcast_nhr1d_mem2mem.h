/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_BROADCAST_NHR_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_BROADCAST_NHR_1D_MEM2MEM_H

#include <memory>
#include <map>
#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

using NHRStepInfo = struct NHRStepInfo {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    NHRStepInfo() : nSlices(0)
    {
    }
};

class CcuKernelArgBroadcastNhr1DMem2Mem : public CcuKernelArg {
public:
    explicit CcuKernelArgBroadcastNhr1DMem2Mem(uint32_t rankId, uint32_t axisId, uint32_t axisSize, std::vector<uint32_t>& dimSize,
                                                const std::vector<NHRStepInfo> stepInfoVector, const std::map<u32, u32> rank2ChannelIdx,
                                                const OpParam& opParam, const std::vector<std::vector<uint32_t>>& subCommRanks)
        : rankId_(rankId),
          axisId_(axisId),
          axisSize_(axisSize),
          dimSize_(dimSize),
          stepInfoVector_(stepInfoVector),
          rank2ChannelIdx_(rank2ChannelIdx),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
    }
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgBroadcastNhr1DMem2Mem", opParam_, subCommRanks_);
        return signature;
    }

    uint32_t                                rankId_;
    uint32_t                                axisId_;
    uint32_t                                axisSize_;
    std::vector<uint32_t>                   dimSize_;
    std::vector<NHRStepInfo>                stepInfoVector_;
    std::map<u32, u32>                      rank2ChannelIdx_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgBroadcastNhr1DMem2Mem : public CcuTaskArg {
public:
    explicit CcuTaskArgBroadcastNhr1DMem2Mem(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                uint64_t die0Size, uint64_t die1Size, uint64_t die0SliceSize, uint64_t die1SliceSize,
                                                uint64_t die0LastSliceSize, uint64_t die1LastSliceSize)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), die0Size_(die0Size), die1Size_(die1Size),
        die0SliceSize_(die0SliceSize), die1SliceSize_(die1SliceSize), die0LastSliceSize_(die0LastSliceSize), die1LastSliceSize_(die1LastSliceSize)
    {
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t die0Size_;
    uint64_t die1Size_;
    uint64_t die0SliceSize_;
    uint64_t die1SliceSize_;
    uint64_t die0LastSliceSize_;
    uint64_t die1LastSliceSize_;
};

class CcuKernelBroadcastNhr1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelBroadcastNhr1DMem2Mem(const CcuKernelArg &arg);
    ~CcuKernelBroadcastNhr1DMem2Mem() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;

private:
    HcclResult InitResources();
    void PostSync();
    void PreSync();
    void LoadArgs();
    void DoScatterNHR();
    void DoScatterNHRSingleStep(const NHRStepInfo &nhrStepInfo);
    void DoAllGatherNHR();
    void DoSendRecvSlice(const u32 &toRank, CcuRep::LocalAddr &src, CcuRep::RemoteAddr &dst,
                         const u32 &sendSliceIdx, u32 signalIndex);
    void DoAllGatherNHRSingleStep(const NHRStepInfo &nhrStepInfo);

    // 构造函数
    uint32_t rankId_{0};
    uint32_t axisId_{0};
    uint32_t axisSize_{0};
    uint32_t dimSize_{0};
    std::vector<NHRStepInfo> stepInfoVector_;
    std::map<u32, u32> rank2ChannelIdx_;
    uint32_t localSize_{0};
    uint32_t myRankIdx_{0};
    std::vector<ChannelHandle> channels_;
    HcclDataType dataType_;

    // load进来参数
    CcuRep::Variable input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable die0Size_;
    CcuRep::Variable die1Size_;
    CcuRep::Variable die0SliceSize_;
    CcuRep::Variable die1SliceSize_;
    CcuRep::Variable die0LastSliceSize_;
    CcuRep::Variable die1LastSliceSize_;

    std::vector<CcuRep::Variable> sliceOffset_;

    CcuRep::LocalAddr localSrc_;
    CcuRep::LocalAddr localDst_;
    CcuRep::RemoteAddr remoteDst_;

    CcuRep::CompletedEvent event_;
};

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_BROADCAST_NHR_1D_MEM2MEM_H
