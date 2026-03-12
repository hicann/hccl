/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_NHR_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_REDUCE_NHR_1D_MEM2MEM_H

#include <vector>
#include <ios>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "ins_temp_all_reduce_nhr.h"

namespace ops_hccl {

class CcuKernelArgReduceNHR1D : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgReduceNHR1D(uint64_t dimSize, uint32_t rankId, uint32_t rootId, uint32_t axisId,
                                     const std::vector<NHRStepInfo> stepInfoVector,
                                     const std::map<u32, u32> rank2ChannelIdx, const OpParam& opParam,
                                     const std::vector<std::vector<uint32_t>>& subCommRanks,
                                     uint32_t axisSize)
        : dimSize_(dimSize),
          rankId_(rankId),
          rootId_(rootId),
          axisId_(axisId),
          stepInfoVector_(stepInfoVector),
          rank2ChannelIdx_(rank2ChannelIdx),
          opParam_(opParam),
          subCommRanks_(subCommRanks),
          axisSize_(axisSize)
    {
        HCCL_DEBUG("[CcuKernelArgReduceMesh2D] dimSize: %lu, rankId: %u, axisId: %u, axisSize: %u, reduceOp: %d, dataType: %d",
                   dimSize_, rankId_, axisId_, axisSize_, opParam.reduceType, opParam.DataDes.dataType);
    }

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceNHR1D", opParam_, subCommRanks_);
        return signature;
    }

    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    uint32_t                                rootId_;
    uint32_t                                axisId_;
    std::vector<NHRStepInfo>                stepInfoVector_;
    std::map<u32, u32>                      rank2ChannelIdx_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
    uint32_t                                axisSize_;
};

class CcuTaskArgReduceNHR1D : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgReduceNHR1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                           uint64_t isInputOutputEqual, uint64_t die0Size, uint64_t die1Size,
                                           uint64_t die0SliceSize, uint64_t die1SliceSize,
                                           uint64_t die0LastSliceSize, uint64_t die1LastSliceSize)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), isInputOutputEqual_(isInputOutputEqual),
          die0Size_(die0Size), die1Size_(die1Size), die0SliceSize_(die0SliceSize), die1SliceSize_(die1SliceSize),
          die0LastSliceSize_(die0LastSliceSize), die1LastSliceSize_(die1LastSliceSize)
    {
        HCCL_DEBUG("[CcuTaskArgReduceNHR1D] inputAddr: %lu, outputAddr: %lu, isInputOutputEqual: %lu, "
                   "die0Size: %lu, die1Size: %lu, die0SliceSize: %lu, die1SliceSize: %lu, "
                   "die0LastSliceSize: %lu, die1LastSliceSize: %lu",
                   inputAddr_, outputAddr_, isInputOutputEqual_, die0Size_, die1Size_,
                   die0SliceSize_, die1SliceSize_, die0LastSliceSize_, die1LastSliceSize_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t isInputOutputEqual_;
    uint64_t die0Size_;
    uint64_t die1Size_;
    uint64_t die0SliceSize_;
    uint64_t die1SliceSize_;
    uint64_t die0LastSliceSize_;
    uint64_t die1LastSliceSize_;
};

class CcuKernelReduceNHR1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelReduceNHR1DMem2Mem(const hcomm::CcuKernelArg &arg);
    ~CcuKernelReduceNHR1DMem2Mem() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    void LoadArgs();
    HcclResult InitResources();
    void PreSync();
    void PostSync();
    void LocalCopySlices();
    void DoLocalCopySlice(CcuRep::LocalAddr &src, CcuRep::LocalAddr &dst,
                          const u32 &copySliceIdx, u32 signalIndex);
    std::vector<u32> GetNonTxSliceIdxs(const std::vector<u32> &txSliceIdxs) const;
    void DoReduceScatterNHR();
    void DoReduceScatterNHRSingleStep(const NHRStepInfo &nhrStepInfo);
    void DoWriteReduceSlice(const u32 &toRank, CcuRep::LocalAddr &src, CcuRep::RemoteAddr &dst,
                            const u32 &sendSliceIdx, u32 signalIndex);
    void DoGatherNHR();
    void DoGatherNHRSingleStep(const NHRStepInfo &nhrStepInfo);
    void DoSendRecvSlice(const u32 &toRank, CcuRep::LocalAddr &src, CcuRep::RemoteAddr &dst,
                         const u32 &sendSliceIdx, u32 signalIndex);

    std::vector<ChannelHandle> channels_;
    uint32_t rankId_{0};
    uint32_t rootId_{0};
    uint64_t dimSize_{0};
    uint32_t axisId_{0};
    uint32_t axisSize_{0};
    uint32_t localSize_{0};  // 本rank所在行或列的总rank数
    uint32_t myRankIdx_{0};
    uint32_t repeatNum_{0};
    HcclDataType dataType_;
    HcclReduceOp reduceOp_;
    std::vector<NHRStepInfo> stepInfoVector_;  //nhr算法执行过程中的参数
    std::map<u32, u32> rank2ChannelIdx_;

    // load进来参数
    CcuRep::Variable input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable isInputOutputEqual_;
    CcuRep::Variable die0Size_;
    CcuRep::Variable die1Size_;
    CcuRep::Variable sliceSize_;
    CcuRep::Variable die0SliceSize_;
    CcuRep::Variable die1SliceSize_;
    CcuRep::Variable die0LastSliceSize_;
    CcuRep::Variable die1LastSliceSize_;

    CcuRep::CompletedEvent event_;

    std::vector<CcuRep::Variable> sliceOffset_;
    CcuRep::Variable repeatInputOffset_;
    CcuRep::Variable repeatOutputOffset_;

    CcuRep::LocalAddr  localSrc_;
    CcuRep::LocalAddr  localDst_;
    CcuRep::RemoteAddr remoteDst_;
};

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_REDUCE_NHR_1D_MEM2MEM_H
