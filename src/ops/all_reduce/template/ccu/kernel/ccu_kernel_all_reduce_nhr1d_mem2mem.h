/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALLREDUCE_NHR_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_ALLREDUCE_NHR_1D_MEM2MEM_H
#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "ins_temp_all_reduce_nhr.h"

namespace ops_hccl {

class CcuKernelArgAllReduceNHR1D : public CcuKernelArg {
public:
    explicit CcuKernelArgAllReduceNHR1D(const std::vector<uint64_t> &dimSize, uint32_t rankId, uint32_t axisId, 
                                     uint32_t axisSize, const std::vector<NHRStepInfo> stepInfoVector,
                                     const std::map<u32, u32> indexMap, const OpParam &opParam,
                                     const std::vector<std::vector<uint32_t>> &tempVTopo)
        : dimSize_(dimSize), rankId_(rankId), axisId_(axisId), axisSize_(axisSize), stepInfoVector_(stepInfoVector),
          indexMap_(indexMap), opParam_(opParam), tempVTopo_(tempVTopo)
    {
    }
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllReduceNHR1D", opParam_, tempVTopo_);
        return signature;
    }
    std::vector<uint64_t>            dimSize_;
    uint32_t                         rankId_;
    uint32_t                         axisId_;
    uint32_t                         axisSize_;
    std::vector<NHRStepInfo>         stepInfoVector_;
    std::map<u32, u32>               indexMap_;
    OpParam                          opParam_;
    std::vector<std::vector<uint32_t>> tempVTopo_;
};

class CcuTaskArgAllReduceNHR1D : public CcuTaskArg {
public:
    explicit CcuTaskArgAllReduceNHR1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                      uint64_t isInputOutputEqual, uint64_t die0Size, uint64_t die1Size,
                                      uint64_t die0SliceSize, uint64_t die1SliceSize,
                                      uint64_t die0LastSliceSize, uint64_t die1LastSliceSize)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), isInputOutputEqual_(isInputOutputEqual),
          die0Size_(die0Size), die1Size_(die1Size), die0SliceSize_(die0SliceSize), die1SliceSize_(die1SliceSize),
          die0LastSliceSize_(die0LastSliceSize), die1LastSliceSize_(die1LastSliceSize)
    {
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

class CcuKernelAllReduceNHR1D : public CcuKernelAlgBase {
public:
    CcuKernelAllReduceNHR1D(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllReduceNHR1D() override {}
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;
private:
    void LoadArgs();
    void InitResources();
    void PreSync();
    void PostSync();
    void LocalCopySlices();
    void DoLocalCopySlice(hcomm::CcuRep::LocalAddr &src, hcomm::CcuRep::LocalAddr &dst,
                          const u32 &copySliceIdx, u32 signalIndex);
    std::vector<u32> GetNonTxSliceIdxs(const std::vector<u32> &txSliceIdxs) const;
    void DoReduceScatterNHR();
    void DoReduceScatterNHRSingleStep(const NHRStepInfo &nhrStepInfo);
    void DoWriteReduceSlice(const u32 &toRank, hcomm::CcuRep::LocalAddr &src, hcomm::CcuRep::RemoteAddr &dst,
                            const u32 &sendSliceIdx, u32 signalIndex);
    void DoAllGatherNHR();
    void DoAllGatherNHRSingleStep(const NHRStepInfo &nhrStepInfo);
    void DoSendRecvSlice(const u32 &toRank, hcomm::CcuRep::LocalAddr &src, hcomm::CcuRep::RemoteAddr &dst,
                         const u32 &sendSliceIdx, u32 signalIndex);

    // 构造函数中
    uint32_t rankId_{0};
    uint64_t dimSize_{0};
    uint32_t axisId_{0};
    uint32_t axisSize_{0};
    uint32_t localSize_{0};  // 本rank所在行或列的总rank数
    uint32_t myRankIdx_{0};
    uint32_t repeatNum_{0};
    HcclDataType dataType_;
    // HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;
    std::vector<NHRStepInfo> stepInfoVector_;   //nhr算法执行过程中的参数
    std::map<u32, u32> indexMap_;

    // load进来参数
    hcomm::CcuRep::Variable input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable isInputOutputEqual_;
    hcomm::CcuRep::Variable die0Size_;
    hcomm::CcuRep::Variable die1Size_;
    hcomm::CcuRep::Variable sliceSize_;
    hcomm::CcuRep::Variable die0SliceSize_;
    hcomm::CcuRep::Variable die1SliceSize_;
    hcomm::CcuRep::Variable die0LastSliceSize_;
    hcomm::CcuRep::Variable die1LastSliceSize_;

    hcomm::CcuRep::CompletedEvent localEvent_;
    std::vector<ChannelHandle> channels_;

    std::vector<hcomm::CcuRep::Variable> sliceOffset_;
    hcomm::CcuRep::Variable repeatInputOffset_;
    hcomm::CcuRep::Variable repeatOutputOffset_;

    hcomm::CcuRep::LocalAddr srcMem_;
    hcomm::CcuRep::RemoteAddr rmtDstMem_;
    hcomm::CcuRep::LocalAddr locDstMem_;
};
} // namespace ops_hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_REDUCE_NHR_1D_H_