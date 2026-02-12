/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_1D
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_1D

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

class CcuKernelArgReduceScatterNHR1D : public CcuKernelArg {
public:
    explicit CcuKernelArgReduceScatterNHR1D(uint64_t dimSize, uint32_t mySubCommRankId, uint32_t axisId,
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
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceScatterNHR1D", opParam_, subCommRanks_);
        // signature.Append<uint8_t>(uint8_t(op.outputDataType));
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

class CcuTaskArgReduceScatterNHR1D : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterNHR1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t die0Size,
                                      uint64_t die1Size, uint64_t inputSliceStride, uint64_t outputSliceStride,
                                      uint64_t inputRepeatStride, uint64_t outputRepeatStride, uint64_t repeatNum)
        : inputAddr_(inputAddr), outputAddr_(outputAddr),
          token_(token), die0Size_(die0Size), die1Size_(die1Size), inputSliceStride_(inputSliceStride),
          outputSliceStride_(outputSliceStride), inputRepeatStride_(inputRepeatStride),
          outputRepeatStride_(outputRepeatStride), repeatNum_(repeatNum)
    {
        HCCL_DEBUG("[CcuTaskArgReduceScatterNHR1D]: inputAddr: %lu, outputAddr: %lu, die0Size: %lu, die1Size: %lu, "
                   "inputSliceStride: %lu, outputSliceStride: %lu, inputRepeatStride: %lu, outputRepeatStride: %lu, "
                   "repeatNum: %lu",
                   inputAddr_, outputAddr_, die0Size_, die1Size_, inputSliceStride_, outputSliceStride_,
                   inputRepeatStride_, outputRepeatStride_, repeatNum_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t die0Size_;
    uint64_t die1Size_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;
    uint64_t repeatNum_;
};

class CcuKernelReduceScatterNHR1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelReduceScatterNHR1DMem2Mem(const CcuKernelArg &arg);
    ~CcuKernelReduceScatterNHR1DMem2Mem() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;
private:
    void LoadArgs();
    HcclResult InitResources();
    void PreSync();
    void PostSync();
    void AxisSync(uint32_t signalIndex);
    void DoRepeatReduceScatterNHR();
    void DoRepeatReduceScatterNHRSingleStep(const NHRStepInfo &nhrStepInfo,
        const std::vector<CcuRep::Variable> &inputSliceOffset);
    void DoRepeatWriteReduceSlices(const u32 &toRank, CcuRep::LocalAddr &src, CcuRep::RemoteAddr &dst);

    // 构造函数中
    uint32_t mySubCommRankId_{0};
    uint64_t dimSize_{0};
    uint32_t axisId_{0};
    uint32_t localSize_{0};  // 本rank所在行或列的总rank数
    uint32_t myRankIdx_{0};
    uint32_t signalNum_{0};  // 需要使用的signal数量
    HcclReduceOp reduceOp_;
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    std::vector<NHRStepInfo> stepInfoVector_;   // nhr算法执行过程中的参数
    std::map<u32, u32> rank2ChannelIdx_;
    uint32_t       axisSize_{0};

    // load进来参数
    std::vector<ChannelHandle> channels_;
    std::vector<CcuRep::Variable> input_;
    CcuRep::Variable output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable die0Size_;
    CcuRep::Variable die1Size_;
    CcuRep::Variable inputSliceStride_;
    CcuRep::Variable outputSliceStride_;
    CcuRep::Variable inputRepeatStride_;
    CcuRep::Variable outputRepeatStride_;
    CcuRep::Variable repeatNumVar_;
    CcuRep::Variable repeatNumVarTemp_;
    // 用于记录每次writereduce的数据
    CcuRep::Variable sliceSize_;

    CcuRep::CompletedEvent event_;

    CcuRep::Variable repeatInputOffset_;
    CcuRep::Variable repeatOutputOffset_;
    CcuRep::Variable myrankInputSliceOffset_;

    CcuRep::LocalAddr   localSrc_;
    CcuRep::LocalAddr   localDst_;
    CcuRep::RemoteAddr  remoteDst_;
    CcuRep::Variable    isRepeatIter_; // 用于判断是否是第一次循环
};
} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_1D