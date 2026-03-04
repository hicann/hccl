/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALLREDUCE_MESH_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_ALLREDUCE_MESH_1D_MEM2MEM_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgAllReduceMeshMem2Mem1D : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllReduceMeshMem2Mem1D(uint64_t dimSize, uint32_t rankId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgAllReduceMeshMem2Mem1D] dimSize: %lu, rankId: %u", dimSize_, rankId_);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllReduceMeshMem2Mem1D", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgAllReduceMeshMem2Mem1D : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllReduceMeshMem2Mem1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                        uint64_t scratchAddr, uint64_t inputSliceStride,
                                                        uint64_t outputSliceStride, uint64_t inputRepeatStride,
                                                        uint64_t outputRepeatStride, uint64_t normalSliceSize,
                                                        uint64_t lastSliceSize, uint64_t mySliceSize,
                                                        uint64_t isInputOutputEqual)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), scratchAddr_(scratchAddr),
          inputSliceStride_(inputSliceStride), outputSliceStride_(outputSliceStride),
          inputRepeatStride_(inputRepeatStride), outputRepeatStride_(outputRepeatStride),
          normalSliceSize_(normalSliceSize), lastSliceSize_(lastSliceSize), mySliceSize_(mySliceSize),
          isInputOutputEqual_(isInputOutputEqual)
    {
        HCCL_DEBUG("[CcuTaskArgAllReduceMeshMem2Mem1D] inputAddr: %lu, outputAddr: %lu, token: %lu, scratchAddr: %lu, "
            "inputSliceStride: %lu, outputSliceStride: %lu, inputRepeatStride: %lu, outputRepeatStride: %lu, normalSliceSize: %lu, "
            "lastSliceSize: %lu, mySliceSize: %lu, isInputOutputEqual: %lu",
            inputAddr_, outputAddr_, token_, scratchAddr_, inputSliceStride_, outputSliceStride_, inputRepeatStride_,
            outputRepeatStride_, normalSliceSize_, lastSliceSize_, mySliceSize_, isInputOutputEqual_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t scratchAddr_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;

    uint64_t normalSliceSize_;
    uint64_t lastSliceSize_;
    uint64_t mySliceSize_;
    uint64_t isInputOutputEqual_;
};

class CcuKernelAllReduceMeshMem2Mem1D : public CcuKernelAlgBase {
public:
    CcuKernelAllReduceMeshMem2Mem1D(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllReduceMeshMem2Mem1D() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

protected:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoRepeatAllReduce();
    void ReduceRmtToLoc(const std::vector<hcomm::CcuRep::Variable> &srcAddr, const hcomm::CcuRep::Variable &dstAddr);
    void BcastLocToRmt(const hcomm::CcuRep::Variable &srcAddr, const std::vector<hcomm::CcuRep::Variable> &dstAddr);

private:
    uint64_t                      rankSize_{0};
    uint32_t                      rankId_{0};
    HcclDataType                      dataType_;
    HcclDataType                      outputDataType_;
    HcclReduceOp                      reduceOp_;
    std::vector<hcomm::CcuRep::Variable> input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    std::vector<hcomm::CcuRep::Variable> scratch_;
    hcomm::CcuRep::Variable              currentRankSliceInputOffset_;
    hcomm::CcuRep::Variable              currentRankSliceOutputOffset_;
    hcomm::CcuRep::Variable              normalSliceSize_;
    hcomm::CcuRep::Variable              lastSliceSize_;
    hcomm::CcuRep::Variable              mySliceSize_;
    hcomm::CcuRep::Variable              sliceOffset_;
    hcomm::CcuRep::Variable              isInputOutputEqual_;
    hcomm::CcuRep::Variable              sliceSize_;
    hcomm::CcuRep::CompletedEvent        locEvent_;

    hcomm::CcuRep::LocalAddr              srcMem_;
    hcomm::CcuRep::LocalAddr              localDstMem_;
    hcomm::CcuRep::RemoteAddr             remoteDstMem_;
    std::vector<hcomm::CcuRep::RemoteAddr> reduceScatterSrc_;
    std::vector<hcomm::CcuRep::LocalAddr> reduceScatterDst_;
    std::vector<ChannelHandle> channels_;

    GroupOpSize localGoSize_;

    std::string GetLoopBlockTag(std::string loopType, int32_t index);
    void CreateReduceLoop(uint32_t size, HcclDataType dataType, HcclDataType outputDataType,
        HcclReduceOp opType);
    void ReduceLoopGroup(hcomm::CcuRep::LocalAddr outDstOrg, hcomm::CcuRep::LocalAddr srcOrg,
        std::vector<hcomm::CcuRep::LocalAddr> &scratchOrg, GroupOpSize goSize, HcclDataType dataType, HcclDataType outputDataType,
        HcclReduceOp opType);
    const std::string LOOP_BLOCK_TAG{"_local_copy_reduce_loop_"};
};
} // namespace ops_hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_1D_H_