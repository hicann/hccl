/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_MESH_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_ALL_GATHER_MESH_1D_MEM2MEM_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgAllGatherMesh1DMem2Mem : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllGatherMesh1DMem2Mem(uint64_t dimSize, uint32_t rankId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgAllGatherMesh1DMem2Mem] dimSize: %lu, rankId: %u",
                   dimSize_, rankId_);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllGatherMesh1DMem2Mem", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgAllGatherMesh1DMem2Mem : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllGatherMesh1DMem2Mem(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                        uint64_t inputSliceStride, uint64_t outputSliceStride,
                                                        uint64_t repeatNum, uint64_t inputRepeatStride,
                                                        uint64_t outputRepeatStride, uint64_t normalSliceSize,
                                                        uint64_t lastSliceSize, uint64_t isInputOutputEqual)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), inputSliceStride_(inputSliceStride),
          outputSliceStride_(outputSliceStride), repeatNum_(repeatNum), inputRepeatStride_(inputRepeatStride),
          outputRepeatStride_(outputRepeatStride), normalSliceSize_(normalSliceSize), lastSliceSize_(lastSliceSize),
          isInputOutputEqual_(isInputOutputEqual)
    {
        HCCL_DEBUG("[CcuTaskArgAllGatherMesh1DMem2Mem] inputAddr: %lu, outputAddr: %lu, token: %lu, inputSliceStride: %lu, "
                   "outputSliceStride: %lu",
                   inputAddr_, outputAddr_, token_, inputSliceStride_, outputSliceStride_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;

    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t repeatNum_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;

    uint64_t normalSliceSize_;
    uint64_t lastSliceSize_;
    uint64_t isInputOutputEqual_;
};

class CcuKernelAllGatherMesh1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelAllGatherMesh1DMem2Mem(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllGatherMesh1DMem2Mem() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoRepeatAllGather();
    void DoAllGather(const hcomm::CcuRep::LocalAddr              &src,
                                                             const std::vector<hcomm::CcuRep::RemoteAddr> &dst,
                                                             const CcuRep::Variable            &sliceSize);

    // CcuKernelAlgDataWrapper algWrapper;
    uint64_t rankSize_{0};
    uint32_t rankId_{0};

    CcuRep::Variable              localInput_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable              currentRankSliceInputOffset_;
    CcuRep::Variable              currentRankSliceOutputOffset_;
    CcuRep::Variable              inputRepeatStride_;
    CcuRep::Variable              outputRepeatStride_;
    CcuRep::Variable              normalSliceSize_;
    CcuRep::Variable              lastSliceSize_;
    CcuRep::Variable              isInputOutputEqual_;
    CcuRep::Variable              repeatTimeflag_;
    CcuRep::Variable              tmpRepeatNum_;
    CcuRep::Variable              constVar1_;

    GroupOpSize localGoSize_;

    hcomm::CcuRep::LocalAddr src;
    hcomm::CcuRep::LocalAddr remote_src;
    std::vector<hcomm::CcuRep::RemoteAddr> dst;
    hcomm::CcuRep::LocalAddr src_loccopy;

    CcuRep::Variable srcOffset_;
    CcuRep::Variable dstOffset_;

    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    CcuRep::CompletedEvent event_;
};

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_ALL_GATHER_MESH_1D_MEM2MEM_H
