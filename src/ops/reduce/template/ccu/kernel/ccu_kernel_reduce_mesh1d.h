/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_MESH_1D_H
#define HCCL_CCU_KERNEL_REDUCE_MESH_1D_H

#include <vector>
#include <ios>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgReduceMesh1D : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgReduceMesh1D(uint64_t dimSize, uint32_t rankId, uint32_t rootId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          rootId_(rootId),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgReduceMesh1D] dimSize: %lu, rankId: %u, rootId: %u, reduceOp: %d, dataType: %d",
                   dimSize_, rankId_, rootId_, opParam.reduceType, opParam.DataDes.dataType);
    }

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceMesh1D", opParam_, subCommRanks_);
        return signature;
    }

    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    uint32_t                                rootId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgReduceMesh1D : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgReduceMesh1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                    uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t repeatNum,
                                    uint64_t inputRepeatStride, uint64_t outputRepeatStride, uint64_t normalSliceSize,
                                    uint64_t lastSliceSize, uint64_t repeatNumVar)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), inputSliceStride_(inputSliceStride),
          outputSliceStride_(outputSliceStride), repeatNum_(repeatNum), inputRepeatStride_(inputRepeatStride),
          outputRepeatStride_(outputRepeatStride), normalSliceSize_(normalSliceSize),
          lastSliceSize_(lastSliceSize), repeatNumVar_(repeatNumVar)
    {
        HCCL_DEBUG("[CcuTaskArgReduceMesh1D] inputAddr: %lu, outputAddr: %lu, inputSliceStride: %lu, "
                   "outputSliceStride: %lu, repeatNum: %lu, inputRepeatStride: %lu, outputRepeatStride: %lu, "
                   "normalSliceSize: %lu, lastSliceSize: %lu, repeatNumVar: %lu",
                   inputAddr_, outputAddr_, inputSliceStride_, outputSliceStride_, repeatNum_,
                   inputRepeatStride_, outputRepeatStride_, normalSliceSize_, lastSliceSize_, repeatNumVar_);
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
    uint64_t repeatNumVar_;
};

class CcuKernelReduceMesh1D : public CcuKernelAlgBase {
public:
    CcuKernelReduceMesh1D(const hcomm::CcuKernelArg &arg);
    ~CcuKernelReduceMesh1D() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void DoRepeatReduce();
    void PostSync();

    std::vector<ChannelHandle> channels_;
    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    uint32_t rootId_{0}; // 当rankid == rootid时，为root节点 则跳过write操作
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;
    std::vector<hcomm::CcuRep::Variable> input_; // 输入地址信息
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable              currentRankSliceInputOffset_;
    hcomm::CcuRep::Variable              currentRankSliceOutputOffset_;
    hcomm::CcuRep::Variable              repeatNum_;
    hcomm::CcuRep::Variable              inputRepeatStride_;
    hcomm::CcuRep::Variable              outputRepeatStride_;
    hcomm::CcuRep::Variable              normalSliceSize_;
    hcomm::CcuRep::Variable              lastSliceSize_;
    hcomm::CcuRep::Variable              repeatNumVar_;
    hcomm::CcuRep::Variable              flag_;

    GroupOpSize groupOpSize_;

    hcomm::CcuRep::Variable srcOffset_;
    hcomm::CcuRep::Variable dstOffset_;

    hcomm::CcuRep::LocalAddr               dstAddr_;
    std::vector<hcomm::CcuRep::RemoteAddr> inputAddr_;

    hcomm::CcuRep::CompletedEvent event_;
};

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_REDUCE_MESH_1D_H
