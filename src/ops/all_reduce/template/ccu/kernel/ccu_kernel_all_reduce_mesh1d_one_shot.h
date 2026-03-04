/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_ONE_SHOT_H_
#define HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_ONE_SHOT_H_

#include <vector>
#include <ios>
#include "log.h"
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_alg_base.h"
#include "alg_param.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {

class CcuKernelArgAllReduceMesh1DOneShot : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllReduceMesh1DOneShot(uint64_t dSize, uint32_t rId, const OpParam& opParam,
                                                const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dSize), rankId_(rId), opParam_(opParam), subCommRanks_(subCommRanks)
    {
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllReduceMesh1DOneShot", opParam_, subCommRanks_);
        return signature;
    }

    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgAllReduceMesh1DOneShot : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllReduceMesh1DOneShot(uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize,
        uint64_t token) :
        inputAddr_(inputAddr), outputAddr_(outputAddr), sliceSize_(sliceSize), token_(token) {}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t sliceSize_;
    uint64_t token_;
};

class CcuKernelAllReduceMesh1DOneShot : public CcuKernelAlgBase {
public:
    CcuKernelAllReduceMesh1DOneShot(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllReduceMesh1DOneShot() override{}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void Presync();
    void Postsync();
    void DoGroupReduce();

    std::vector<ChannelHandle> channels_;

    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;

    std::vector<hcomm::CcuRep::Variable> input_;
    hcomm::CcuRep::Variable output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable offSet_;
    GroupOpSize groupOpSize_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_1D_H_