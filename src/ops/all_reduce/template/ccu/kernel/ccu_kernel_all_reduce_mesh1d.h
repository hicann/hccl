/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALLREDUCE_MESH_1D_H
#define HCCL_CCU_KERNEL_ALLREDUCE_MESH_1D_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
class CcuKernelArgAllReduceMesh1D : public CcuKernelArg {
public:
    explicit CcuKernelArgAllReduceMesh1D(const std::vector<uint64_t> &dSize, uint32_t rId, const OpParam &opParam,
        const std::vector<std::vector<uint32_t>> &tempVTopo) :
            dimSize_(dSize), rankId_(rId), opParam_(opParam), tempVTopo_(tempVTopo) {}
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllReduceMesh1D", opParam_, tempVTopo_);
        return signature;
    }
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_;
    OpParam opParam_;
    std::vector<std::vector<uint32_t>> tempVTopo_;
};

class CcuTaskArgAllReduceMesh1D : public CcuTaskArg {
public:
    explicit CcuTaskArgAllReduceMesh1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize, uint64_t offSet,
        uint64_t token) :
        inputAddr_(inputAddr), outputAddr_(outputAddr), sliceSize_(sliceSize), offSet_(offSet), token_(token) {}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t sliceSize_;
    uint64_t offSet_;
    uint64_t token_;
};

class CcuKernelAllReduceMesh1D : public CcuKernelAlgBase {
public:
    CcuKernelAllReduceMesh1D(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllReduceMesh1D() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;
private:
    void RunBroadcast(std::vector<CcuRep::RemoteAddr> &dst, CcuRep::LocalAddr &src);
    void RunReduce(CcuRep::LocalAddr &dst, std::vector<CcuRep::RemoteAddr> &src);

    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable offSet_;
    GroupOpSize groupOpSize_;
    GroupOpSize groupOpSizeV2_;
    std::vector<ChannelHandle> channels_;
};
}

#endif