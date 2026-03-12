/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_BROADCAST_MESH_1D_H
#define HCCL_CCU_KERNEL_BROADCAST_MESH_1D_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgBroadcastMesh1D : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgBroadcastMesh1D(uint64_t dimSize, uint32_t rankId, uint32_t rootId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          rootId_(rootId),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgBroadcastMesh1D", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    uint32_t                                rootId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgBroadcastMesh1D : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgBroadcastMesh1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                uint64_t offSet, uint64_t sliceSize)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), offSet_(offSet), sliceSize_(sliceSize)
    {
        HCCL_DEBUG("[CcuTaskArgBroadcastMesh1D] inputAddr: %lu, outputAddr: %lu, offSet_: %lu, sliceSize_: %lu",
                   inputAddr_, outputAddr_, offSet_, sliceSize_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t offSet_;
    uint64_t sliceSize_;
};

class CcuKernelBroadcastMesh1D : public CcuKernelAlgBase {
public:
    CcuKernelBroadcastMesh1D(const hcomm::CcuKernelArg &arg);
    ~CcuKernelBroadcastMesh1D() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void BroadcastFromRootToAll();

    uint32_t rankId_{0};
    uint32_t rootId_{0};
    uint64_t rankSize_{0};
    std::vector<ChannelHandle> channels_;
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    hcomm::CcuRep::Variable input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable offSet_;
    hcomm::CcuRep::Variable slicesize_;
    GroupOpSize groupOpSize_;
};

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_BROADCAST_MESH_1D_H
