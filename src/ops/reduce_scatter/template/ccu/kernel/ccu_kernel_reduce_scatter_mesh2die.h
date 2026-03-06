/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_KERNEL_REDUCE_SCATTER_MESH_2Die_H_
#define HCCLV2_CCU_KERNEL_REDUCE_SCATTER_MESH_2Die_H_

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using RankId = u32;

class CcuKernelArgReduceScatterMesh2Die : public hcomm::CcuKernelArg{
public:
    CcuKernelArgReduceScatterMesh2Die(uint64_t rankSize, uint32_t rankId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks, bool rmtReduceWithMyRank)
        : rankSize_(rankSize),
          rankId_(rankId),
          opParam_(opParam),
          subCommRanks_(subCommRanks),
          rmtReduceWithMyRank_(rmtReduceWithMyRank)
          {}
    ~CcuKernelArgReduceScatterMesh2Die() override {}

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelReduceScatterMesh2Die", opParam_, subCommRanks_);
        return signature;
    }

    uint64_t rankSize_;
    uint32_t rankId_;
    OpParam opParam_;
    std::vector<std::vector<RankId>> subCommRanks_;
    bool rmtReduceWithMyRank_;
};

class CcuTaskArgReduceScatterMesh2Die : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterMesh2Die(uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t scratchAddr,
                                               uint64_t sliceSize, uint64_t inputSliceStride)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), scratchAddr_(scratchAddr), sliceSize_(sliceSize),
         inputSliceStride_(inputSliceStride)
    {
    }

    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t token_{0};
    uint64_t scratchAddr_{0};
    uint64_t sliceSize_{0};
    uint64_t inputSliceStride_{0};
};

class CcuKernelReduceScatterMesh2Die : public CcuKernelAlgBase {
public:
    CcuKernelReduceScatterMesh2Die(const hcomm::CcuKernelArg &arg);
    ~CcuKernelReduceScatterMesh2Die() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResources();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void RmtReduce();

    bool rmtReduceWithMyRank_{true};

    uint32_t myRankId_{0};
    uint32_t rankSize_{0};

    uint32_t rmtReduceRankNum_{0};
    uint32_t rmtSyncMyBit_{0};
    uint32_t rmtSyncWaitBit_{0};

    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;

    hcomm::CcuRep::Variable myInput_;
    hcomm::CcuRep::Variable myOutput_;
    hcomm::CcuRep::Variable myScratch_;
    hcomm::CcuRep::Variable myToken_;
    std::vector<hcomm::CcuRep::Variable> peerInput_;
    std::vector<hcomm::CcuRep::Variable> peerToken_;

    hcomm::CcuRep::Variable sliceSize_;
    hcomm::CcuRep::Variable rmtReduceSliceOffset_;
    GroupOpSize rmtReduceGoSize_;
    std::vector<ChannelHandle> channels_;
};
}// namespace ops_hccl

#endif // HCCLV2_CCU_KERNEL_REDUCE_SCATTER_MESH_2Die_H_
