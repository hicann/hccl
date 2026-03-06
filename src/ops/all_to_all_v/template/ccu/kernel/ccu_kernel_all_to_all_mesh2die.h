/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_KERNEL_ALL_TO_ALL_MESH_2DIE_H_
#define HCCLV2_CCU_KERNEL_ALL_TO_ALL_MESH_2DIE_H_

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using RankId = u32;

class CcuKernelArgAllToAllMesh2Die : public hcomm::CcuKernelArg{
public:
    CcuKernelArgAllToAllMesh2Die(uint64_t rankSize, uint32_t rankId, const OpParam &opParam,
        const std::vector<std::vector<RankId>> &subCommRanks, bool withMyRank, const std::vector<RankId> &rankGroup)
        : rankSize_(rankSize), rankId_(rankId), opParam_(opParam), subCommRanks_(subCommRanks), withMyRank_(withMyRank),
          rankGroup_(rankGroup) {}
    ~CcuKernelArgAllToAllMesh2Die() override {}

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelAllToAllMesh2Die", opParam_, subCommRanks_);
        return signature;
    }

    uint64_t rankSize_;
    uint32_t rankId_;
    OpParam opParam_;
    std::vector<std::vector<RankId>> subCommRanks_;
    bool withMyRank_;
    std::vector<RankId> rankGroup_;
};

class CcuTaskArgAllToAllMesh2Die : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllToAllMesh2Die(uint64_t inputAddr, uint64_t outputAddr,
        uint64_t token,  uint64_t sliceSize, uint64_t inputSliceStride, uint64_t outputSliceStride) :
        inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), sliceSize_(sliceSize),
        inputSliceStride_(inputSliceStride), outputSliceStride_(outputSliceStride){}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t sliceSize_;
    uint64_t token_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
};

class CcuKernelAllToAllMesh2Die : public CcuKernelAlgBase {
public:
    CcuKernelAllToAllMesh2Die(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllToAllMesh2Die() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResources();
    void LoadArgs();
    void PreSync();
    void PostSync();
    uint32_t CalcDstRank(uint32_t peerId) const;
    void DoRepeatAllToAll();

    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    uint16_t virRankSize{0};
    uint64_t logicRankSize{0};
    hcomm::CcuRep::Variable input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    std::vector<ChannelHandle> channels_;
    bool withMyRank_ = true;  // 发数据是否包含本rank
    std::vector<RankId> rankGroup_;
    hcomm::CcuRep::Variable sliceSize_;
    hcomm::CcuRep::Variable inputSliceStride_;
    hcomm::CcuRep::Variable outputoffset_;
    GroupOpSize groupOpSize_;

    // 在本地的搬运完成标记
    hcomm::CcuRep::CompletedEvent event_;
};
} // namespace ops_hccl

#endif // HCCLV2_CCU_KERNEL_ALL_TO_ALL_MESH_2DIE_H_