/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALLREDUCE_MESH_1D_2DIE_ONE_SHOT_H
#define HCCL_CCU_KERNEL_ALLREDUCE_MESH_1D_2DIE_ONE_SHOT_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgAllreduceMesh1D2DieOneShot : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllreduceMesh1D2DieOneShot(uint64_t dimSize, uint32_t rankId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks,
                                                    bool rmtReduceWithMyRank)
        : dimSize_(dimSize),
          rankId_(rankId),
          opParam_(opParam),
          subCommRanks_(subCommRanks),
          rmtReduceWithMyRank_(rmtReduceWithMyRank)
    {
        HCCL_DEBUG("[CcuKernelArgAllreduceMesh1D2DieOneShot] dimSize: %lu, rankId: %u",
                   dimSize_, rankId_, rmtReduceWithMyRank_);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllreduceMesh1D2DieOneShot", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
    bool                                    rmtReduceWithMyRank_;
};

class CcuTaskArgAllreduceMesh1D2DieOneShot : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllreduceMesh1D2DieOneShot(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                uint64_t scratchAddr,
                                                uint64_t sliceSize)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), scratchAddr_(scratchAddr),
        sliceSize_(sliceSize)
    {
        HCCL_DEBUG("[CcuTaskArgAllreduceMesh1D2DieOneShot] inputAddr: %lu, outputAddr: %lu, token: %lu, scratchAddr: %lu, "
                   "sliceSize: %lu, ", inputAddr_, outputAddr_, token_, scratchAddr_, sliceSize_); 
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t scratchAddr_;
    uint64_t sliceSize_;
};

class CcuKernelAllreduceMesh1D2DieOneShot : public CcuKernelAlgBase {
public:
    CcuKernelAllreduceMesh1D2DieOneShot(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllreduceMesh1D2DieOneShot() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync(uint32_t signalIndex);
    void MissionSync(uint32_t maskIndex);
    void DoRmtReduce();
    std::string GetLoopBlockTag(std::string loopType, int32_t index) const;
    void DoLocalReduce();
    void ReduceLoopGroup(CcuRep::LocalAddr &outDstOrg, std::vector<CcuRep::LocalAddr> &srcOrg,
        GroupOpSize goSize, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType);
    void CreateReduceLoop(uint32_t size, HcclDataType dataType, HcclDataType outputDataType,
                                                         HcclReduceOp opType);

    bool rmtReduceWithMyRank_;
    uint32_t myRankId_;
    uint32_t rankSize_;
 
    uint32_t rmtReduceRankNum_;
    uint32_t rmtSyncMyBit_;
    uint32_t rmtSyncWaitBit_;
 
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;
 
    uint32_t missionSyncMybit_;
    uint32_t missionSyncWaitBit_;

    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    
    hcomm::CcuRep::Variable myInput_;
    hcomm::CcuRep::Variable myOutput_;
    hcomm::CcuRep::Variable myScratch_;
    hcomm::CcuRep::Variable myToken_;

    // hcomm::CcuRep::LocalAddr scratchMem_;
    std::vector<hcomm::CcuRep::Variable> input_;
    // std::vector<hcomm::CcuRep::RemoteAddr> remoteInput_;
    std::vector<hcomm::CcuRep::Variable> remoteToken_;
 
    hcomm::CcuRep::Variable scratchBaseOffset0_;
    hcomm::CcuRep::Variable scratchBaseOffset1_;
 
    hcomm::CcuRep::Variable localReduceSliceOffset0_;
    hcomm::CcuRep::Variable localReduceSliceOffset1_;
 
    GroupOpSize rmtReduceGoSize_;
    GroupOpSize localReduceGoSize0_;
    GroupOpSize localReduceGoSize1_;
    std::vector<ChannelHandle> channels_;
};

} // namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_ALLREDUCE_MESH_1D_2DIE_ONE_SHOT_H

