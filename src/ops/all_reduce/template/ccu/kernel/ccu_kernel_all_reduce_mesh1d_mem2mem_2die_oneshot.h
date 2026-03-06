/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_MEM2MEM_2DIE_ONESHOT
#define HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_MEM2MEM_2DIE_ONESHOT
 
#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
 
namespace ops_hccl {
using namespace hcomm;
 
class CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot : public CcuKernelArg {
public:
    explicit CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot(uint64_t dimSize, uint32_t rankId, const OpParam& opParam,
                                                    const std::vector<uint32_t>& kernelRanks, 
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks,
                                                    bool rmtReduceWithMyRank)
        : dimSize_(dimSize),
          rankId_(rankId),
          opParam_(opParam),
          kernelRanks_(kernelRanks),
          subCommRanks_(subCommRanks),
          rmtReduceWithMyRank_(rmtReduceWithMyRank)
    {
        HCCL_DEBUG("[CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot] dimSize: %lu, rankId: %u, reduceOp: %d, dataType: %d",
                   dimSize_, rankId_, opParam.reduceType, opParam.DataDes.dataType);
    }
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    OpParam                                 opParam_;
    std::vector<uint32_t>                   kernelRanks_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
    bool                                    rmtReduceWithMyRank_;
    std::vector<ChannelHandle>              kernelChannels;
};
 
class CcuTaskArgAllReduceMesh1DMem2Mem2DieOneShot : public CcuTaskArg {
public:
    explicit CcuTaskArgAllReduceMesh1DMem2Mem2DieOneShot(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                uint64_t scratchAddr, uint64_t normalSliceSize)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), scratchAddr_(scratchAddr), normalSliceSize_(normalSliceSize)
    {
        HCCL_DEBUG("[CcuTaskArgAllReduceMesh1DMem2Mem2DieOneShot] inputAddr: %lu, outputAddr: %lu, scratchAddr: %lu, "
                   "normalSliceSize: %lu", inputAddr_, outputAddr_, scratchAddr_, normalSliceSize_);
    }
 
    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t scratchAddr_;
    uint64_t normalSliceSize_;
};
 
class CcuKernelAllReduceMesh1DMem2Mem2DieOneShot : public CcuKernelAlgBase {
public:
    CcuKernelAllReduceMesh1DMem2Mem2DieOneShot(const CcuKernelArg &arg);
    ~CcuKernelAllReduceMesh1DMem2Mem2DieOneShot() override {}
 
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;
 
private:
    HcclResult InitResource();
    void LoadArgs();
    HcclResult PreSync();
    HcclResult PostSync();
    HcclResult RmtReduce();
    void DoLocalReduce();
    void ReduceLoopGroup(CcuRep::LocalAddr &outDstOrg, std::vector<CcuRep::LocalAddr> &srcOrg,
                                                        GroupOpSize goSize, HcclDataType dataType, HcclDataType outputDataType,
                                                        HcclReduceOp opType, std::string loopName);
    std::string GetLoopBlockTag(std::string loopType, int32_t index);
    void CreateReduceLoop(uint32_t size, HcclDataType dataType, HcclDataType outputDataType,
                                                         HcclReduceOp opType, std::string loopName);
    HcclResult MissionSync(uint32_t maskIndex);
 
    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    std::vector<ChannelHandle> channels_;
 
    bool rmtReduceWithMyRank_;
    uint32_t rmtReduceRankNum_;

    uint32_t missionSyncMybit_;
    uint32_t missionSyncWaitBit_;
 
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;
 
    CcuRep::Variable myInput_;
    CcuRep::Variable myOutput_;
    CcuRep::Variable myScratch_;
    CcuRep::Variable myToken_;
    std::vector<CcuRep::Variable> peerInput_;
    std::vector<CcuRep::Variable> peerToken_;
 
    CcuRep::Variable scratchBaseOffset0_;
    CcuRep::Variable scratchBaseOffset1_;
    CcuRep::Variable normalSliceSize_;
    CcuRep::Variable localReduceSliceOffset0_;
    CcuRep::Variable localReduceSliceOffset1_;
 
    GroupOpSize localReduceGoSize_;
    GroupOpSize localReduceGoSize0_;
    GroupOpSize localReduceGoSize1_;
 
    CcuRep::CompletedEvent event_;
};
 
} // namespace ops_hccl
#endif // HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_MEM2MEM_2DIE_ONESHOT