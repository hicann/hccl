/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_MEM2MEM
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_MEM2MEM

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

class CcuKernelArgReduceScatterMesh1DMem2Mem : public CcuKernelArg {
public:
    explicit CcuKernelArgReduceScatterMesh1DMem2Mem(uint64_t dimSize, uint32_t rankId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgReduceScatterMesh1DMem2Mem] dimSize: %lu, rankId: %u, reduceOp: %d, dataType: %d",
                   dimSize_, rankId_, opParam.reduceType, opParam.DataDes.dataType);
    }
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceScatterMesh1DMem2Mem", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgReduceScatterMesh1DMem2Mem : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterMesh1DMem2Mem(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                uint64_t scratchAddr,
                                                uint64_t inputSliceStride,
                                                uint64_t inputRepeatStride, uint64_t outputRepeatStride,
                                                uint64_t normalSliceSize, uint64_t lastSliceSize, uint64_t repeatNum)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), scratchAddr_(scratchAddr),
        inputSliceStride_(inputSliceStride), inputRepeatStride_(inputRepeatStride), outputRepeatStride_(outputRepeatStride),
        normalSliceSize_(normalSliceSize), lastSliceSize_(lastSliceSize), repeatNum_(repeatNum) 
    {
        HCCL_DEBUG("[CcuTaskArgReduceScatterMesh1DMem2Mem] inputAddr: %lu, outputAddr: %lu, scratchAddr: %lu, "
                   "inputSliceStride: %lu, inputRepeatStride: %lu, outputRepeatStride: %lu, normalSliceSize: %lu, "
                   "lastSliceSize: %lu, repeatNum: %lu",
                   inputAddr_, outputAddr_, scratchAddr_, inputSliceStride_, inputRepeatStride_,
                   outputRepeatStride_, normalSliceSize_, lastSliceSize_, repeatNum_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t scratchAddr_;
    uint64_t inputSliceStride_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;
    uint64_t normalSliceSize_;
    uint64_t lastSliceSize_;
    uint64_t repeatNum_;
};

class CcuKernelReduceScatterMesh1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelReduceScatterMesh1DMem2Mem(const CcuKernelArg &arg);
    ~CcuKernelReduceScatterMesh1DMem2Mem() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoRepeatReduceScatter();
    void DoReduceScatter();
    
    std::string GetLoopBlockTag(std::string loopType, int32_t index);
    void CreateReduceLoop(uint32_t size, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType);
    void ReduceLoopGroup(CcuRep::LocalAddr outDstOrg, CcuRep::LocalAddr srcOrg, std::vector<CcuRep::LocalAddr> &scratchOrg,
        GroupOpSize goSize, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType);
 
    const std::string LOOP_BLOCK_TAG{"_local_copy_reduce_loop_"};
 

    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    CcuRep::Variable repeatNum_;
    HcclReduceOp reduceOp_;
    std::vector<CcuRep::Variable> input_;
    CcuRep::Variable output_;
    std::vector<CcuRep::Variable> scratch_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable currentRankSliceInputOffset_;
    CcuRep::Variable inputRepeatStride_;
    CcuRep::Variable outputRepeatStride_;
    CcuRep::Variable normalSliceSize_;
    GroupOpSize normalGoSize_;
    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    CcuRep::LocalAddr                   myInput_;
    std::vector<CcuRep::RemoteAddr>     remoteInput_;
    std::vector<CcuRep::LocalAddr>      scratchMem_;
    CcuRep::CompletedEvent event_;
    CcuRep::Variable flag_; // 用以判断是否是第一次重复
};

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_MEM2MEM
