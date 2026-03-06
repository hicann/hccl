/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_DUO_MEM2MEM_H
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_DUO_MEM2MEM_H
 
#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
 
namespace ops_hccl {
using namespace hcomm;
 
class CcuKernelArgReduceScatterMeshMem2Mem1D2Die : public CcuKernelArg {
public:
    explicit CcuKernelArgReduceScatterMeshMem2Mem1D2Die(uint64_t gRankSize,
                                                        uint64_t rankSize,
                                                        bool isReduceToOutput,
                                                        uint32_t rankId,
                                                        const OpParam& opParam,
                                                        const std::vector<uint32_t>& subRankGroup,
                                                        const std::vector<std::vector<uint32_t>>& subCommRanks)
        : gRankSize_(gRankSize),
          rankSize_(rankSize),
          isReduceToOutput_(isReduceToOutput),
          rankId_(rankId),
          opParam_(opParam),
          subRankGroup_(subRankGroup),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgReduceScatterMeshMem2Mem1D2Die] dimSize: %lu, rankId: %u, reduceOp: %d, dataType: %d",
                   rankSize_, rankId_, opParam.reduceType, opParam.DataDes.dataType);
    }
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceScatterMeshMem2Mem1D2Die", opParam_, subCommRanks_);
        return signature;
    }
    uint32_t                                rankId_;
    uint32_t                                rankSize_;
    uint32_t                                gRankSize_;
    bool                                    isReduceToOutput_;
    OpParam                                 opParam_;
    std::vector<uint32_t>                   subRankGroup_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};
 
class CcuTaskArgReduceScatterMeshMem2Mem1D2Die : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterMeshMem2Mem1D2Die(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                uint64_t scratchAddr,
                                                uint64_t inputSliceStride,
                                                uint64_t inputRepeatStride, uint64_t outputRepeatStride,
                                                uint64_t normalSliceSize, uint64_t lastSliceSize, uint64_t repeatNum)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), scratchAddr_(scratchAddr),
        inputSliceStride_(inputSliceStride), inputRepeatStride_(inputRepeatStride), outputRepeatStride_(outputRepeatStride),
        normalSliceSize_(normalSliceSize), sliceSize_(lastSliceSize), repeatNum_(repeatNum) 
    {
        HCCL_DEBUG("[CcuTaskArgReduceScatterMeshMem2Mem1D2Die] inputAddr: %lu, outputAddr: %lu, token: %lu, scratchAddr: %lu, "
                   "inputSliceStride: %lu, inputRepeatStride: %lu, outputRepeatStride: %lu, normalSliceSize: %lu, "
                   "lastSliceSize: %lu, repeatNum: %lu",
                   inputAddr_, outputAddr_, token_, scratchAddr_, inputSliceStride_, inputRepeatStride_,
                   outputRepeatStride_, normalSliceSize_, sliceSize_, repeatNum_);
    }
 
    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t scratchAddr_;
    uint64_t inputSliceStride_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;
    uint64_t normalSliceSize_;
    uint64_t sliceSize_;
    uint64_t repeatNum_;
};
 
class CcuKernelReduceScatterMesh1D2DieMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelReduceScatterMesh1D2DieMem2Mem(const CcuKernelArg &arg);
    ~CcuKernelReduceScatterMesh1D2DieMem2Mem() override {}
 
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;
 
private:
    HcclResult InitResource();
    void LoadArgs();
    HcclResult PreSync();
    HcclResult RmtReduce();
    HcclResult PostSync();
    HcclResult DoReduceScatter();
    void ReduceLoopGroup(CcuRep::LocalAddr outDstOrg, std::vector<CcuRep::LocalAddr> &scratchOrg);
    std::string GetLoopBlockTag(std::string loopType, int32_t index);
    void GroupCopy(CcuRep::LocalAddr dst, CcuRep::LocalAddr src);
    void CreateGroupCopy();
    void CreateReduceLoop(uint32_t size);
 
    uint64_t rankSize_{0};
    uint64_t gRankSize_{0};
    uint32_t rankId_{0};
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    CcuRep::Variable repeatNum_;
    HcclReduceOp reduceOp_;
    std::vector<ChannelHandle> channels_;
    std::vector<CcuRep::Variable> input_;
    CcuRep::Variable output_;
    std::vector<CcuRep::Variable> scratch_;
    std::vector<CcuRep::Variable> token_;
    std::vector<uint32_t> subRankGroup_;
    CcuRep::Variable currentRankSliceInputOffset_;
    CcuRep::Variable inputRepeatStride_;
    CcuRep::Variable outputRepeatStride_;
    CcuRep::Variable sliceSize_;
    CcuRep::LocalAddr                   myInput_;
    std::vector<CcuRep::RemoteAddr>     remoteInput_;
    std::vector<CcuRep::LocalAddr>      scratchMem_;
    CcuRep::LocalAddr                   outputTmp_;
    CcuRep::CompletedEvent event_;
    CcuRep::Variable flag_; // 用以判断是否是第一次重复
    bool isReduceToOutput_;
    GroupOpSize sliceGoSize_;
    CcuRep::Variable scratchOffset0_;
    CcuRep::Variable scratchOffset1_;
    uint32_t myRankIdx_{0};
};
 
}// namespace ops_hccl
#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_DUO_MEM2MEM_H