/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALLGATHER_2DIES_MESH_1D_H
#define HCCL_CCU_KERNEL_ALLGATHER_2DIES_MESH_1D_H

#include <cstdint>
#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
class CcuKernelArgAllGather2DiesMesh1D : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllGather2DiesMesh1D(uint64_t dimSize, uint32_t rankId,
            std::vector<uint32_t> &rankIdGroup, bool ifHandleSelfRank, const std::vector<std::vector<uint32_t>>& subCommRanks, const OpParam& opParam)
            : dimSize_(dimSize), rankId_(rankId), rankIdGroup_(rankIdGroup), ifHandleSelfRank_(ifHandleSelfRank),
              subCommRanks_(subCommRanks), opParam_(opParam)
        {
            HCCL_DEBUG("[CcuKernelArgAllGather2DiesMesh1D] dimSize: %lu, rankId: %u",
                   dimSize_, rankId_);
        }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllGather2DiesMesh1D", opParam_, subCommRanks_);
        return signature;
    }

    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    std::vector<uint32_t>                   rankIdGroup_;
    bool                                    ifHandleSelfRank_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
    OpParam                                 opParam_;
};

class CcuTaskArgAllGather2DiesMesh1D : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllGather2DiesMesh1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize,
                                                uint64_t offSet, uint64_t token)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), sliceSize_(sliceSize), offSet_(offSet), token_(token)
    {
        HCCL_DEBUG("[CcuTaskArgReduceScatterMeshMem2Mem1D] inputAddr: %lu, outputAddr: %lu, sliceSize: %u, offSet: %u",
                   inputAddr_, outputAddr_, sliceSize_, offSet_);
    }
 
    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t sliceSize_;
    uint64_t offSet_;
    uint64_t token_;
};

class CcuKernelAllGather2DiesMesh1D : public CcuKernelAlgBase {
public:
    CcuKernelAllGather2DiesMesh1D(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllGather2DiesMesh1D() override {}
 
    HcclResult Algorithm();
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;
 
private:
    HcclResult InitResource();//资源初始化 check
    void LoadArgs();//参数载入 check
    void PreSync();//前同步 check
    void PostSync();//后同步 check
    void DoAllGather();//AllGather计算
    
    void CreateLocalCopyLoop();//本地复制的工具函数 check?
    void LocalCopyByLoopGroup(CcuRep::LocalAddr dst, CcuRep::LocalAddr src);//本地复制 HBM->ccu ms->HBM check

    uint32_t rankId_{0};
    uint64_t rankSize_{0};
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable offSet_;
    CcuRep::Variable sliceSize_;
    std::vector<uint32_t> rankIdGroup_;
    bool ifHandleSelfRank_;
    std::vector<ChannelHandle> channels_;
    hcomm::CcuRep::CompletedEvent event_;// signal, used to controll the broadcast sync
    GroupOpSize groupOpSize_;
};

}//namespace ops_hccl

#endif//HCCL_CCU_KERNEL_ALLGATHER_2DIES_MESH_1D_H