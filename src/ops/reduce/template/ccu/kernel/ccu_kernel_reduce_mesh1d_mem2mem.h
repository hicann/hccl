/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_MESH_1D_MEM2MEM
#define HCCL_CCU_KERNEL_REDUCE_MESH_1D_MEM2MEM

#include <vector>
#include <ios>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgReduceMesh1DMem2Mem : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgReduceMesh1DMem2Mem(uint64_t dimSize, uint32_t rankId, uint32_t rootId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          rootId_(rootId),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgReduceMesh1DMem2Mem] dimSize: %lu, rankId: %u, rootId: %u, reduceOp: %d, dataType: %d",
                   dimSize_, rankId_, rootId_, opParam.reduceType, opParam.DataDes.dataType);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceMesh1DMem2Mem", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    uint32_t                                rootId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgReduceMeshMem2Mem1D : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgReduceMeshMem2Mem1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                           uint64_t bigDataSliceNum, uint64_t bigDataSliceSize, uint64_t smallDataSliceNum,
                                           uint64_t smallDataSliceSize, uint64_t inputRepeatStride, uint64_t outputRepeatStride,
                                           uint64_t normalSliceSize, uint64_t lastSliceSize, uint64_t repeatNumVar)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), bigDataSliceNum_(bigDataSliceNum),
          bigDataSliceSize_(bigDataSliceSize), smallDataSliceNum_(smallDataSliceNum), smallDataSliceSize_(smallDataSliceSize),
          inputRepeatStride_(inputRepeatStride), outputRepeatStride_(outputRepeatStride),
          normalSliceSize_(normalSliceSize), lastSliceSize_(lastSliceSize), repeatNumVar_(repeatNumVar)
    {
        HCCL_DEBUG("[CcuTaskArgReduceMeshMem2Mem1D] inputAddr: %lu, outputAddr: %lu, bigDataSliceNum: %lu, "
                   "bigDataSliceSize: %lu, smallDataSliceNum: %lu, smallDataSliceSize: %lu, inputRepeatStride: %lu, "
                   "outputRepeatStride: %lu, normalSliceSize: %lu, lastSliceSize: %lu, repeatNumVar: %lu",
                   inputAddr_, outputAddr_, bigDataSliceNum_, bigDataSliceSize_, smallDataSliceNum_,
                   smallDataSliceSize_, inputRepeatStride_, outputRepeatStride_, normalSliceSize_, lastSliceSize_, repeatNumVar_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t bigDataSliceNum_;
    uint64_t bigDataSliceSize_;
    uint64_t smallDataSliceNum_;
    uint64_t smallDataSliceSize_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;
    uint64_t normalSliceSize_;
    uint64_t lastSliceSize_;
    uint64_t repeatNumVar_;
};

class CcuKernelReduceMesh1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelReduceMesh1DMem2Mem(const hcomm::CcuKernelArg &arg);
    ~CcuKernelReduceMesh1DMem2Mem() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void DoRepeatReduce(const std::vector<hcomm::CcuRep::Variable> &srcAddr, const hcomm::CcuRep::Variable &dstAddr);
    void PostSync();

    std::vector<ChannelHandle> channels_;
    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    uint32_t rootId_{0}; // 当rankid == rootid时，为root节点 则跳过write操作
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;
    std::vector<hcomm::CcuRep::Variable> input_; // 输入地址信息
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable inputRepeatStride_;
    hcomm::CcuRep::Variable outputRepeatStride_;
    hcomm::CcuRep::Variable normalSliceSize_;
    hcomm::CcuRep::Variable lastSliceSize_;
    hcomm::CcuRep::Variable repeatNumVar_;
    hcomm::CcuRep::Variable flag_; // 用以判断是否是第一次重复

    hcomm::CcuRep::CompletedEvent event_;

    hcomm::CcuRep::LocalAddr                   myInputAddr_;
    hcomm::CcuRep::RemoteAddr                  remoteInputAddr_;
    hcomm::CcuRep::LocalAddr                   dstAddr_;
    GroupOpSize                                localGoSize_;

    hcomm::CcuRep::Variable              isInputOutputEqual_;

    std::vector<hcomm::CcuRep::Variable> chunkSize_;
    hcomm::CcuRep::Variable              chunkOffset_;
    void                                 CreateLocalCopyLoop();  
    void                                 LocalCopyByLoopGroup(hcomm::CcuRep::LocalAddr dst, hcomm::CcuRep::LocalAddr src);
};

}// namespace ops_hccl
#endif // HCCL_CCU_KERNEL_REDUCE_MESH_1D_MEM2MEM
