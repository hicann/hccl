/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_MESH_2D_MEM2MEM_H
#define HCCL_CCU_KERNEL_REDUCE_MESH_2D_MEM2MEM_H

#include <vector>
#include <ios>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgReduceMeshMem2Mem2D : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgReduceMeshMem2Mem2D(const std::vector<uint64_t> dimSize, uint64_t rankId, 
                                             uint32_t rootId, uint32_t axisId, const OpParam& opParam,
                                             const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          rootId_(rootId),
          axisId_(axisId),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgReduceMeshMem2Mem2D] dimSize[0]: %lu, dimSize[1]: %lu, "
                   "rankId: %u, rootId: %u, axisId: %u, reduceOp: %d, dataType: %d",
                   dimSize_[0], dimSize_[1], rankId_, rootId_, axisId_, 
                   opParam.reduceType, opParam.DataDes.dataType);
    }

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceMeshMem2Mem2D", opParam_, subCommRanks_);
        return signature;
    }
    std::vector<uint64_t>                   dimSize_;
    uint32_t                                rankId_;
    uint32_t                                rootId_;
    uint32_t                                axisId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgReduceMeshMem2Mem2D : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgReduceMeshMem2Mem2D(uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize,
                                           uint64_t xAxisSize, uint64_t yAxisSize, uint64_t token)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), sliceSize_(sliceSize), xAxisSize_(xAxisSize),
          yAxisSize_(yAxisSize), token_(token)
    {
        HCCL_DEBUG("[CcuTaskArgReduceMeshMem2Mem2D] inputAddr: %lu, outputAddr: %lu, sliceSize: %lu, xAxisSize: %lu, "
                   "yAxisSize: %lu",
                   inputAddr_, outputAddr_, sliceSize_, xAxisSize_, yAxisSize_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t sliceSize_;
    uint64_t xAxisSize_;
    uint64_t yAxisSize_;
    uint64_t token_;
};

class CcuKernelReduceMesh2DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelReduceMesh2DMem2Mem(const hcomm::CcuKernelArg &arg);
    ~CcuKernelReduceMesh2DMem2Mem() override {}

    HcclResult Algorithm();
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResources();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void AxisSync(uint32_t signalIndex);
    void ReduceStep1();
    void ReduceStep2();
    std::vector<uint64_t> CalMeshChunkSlice(uint64_t dataSize, uint64_t sliceNum); // for mesh-chunk

    std::vector<ChannelHandle>           channels_;
    std::vector<uint64_t>                dimSize_;
    uint32_t                             rankId_{0}; // 全局rankid
    uint32_t                             rootId_{0}; // 全局rootid
    std::vector<uint32_t>                dimId_;     // 本rank所在行或列的编号
    std::vector<uint32_t>                rootDimId_; // root所在行或列的编号
    uint32_t                             localId_{0}; // 当前轴的rankId
    uint32_t                             localSize_{0}; // 当前轴的rankSize
    uint32_t                             axisId_{0};    // 0 : X轴， 1 : Y轴
    HcclDataType                         dataType_;
    HcclDataType                         outputDataType_;
    HcclReduceOp                         reduceOp_;
    std::vector<hcomm::CcuRep::Variable> input_; // 输入地址信息
    hcomm::CcuRep::Variable              output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable              xAxisSize_;
    hcomm::CcuRep::Variable              yAxisSize_;
    hcomm::CcuRep::Variable              yAxisOffset_;
    hcomm::CcuRep::CompletedEvent        event_;

    GroupOpSize                          xAxisGroupOpSize_;
    GroupOpSize                          yAxisGroupOpSize_;
    GroupOpSize                          curGoSize_; // for loop-group local copy
    // variables for mesh-chunk
    std::vector<hcomm::CcuRep::Variable> xChunkSize_; // for xsliceszie
    std::vector<hcomm::CcuRep::Variable> yChunkSize_; // for ysliceszie
    std::vector<hcomm::CcuRep::Variable> chunkSize_;  // for current axis
    hcomm::CcuRep::Variable              chunkOffset_;
};

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_REDUCE_MESH_2D_MEM2MEM_H
