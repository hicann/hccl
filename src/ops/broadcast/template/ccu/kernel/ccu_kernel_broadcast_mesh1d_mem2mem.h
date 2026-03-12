/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_BROADCAST_MESH_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_BROADCAST_MESH_1D_MEM2MEM_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

class CcuKernelArgBroadcastMesh1DMem2Mem : public CcuKernelArg {
public:
    explicit CcuKernelArgBroadcastMesh1DMem2Mem(uint64_t dimSize, uint32_t rankId, uint32_t rootId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          rootId_(rootId),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_INFO("[CcuKernelArgBroadcastMesh1DMem2Mem] dimSize[%llu] rankId[%u] rootId[%u] dataType[%d] outputType[%d] "
                    "subCommRanksSize[%zu]",
            dimSize, rankId, rootId, opParam.DataDes.dataType, opParam_.DataDes.outputType, subCommRanks.size());
    }
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgBroadcastMesh1DMem2Mem", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    uint32_t                                rootId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgBroadcastMesh1DMem2Mem : public CcuTaskArg {
public:
    explicit CcuTaskArgBroadcastMesh1DMem2Mem(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                uint64_t inputSliceStride,
                                                uint64_t outputSliceStride,
                                                uint64_t inputRepeatStride, uint64_t outputRepeatStride,
                                                uint64_t normalSliceSize, uint64_t lastSliceSize, uint64_t repeatNumVar)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), inputSliceStride_(inputSliceStride),
        outputSliceStride_(outputSliceStride), inputRepeatStride_(inputRepeatStride), outputRepeatStride_(outputRepeatStride),
        normalSliceSize_(normalSliceSize), lastSliceSize_(lastSliceSize), repeatNumVar_(repeatNumVar)
    {
        HCCL_DEBUG("[CcuTaskArgBroadcastMesh1DMem2Mem] inputAddr: %lu, outputAddr: %lu, inputSliceStride: %lu, "
                   "outputSliceStride: %lu, inputRepeatStride: %lu, outputRepeatStride: %lu, normalSliceSize: %lu, "
                   "lastSliceSize: %lu, repeatNumVar: %lu",
                   inputAddr_, outputAddr_, inputSliceStride_, outputSliceStride_, inputRepeatStride_,
                   outputRepeatStride_, normalSliceSize_, lastSliceSize_, repeatNumVar_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;
    uint64_t normalSliceSize_;
    uint64_t lastSliceSize_;
    uint64_t repeatNumVar_;
};

class CcuKernelBroadcastMesh1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelBroadcastMesh1DMem2Mem(const CcuKernelArg &arg);
    ~CcuKernelBroadcastMesh1DMem2Mem() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void DoRepeaScatterMem2Mem();
    void DoRepeatAllGatherMem2Mem();
    void DoScatter(const std::vector<CcuRep::RemoteAddr> &dst);
    void DoAllGather(const CcuRep::LocalAddr &src, const std::vector<CcuRep::RemoteAddr> &dst);
    void PostSync(int CKE_id);

    uint32_t rankId_{0};
    uint32_t rootId_{0};
    uint64_t rankSize_{0};
    std::vector<ChannelHandle> channels_;
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable currentRankSliceInputOffset_;
    CcuRep::Variable currentRankSliceOutputOffset_;
    CcuRep::Variable inputRepeatStride_;
    CcuRep::Variable outputRepeatStride_;
    CcuRep::Variable normalSliceSize_;
    CcuRep::Variable lastSliceSize_;
    CcuRep::Variable allgatherOffset_;
    CcuRep::Variable repeatNumVar_;
    CcuRep::Variable flag_;
    CcuRep::Variable SliceOffset_;
    CcuRep::LocalAddr                   myScatterDst_;
    CcuRep::LocalAddr                   myAllgatherDst_;
    std::vector<CcuRep::LocalAddr>      scattersrcMem_;
    std::vector<CcuRep::RemoteAddr>     scatterdstMem_;
    std::vector<CcuRep::RemoteAddr>      allgatherdstMem_;
    CcuRep::CompletedEvent event_;
};

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_BROADCAST_MESH_1D_MEM2MEM_H
