/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_SCATTER_MESH_1D__H
#define HCCL_CCU_KERNEL_SCATTER_MESH_1D__H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgScatterMesh1D : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgScatterMesh1D(uint64_t dimSize, uint32_t rankId, uint32_t root,
                                              const OpParam& opParam,
                                              const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          rootId_(root),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgScatterMesh1D] dimSize: %lu, rankId: %u, root: %u, dataType: %d",
                   dimSize_, rankId_, rootId_, opParam.DataDes.dataType);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgScatterMesh1D", opParam_, subCommRanks_);
        return signature;
    }

    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    uint32_t                                rootId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

/**
 * @brief CCU Task Scatter Mesh 1D 的参数类
 * 
 * 用于执行CCU kernel任务时的参数，包含地址、大小、步长等信息
 */
class CcuTaskArgScatterMesh1D : public hcomm::CcuTaskArg {
public:
    /**
     * @brief 构造函数
     * @param inputAddr 输入缓冲区地址
     * @param outputAddr 输出缓冲区地址
     * @param token Token信息（用于同步）
     * @param inputSliceStride 输入切片步长（每个rank的数据大小）
     * @param inputRepeatStride 输入重复步长（用于多次重复操作）
     * @param outputRepeatStride 输出重复步长
     * @param normalSliceSize 正常切片大小（除最后一个切片外）
     * @param lastSliceSize 最后一个切片大小
     * @param repeatNum 重复次数
     */
    explicit CcuTaskArgScatterMesh1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                            uint64_t inputSliceStride, uint64_t inputRepeatStride,
                                            uint64_t outputRepeatStride, uint64_t normalSliceSize,
                                            uint64_t lastSliceSize, uint64_t repeatNum)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          token_(token),
          inputSliceStride_(inputSliceStride),
          inputRepeatStride_(inputRepeatStride),
          outputRepeatStride_(outputRepeatStride),
          normalSliceSize_(normalSliceSize),
          lastSliceSize_(lastSliceSize),
          repeatNum_(repeatNum)
    {
        HCCL_DEBUG("[CcuTaskArgScatterMesh1D] inputAddr: %lu, outputAddr: %lu, token: %lu, "
                   "inputSliceStride: %lu, inputRepeatStride: %lu, outputRepeatStride: %lu, normalSliceSize: %lu, "
                   "lastSliceSize: %lu, repeatNum: %lu",
                   inputAddr_, outputAddr_, token_, inputSliceStride_, inputRepeatStride_, outputRepeatStride_,
                   normalSliceSize_, lastSliceSize_, repeatNum_);
    }

    uint64_t inputAddr_;           // 输入缓冲区地址
    uint64_t outputAddr_;          // 输出缓冲区地址
    uint64_t token_;               // Token信息（用于同步）
    uint64_t inputSliceStride_;    // 输入切片步长（每个rank的数据大小）
    uint64_t inputRepeatStride_;   // 输入重复步长
    uint64_t outputRepeatStride_;  // 输出重复步长
    uint64_t normalSliceSize_;     // 正常切片大小
    uint64_t lastSliceSize_;       // 最后一个切片大小
    uint64_t repeatNum_;           // 重复次数
};

class CcuKernelScatterMesh1D : public CcuKernelAlgBase {
public:
    CcuKernelScatterMesh1D(const hcomm::CcuKernelArg &arg);
    ~CcuKernelScatterMesh1D() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoRepeatScatter();
    void DoScatter();

    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    uint32_t rootId_{0};
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    hcomm::CcuRep::Variable repeatNum_;
    std::vector<ChannelHandle> channels_;
    hcomm::CcuRep::Variable input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable currentRankSliceInputOffset_;
    hcomm::CcuRep::Variable inputRepeatStride_;
    hcomm::CcuRep::Variable outputRepeatStride_;
    hcomm::CcuRep::Variable normalSliceSize_;
    hcomm::CcuRep::Variable lastSliceSize_;
    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    hcomm::CcuRep::LocalAddr myInput_;
    std::vector<hcomm::CcuRep::RemoteAddr> outputMem_;
    std::vector<hcomm::CcuRep::LocalAddr> inputMem_;
    hcomm::CcuRep::CompletedEvent event_;
    hcomm::CcuRep::Variable flag_; // 用以判断是否是第一次重复
};
}// namespace ops_hccl
#endif // HCCL_CCU_KERNEL_SCATTER_MESH_1D_
