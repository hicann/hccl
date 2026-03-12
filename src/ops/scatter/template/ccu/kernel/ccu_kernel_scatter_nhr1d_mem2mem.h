/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_SCATTER_NHR_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_SCATTER_NHR_1D_MEM2MEM_H

#include <vector>
#include <ios>
#include <map>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "template_utils.h"

namespace ops_hccl {
using namespace hcomm;

using NHRStepInfo = struct NHRStepInfo {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    NHRStepInfo() : nSlices(0)
    {
    }
};

/**
 * @brief CCU Kernel Scatter NHR 1D Mem2Mem的参数类
 * 
 * 用于初始化CCU kernel的参数，包含rank信息、NHR步骤信息、rank2ChannelIdx等
 */
class CcuKernelArgScatterNHRMem2Mem1D : public CcuKernelArg {
public:
    /**
     * @brief 构造函数
     * 
     * @param dimSize 维度大小（rank数量）
     * @param rankId 当前rank ID（在子通信域中的rank ID）
     * @param root 根节点rank ID（在通信域中的rank ID）
     * @param axisId 轴ID（0表示die0，1表示die1）
     * @param axisSize 轴数量（1或2）
     * @param stepInfoVector NHR步骤信息向量
     * @param rank2ChannelIdx 虚拟rank ID到channel索引的映射
     * @param opParam 操作参数
     * @param subCommRanks 子通信域rank列表
     */
    explicit CcuKernelArgScatterNHRMem2Mem1D(uint64_t dimSize, uint32_t rankId, uint32_t root, uint32_t axisId,
                                             uint32_t axisSize, const std::vector<NHRStepInfo> stepInfoVector,
                                             const std::map<u32, u32> &rank2ChannelIdx, const OpParam &opParam,
                                             const std::vector<std::vector<uint32_t>> &subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          rootId_(root),
          axisId_(axisId),
          axisSize_(axisSize),
          stepInfoVector_(stepInfoVector),
          rank2ChannelIdx_(rank2ChannelIdx),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgScatterNHRMem2Mem1D] dimSize: %lu, rankId: %u, root: %u, axisId: %u, axisSize: %u, "
                   "dataType: %d, stepInfoVector.size(): %lu, rank2ChannelIdx.size(): %lu",
                   dimSize_, rankId_, rootId_, axisId_, axisSize_, opParam.DataDes.dataType, stepInfoVector_.size(),
                   rank2ChannelIdx_.size());
    }
    
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgScatterNHRMem2Mem1D", opParam_, subCommRanks_);
        return signature;
    }
    
    uint64_t dimSize_;                              // 维度大小
    uint32_t rankId_;                               // 当前rank ID（在子通信域中的rank ID）
    uint32_t rootId_;                               // 根节点rank ID（在通信域中的rank ID）
    uint32_t axisId_;                               // 轴ID（0=die0，1=die1）
    uint32_t axisSize_;                             // 轴数量（1或2）
    std::vector<NHRStepInfo> stepInfoVector_;  // NHR步骤信息向量
    std::map<u32, u32> rank2ChannelIdx_;                   // 虚拟rank ID到channel索引的映射
    OpParam opParam_;                                // 操作参数
    std::vector<std::vector<uint32_t>> subCommRanks_;  // 子通信域rank列表
};

/**
 * @brief CCU Task Scatter NHR 1D Mem2Mem的参数类
 * 
 * 用于执行CCU kernel任务时的参数，包含地址、大小、步长等信息
 */
class CcuTaskArgScatterNHRMem2Mem1D : public CcuTaskArg {
public:
    /**
     * @brief 构造函数
     * @param inputAddr 输入缓冲区地址
     * @param outputAddr 输出缓冲区地址
     * @param scratchAddr Scratch缓冲区地址
     * @param token Token信息（用于同步）
     * @param sliceSize 切片大小
     * @param die0Size Die0切片大小
     * @param die1Size Die1切片大小
     * @param inputSliceStride 输入切片步长（每个rank的数据大小）
     * @param inputRepeatStride 输入重复步长（用于多次重复操作）
     * @param outputRepeatStride 输出重复步长
     * @param repeatNum 重复次数
     * @param isOutputScratch 输出是否使用scratch buffer
     */
    explicit CcuTaskArgScatterNHRMem2Mem1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr,
                                           uint64_t token, uint64_t sliceSize, uint64_t die0Size, uint64_t die1Size,
                                           uint64_t inputSliceStride, uint64_t inputRepeatStride,
                                           uint64_t outputRepeatStride, uint64_t repeatNum, uint64_t isOutputScratch)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          scratchAddr_(scratchAddr),
          token_(token),
          sliceSize_(sliceSize),
          die0Size_(die0Size),
          die1Size_(die1Size),
          inputSliceStride_(inputSliceStride),
          inputRepeatStride_(inputRepeatStride),
          outputRepeatStride_(outputRepeatStride),
          repeatNum_(repeatNum),
          isOutputScratch_(isOutputScratch)
    {
        HCCL_DEBUG("[CcuTaskArgScatterNHRMem2Mem1D] inputAddr: %lu, outputAddr: %lu, scratchAddr: %lu, token: %lu, "
                   "sliceSize: %lu, die0Size: %lu, die1Size: %lu, inputSliceStride: %lu, inputRepeatStride: %lu, "
                   "outputRepeatStride: %lu, repeatNum: %lu, isOutputScratch: %lu",
                   inputAddr_, outputAddr_, scratchAddr_, token_, sliceSize_, die0Size_, die1Size_,
                   inputSliceStride_, inputRepeatStride_, outputRepeatStride_, repeatNum_, isOutputScratch_);
    }

    uint64_t inputAddr_;           // 输入缓冲区地址
    uint64_t outputAddr_;          // 输出缓冲区地址
    uint64_t scratchAddr_;         // Scratch缓冲区地址
    uint64_t token_;               // Token信息（用于同步）
    uint64_t sliceSize_;           // 切片大小
    uint64_t die0Size_;            // Die0切片大小
    uint64_t die1Size_;            // Die1切片大小
    uint64_t inputSliceStride_;    // 输入切片步长（每个rank的数据大小）
    uint64_t inputRepeatStride_;   // 输入重复步长
    uint64_t outputRepeatStride_;  // 输出重复步长
    uint64_t repeatNum_;           // 重复次数
    uint64_t isOutputScratch_;     // 输出是否使用scratch buffer
};

/**
 * @brief CCU Kernel Scatter NHR 1D Mem2Mem实现类
 * 
 * 实现基于NHR算法的1D Scatter操作，使用内存到内存的传输方式
 * Scatter操作：root节点将数据分发给各个rank
 * NHR算法：通过多步骤的通信，逐步将数据从root节点分发到所有rank
 * 特点：
 * 1. 支持多轴（axisSize可能为1或2），当axisSize=2时，数据在die0和die1上并行处理
 * 2. 使用scratch buffer作为中间存储
 * 3. 需要执行多个NHR步骤，每个步骤完成部分数据的传输
 */
class CcuKernelScatterNHR1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelScatterNHR1DMem2Mem(const CcuKernelArg &arg);
    ~CcuKernelScatterNHR1DMem2Mem() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void DoScatterNHR();
    void DoScatterNHRSingleStep(const NHRStepInfo &nhrStepInfo);
    void DoSendRecvSlice(const u32 &toRank, CcuRep::LocalAddr &src, CcuRep::RemoteAddr &dst,
                        u32 signalIndex);

    uint64_t dimSize_{0};                                    // Rank数量
    uint32_t rankId_{0};                                      // 当前rank ID（在子通信域中的rank ID）
    uint32_t rootId_{0};                                      // 根节点rank ID（在通信域中的rank ID）
    uint32_t axisId_{0};                                      // 轴ID（0=die0，1=die1）
    uint32_t axisSize_{0};                                    // 轴数量（1或2）
    uint32_t localSize_{0};                                   // 本rank需要通信的rank数量（rank2ChannelIdx的大小）
    uint32_t myRankIdx_{0};                                   // 本rank在rank2ChannelIdx中的索引
    uint32_t signalNum_{0};                                   // 需要使用的signal数量
    HcclDataType dataType_;                                   // 数据类型
    std::vector<NHRStepInfo> stepInfoVector_;            // NHR步骤信息向量
    std::map<u32, u32> rank2ChannelIdx_;                             // 虚拟rank ID到channel索引的映射
    std::vector<ChannelHandle> channels_;                     // 通信通道句柄列表（排除自身）

    // 资源变量
    CcuRep::Variable input_;                          // 输入变量
    CcuRep::Variable output_;                          // 输出变量
    std::vector<CcuRep::Variable> scratch_;          // Scratch缓冲区变量列表
    std::vector<CcuRep::Variable> token_;             // Token变量列表（用于同步）
    CcuRep::Variable die0Size_;                       // Die0切片大小变量
    CcuRep::Variable die1Size_;                       // Die1切片大小变量
    CcuRep::Variable inputSliceStride_;               // 输入切片步长变量
    CcuRep::Variable curScratchStride_;               // 当前Scratch步长变量
    CcuRep::Variable inputRepeatStride_;              // 输入重复步长变量
    CcuRep::Variable outputRepeatStride_;             // 输出重复步长变量
    CcuRep::Variable repeatNumVar_;                    // 重复次数变量
    CcuRep::Variable isOutputScratch_;                // 输出是否使用scratch buffer变量

    // 临时变量
    CcuRep::Variable repeatNumVarTemp_;                // 临时重复次数变量
    CcuRep::Variable repeatTimeflag_;                 // 重复时间标志
    std::vector<CcuRep::Variable> inputOffset_;       // 输入偏移量列表
    std::vector<CcuRep::Variable> scratchOffset_;     // Scratch偏移量列表
    CcuRep::Variable curInputOffset_;                 // 当前输入偏移量
    CcuRep::Variable curScratchOffset_;               // 当前Scratch偏移量
    CcuRep::Variable cursliceSize_;                   // 当前切片大小
    CcuRep::LocalAddr srcMem_;                        // 源内存（本地地址）
    CcuRep::LocalAddr dstMem_;                        // 目标内存（本地地址，用于本地复制）
    CcuRep::RemoteAddr dstRemoteMem_;                 // 目标内存（远程地址，用于远程写入）
    CcuRep::CompletedEvent event_;
};

}  // namespace ops_hccl
#endif  // HCCL_CCU_KERNEL_SCATTER_NHR_1D_MEM2MEM_H
