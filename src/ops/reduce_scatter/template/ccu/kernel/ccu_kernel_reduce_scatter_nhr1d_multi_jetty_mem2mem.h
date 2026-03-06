/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_MUTILJETTY_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_MUTILJETTY_1D_MEM2MEM_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
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

namespace ops_hccl {
class CcuKernelArgReduceScatterNhrMutilJettyMem2Mem1D : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgReduceScatterNhrMutilJettyMem2Mem1D(uint64_t dimSize, uint32_t rankId, uint16_t portNum,
                                const std::vector<NHRStepInfo> stepInfoVector, const std::map<u32, u32> rank2ChannelIdx,
                                const OpParam& opParam, const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          portNum_(portNum),
          stepInfoVector_(stepInfoVector),
          rank2ChannelIdx_(rank2ChannelIdx),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgReduceScatterNhrMutilJettyMem2Mem1D] dimSize: %lu, rankId: %u, portNum: %u, reduceOp: %d, dataType: %d",
                   dimSize_, rankId_, portNum_, opParam_.reduceType, opParam_.DataDes.dataType);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceScatterNhrMutilJettyMem2Mem1D", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t  dimSize_{0};
    uint32_t rankId_{0};
    uint16_t portNum_{0}; // 一个channel使用的端口数
    OpParam opParam_;
    std::vector<NHRStepInfo> stepInfoVector_;   // nhr算法执行过程中的参数
    std::map<u32, u32> rank2ChannelIdx_; // 将rankId和channel对应起来<rankId,channelId>
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

class CcuTaskArgReduceScatterNhrMutilJettyMem2Mem1D : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterNhrMutilJettyMem2Mem1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                uint64_t sliceSize, uint64_t inputSliceStride, uint64_t outputSliceStride,
                                                uint64_t sliceOneJettySize, uint64_t sliceLastJettySize,
                                                uint64_t repeatNum, uint64_t inputRepeatStride,
                                                uint64_t outputRepeatStride)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), sliceSize_(sliceSize),
        inputSliceStride_(inputSliceStride), outputSliceStride_(outputSliceStride),  sliceOneJettySize_(sliceOneJettySize),
        sliceLastJettySize_(sliceLastJettySize), repeatNum_(repeatNum),
        inputRepeatStride_(inputRepeatStride), outputRepeatStride_(outputRepeatStride)
    {
        HCCL_DEBUG("[CcuTaskArgReduceScatterNhrMutilJettyMem2Mem1D] inputAddr: %lu, outputAddr: %lu, token: %lu, "
                    "sliceSize: %lu, inputSliceStride: %lu, outputSliceStride: %lu, sliceOneJettySize_: %lu, "
                    "sliceLastJettySize_: %lu, repeatNum: %lu, "
                    "inputRepeatStride: %lu, outputRepeatStride: %lu",
                   inputAddr_, outputAddr_, token_, sliceSize_, inputSliceStride_, outputSliceStride_,
                   sliceOneJettySize_, sliceLastJettySize_, repeatNum, inputRepeatStride_,
                   outputRepeatStride_);
    }

    uint64_t inputAddr_; // 输入地址
    uint64_t outputAddr_; // 输出地址
    uint64_t token_;
    uint64_t sliceSize_;
    uint64_t inputSliceStride_; 
    uint64_t outputSliceStride_;
    uint64_t sliceOneJettySize_;
    uint64_t sliceLastJettySize_;
    uint64_t repeatNum_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;
};

class CcuKernelReduceScatterNhrMutilJettyMem2Mem1D : public CcuKernelAlgBase {
public:
    CcuKernelReduceScatterNhrMutilJettyMem2Mem1D(const hcomm::CcuKernelArg &arg);
    ~CcuKernelReduceScatterNhrMutilJettyMem2Mem1D() override {}

    HcclResult Algorithm();
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    HcclResult LoadArgs();
    HcclResult PreSync();
    HcclResult PostSync();
    HcclResult DoRepeatReduceScatter();
    HcclResult DoRepeatReduceScatterNHRSingleStep(const NHRStepInfo &nhrStepInfo, const std::vector<CcuRep::Variable> &inputSliceOffset);
    HcclResult DoRepeatSendRecvSlices(const u32 &toRank, CcuRep::LocalAddr &src, CcuRep::RemoteAddr &dst);

    // 构造函数中
    uint64_t dimSize_{0};
    uint32_t rankId_{0};
    uint32_t localSize_{0};  // 本rank所在行或列的总rank数
    uint32_t myRankIdx_{0};
    uint32_t portNum_{0}; // 端口个数
    HcclReduceOp reduceOp_;
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    std::vector<NHRStepInfo> stepInfoVector_;   // nhr算法执行过程中的参数
    std::map<u32, u32> rank2ChannelIdx_;
    hcomm::CcuRep::Variable repeatNum_;
    std::vector<ChannelHandle> channels_;
    
    // load 
    std::vector<hcomm::CcuRep::Variable> input_;
    hcomm::CcuRep::Variable output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable sliceSize_;
    hcomm::CcuRep::Variable inputSliceStride_;
    hcomm::CcuRep::Variable outputSliceStride_;
    hcomm::CcuRep::Variable inputRepeatStride_;
    hcomm::CcuRep::Variable outputRepeatStride_;
    hcomm::CcuRep::Variable sliceOneJettySize_;
    hcomm::CcuRep::Variable sliceLastJettySize_;
    hcomm::CcuRep::Variable repeatNumVar_;

    std::vector<hcomm::CcuRep::CompletedEvent> jettyEvent_;
    hcomm::CcuRep::CompletedEvent event_;
    hcomm::CcuRep::Variable repeatNumVarTemp_;
    CcuRep::LocalAddr   localSrc_;
    CcuRep::LocalAddr   localDst_;
    CcuRep::RemoteAddr  remoteDst_;
    CcuRep::Variable flag_; // 用于判断是否是第一次循环
};

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_REDUCE_SCATTER_NHR_1D_MUTIL_JETTY_MEM2MEM_H
