/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_REDUCE_NHR_1D_MEM2MEM_MULTI_JETTY_H
#define HCCL_CCU_KERNEL_ALL_REDUCE_NHR_1D_MEM2MEM_MULTI_JETTY_H

#include <vector>
#include <ios>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

enum class XnId {
    OUTPUT = 0,
    TOKEN
};

enum class SignalBit {
    PRE_SYNC_OUTPUT = 0,
    PRE_SYNC_TOKEN,
    READY_TO_RECV_RS,
    READY_TO_RECV_AG,
    SEND_DONE_RS,
    SEND_DONE_AG,
    POST_SYNC
};

using NHRStepInfo = struct NHRStepInfoDef {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    NHRStepInfoDef() : nSlices(0)
    {
    }
};

namespace ops_hccl {

class CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty(uint32_t rankSize, uint32_t rankId, uint32_t portSize,
                                                         const OpParam &opParam,
                                                         const std::vector<NHRStepInfo> &algStepInfoList,
                                                         const std::map<u32, u32> &channelIdxMap,
                                                         const std::vector<std::vector<uint32_t>> &subCommRanks)
        : rankSize_(rankSize),
          rankId_(rankId),
          portNum_(portSize),
          opParam_(opParam),
          algStepInfoList_(algStepInfoList),
          channelIdxMap_(channelIdxMap),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty] rankSize[%u], rankId[%u], portSize[%u]", rankSize_,
                   rankId_, portNum_);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty", opParam_, subCommRanks_);
        return signature;
    }
    uint32_t rankSize_{0};
    uint32_t rankId_{0};
    uint32_t portNum_{0};  // 端口数量
    OpParam opParam_;
    std::vector<NHRStepInfo> algStepInfoList_;
    std::map<u32, u32> channelIdxMap_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

class CcuTaskArgAllReduceNhrMem2Mem1DMultiJetty : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllReduceNhrMem2Mem1DMultiJetty(uint64_t inputAddr, uint64_t outputAddr, uint64_t outputToken,
                                                       uint64_t isInplace, uint64_t dataSizePerRank,
                                                       uint64_t dataSizePerPort, uint64_t lastRankSliceSize,
                                                       uint64_t lastPortSliceSize)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          outputToken_(outputToken),
          isInplace_(isInplace),
          dataSizePerRank_(dataSizePerRank),
          dataSizePerPort_(dataSizePerPort),
          lastRankSliceSize_(lastRankSliceSize),
          lastPortSliceSize_(lastPortSliceSize)
    {
        HCCL_DEBUG("[CcuTaskArgAllReduceNhrMem2Mem1DMultiJetty] inputAddr: %llu, outputAddr: %llu, outputToken: %llu, "
                   "isInplace: %llu, dataSizePerRank: %llu, dataSizePerPort: %llu, lastRankSliceSize: "
                   "%llu, lastPortSliceSize: %llu",
                   inputAddr_, outputAddr_, outputToken_, isInplace_, dataSizePerRank_, dataSizePerPort_,
                   lastRankSliceSize_, lastPortSliceSize_);
    }

    uint64_t inputAddr_;    // input地址
    uint64_t outputAddr_;   // output地址
    uint64_t outputToken_;  // output地址的token，用于授权对端写入output
    uint64_t isInplace_;  // 是否为原地操作，即input地址 == output地址
    uint64_t dataSizePerRank_;    // dataSizePerPort_ * portSize，用于计算发送数据的偏移地址
    uint64_t dataSizePerPort_;    // 将整块数据平分成（rankSize * portSize）块，向上取整
    uint64_t lastRankSliceSize_;  // 按rankSize平分后最后一块的数据大小，小于等于dataSizePerRank_
    uint64_t lastPortSliceSize_;  // 按rankSize * portSize平分后最后一块的数据大小，小于等于dataSizePerPort_
};

class CcuKernelAllReduceNhr1DMem2MemMultiJetty : public CcuKernelAlgBase {
public:
    CcuKernelAllReduceNhr1DMem2MemMultiJetty(const CcuKernelArg &arg);
    ~CcuKernelAllReduceNhr1DMem2MemMultiJetty() override {}

    HcclResult Algorithm();
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;

private:
    HcclResult InitResource();                                  // 申请Variables
    HcclResult LoadArgs();                                            // load task参数
    HcclResult PreSync();                                             // 前同步+地址信息交换
    HcclResult LocalWaitAllEvent(const uint16_t mask);

    HcclResult LocalCopySlices();  // 本地拷贝
    std::vector<u32> GetNonTxSliceIdxs(
        const std::vector<u32> &txSliceIdxs) const;  // 计算出首轮本地拷贝可以跳过的slice索引
    HcclResult DoLocalCopySlice(hcomm::CcuRep::LocalAddr &src, hcomm::CcuRep::LocalAddr &dst, const u32 &copySliceIdx,
                          hcomm::CcuRep::CompletedEvent &event);

    HcclResult DoReduceScatterNHR();
    HcclResult DoReduceScatterNHRSingleStep(const NHRStepInfo &nhrStepInfo);
    HcclResult DoWriteReduceSlice(const u32 toRank, hcomm::CcuRep::LocalAddr &src, hcomm::CcuRep::RemoteAddr &dst,
                            const u32 sendSliceIdx, const u32 signalIndex);

    HcclResult DoAllGatherNHR();
    HcclResult DoAllGatherNHRSingleStep(const NHRStepInfo &nhrStepInfo);
    HcclResult DoSendRecvSlice(const u32 toRank, hcomm::CcuRep::LocalAddr &src, hcomm::CcuRep::RemoteAddr &dst,
                         const u32 &sendSliceIdx, u32 signalIndex);

    uint32_t rankSize_{0};
    uint32_t rankId_{0};
    uint32_t portNum_{0};  // 端口数
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;
    std::vector<ChannelHandle> channels_;       // 对端的channel，按rank号排序，不包含本rank
    std::vector<NHRStepInfo> algStepInfoList_;  // 描述NHR算法步骤信息
    std::map<u32, u32> channelIdxMap_;

    hcomm::CcuRep::Variable inputAddr_;
    std::vector<hcomm::CcuRep::Variable> outputAddrs_;
    std::vector<hcomm::CcuRep::Variable> outputTokens_;
    hcomm::CcuRep::Variable isInplace_;
    hcomm::CcuRep::Variable dataSize_;
    hcomm::CcuRep::Variable dataSizePerRank_;
    hcomm::CcuRep::Variable dataSizePerPort_;
    hcomm::CcuRep::Variable lastRankSliceSize_;
    hcomm::CcuRep::Variable lastPortSliceSize_;
    std::vector<hcomm::CcuRep::Variable> sliceOffset_;
    GroupOpSize localCopyGoSize_;
    GroupOpSize localCopyGoSizeLastSlice_;

    hcomm::CcuRep::LocalAddr localInput_;     // 临时本地输入
    hcomm::CcuRep::LocalAddr localOutput_;    // 临时本地输出
    hcomm::CcuRep::RemoteAddr remoteOutput_;  // 临时远端输出

    std::vector<hcomm::CcuRep::CompletedEvent> events_;
};

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_MEM2MEM_H
