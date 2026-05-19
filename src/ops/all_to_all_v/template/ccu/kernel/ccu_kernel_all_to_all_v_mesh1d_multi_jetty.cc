/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_base.h"
#include "ccu_kernel_all_to_all_v_mesh1d_multi_jetty.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int POST_SYNC_ID  = 3;
constexpr int CKE_IDX_0    = 0;

CcuKernelAllToAllVMesh1DMultiJetty::CcuKernelAllToAllVMesh1DMultiJetty(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllToAllVMesh1DMultiJetty *kernelArg
        = dynamic_cast<const CcuKernelArgAllToAllVMesh1DMultiJetty *>(&arg);
    rankId_         = kernelArg->rankId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;
    jettyNums_      = kernelArg->jettyNums_;
    
    HCCL_INFO(
        "[CcuKernelAllToAllVMesh1DMultiJetty] Init, KernelArgs are rankId[%u], rankSize_[%u]",
        rankId_, rankSize_);
}

HcclResult CcuKernelAllToAllVMesh1DMultiJetty::PreSync()
{
    CcuRep::Variable tempDst = CreateVariable();
    u32 channelIdx = 0;
    for (u32 id = 0; id < rankSize_; id++) {
        if (id == rankId_) {
            continue;
        }
        tempDst = output_[rankId_];
        tempDst += sendRecvInfo_[id].recvOffset;
        // index = 0，传递output信息
        CHK_RET(NotifyRecord(channels_[channelIdx], CKE_IDX_0, OUTPUT_XN_ID, tempDst, 1 << OUTPUT_XN_ID));
        // index = 1，传递token信息
        CHK_RET(NotifyRecord(channels_[channelIdx], CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID));
        channelIdx++;
    }

    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle ch : channels_) {
        CHK_RET(NotifyWait(ch, CKE_IDX_0, allBit));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllToAllVMesh1DMultiJetty::PostSync()
{
    for (ChannelHandle ch : channels_) {
        CHK_RET(NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (ChannelHandle ch : channels_) {
        CHK_RET(NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllToAllVMesh1DMultiJetty::InitResource()
{
    u32 channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAllToAllVMesh1DMultiJetty] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    input_.push_back(CreateVariable());
    output_.reserve(rankSize_);
    token_.reserve(rankSize_);
    for (u32 id = 0; id < rankSize_; id++) {
        if (id == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        }
        else { // 非本地，使用远端Variable
            HCCL_DEBUG("[CcuKernelAllToAllVMesh1DMultiJetty] MyRank[%u], PeerId[%u], ChannelId[%u]",
                rankId_, id, channelIdx);
            CcuRep::Variable outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar); // 获取channel中id=0的Var来传递output
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }

    src_.reserve(rankSize_);
    remoteDst_.reserve(rankSize_);
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        src_.push_back(CreateLocalAddr());
        if (rankIdx == rankId_) {
            myDst_ = CreateLocalAddr();
            remoteDst_.push_back({});
        } else {
            remoteDst_.push_back(CreateRemoteAddr());
        }
    }

    srcOffset_ = CreateVariable();
    dstOffset_ = CreateVariable();

    //  all2allv 数据搬运
    completedRankCount_ = CreateVariable();
    xnMaxTransportSize_ = CreateVariable();
    xnMaxTransportGoSize_ = CreateGroupOpSize();
    xnConst1_ = CreateVariable();

    eventList_.reserve(rankSize_);
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        eventList_.push_back(CreateCompletedEvent());
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllToAllVMesh1DMultiJetty::LoadArgs()
{
    // 从SQE load args，本rank需要的input、output地址等信息
    // inputAddr, outputAddr, tokenInfo, srcOffset, dstOffset
    Load(input_[0]);
    Load(output_[rankId_]); // load的目的存放寄存器
    Load(token_[rankId_]);
    Load(srcOffset_);
    Load(dstOffset_);
    Load(xnMaxTransportGoSize_);
    
    // 恢复当前卡对所有卡的收发信息
    sendRecvInfo_.resize(rankSize_);
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        CHK_RET(LoadAll2allSendRecvInfo(sendRecvInfo_[rankIdx]));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllToAllVMesh1DMultiJetty::CalcGroupSrcDst()
{
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        src_[rankIdx].token = token_[rankIdx];

        // src_[rankIdx] = usrInAddr + sendoffset + srcOffset_
        src_[rankIdx].addr = input_[0];
        src_[rankIdx].addr += sendRecvInfo_[rankIdx].sendOffset;
        src_[rankIdx].addr += srcOffset_;

        if (rankIdx == rankId_) {
            myDst_.token = token_[rankIdx];

            // dst_[r] = recvBuf[r] + recvOffset + dstOffset_
            myDst_.addr = output_[rankIdx];
            myDst_.addr += sendRecvInfo_[rankIdx].recvOffset;
            myDst_.addr += dstOffset_;
        } else {
            remoteDst_[rankIdx].token = token_[rankIdx];

            // 对端交换的接收块起始地址 + 已接收的数据偏移
            remoteDst_[rankIdx].addr = output_[rankIdx];
            remoteDst_[rankIdx].addr += dstOffset_;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllToAllVMesh1DMultiJetty::DoAll2AllVMultiLoop()
{
    HCCL_DEBUG("[CcuKernelAllToAllVMesh1DMultiJetty] alltoallv mesh 1d use GroupCopy start");
    xnMaxTransportSize_ = UB_MAX_TRANS_SIZE;
    completedRankCount_ = 0;
    xnConst1_ = 1;
    u32 channelIdx = 0;
    CCU_WHILE(completedRankCount_ != rankSize_) {  // 循环发送数据，直到所有对端数据都发送完成
        for(uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {  // 循环发送所有对端数据
            if (rankIdx == rankId_) {
                continue;
            }
            CCU_IF(sendRecvInfo_[rankIdx].loopNum == UINT64_MAX) { // 已经完成，直接置位完成信号
                eventList_[rankIdx].SetMask((1 << jettyNums_[rankIdx]) - 1);
                CHK_RET(RecordEvent(eventList_[rankIdx]));
            }
            CCU_IF(sendRecvInfo_[rankIdx].loopNum != UINT64_MAX) {
                CCU_IF(sendRecvInfo_[rankIdx].loopNum != UINT64_MAX - jettyNums_[rankIdx]) {
                    CHK_RET(DoAll2AllVBlock(rankIdx, channelIdx));  // 处理非最后一轮数据块
                }
                CCU_IF(sendRecvInfo_[rankIdx].loopNum == UINT64_MAX - jettyNums_[rankIdx]) {
                    CHK_RET(DoAll2AllVLastBlock(rankIdx, channelIdx));  // 处理最后一轮数据块
                }
            }
            channelIdx++;
        }
        CCU_IF(sendRecvInfo_[rankId_].loopNum == UINT64_MAX) { // 已经完成，直接置位完成信号
            eventList_[rankId_].SetMask((1 << jettyNums_[rankId_]) - 1);
            CHK_RET(RecordEvent(eventList_[rankId_]));
        }

        CCU_IF(sendRecvInfo_[rankId_].loopNum != UINT64_MAX) {  // 还没有完成，则继续循环
            CCU_IF(sendRecvInfo_[rankId_].loopNum == UINT64_MAX - jettyNums_[rankId_]) { // 本地ccu ms拷贝，jetty数量为1
                CCU_IF(sendRecvInfo_[rankId_].lastTailSliceSize == 0) { // 尾块数据量为 0，则不需要发送尾块数据
                    eventList_[rankId_].SetMask((1 << jettyNums_[rankId_]) - 1);
                    CHK_RET(RecordEvent(eventList_[rankId_]));
                }
                CCU_IF(sendRecvInfo_[rankId_].lastTailSliceSize != 0) { // 尾块数据量不为 0，则需要发送尾块数据
                    CHK_RET(GroupCopy(myDst_, src_[rankId_], sendRecvInfo_[rankId_].tailGoSize));
                    eventList_[rankId_].SetMask((1 << jettyNums_[rankId_]) - 1);
                    CHK_RET(RecordEvent(eventList_[rankId_]));
                }
                completedRankCount_ += xnConst1_;  // 之后一轮循环完成，更新已完成的rank数
            }
            CCU_IF(sendRecvInfo_[rankId_].loopNum != UINT64_MAX - jettyNums_[rankId_]) {
                CCU_IF(sendRecvInfo_[rankId_].tailSliceSize == 0) { // 当前数据块为 0，则不需要发送数据
                    eventList_[rankId_].SetMask((1 << jettyNums_[rankId_]) - 1);
                    CHK_RET(RecordEvent(eventList_[rankId_]));
                }
                CCU_IF(sendRecvInfo_[rankId_].tailSliceSize != 0) { // 当前数据块不为 0，需要发送数据
                    CHK_RET(GroupCopy(myDst_, src_[rankId_], xnMaxTransportGoSize_));
                    eventList_[rankId_].SetMask((1 << jettyNums_[rankId_]) - 1);
                    CHK_RET(RecordEvent(eventList_[rankId_]));
                    // 更新偏移
                    src_[rankId_].addr += xnMaxTransportSize_;
                    myDst_.addr += xnMaxTransportSize_;
                }
            }
            sendRecvInfo_[rankId_].loopNum += xnConst1_;
        }
        // 等待本轮发送完成
        for(uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
            eventList_[rankIdx].SetMask((1 << jettyNums_[rankIdx]) - 1);
            CHK_RET(WaitEvent(eventList_[rankIdx]));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllToAllVMesh1DMultiJetty::DoAll2AllVLastBlock(u32 rankIdx, u32 channelIdx)
{
    for(uint32_t i = 0; i < jettyNums_[rankIdx]; i++) {
        CCU_IF(sendRecvInfo_[rankIdx].loopNum != UINT64_MAX - 1) { // 处理最后一轮数据块的切片数据（非尾片）
            CCU_IF(sendRecvInfo_[rankIdx].lastSliceSize == 0) { // 切片数据量为 0，则不需要发送切片数据
                eventList_[rankIdx].SetMask(1 << i);
                CHK_RET(RecordEvent(eventList_[rankIdx]));
            }
            CCU_IF(sendRecvInfo_[rankIdx].lastSliceSize != 0) { // 切片数据量不为 0，则需要发送切片数据
                eventList_[rankIdx].SetMask(1 << i);
                CHK_RET(WriteNb(channels_[channelIdx], remoteDst_[rankIdx], src_[rankIdx], sendRecvInfo_[rankIdx].lastSliceSize,
                    eventList_[rankIdx]));
                // 更新偏移
                src_[rankIdx].addr += sendRecvInfo_[rankIdx].lastSliceSize;
                remoteDst_[rankIdx].addr += sendRecvInfo_[rankIdx].lastSliceSize;
            }
        }
        CCU_IF(sendRecvInfo_[rankIdx].loopNum == UINT64_MAX - 1) { // 处理最后一轮数据块的尾片数据
            CCU_IF(sendRecvInfo_[rankIdx].lastTailSliceSize == 0) { // 尾片数据量为 0，则不需要发送尾片数据
                eventList_[rankIdx].SetMask(1 << i);
                CHK_RET(RecordEvent(eventList_[rankIdx]));
            }
            CCU_IF(sendRecvInfo_[rankIdx].lastTailSliceSize != 0) { // 尾片数据量不为 0，则需要发送尾片数据
                eventList_[rankIdx].SetMask(1 << i);
                CHK_RET(WriteNb(channels_[channelIdx], remoteDst_[rankIdx], src_[rankIdx], sendRecvInfo_[rankIdx].lastTailSliceSize,
                    eventList_[rankIdx]));
            }
            completedRankCount_ += xnConst1_;  // 最后一轮数据块的尾片处理完成，更新已完成的rank数
        }
        sendRecvInfo_[rankIdx].loopNum += xnConst1_; // 处理完一片数据，更新loopNum
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllToAllVMesh1DMultiJetty::DoAll2AllVBlock(u32 rankIdx, u32 channelIdx)
{
    for(uint32_t i = 0; i < jettyNums_[rankIdx]; i++) {
        if (i != jettyNums_[rankIdx] - 1) {
            CCU_IF(sendRecvInfo_[rankIdx].sliceSize == 0) { // 切片数据量为 0，则不需要发送切片数据
                eventList_[rankIdx].SetMask(1 << i);
                CHK_RET(RecordEvent(eventList_[rankIdx]));
            }
            CCU_IF(sendRecvInfo_[rankIdx].sliceSize != 0) { // 切片数据量不为 0，则需要发送切片数据
                eventList_[rankIdx].SetMask(1 << i);
                CHK_RET(WriteNb(channels_[channelIdx], remoteDst_[rankIdx], src_[rankIdx], sendRecvInfo_[rankIdx].sliceSize,
                    eventList_[rankIdx]));
                // 更新偏移
                src_[rankIdx].addr += sendRecvInfo_[rankIdx].sliceSize;
                remoteDst_[rankIdx].addr += sendRecvInfo_[rankIdx].sliceSize;
            }
        } else {
            CCU_IF(sendRecvInfo_[rankIdx].tailSliceSize == 0) { // 尾片数据量为 0，则不需要发送尾片数据
                eventList_[rankIdx].SetMask(1 << i);
                CHK_RET(RecordEvent(eventList_[rankIdx]));
            }
            CCU_IF(sendRecvInfo_[rankIdx].tailSliceSize != 0) { // 尾片数据量不为 0，则需要发送尾片数据
                eventList_[rankIdx].SetMask(1 << i);
                CHK_RET(WriteNb(channels_[channelIdx], remoteDst_[rankIdx], src_[rankIdx], sendRecvInfo_[rankIdx].tailSliceSize,
                    eventList_[rankIdx]));
                // 更新偏移
                src_[rankIdx].addr += sendRecvInfo_[rankIdx].tailSliceSize;
                remoteDst_[rankIdx].addr += sendRecvInfo_[rankIdx].tailSliceSize;
            }
        }
        sendRecvInfo_[rankIdx].loopNum += xnConst1_; // 处理完一片数据，更新loopNum
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllToAllVMesh1DMultiJetty::Algorithm()
{
    HCCL_INFO("[AllToAllVAlgo] AllToAllVMesh1DMultiJetty run");
    CHK_RET(InitResource());
    CHK_RET(LoadArgs());
    CHK_RET(PreSync());
    // 创建GSA， src为本地的各片HBM地址GSA列表，dst为所有对端的HBM地址GSA列表
    CHK_RET(CalcGroupSrcDst());
    CHK_RET(DoAll2AllVMultiLoop());
    //  后同步
    CHK_RET(PostSync());
    HCCL_INFO("[AllToAllVAlgo] AllToAllVMesh1DMultiJetty end");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllToAllVMesh1DMultiJetty::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllToAllVMesh1DMultiJetty *taskArg = dynamic_cast<const CcuTaskArgAllToAllVMesh1DMultiJetty *>(&arg);

    uint64_t inputAddr  = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo  = taskArg->token_;
    uint64_t srcOffset = taskArg->srcOffset_;
    uint64_t dstOffset = taskArg->dstOffset_;

    HCCL_INFO("[AllToAllVAlgo] inputAddr[%llu], outputAddr[%llu],"
              "srcOffset[%llu], dstOffset[%llu]",
              inputAddr, outputAddr, srcOffset, dstOffset);
    std::vector<uint64_t> processReturn = {inputAddr, outputAddr, tokenInfo, srcOffset, dstOffset};

    uint64_t xnMaxTransportSize   = UB_MAX_TRANS_SIZE;
    auto     xnMaxTransportGoSize = CalGoSize(xnMaxTransportSize); // 用于本地ccu ms拷贝
    for (auto val : xnMaxTransportGoSize) {
        processReturn.push_back(val);
    }

    uint64_t rankSize = jettyNums_.size();
    for (uint64_t i = 0; i < rankSize; i++) {
        uint64_t blockSize = UB_MAX_TRANS_SIZE;
        uint64_t loopNum = UINT64_MAX - (taskArg->localSendRecvInfo_.sendLength[i] / blockSize + 1) * jettyNums_[i];
        uint64_t sliceSize = 0;
        uint64_t tailSliceSize = 0;
        if(taskArg->localSendRecvInfo_.sendLength[i] >= blockSize) {
            // 切分blockSize大小的数据块
            sliceSize = blockSize / jettyNums_[i] / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
            tailSliceSize = blockSize - sliceSize * (jettyNums_[i] - 1);
            if (jettyNums_[i] == 1) {
                sliceSize = 0;
                tailSliceSize = blockSize;
            }
        }
        // 切分最后一块数据块
        uint64_t lastBlockSize = taskArg->localSendRecvInfo_.sendLength[i] % blockSize;
        uint64_t lastSliceSize = lastBlockSize / jettyNums_[i] / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        uint64_t lastTailSliceSize = lastBlockSize - lastSliceSize * (jettyNums_[i] - 1);
        if (jettyNums_[i] == 1) {
            lastSliceSize = 0;
            lastTailSliceSize = lastBlockSize;
        }

        uint64_t sendOffset = taskArg->localSendRecvInfo_.sendOffset[i];
        uint64_t recvOffset = taskArg->localSendRecvInfo_.recvOffset[i];
        processReturn.push_back(sliceSize);
        processReturn.push_back(tailSliceSize);
        processReturn.push_back(lastSliceSize);
        processReturn.push_back(lastTailSliceSize);
        processReturn.push_back(loopNum);
        processReturn.push_back(sendOffset);
        processReturn.push_back(recvOffset);

        auto tailGoSize = CalGoSize(lastBlockSize); // 用于ccu ms搬运本地数据尾块，jetty数为1
        for (auto val : tailGoSize) {
            processReturn.push_back(val);
        }
        HCCL_INFO("[AllToAllVAlgo] rankIdx[%llu] " \
            "taskArg->localSendRecvInfo.sendLength[%llu]," \
            "taskArg->localSendRecvInfo.sendOffset[%llu]," \
            "taskArg->localSendRecvInfo.recvLength[%llu]," \
            "taskArg->localSendRecvInfo.recvOffset[%llu]", i,
            taskArg->localSendRecvInfo_.sendLength[i],
            taskArg->localSendRecvInfo_.sendOffset[i],
            taskArg->localSendRecvInfo_.recvLength[i],
            taskArg->localSendRecvInfo_.recvOffset[i]);
    }

    return processReturn;
}

HcclResult CcuKernelAllToAllVMesh1DMultiJetty::LoadAll2allSendRecvInfo(A2AsingleSendRecvInfo &sendRecvInfo)
{
    sendRecvInfo.sliceSize          = CreateVariable();
    sendRecvInfo.tailSliceSize      = CreateVariable();
    sendRecvInfo.lastSliceSize      = CreateVariable();
    sendRecvInfo.lastTailSliceSize  = CreateVariable();
    sendRecvInfo.loopNum            = CreateVariable();
    sendRecvInfo.sendOffset         = CreateVariable();
    sendRecvInfo.recvOffset         = CreateVariable();
    sendRecvInfo.tailGoSize         = CreateGroupOpSize();

    Load(sendRecvInfo.sliceSize);
    Load(sendRecvInfo.tailSliceSize);
    Load(sendRecvInfo.lastSliceSize);
    Load(sendRecvInfo.lastTailSliceSize);
    Load(sendRecvInfo.loopNum);
    Load(sendRecvInfo.sendOffset);
    Load(sendRecvInfo.recvOffset);
    Load(sendRecvInfo.tailGoSize);
    return HcclResult::HCCL_SUCCESS;
}
}