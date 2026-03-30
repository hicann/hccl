/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_all_to_all_v_mesh1d.h"

namespace ops_hccl {
using namespace hcomm;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int CKE_IDX_0    = 0; // pre
constexpr int CKE_IDX_1    = 1; // post
constexpr int CKE_IDX_2    = 2;
constexpr int CONST_ONE    = 1;

CcuKernelAlltoAllVMesh1D::CcuKernelAlltoAllVMesh1D(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAlltoAllVMesh1D *kernelArg
        = dynamic_cast<const CcuKernelArgAlltoAllVMesh1D *>(&arg);
    rankId_ = kernelArg->rankId_;
    rankSize_ = kernelArg->dimSize_;
    channels_ = kernelArg->channels;
    loadFromMem_ = kernelArg->loadFromMem_;
    HCCL_INFO( "[CcuKernelAlltoAllVMesh1D] rankId_ = %d, rankSize_ = %d, loadFromMem_ = %d",rankId_, rankSize_, loadFromMem_);
}

HcclResult CcuKernelAlltoAllVMesh1D::InitResource()
{   
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] InitResource!");
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAlltoAllVMesh1D] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    input_.push_back(CreateVariable());
    HCCL_INFO("rankSize_ = %d, rankId_ = %d", rankSize_, rankId_);
    for (u32 id = 0; id < rankSize_; id++) {
        if (id == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        }
        else { // 非本地，使用远端Variable
            CcuRep::Variable outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    HCCL_INFO("output size: %d, token size: %d", output_.size(), token_.size());

    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        src_.push_back(CreateLocalAddr());
        if (rankIdx == rankId_) {
            myDst_ = CreateLocalAddr();
            dst_.push_back({});
        } else {
            dst_.push_back(CreateRemoteAddr());
        }
    }

    srcOffset_ = CreateVariable();
    dstOffset_ = CreateVariable();
    a2avXnAddr_ = CreateVariable();

    // 前同步。交换信息，将本Rank load的in\out等地址信息写到所有对端的对应Variable中，并同步
    selfBit_ = 1 << rankId_;  // 本rank的mask
    allBit_  = (1 << rankSize_) - 1;  // 等待包含自身的全部对端
    allOtherBit_ = ((1 << rankSize_) - 1) & (~(1 << rankId_)); // 等待其他所有对端

    //  all2allv 数据搬运
    completedRankCount_ = CreateVariable();
    xnMaxTransportSize_ = CreateVariable();
    xnMaxTransportGoSize_ = CreateGroupOpSize();
    xnConst1_ = CreateVariable();

    xnLength_ = CreateVariable();
    xnLength_ = 8; // xn长度为8byte

    event_ = CreateCompletedEvent();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAlltoAllVMesh1D::PreSync()
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] PreSync!");
    CcuRep::Variable tempDst = CreateVariable();
    uint16_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    u32 channelIdx = 0;
    for (u32 id = 0; id < rankSize_; id++) {
        if (id == rankId_) {
            continue;
        }
        tempDst = output_[rankId_];
        tempDst += sendRecvInfo_[id].recvOffset;
        // index = 0，传递output信息
        NotifyRecord(channels_[channelIdx], CKE_IDX_0, OUTPUT_XN_ID, tempDst, 1 << OUTPUT_XN_ID);
        // index = 1，传递token信息
        NotifyRecord(channels_[channelIdx], CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
        channelIdx++;
    }

    for (auto &ch: channels_) {
        NotifyWait(ch, CKE_IDX_0, allBit);
    }
    HCCL_INFO( "[CcuKernelAlltoAllVMesh1D] PreSync end");
}

void CcuKernelAlltoAllVMesh1D::PostSync()
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] PostSync!");
    for (auto &ch : channels_) {
        NotifyRecord(ch, CKE_IDX_1, 1 << CONST_ONE);
    }

    for (auto &ch: channels_) {
        NotifyWait(ch, CKE_IDX_1, 1 << CONST_ONE);
    }
}

void CcuKernelAlltoAllVMesh1D::LoadArgs()
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] LoadArgs!");
    // 从SQE load args，本rank需要的input、output地址等信息
    // inputAddr, outputAddr, tokenInfo, srcStride, dstStride, srcOffset, dstOffset
    Load(input_[0]);
    Load(output_[rankId_]); // load的目的存放寄存器
    Load(token_[rankId_]);
    Load(srcOffset_);
    Load(dstOffset_);
    if (loadFromMem_) {
        Load(a2avXnAddr_);
    } else {
        Load(xnMaxTransportGoSize_);
    }
    // 恢复当前卡对所有卡的收发信息
    sendRecvInfo_.resize(rankSize_);
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        LoadAll2allSendRecvInfo(sendRecvInfo_[rankIdx]);
    }
}

void CcuKernelAlltoAllVMesh1D::CalcGroupSrcDst()
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] CalcGroupSrcDst!");
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        src_[rankIdx].token = token_[rankIdx];

        // src_[rankIdx] = usrInAddr + sendoffset + srcOffset_
        src_[rankIdx].addr = input_[0];
        src_[rankIdx].addr += sendRecvInfo_[rankIdx].sendOffset;
        src_[rankIdx].addr += srcOffset_;

        // dst_[r] = recvBuf[r] + recvOffset + dstOffset_
        if (rankIdx == rankId_) {
            // 写目的端为本端时需要特殊处理：使用接收基地址 + 块地址offset + 已发送数据量
            myDst_.token = token_[rankIdx];
            myDst_.addr = output_[rankIdx];
            myDst_.addr += sendRecvInfo_[rankIdx].recvOffset;
            myDst_.addr += dstOffset_;
        } else {
            // 对端交换的接收块起始地址 + 已接收的数据偏移
            dst_[rankIdx].token = token_[rankIdx];
            dst_[rankIdx].addr = output_[rankIdx];
            dst_[rankIdx].addr += dstOffset_;
        }
    }
}

void CcuKernelAlltoAllVMesh1D::DoAll2AllVMultiLoop()
{
    HCCL_DEBUG("[CcuKernelAlltoAllVMesh1D] alltoallv mesh 1d use GroupCopy start");
    xnMaxTransportSize_ = UB_MAX_TRANS_SIZE;
    completedRankCount_ = 0;
    xnConst1_ = 1;
    u32 channelId = 0;
    CCU_WHILE(completedRankCount_ != rankSize_) {  // 循环发送数据，直到所有对端数据都发送完成
        for(uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {  // 循环发送所有对端数据
            event_.SetMask(1 <<rankIdx);
            if (rankIdx == rankId_) {
                continue;
            }
            CCU_IF(sendRecvInfo_[rankIdx].loopNum == UINT64_MAX) { // 已经完成，直接置位完成信号
                RecordEvent(event_);
            }
            CCU_IF(sendRecvInfo_[rankIdx].loopNum != UINT64_MAX) {  // 还没有完成，则继续循环
                CCU_IF(sendRecvInfo_[rankIdx].loopNum == UINT64_MAX - 1) { // 最后一轮循环, 发送尾块数据
                    CCU_IF(sendRecvInfo_[rankIdx].tailSize == 0) { // 尾块数据量为 0，则不需要发送尾块数据
                        RecordEvent(event_);
                    }
                    CCU_IF(sendRecvInfo_[rankIdx].tailSize != 0) { // 尾块数据量不为 0，则需要发送尾块数据
                        WriteNb(channels_[channelId], dst_[rankIdx], src_[rankIdx], sendRecvInfo_[rankIdx].tailSize,
                              event_);
                    }
                    completedRankCount_ += xnConst1_;  // 之后一轮循环完成，更新已完成的rank数
                }
                CCU_IF(sendRecvInfo_[rankIdx].loopNum != UINT64_MAX - 1) { // 未完成，则继续循环，发送整块数据
                    WriteNb(channels_[channelId], dst_[rankIdx], src_[rankIdx], xnMaxTransportSize_, event_);
                    // 更新偏移
                    src_[rankIdx].addr += xnMaxTransportSize_;
                    dst_[rankIdx].addr += xnMaxTransportSize_;
                }
                sendRecvInfo_[rankIdx].loopNum += xnConst1_;
            }
                channelId++;
        }

        event_.SetMask(1 << rankId_);
        CCU_IF(sendRecvInfo_[rankId_].loopNum == UINT64_MAX) { // 已经完成，直接置位完成信号
                RecordEvent(event_);
        }

        CCU_IF(sendRecvInfo_[rankId_].loopNum != UINT64_MAX) {  // 还没有完成，则继续循环
                CCU_IF(sendRecvInfo_[rankId_].loopNum == UINT64_MAX - 1) { // 最后一轮循环, 发送尾块数据
                    CCU_IF(sendRecvInfo_[rankId_].tailSize == 0) { // 尾块数据量为 0，则不需要发送尾块数据
                        RecordEvent(event_);
                    }
                    CCU_IF(sendRecvInfo_[rankId_].tailSize != 0) { // 尾块数据量不为 0，则需要发送尾块数据
                        if (loadFromMem_) {
                            LocalCopyNb(myDst_, src_[rankId_], sendRecvInfo_[rankId_].tailSize, event_);
                        } else {
                            GroupCopy(myDst_, src_[rankId_], sendRecvInfo_[rankId_].tailGoSize);
                            RecordEvent(event_);
                        }
                    }
                    completedRankCount_ += xnConst1_;  // 之后一轮循环完成，更新已完成的rank数
                }
                CCU_IF(sendRecvInfo_[rankId_].loopNum != UINT64_MAX - 1) { // 未完成，则继续循环，发送整块数据
                    if (loadFromMem_) {
                        LocalCopyNb(myDst_, src_[rankId_], xnMaxTransportSize_, event_);
                    } else {
                        GroupCopy(myDst_, src_[rankId_], xnMaxTransportGoSize_);
                        RecordEvent(event_);
                    }
                    // 更新偏移
                    src_[rankId_].addr += xnMaxTransportSize_;
                    myDst_.addr += xnMaxTransportSize_;
                }
                sendRecvInfo_[rankId_].loopNum += xnConst1_;
        }
        // 等待本轮发送完成
        event_.SetMask(allBit_);
        WaitEvent(event_);
    }
}

HcclResult CcuKernelAlltoAllVMesh1D::Algorithm()
{
    HCCL_INFO("[ccuAllToAllMesh1D_Kernel] AllToAllMesh1D run.");

    CHK_RET(InitResource());

    LoadArgs();

    PreSync();

    // 创建GSA， src为本地的各片HBM地址GSA列表，dst为所有对端的HBM地址GSA列表
    CalcGroupSrcDst();

    DoAll2AllVMultiLoop();

    PostSync();

    HCCL_INFO("[AllToAllAlgo] AllToAllMesh1D end");
    
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAlltoAllVMesh1D::GeneArgs(const CcuTaskArg &arg)
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] GeneArgs!");
    const CcuTaskArgAlltoAllVMesh1D *taskArg
        = dynamic_cast<const CcuTaskArgAlltoAllVMesh1D *>(&arg);
    uint64_t inputAddr  = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo  = taskArg->token_;

    uint64_t srcOffset = taskArg->srcOffset_;
    uint64_t dstOffset = taskArg->dstOffset_;

    HCCL_INFO("[AllToAllVAlgo] inputAddr[%llu], outputAddr[%llu],"
              "srcOffset[%llu], dstOffset[%llu]",
              inputAddr, outputAddr, srcOffset, dstOffset);
    std::vector<uint64_t> processReturn = {inputAddr, outputAddr, tokenInfo, srcOffset, dstOffset};

    if (loadFromMem_) {
        processReturn.push_back(0);  // 空地址占位，保证参数个数与load个数一致
        return processReturn;
    }
    uint64_t xnMaxTransportSize   = UB_MAX_TRANS_SIZE;
    auto     xnMaxTransportGoSize = CalGoSize(xnMaxTransportSize);
    for (auto val : xnMaxTransportGoSize) {
        processReturn.push_back(val);
    }
    uint64_t rankSize = taskArg->sliceSize_.size();
    for (uint64_t i = 0; i < rankSize; i++) {
        uint64_t tailSize = taskArg->localSendRecvInfo_.sendLength[i] % UB_MAX_TRANS_SIZE;
        uint64_t loopNum = UINT64_MAX - 1 - (taskArg->localSendRecvInfo_.sendLength[i] / UB_MAX_TRANS_SIZE);
        uint64_t sendOffset = taskArg->localSendRecvInfo_.sendOffset[i];
        uint64_t recvOffset = taskArg->localSendRecvInfo_.recvOffset[i];
        auto tailGoSize = CalGoSize(tailSize);
        processReturn.push_back(tailSize);
        processReturn.push_back(loopNum);
        processReturn.push_back(sendOffset);
        processReturn.push_back(recvOffset);
        for (auto val : tailGoSize) {
            processReturn.push_back(val);
        }
        HCCL_INFO("[AllToAllVAlgo] rankIdx[i] taskArg->sliceSize[%llu]," \
            "taskArg->localSendRecvInfo.sendOffset[%llu]," \
            "taskArg->localSendRecvInfo.recvOffset[%llu]",
            taskArg->sliceSize_[i], taskArg->localSendRecvInfo_.sendOffset[i],
            taskArg->localSendRecvInfo_.recvOffset[i]);
    }

    return processReturn;
}

void CcuKernelAlltoAllVMesh1D::LoadAll2allSendRecvInfo(A2AsingleSendRecvInfo &sendRecvInfo)
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] LoadAll2allSendRecvInfo!");
    sendRecvInfo.tailSize   = CreateVariable();
    sendRecvInfo.loopNum    = CreateVariable();
    sendRecvInfo.sendOffset = CreateVariable();
    sendRecvInfo.recvOffset = CreateVariable();
    sendRecvInfo.tailGoSize = CreateGroupOpSize();
    if (loadFromMem_) {
        HCCL_INFO("[CcuKernelAlltoAllVMesh1D] Load Args from Mem");
        sendRecvInfo.loopNum = UINT64_MAX - 1; // MC2 场景 loop num 默认为 1

        // 要求client端排列内存为[size,send,recv][size,send,recv]...
        LoadVariable(a2avXnAddr_, sendRecvInfo.tailSize);
        a2avXnAddr_ += xnLength_;

        LoadVariable(a2avXnAddr_, sendRecvInfo.sendOffset);
        a2avXnAddr_ += xnLength_;

        // 跳过recvSize
        a2avXnAddr_ += xnLength_;

        LoadVariable(a2avXnAddr_, sendRecvInfo.recvOffset);
        a2avXnAddr_ += xnLength_;
    } else {
        Load(sendRecvInfo.tailSize);
        Load(sendRecvInfo.loopNum);
        Load(sendRecvInfo.sendOffset);
        Load(sendRecvInfo.recvOffset);
        Load(sendRecvInfo.tailGoSize);
    }
}
}