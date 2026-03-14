/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_mesh1d_mem2mem.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID    = 0;
constexpr int OUTPUT_XN_ID   = 1;
constexpr int TOKEN_XN_ID    = 2;
constexpr int POST_SYNC_ID   = 3;
constexpr int CKE_IDX_0      = 0;
constexpr uint64_t CCU_MS_SIZE   = 4096;
constexpr uint64_t LOCAL_COPY_MS = 8;

CcuKernelReduceMesh1DMem2Mem::CcuKernelReduceMesh1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{   
    const CcuKernelArgReduceMesh1DMem2Mem *kernelArg
        = dynamic_cast<const CcuKernelArgReduceMesh1DMem2Mem *>(&arg);
    rankId_         = kernelArg->rankId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelReduceMesh1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_       = kernelArg->opParam_.reduceType;
    rootId_         = kernelArg->rootId_;
    HCCL_INFO(
        "[CcuKernelReduceMesh1DMem2Mem] Init, KernelArgs are rankId[%u], rootId[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d]",
        rankId_, rootId_, rankSize_, dataType_, outputDataType_, reduceOp_);
}

void CcuKernelReduceMesh1DMem2Mem::CreateLocalCopyLoop()
{
    std::string loopType = "reduce";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }

    for (uint32_t index = 0; index < 2; index++) { // 需要2个Loop
        CcuRep::LocalAddr src = CreateLocalAddr();
        CcuRep::LocalAddr dst = CreateLocalAddr();
        CcuRep::Variable  len = CreateVariable();
        CcuRep::LoopBlock lb(this, loopType + "_localcopy_loop_" + std::to_string(index));
        lb(src, dst, len);
        CcuRep::CompletedEvent event = moRes.completedEvent[index];
        std::vector<CcuRep::CcuBuf> bufs;
        for (uint32_t i = 0; i < LOCAL_COPY_MS; i++) {
            bufs.push_back(moRes.ccuBuf[i]);
        }

        LocalCopyNb(bufs[0], src, len, event);
        WaitEvent(event);
        LocalCopyNb(dst, bufs[0], len, event);
        WaitEvent(event);
    }
    registeredLoop.insert(loopType);
    return;
}

void CcuKernelReduceMesh1DMem2Mem::LocalCopyByLoopGroup(CcuRep::LocalAddr dst, CcuRep::LocalAddr src)
{
    CreateLocalCopyLoop();

    CCU_IF(localGoSize_.addrOffset != 0)
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam                  = GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += localGoSize_.loopParam;

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize                  = moConfig.memSlice;
        auto lc                    = Loop("reduce_localcopy_loop_0")(src, dst, sliceSize);

        CcuRep::Variable paraCfg   = CreateVariable();
        paraCfg                    = GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg                  = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }

    CCU_IF(localGoSize_.parallelParam != 0)
    {
        CcuRep::Condition cond(this, localGoSize_.parallelParam != 0);

        src.addr += localGoSize_.addrOffset;
        dst.addr += localGoSize_.addrOffset;
        auto lc0 = Loop("reduce_localcopy_loop_0")(src, dst, localGoSize_.residual);

        src.addr += localGoSize_.residual;
        dst.addr += localGoSize_.residual;
        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize                  = moConfig.memSlice;
        auto lc1                   = Loop("reduce_localcopy_loop_1")(src, dst, sliceSize);

        CcuRep::Variable loopCfg0  = CreateVariable();
        loopCfg0                   = GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1  = CreateVariable();
        loopCfg1                   = GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg                  = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, localGoSize_.parallelParam, offsetCfg);
    }
}

HcclResult CcuKernelReduceMesh1DMem2Mem::InitResource()
{
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelReduceMesh1DMem2Mem] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem]channels.size: [%u]", channels_.size());
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelReduceMesh1DMem2Mem] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
            CcuRep::Variable inputVar, outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    for (uint16_t roundId = 0; roundId < (rankSize_ - 1); roundId++) {
        chunkSize_.push_back(CreateVariable());
    }
    inputRepeatStride_            = CreateVariable();
    outputRepeatStride_           = CreateVariable();
    normalSliceSize_              = CreateVariable();
    lastSliceSize_                = CreateVariable();
    repeatNumVar_                 = CreateVariable();
    flag_                         = CreateVariable();
    isInputOutputEqual_           = CreateVariable();
    myInputAddr_                  = CreateLocalAddr();
    remoteInputAddr_              = CreateRemoteAddr();
    dstAddr_                      = CreateLocalAddr();
    event_                        = CreateCompletedEvent();
    localGoSize_                  = CreateGroupOpSize();
    chunkOffset_                  = CreateVariable();

    moConfig.loopCount    = 8;                           // loop展开8次、16次
    moConfig.msInterleave = LOCAL_COPY_MS;               // 一个loop 8个MS
    moConfig.memSlice     = LOCAL_COPY_MS * CCU_MS_SIZE; // 32k
    if (moRes.executor.size() == 0) {
        moRes.executor   = CreateBlockExecutor(moConfig.loopCount);
        moRes.completedEvent = CreateBlockCompletedEvent(moConfig.loopCount);
        moRes.ccuBuf  = CreateBlockCcuBuf(moConfig.loopCount * moConfig.msInterleave);
    }

    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelReduceMesh1DMem2Mem::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(isInputOutputEqual_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(repeatNumVar_);
    for (uint16_t i = 0; i < (rankSize_ - 1); i++) {
        Load(chunkSize_[i]);
    }
    Load(localGoSize_);
    return;
}

void CcuKernelReduceMesh1DMem2Mem::PreSync()
{
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMeshMem2Mem1D LocalPost begin");
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID); // index = 1，传递input信息
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    uint16_t allBit = 1 << INPUT_XN_ID | 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit); // index = 1，传递input信息
    }
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMeshMem2Mem1D wait all end");
}

void CcuKernelReduceMesh1DMem2Mem::PostSync()
{
    for (auto &ch : channels_) {
        NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto &ch : channels_) {
        NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMesh1D Reduce groupwait end");
}

void CcuKernelReduceMesh1DMem2Mem::DoRepeatReduce(const std::vector<CcuRep::Variable> &srcAddr,
                                                  const CcuRep::Variable &dstAddr)
{
    // dstAddr = output_[rankId_]
    dstAddr_.addr  = dstAddr;
    dstAddr_.token = token_[rankId_];

    // srcAddr = input_,
    myInputAddr_.addr  = srcAddr[rankId_];
    myInputAddr_.token = token_[rankId_];

    CCU_IF (flag_ != 0) {
        // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
        dstAddr_.addr += outputRepeatStride_;
        myInputAddr_.addr += inputRepeatStride_;
    }
    // 若root节点的输入输出地址不一致，则本地拷贝
    CCU_IF (isInputOutputEqual_ == 0) {
        LocalCopyByLoopGroup(dstAddr_, myInputAddr_);
    }

    for (uint16_t i = 0; i < (rankSize_ - 1); i++) { // 外层循环控制step
        // 读不同rank的不同chunk
        for (uint16_t rmtId = 0; rmtId < rankSize_; ++rmtId) {
            if (rmtId == rootId_) {
                continue;
            }
            chunkOffset_  = 0;
            dstAddr_.addr  = dstAddr;
            remoteInputAddr_.addr  = srcAddr[rmtId];
            remoteInputAddr_.token = token_[rmtId];

            CCU_IF (flag_ != 0) {
                // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
                dstAddr_.addr += outputRepeatStride_;
                remoteInputAddr_.addr += inputRepeatStride_;
            }
            uint16_t chkId = 0;
            if (rmtId < rankId_) {
                chkId = (i + rmtId) % (rankSize_ - 1);
            } else {
                chkId = (i + rmtId - 1) % (rankSize_ - 1);
            }
            // 获取链接Id，跳过本端
            uint16_t channelId = rmtId < rootId_ ? rmtId : rmtId - 1;
            HCCL_DEBUG(
                "[ReadReduceRmtToLoc] debug rankId[%llu], root[%llu] chkId[%llu], rmtId[%llu] channelId[%llu]",
                rankId_, rootId_, chkId, rmtId, channelId);

            // 计算一下offset 0~(chikd-1)
            for (uint16_t j = 0; j < chkId; ++j) {
                chunkOffset_ += chunkSize_[j];
            }
            // 更新对应的addr
            remoteInputAddr_.addr += chunkOffset_;
            dstAddr_.addr += chunkOffset_;

            CCU_IF(chunkSize_[chkId] == 0)
            {
                event_.SetMask(1 << rmtId);
                RecordEvent(event_);
            }

            CCU_IF(chunkSize_[chkId] != 0)
            {
                // 读远端内存并Reduce, 将远端内存中的数据，和本端内存中的数据进行Reduce操作，结果保存在本端内存中
                event_.SetMask(1 << rmtId);
                ReadReduceNb(channels_[channelId], dstAddr_, remoteInputAddr_, chunkSize_[chkId], dataType_, reduceOp_, event_);
            }
        }
        uint16_t allBit = ((1 << rankSize_) - 1) & (~(1 << rankId_));
        event_.SetMask(allBit);
        WaitEvent(event_);
    }
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMeshMem2Mem1D ReadReduce end");
}

HcclResult CcuKernelReduceMesh1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMesh1DMem2Mem run");

    CHK_RET(InitResource());

    LoadArgs();

    PreSync();

    CCU_IF(normalSliceSize_ != 0)
    {
        if (rankId_ == rootId_) {
            CcuRep::Variable repeatNumAdd = CreateVariable();
            repeatNumAdd  = 1;
            flag_ = 0;
            CCU_WHILE(repeatNumVar_ != UINT64_MAX) { // 循环repeatNum_次
                // root要去读每个rank每个chunk的数据
                DoRepeatReduce(input_, output_[rankId_]);
                repeatNumVar_ += repeatNumAdd;
                flag_ = 1;
            }
        }
    }

    PostSync();

    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMesh1DMem2Mem end");
    
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelReduceMesh1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceMeshMem2Mem1D *taskArg
        = dynamic_cast<const CcuTaskArgReduceMeshMem2Mem1D *>(&arg);
    uint64_t inputAddr                   = taskArg->inputAddr_;
    uint64_t outputAddr                  = taskArg->outputAddr_;
    uint64_t tokenInfo                   = taskArg->token_;
    uint64_t bigDataSliceNum             = taskArg->bigDataSliceNum_;
    uint64_t bigDataSliceSize            = taskArg->bigDataSliceSize_;
    uint64_t smallDataSliceNum           = taskArg->smallDataSliceNum_;
    uint64_t smallDataSliceSize          = taskArg->smallDataSliceSize_;
    uint64_t inputRepeatStride           = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride          = taskArg->outputRepeatStride_;
    uint64_t normalSliceSize             = taskArg->normalSliceSize_;
    uint64_t lastSliceSize               = taskArg->lastSliceSize_;
    uint64_t repeatNumVar                = taskArg->repeatNumVar_;
    uint64_t isInputOutputEqual          = (inputAddr == outputAddr) ? 1: 0;

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        tokenInfo,
        isInputOutputEqual,
        inputRepeatStride,
        outputRepeatStride,
        normalSliceSize,
        lastSliceSize,
        repeatNumVar,
    };

    for (uint64_t i = 0; i < bigDataSliceNum; i++) {
        taskArgs.push_back(bigDataSliceSize);
    }
    for (uint64_t i = 0; i < smallDataSliceNum; i++) {
        taskArgs.push_back(smallDataSliceSize);
    }
    auto localGoSize = CalGoSize(normalSliceSize);
    taskArgs.push_back(localGoSize[0]);
    taskArgs.push_back(localGoSize[1]);
    taskArgs.push_back(localGoSize[2]);
    taskArgs.push_back(localGoSize[3]);

    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], inputRepeatStride[%llu], "
        "outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu], repeatNumVar[%llu], "
        "bigDataSliceNum[%llu], bigDataSliceSize[%llu], smallDataSliceNum[%llu], smallDataSliceSize[%llu], ",
        inputAddr, outputAddr, inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar,
        bigDataSliceNum, bigDataSliceSize, smallDataSliceNum, smallDataSliceSize);
    return taskArgs;
}

} // namespace ops_hccl