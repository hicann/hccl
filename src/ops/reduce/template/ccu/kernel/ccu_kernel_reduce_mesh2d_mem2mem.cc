/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_mesh2d_mem2mem.h"


namespace ops_hccl {
using namespace hcomm;

constexpr int      INPUT_XN_ID    = 1;
constexpr int      TOKEN_XN_ID    = 2;
constexpr int      POST_SYNC_ID   = 3;
constexpr int      CKE_IDX_0      = 0; // 前后同步
constexpr int      CKE_IDX_1      = 0; // 轴同步
constexpr uint64_t CCU_MS_SIZE    = 4096;
constexpr uint64_t LOCAL_COPY_MS  = 8;
constexpr int      X_AXIS_ID      = 0;
constexpr int      Y_AXIS_ID      = 1;

CcuKernelReduceMesh2DMem2Mem::CcuKernelReduceMesh2DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgReduceMeshMem2Mem2D *kernelArg
        = dynamic_cast<const CcuKernelArgReduceMeshMem2Mem2D *>(&arg);
    axisId_          = kernelArg->axisId_;          // 要进行操作的是 行或列
    rankId_          = kernelArg->rankId_;          // 全局rankid
    rootId_          = kernelArg->rootId_;          // 全局rootid
    dimSize_         = kernelArg->dimSize_;         // 2维dim
    dimId_.emplace_back(rankId_ % dimSize_[0]);     // rank的x
    dimId_.emplace_back(rankId_ / dimSize_[0]);     // rank的y
    rootDimId_.emplace_back(rootId_ % dimSize_[0]); // root的x
    rootDimId_.emplace_back(rootId_ / dimSize_[0]); // root的y
    localId_         = dimId_[axisId_];             // 本rank所在的行/列
    localSize_       = dimSize_[axisId_];           // 本rank所在的行/列的总数
    channels_        = kernelArg->channels;
    
    HCCL_INFO("[CcuContextReduceMeshMem2Mem2D] RankId[%u], DimSize0[%u], DimSize1[%u], localId[%u], lcoalSize[%u]",
              rankId_, dimSize_[0], dimSize_[1], localId_, localSize_);
    
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelReduceMesh2DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_       = kernelArg->opParam_.reduceType;
    HCCL_INFO("[CcuContextReduceMeshMem2Mem2D] init end, kernelArg->dimSize size[%zu] localSize_[%u]",
              dimSize_.size(), localSize_);
}

HcclResult CcuKernelReduceMesh2DMem2Mem::InitResources()
{
    if (channels_.size() == 0) {
        HCCL_ERROR("CcuKernelReduceMesh2DMem2Mem channels is empty");
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint16_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < localSize_; peerId++) {
        if (peerId == localId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO("[CcuKernelReduceMesh2DMem2Mem] MyRank[%u], PeerId[%u], ChannelId[%u]", localId_, peerId,
                      channelIdx);
            CcuRep::Variable inputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    
    output_               = CreateVariable();
    event_             = CreateCompletedEvent();
    xAxisGroupOpSize_     = CreateGroupOpSize();
    yAxisGroupOpSize_     = CreateGroupOpSize();
    xAxisSize_            = CreateVariable();
    yAxisSize_            = CreateVariable();
    yAxisOffset_          = CreateVariable();
    curGoSize_            = CreateGroupOpSize();
    for (uint16_t roundId = 0; roundId < (localSize_ - 1); roundId++) {
        xChunkSize_.push_back(CreateVariable());
        yChunkSize_.push_back(CreateVariable());
        chunkSize_.push_back(CreateVariable());
    }
    chunkOffset_          = CreateVariable();
    HCCL_INFO("[CcuKernelReduceMesh2DMem2Mem] InitResources finished");
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelReduceMesh2DMem2Mem::LoadArgs()
{
    Load(input_[localId_]);
    Load(output_);
    Load(token_[localId_]);
    Load(xAxisSize_);
    Load(yAxisSize_);
    Load(yAxisOffset_);
    for (uint16_t i = 0; i < (localSize_ - 1); i++) {
        Load(xChunkSize_[i]);
    }
    for (uint16_t i = 0; i < (localSize_ - 1); i++) {
        Load(yChunkSize_[i]);
    }
    Load(xAxisGroupOpSize_);
    Load(yAxisGroupOpSize_);
    // 只有step2会用到localcopy
    curGoSize_ = (axisId_ == X_AXIS_ID) ? yAxisGroupOpSize_ : xAxisGroupOpSize_;
    HCCL_INFO("[CcuKernelReduceMesh2DMem2Mem] LoadArgs run finished");
}

void CcuKernelReduceMesh2DMem2Mem::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[localId_], 1 << INPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[localId_], 1 << TOKEN_XN_ID);
    }
    uint16_t allBit  = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    
    HCCL_INFO("[CcuKernelReduceMesh2DMem2Mem] PreSync run finished");
}

void CcuKernelReduceMesh2DMem2Mem::PostSync()
{
    for (auto &ch : channels_) {
        NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto &ch : channels_) {
        NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelReduceMesh2DMem2Mem] PostSync run finished");
}

void CcuKernelReduceMesh2DMem2Mem::AxisSync(uint32_t signalIndex) // 轴间同步
{
    const uint32_t DIE_NUM = 2;
    LocalNotifyRecord(1 - axisId_, CKE_IDX_1, 1 << (axisId_ + signalIndex * DIE_NUM));
    // TODO x30067372 还没接口
    LocalNotifyWait(axisId_, CKE_IDX_1, 1 << (1 - axisId_ + signalIndex * DIE_NUM));
    HCCL_INFO("[CcuKernelReduceMesh2DMem2Mem] AxisSync run finished");
    return;
}

void CcuKernelReduceMesh2DMem2Mem::ReduceStep1()
{
    HCCL_INFO("[CcuKernelReduceMesh2DMem2Mem] localId [%u], axisId [%u],Reduce Step1 starts", localId_, axisId_);
    CcuRep::LocalAddr dst      = CreateLocalAddr();
    CcuRep::RemoteAddr src     = CreateRemoteAddr();
    dst.addr               = input_[localId_]; // step1 reduce到input
    dst.token              = token_[localId_];
    bool              isYAxis = (axisId_ == Y_AXIS_ID);
    CcuRep::LocalAddr tmpDst  = CreateLocalAddr();
    chunkSize_                = isYAxis ? yChunkSize_ : xChunkSize_;
    for (uint16_t i = 0; i < (localSize_ - 1); i++) { // 外层循环控制步数=chunk数量
        // 读不同rank的不同chunk
        for (uint16_t rmtId = 0; rmtId < localSize_; ++rmtId) {
            if (rmtId == localId_) {
                continue;
            }
            src.addr     = input_[rmtId];
            src.token    = token_[rmtId];
            tmpDst.addr  = dst.addr;
            tmpDst.token = dst.token;
            if (isYAxis) { // 第一步yslicesize要在y轴方向reduce
                src.addr += yAxisOffset_;
                tmpDst.addr += yAxisOffset_;
            }
            chunkOffset_   = 0;
            uint16_t chkId = 0;
            if (rmtId < localId_) {
                chkId = (i + rmtId) % (localSize_ - 1);
            } else {
                chkId = (i + rmtId - 1) % (localSize_ - 1);
            }
            // 计算一下offset 0~(chikd-1)
            for (uint16_t j = 0; j < chkId; ++j) {
                chunkOffset_ += chunkSize_[j];
            }
            // 更新对应的addr
            src.addr += chunkOffset_;
            tmpDst.addr += chunkOffset_;
            CCU_IF(chunkSize_[chkId] == 0)
            {
                event_.SetMask(1 << rmtId);
                RecordEvent(event_);
            }

            CCU_IF(chunkSize_[chkId] != 0)
            {
                uint16_t channelId = rmtId < localId_ ? rmtId : rmtId - 1;
                event_.SetMask(1 << rmtId);
                ReadReduceNb(channels_[channelId], tmpDst, src, chunkSize_[chkId], dataType_, reduceOp_, event_);
            }
        }
        uint16_t allBit  = ((1 << localSize_) - 1) & (~(1 << localId_));
        event_.SetMask(allBit);
        WaitEvent(event_);
    }
    HCCL_INFO("[CcuKernelReduceMesh2DMem2Mem] Reduce Step1 ends");
}

void CcuKernelReduceMesh2DMem2Mem::ReduceStep2()
{
    HCCL_INFO("[CcuKernelReduceMesh2DMem2Mem] localId [%u] Reduce Step2 starts", localId_);
    CcuRep::LocalAddr dst            = CreateLocalAddr();
    CcuRep::LocalAddr myInput        = CreateLocalAddr();
    CcuRep::RemoteAddr remoteInput   = CreateRemoteAddr();
    dst.addr              = output_; // 第二步reduce是从input reduce到root的output
    dst.token             = token_[localId_];

    myInput.addr     = input_[localId_];
    myInput.token    = token_[localId_];
    bool isXAxis = (axisId_ == X_AXIS_ID);
    chunkSize_   = isXAxis ? yChunkSize_ : xChunkSize_;
    if (isXAxis) // 第二步yslicesize要在x轴方向reduce
    {
        myInput.addr += yAxisOffset_;
        dst.addr += yAxisOffset_;
    }
    // TODO x30067372 还没接口
    GroupCopy(dst, myInput, curGoSize_);
    for (uint16_t i = 0; i < (localSize_ - 1); i++) {
        for (uint16_t rmtId = 0; rmtId < localSize_; ++rmtId) {
            if (rmtId == localId_) {
                continue;
            }
            dst.addr  = output_;
            remoteInput.addr  = input_[rmtId];
            remoteInput.token = token_[rmtId];
            if (isXAxis) {
                remoteInput.addr += yAxisOffset_;
                dst.addr += yAxisOffset_;
            }
            chunkOffset_   = 0;
            uint16_t chkId = 0;
            if (rmtId < localId_) {
                chkId = (i + rmtId) % (localSize_ - 1);
            } else {
                chkId = (i + rmtId - 1) % (localSize_ - 1);
            }
            // 计算一下offset 0~(chikd-1)
            for (uint16_t j = 0; j < chkId; ++j) {
                chunkOffset_ += chunkSize_[j];
            }
            // 更新对应的addr
            remoteInput.addr += chunkOffset_;
            dst.addr += chunkOffset_;
            CCU_IF(chunkSize_[chkId] == 0)
            {
                event_.SetMask(1 << rmtId);
                RecordEvent(event_);
            }
            CCU_IF(chunkSize_[chkId] != 0)
            {
                uint16_t channelId = rmtId < localId_ ? rmtId : rmtId - 1;
                event_.SetMask(1 << rmtId);
                ReadReduceNb(channels_[channelId], dst, remoteInput, chunkSize_[chkId], dataType_, reduceOp_, event_);
            }
        }
        uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));
        event_.SetMask(allBit);
        WaitEvent(event_);
    }
    HCCL_INFO("[CcuKernelReduceMesh2DMem2Mem] Reduce Step2 ends");
}

HcclResult CcuKernelReduceMesh2DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelReduceMesh2DMem2Mem] ReduceMeshMem2Mem2D run");
    CHK_RET(InitResources());
    LoadArgs();
    PreSync(); // 前同步
    if (rankId_ == rootId_ || 
       (dimId_[1] == rootDimId_[1] && axisId_ == Y_AXIS_ID) || 
       (dimId_[0] == rootDimId_[0] && axisId_ == X_AXIS_ID)) {
        // 与root同行的元素要在Y方向规约 同列元素要在X方向规约
        ReduceStep1();
    }
    AxisSync(0);
    PostSync();
    AxisSync(1);
    if (rankId_ == rootId_) { // 第二步只有root进行readreduce
        ReduceStep2();
    }
    AxisSync(0);
    PostSync();
    AxisSync(1);
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelReduceMesh2DMem2Mem::CalMeshChunkSlice(uint64_t dataSize, uint64_t sliceNum)
{
    uint64_t dataCount          = dataSize / DataTypeSizeGet(dataType_);
    uint64_t bigDataSliceNum    = dataCount % sliceNum;
    uint64_t bigDataSliceSize   = (dataCount / sliceNum + 1) * DataTypeSizeGet(dataType_);
    uint64_t smallDataSliceNum  = sliceNum - dataCount % sliceNum;
    uint64_t smallDataSliceSize = dataCount / sliceNum * DataTypeSizeGet(dataType_);
    return {bigDataSliceNum, bigDataSliceSize, smallDataSliceNum, smallDataSliceSize};
}

std::vector<uint64_t> CcuKernelReduceMesh2DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceMeshMem2Mem2D *taskArg = dynamic_cast<const CcuTaskArgReduceMeshMem2Mem2D *>(&arg);
    uint64_t              inputAddr     = taskArg->inputAddr_;
    uint64_t              outputAddr    = taskArg->outputAddr_;
    uint64_t              tokenInfo     = taskArg->token_;
    uint64_t              xAxisSize     = taskArg->xAxisSize_;
    uint64_t              yAxisSize     = taskArg->yAxisSize_;
    uint64_t              yAxisOffset   = xAxisSize;
    auto                  xAxisGoSize   = CalGoSize(xAxisSize);
    auto                  yAxisGoSize   = CalGoSize(yAxisSize);
    std::vector<uint64_t> processReturn = {inputAddr, outputAddr, tokenInfo, xAxisSize, yAxisSize, yAxisOffset};
    HCCL_INFO("[CcuKernelReduceMesh2DMem2Mem] ReduceMeshMem2Mem2D inputAddr [%llu] outputAddr [%llu] "
              "xAxisSize [%llu] yAxisSize [%llu],yAxisOffset[%llu]",
              inputAddr, outputAddr, xAxisSize, yAxisSize, yAxisOffset);
    // mesh chunk for xslicesize
    std::vector<uint64_t> xChunkVec = CalMeshChunkSlice(xAxisSize, localSize_ - 1);
    for (uint64_t i = 0; i < xChunkVec[0]; i++) {
        processReturn.push_back(xChunkVec[1]);
    }
    for (uint64_t i = 0; i < xChunkVec[2]; i++) {
        processReturn.push_back(xChunkVec[3]);
    }
    // mesh chunk for yslicesize
    std::vector<uint64_t> yChunkVec = CalMeshChunkSlice(yAxisSize, localSize_ - 1);
    for (uint64_t i = 0; i < yChunkVec[0]; i++) {
        processReturn.push_back(yChunkVec[1]);
    }
    for (uint64_t i = 0; i < yChunkVec[2]; i++) {
        processReturn.push_back(yChunkVec[3]);
    }
    // for gosize
    processReturn.insert(processReturn.end(), xAxisGoSize.begin(), xAxisGoSize.end());
    processReturn.insert(processReturn.end(), yAxisGoSize.begin(), yAxisGoSize.end());
    return processReturn;
}
} // namespace ops_hccl