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
constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int CKE_IDX_0    = 0; // pre
constexpr int CKE_IDX_1    = 1; // post
constexpr int CKE_IDX_2    = 2;
constexpr int CONST_ONE    = 1;
constexpr int CONST_EIGHT  = 8;
constexpr int CONST_NINE   = 9;

static CcuResult ParseKernelArg(AlltoAllVMesh1DContext &ctx, CcuKernelArgAlltoAllVMesh1D *kernelArg)
{
    return CCU_SUCCESS;
}

static CcuResult LoadAll2allSendRecvInfo(AlltoAllVMesh1DContext &ctx, A2AsingleSendRecvInfo &sendRecvInfo, uint16_t index)
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] LoadAll2allSendRecvInfo!");
    const auto *arg = ctx.arg;

    if (arg->loadFromMem) {
        HCCL_INFO("[CcuKernelAlltoAllVMesh1D] Load Args from Mem");
        sendRecvInfo.loopNum = UINT64_MAX - 1; // MC2 场景 loop num 默认为 1

        // 要求client端排列内存为[size,send,recv][size,send,recv]...
        ccu::Load(ctx.a2avXnAddr, sendRecvInfo.tailSize);
        ctx.a2avXnAddr += ctx.xnLength;

        ccu::Load(ctx.a2avXnAddr, sendRecvInfo.sendOffset);
        ctx.a2avXnAddr += ctx.xnLength;

        // 跳过recvSize
        ctx.a2avXnAddr += ctx.xnLength;

        ccu::Load(ctx.a2avXnAddr, sendRecvInfo.recvOffset);
        ctx.a2avXnAddr += ctx.xnLength;
    } else {
        uint32_t curIndex = index;
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.tailSize, curIndex++));
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.loopNum, curIndex++));
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.sendOffset, curIndex++));
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.recvOffset, curIndex++));
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.tailGoSize.addrOffset, curIndex++));
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.tailGoSize.loopParam, curIndex++));
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.tailGoSize.parallelParam, curIndex++));
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.tailGoSize.residual, curIndex++));
    }

    return CCU_SUCCESS;
}

static CcuResult InitResource(AlltoAllVMesh1DContext &ctx)
{   
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] InitResource!");
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAlltoAllVMesh1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    HCCL_INFO("arg->rankSize = %d, arg->rankId = %d", arg->rankSize, arg->rankId);
    ctx.input.resize(arg->rankSize);
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId != arg->rankId) { // 非本地，使用远端Variable
            ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }
    HCCL_INFO("output size: %d, token size: %d", ctx.output.size(), ctx.token.size());

    ctx.src.resize(arg->rankSize);
    ctx.dst.resize(arg->rankSize);

    // 前同步。交换信息，将本Rank load的in\out等地址信息写到所有对端的对应Variable中，并同步

    //  all2allv 数据搬运
    ctx.xnLength = 8; // xn长度为8byte

    ctx.resourceAllocated = false;

    return CCU_SUCCESS;
}

static CcuResult PreSync(AlltoAllVMesh1DContext &ctx)
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] PreSync begain!");
    const auto *arg = ctx.arg;
    ccu::Variable tempDst;
    
    u32 channelIdx = 0;
    for (u32 id = 0; id < arg->rankSize; id++) {
        if (id == arg->rankId) {
            continue;
        }
        tempDst = ctx.output[arg->rankId];
        tempDst += ctx.sendRecvInfo[id].recvOffset;
        // index = 0，传递output信息
        ccu::WriteVariableWithNotify(arg->channels[channelIdx], tempDst, OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        // index = 1，传递token信息
        ccu::WriteVariableWithNotify(arg->channels[channelIdx], ctx.token[arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
        channelIdx++;
    }

    uint16_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
    HCCL_INFO( "[CcuKernelAlltoAllVMesh1D] PreSync end");

    return CCU_SUCCESS;
}

static void PostSync(AlltoAllVMesh1DContext &ctx)
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] PostSync!");
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_1, 1 << CONST_ONE);
    }

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_1, 1 << CONST_ONE);
    }
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] PostSync End!");
}

static CcuResult LoadArgs(AlltoAllVMesh1DContext &ctx)
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] LoadArgs!");
    // 从SQE load args，本rank需要的input、output地址等信息
    // inputAddr, outputAddr, tokenInfo, srcStride, dstStride, srcOffset, dstOffset
    const auto *arg = ctx.arg;
    uint32_t cnt = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[0], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.srcOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.dstOffset, cnt++));
    if (arg->loadFromMem) {
        CCU_CHK_RET(ccu::LoadArg(ctx.a2avXnAddr, cnt++));
    } else {
        CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.addrOffset, cnt++));
        CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.loopParam, cnt++));
        CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.parallelParam, cnt++));
        CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.residual, cnt++));
    }
    // 恢复当前卡对所有卡的收发信息
    ctx.sendRecvInfo.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        uint16_t index = CONST_NINE + CONST_EIGHT * peerId;
        CCU_CHK_RET(LoadAll2allSendRecvInfo(ctx, ctx.sendRecvInfo[peerId], index));
    }

    return CCU_SUCCESS;
}

static void CalcGroupSrcDst(AlltoAllVMesh1DContext &ctx)
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] CalcGroupSrcDst!");
    const auto *arg = ctx.arg;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        ctx.src[rankIdx].token = ctx.token[rankIdx];

        // ctx.src[rankIdx] = usrInAddr + sendoffset + srcOffset_
        ctx.src[rankIdx].addr = ctx.input[0];
        ctx.src[rankIdx].addr += ctx.sendRecvInfo[rankIdx].sendOffset;
        ctx.src[rankIdx].addr += ctx.srcOffset;

        // ctx.dst[r] = recvBuf[r] + recvOffset + ctx.dstOffset
        if (rankIdx == arg->rankId) {
            // 写目的端为本端时需要特殊处理：使用接收基地址 + 块地址offset + 已发送数据量
            ctx.myDst.token = ctx.token[rankIdx];
            ctx.myDst.addr = ctx.output[rankIdx];
            ctx.myDst.addr += ctx.sendRecvInfo[rankIdx].recvOffset;
            ctx.myDst.addr += ctx.dstOffset;
        } else {
            // 对端交换的接收块起始地址 + 已接收的数据偏移
            ctx.dst[rankIdx].token = ctx.token[rankIdx];
            ctx.dst[rankIdx].addr = ctx.output[rankIdx];
            ctx.dst[rankIdx].addr += ctx.dstOffset;
        }
    }
}

static CcuResult DoAll2AllVMultiLoop(AlltoAllVMesh1DContext &ctx)
{
    HCCL_DEBUG("[CcuKernelAlltoAllVMesh1D] alltoallv mesh 1d use GroupCopy start");
    const auto *arg = ctx.arg;
    ctx.xnMaxTransportSize = UB_MAX_TRANS_SIZE;
    ctx.completedRankCount = 0;
    ctx.xnConst1 = 1;
    uint16_t allBit  = (1 << arg->rankSize) - 1;
    CCU_WHILE(ctx.completedRankCount != arg->rankSize) {
        u32 channelId = 0;
        for(uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
            if (rankIdx == arg->rankId) {
                continue;
            }
            CCU_IF(ctx.sendRecvInfo[rankIdx].loopNum == UINT64_MAX) { // 已经完成，直接置位完成信号
                ccu::EventRecord(ctx.event, 1 <<rankIdx);
            }
            CCU_IF(ctx.sendRecvInfo[rankIdx].loopNum != UINT64_MAX) {  // 还没有完成，则继续循环
                CCU_IF(ctx.sendRecvInfo[rankIdx].loopNum == UINT64_MAX - 1) { // 最后一轮循环, 发送尾块数据
                    CCU_IF(ctx.sendRecvInfo[rankIdx].tailSize == 0) { // 尾块数据量为 0，则不需要发送尾块数据
                        ccu::EventRecord(ctx.event, 1 <<rankIdx);
                    }
                    CCU_IF(ctx.sendRecvInfo[rankIdx].tailSize != 0) { // 尾块数据量不为 0，则需要发送尾块数据
                        ccu::Write(arg->channels[channelId], ctx.dst[rankIdx], ctx.src[rankIdx], ctx.sendRecvInfo[rankIdx].tailSize,
                              ctx.event, 1 <<rankIdx);
                    }
                    ctx.completedRankCount += ctx.xnConst1;  // 之后一轮循环完成，更新已完成的rank数
                }
                CCU_IF(ctx.sendRecvInfo[rankIdx].loopNum != UINT64_MAX - 1) { // 未完成，则继续循环，发送整块数据
                    ccu::Write(arg->channels[channelId], ctx.dst[rankIdx], ctx.src[rankIdx], ctx.xnMaxTransportSize, ctx.event, 1 <<rankIdx);
                    // 更新偏移
                    ctx.src[rankIdx].addr += ctx.xnMaxTransportSize;
                    ctx.dst[rankIdx].addr += ctx.xnMaxTransportSize;
                }
                ctx.sendRecvInfo[rankIdx].loopNum += ctx.xnConst1;
            }
                channelId++;
        }

        CCU_IF(ctx.sendRecvInfo[arg->rankId].loopNum == UINT64_MAX) { // 已经完成，直接置位完成信号
                ccu::EventRecord(ctx.event, 1 << arg->rankId);
        }

        CCU_IF(ctx.sendRecvInfo[arg->rankId].loopNum != UINT64_MAX) {  // 还没有完成，则继续循环
                CCU_IF(ctx.sendRecvInfo[arg->rankId].loopNum == UINT64_MAX - 1) { // 最后一轮循环, 发送尾块数据
                    CCU_IF(ctx.sendRecvInfo[arg->rankId].tailSize == 0) { // 尾块数据量为 0，则不需要发送尾块数据
                        ccu::EventRecord(ctx.event, 1 << arg->rankId);
                    }
                    CCU_IF(ctx.sendRecvInfo[arg->rankId].tailSize != 0) { // 尾块数据量不为 0，则需要发送尾块数据
                        if (arg->loadFromMem) {
                            ccu::LocalCopy(ctx.myDst, ctx.src[arg->rankId], ctx.sendRecvInfo[arg->rankId].tailSize, ctx.event, 1 << arg->rankId);
                        } else {
                            GroupCopy(ctx, ctx.myDst, ctx.src[arg->rankId], ctx.sendRecvInfo[arg->rankId].tailGoSize);
                            ccu::EventRecord(ctx.event, 1 <<arg->rankId);
                        }
                    }
                    ctx.completedRankCount += ctx.xnConst1;  // 之后一轮循环完成，更新已完成的rank数
                }
                CCU_IF(ctx.sendRecvInfo[arg->rankId].loopNum != UINT64_MAX - 1) { // 未完成，则继续循环，发送整块数据
                    if (arg->loadFromMem) {
                        ccu::LocalCopy(ctx.myDst, ctx.src[arg->rankId], ctx.xnMaxTransportSize, ctx.event, 1 <<arg->rankId);
                    } else {
                        GroupCopy(ctx, ctx.myDst, ctx.src[arg->rankId], ctx.xnMaxTransportGoSize);
                        ccu::EventRecord(ctx.event, 1 <<arg->rankId);
                    }
                    // 更新偏移
                    ctx.src[arg->rankId].addr += ctx.xnMaxTransportSize;
                    ctx.myDst.addr += ctx.xnMaxTransportSize;
                }
                ctx.sendRecvInfo[arg->rankId].loopNum += ctx.xnConst1;
        }
        // 等待本轮发送完成
        ccu::EventWait(ctx.event, allBit);
    }

    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuAlltoAllVMesh1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAlltoAllVMesh1D *>(arg);

    AlltoAllVMesh1DContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelAlltoAllMesh1D] AlltoAllMesh1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));
    CalcGroupSrcDst(ctx);

    CCU_CHK_RET(DoAll2AllVMultiLoop(ctx));

    PostSync(ctx);
    HCCL_INFO("[CcuKernelAlltoAllMesh1D] AlltoAllMesh1D end");

    return CCU_SUCCESS;
}
}