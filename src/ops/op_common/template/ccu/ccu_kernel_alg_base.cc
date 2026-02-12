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
#include "ccu_kernel_utils.h"

namespace ops_hccl {
using namespace hcomm;

HcclResult CcuKernelAlgBase::LocalReduceNb(const std::vector<CcuRep::CcuBuf> &bufs, uint32_t count, HcclDataType dataType,
                     HcclDataType outputDataType, HcclReduceOp opType,
                     const CcuRep::Variable &len, CcuRep::CompletedEvent event)
{
    return CcuKernel::LocalReduceNb(bufs.data(), bufs.size(), dataType, outputDataType, opType, len, event);
}

void CcuKernelAlgBase::AllocGoResource(uint32_t parallelDim, uint32_t msPerLoop)
{
    if (moConfig.loopCount != 0xFFFFFFFF && moConfig.msInterleave != 0xFFFFFFFF &&
        moConfig.memSlice != 0xFFFFFFFFFFFFFFFF) {
        // 已经配置过，略过
        return;
    } else {
        // 采用默认配置
        moConfig = {CcuRep::CCU_MS_INTERLEAVE, CcuRep::CCU_MS_DEFAULT_LOOP_COUNT, CcuRep::CCU_MS_SIZE};
    }
    // 算法配置的loop数覆盖默认配置，parallelDim默认为CCU_MS_DEFAULT_LOOP_COUNT
    moConfig.loopCount = parallelDim;
    // 算法配置的msPerLoop * CcuRep::CCU_MS_SIZE覆盖默认配置，msPerLoop默认为1
    moConfig.memSlice = msPerLoop * CcuRep::CCU_MS_SIZE;

    HCCL_INFO("[AllocGoResource]moConfig: loopCount = %u, msInterleave = %u", moConfig.loopCount, moConfig.msInterleave);

    // 简单实现,只需要申请一次资源
    if (moRes.executor.size() == 0) {
        moRes.executor = CreateBlockExecutor(moConfig.loopCount);
        moRes.completedEvent = CreateBlockCompletedEvent(moConfig.loopCount);
        moRes.ccuBuf = CreateBlockCcuBuf(moConfig.loopCount * moConfig.msInterleave);
    }
}

void CcuKernelAlgBase::Load(GroupOpSize moSize)
{
    Load(moSize.addrOffset);
    Load(moSize.loopParam);
    Load(moSize.parallelParam);
    Load(moSize.residual);
}

std::vector<uint64_t> CcuKernelAlgBase::CalGoSize(uint64_t size)
{
    uint64_t offset        = 0;
    uint64_t loopIterNum   = 0;
    uint64_t loopExtendNum = 0;
    uint64_t tailSize      = 0;

    uint64_t loopSize = moConfig.loopCount * moConfig.memSlice;     // loop展开后，一次并行搬运的size
    uint64_t maxSize = loopSize * (GetMaxLoopIterNum() + 1);

    uint64_t m = size / loopSize;                               // 能被loopSize(所有loop并行一次的大小)整除的部分。m为第一个loopgroup展开后的loop串行次数。
    uint64_t n = (size - m * loopSize) / moConfig.memSlice;     // 能被一个loop搬运大小整除的部分。n为第二个loopgroup的展开后的并行loop数。
    uint64_t p = size - m * loopSize - n * moConfig.memSlice;   // 尾数据。p为单独搬运的数据大小。

    if (size == maxSize) {
        m = GetMaxLoopIterNum();
        n = moConfig.loopCount - 1;
        p = moConfig.memSlice;
    }

    HCCL_INFO("Ccu Slice Split: m = %lu, n = %lu, p = %lu", m, n, p);

    // 数据量 < 256K, 跳过LoopGroup0
    // 此时loopIterNum == 0
    // 可以以此做为跳过LoopGroup0的条件
    offset = moConfig.memSlice * moConfig.loopCount * m;
    // 未实现, 这里可以只传入m, 在内部通过加法获得完整的参数
    loopIterNum = m;

    if (n == 0 && p == 0) {
        // 数据量为256K的整数倍，跳过LoopGroup1
        // 此时tailSize = 0，可以依次做为跳过LoopGroup1的条件
        loopExtendNum = 0; // loopExtendNum 赋值
        tailSize      = 0; // tailSize 赋值
    } else if (n != 0 && p == 0) {
        // 数据量为256K * m + 4K * n
        // 因为p == 0, 所以只需要使用第一个Loop, 数据量4K, 展开成n次
        loopExtendNum = GetParallelParam(n - 1, 0, 1); // loopExtendNum 赋值
        tailSize      = moConfig.memSlice;                     // tailSize 赋值
    } else if (n == 0 && p != 0) {
        // 数据量为256K * m + p
        // 因为n == 0, 所以只需要使用第一个Loop, 数据量p, 不展开
        loopExtendNum = GetParallelParam(0, 0, 1); // loopExtendNum 赋值
        tailSize      = p;                                 // tailSize 赋值
    } else {
        loopExtendNum = GetParallelParam(n - 1, 1, 2); // loopExtendNum 赋值, 为2
        tailSize      = p;                                     // tailSize 赋值
    }

    HCCL_INFO("[CcuKernelAlgBase::CalGoSize] offset = %lu, loopIterNum = %lu, loopExtendNum = %lu, tailSize = %lu", offset, loopIterNum,
               loopExtendNum, tailSize);

    return {offset, loopIterNum, loopExtendNum, tailSize};
}

HcclResult CcuKernelAlgBase::CreateMultiOpBroadcast(const std::vector<ChannelHandle> &channels)
{
    AllocGoResource();

    std::string loopType = "broadcast";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return HCCL_SUCCESS;
    }

    uint32_t size = channels.size() + 1;

    for (uint32_t index = 0; index < 2; index++) { // 需要实现化2个Loop
        CcuRep::LocalAddr src = CreateLocalAddr();
        std::vector<CcuRep::RemoteAddr> dst;
        for (uint32_t i = 0; i < size; i++) {
            CcuRep::LocalAddr tmp = CreateLocalAddr();
            dst.emplace_back(*reinterpret_cast<CcuRep::RemoteAddr*>(&tmp));
        }
        CcuRep::Variable            len = CreateVariable();
        CcuRep::LoopBlock           lb(this, loopType + "_loop_" + std::to_string(index));
        lb(src, dst, len);

        CcuRep::CcuBuf &buf = moRes.ccuBuf[index * moConfig.msInterleave];
        CcuRep::CompletedEvent &event = moRes.completedEvent[index];

        event.mask = 1;
        LocalCopyNb(buf, src, len, event);
        WaitEvent(event);

        for (uint32_t i = 0; i < channels.size(); i++) {
            if (channels[i] == 0) {
                return HCCL_E_PTR;
            }
            event.mask = 1 << i;
            CHK_RET(WriteNb(channels[i], dst[i], buf, len, event));
        }
        CcuRep::LocalAddr &localDst = *reinterpret_cast<CcuRep::LocalAddr*>(&dst[size - 1]);
        event.mask = 1 << (size - 1);
        LocalCopyNb(localDst, buf, len, event);
        event.mask = (1 << size) - 1;
        WaitEvent(event);
    }

    registeredLoop.insert(loopType);
    return HCCL_SUCCESS;
}

HcclResult CcuKernelAlgBase::GroupBroadcast(const std::vector<ChannelHandle> &channels, std::vector<CcuRep::RemoteAddr> dst,
                                CcuRep::LocalAddr src, GroupOpSize goSize)
{
    CHK_RET(CreateMultiOpBroadcast(channels));

    uint32_t size = channels.size() + 1;

    CCU_IF(goSize.addrOffset != 0)
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam = GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        auto lc   = Loop("broadcast_loop_0")(src, dst, sliceSize);

        CcuRep::Variable paraCfg = CreateVariable();
        paraCfg = GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }

    CCU_IF(goSize.parallelParam != 0)
    {
        src.addr += goSize.addrOffset;
        for (uint32_t i = 0; i < size; i++) {
            dst[i].addr += goSize.addrOffset;
        }

        auto lc0 = Loop("broadcast_loop_0")(src, dst, goSize.residual);

        src.addr += goSize.residual;
        for (uint32_t i = 0; i < size; i++) {
            dst[i].addr += goSize.residual;
        }

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        auto lc1  = Loop("broadcast_loop_1")(src, dst, sliceSize);

        CcuRep::Variable loopCfg0 = CreateVariable();
        loopCfg0 = GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1 = CreateVariable();
        loopCfg1 = GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
    }
    return HCCL_SUCCESS;
}

HcclResult CcuKernelAlgBase::CreateMultiOpReduce(const std::vector<ChannelHandle> &channels, HcclDataType dataType,
                                     HcclDataType outputDataType, HcclReduceOp opType)
{
    AllocGoResource();

    std::string loopType = GetReduceTypeStr(dataType, opType);
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return HCCL_SUCCESS;
    }

    uint32_t size         = channels.size() + 1;
    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < 2; index++) { // 需要实现化2个Loop
        std::vector<CcuRep::RemoteAddr> src;
        src.reserve(size);
        for (uint32_t i = 0; i < size; i++) {
            CcuRep::LocalAddr tmp = CreateLocalAddr();
            src.emplace_back(*reinterpret_cast<CcuRep::RemoteAddr*>(&tmp));
        }
        CcuRep::LocalAddr dst = CreateLocalAddr();
        CcuRep::Variable  len = CreateVariable();
        CcuRep::Variable  lenForExpansion = CreateVariable();
        CcuRep::LoopBlock lb(this, loopType + "_loop_" + std::to_string(index));
        lb(src, dst, len, lenForExpansion);

        std::vector<CcuRep::CcuBuf> bufs = {moRes.ccuBuf.begin() + index * moConfig.msInterleave,
                                               moRes.ccuBuf.begin() + index * moConfig.msInterleave + usedBufNum};
        CcuRep::CompletedEvent &event = moRes.completedEvent[index];
        for (uint32_t i = 0; i < channels.size(); i++) {
            event.mask = 1 << i;
            ReadNb(channels[i], bufs[i], src[i], len, event);
        }

        CcuRep::LocalAddr &localSrc = *reinterpret_cast<CcuRep::LocalAddr*>(&src[size - 1]);
        event.mask = 1 << (size - 1);
        LocalCopyNb(bufs[size - 1], localSrc, len, event);
        event.mask = (1 << size) - 1;
        WaitEvent(event);

        if (size > 1) {
            event.mask = 1;
            LocalReduceNb(bufs, size, dataType, outputDataType, opType, len, event);
            WaitEvent(event);
        }

        event.mask = 1;
        LocalCopyNb(dst, bufs[0], lenForExpansion, event);
        WaitEvent(event);
    }

    registeredLoop.insert(loopType);
    return HCCL_SUCCESS;
}

HcclResult CcuKernelAlgBase::GroupReduce(const std::vector<ChannelHandle> &channels, CcuRep::LocalAddr dst,
                             std::vector<CcuRep::RemoteAddr> src, GroupOpSize goSize, HcclDataType dataType,
                             HcclDataType outputDataType, HcclReduceOp opType)
{
    CHK_RET(CreateMultiOpReduce(channels, dataType, outputDataType, opType));

    uint32_t         size         = channels.size() + 1;
    uint32_t         expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    CcuRep::Variable sliceSizeExpansion = CreateVariable();

    if (expansionNum != 1) {
        CcuRep::Variable tmp = CreateVariable();
        tmp = GetExpansionParam(expansionNum);
        dst.token += tmp;
    }

    // 第一个loopgroup，只包含1个loop，搬运m部分数据。
    // loopgroup的parallel参数自己生成
    CCU_IF(goSize.loopParam != 0)
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam = GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize          = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;

        auto lc = Loop(GetReduceTypeStr(dataType, opType) + "_loop_0")(src, dst, sliceSize, sliceSizeExpansion);

        CcuRep::Variable paraCfg = CreateVariable();
        paraCfg = GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);

    }

    // 第二个loopgroup，包含1或2个loop，搬运n和p部分数据。
    // loopgroup的parallel参数使用基类中的moConfig，需要提前调用CalGoSize计算出来。
    CCU_IF(goSize.parallelParam != 0)
    {
        for (uint32_t i = 0; i < size; i++) {
            src[i].addr += goSize.addrOffset;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion += goSize.residual;
        }

        auto lc0 = Loop(GetReduceTypeStr(dataType, opType) + "_loop_0")(src, dst, goSize.residual, sliceSizeExpansion);

        for (uint32_t i = 0; i < size; i++) {
            src[i].addr += goSize.residual;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.residual;
        }

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize          = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;

        auto lc1 = Loop(GetReduceTypeStr(dataType, opType) + "_loop_1")(src, dst, sliceSize, sliceSizeExpansion);

        CcuRep::Variable loopCfg0 = CreateVariable();
        loopCfg0 = GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1 = CreateVariable();
        loopCfg1 = GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);

    }
    return HCCL_SUCCESS;
}

HcclResult CcuKernelAlgBase::CreateMultiOpCopy()
{
    AllocGoResource(CCU_MS_LOCAL_COPY_LOOP_COUNT, LOCAL_COPY_MS_PER_LOOP);
    std::string loopType = "localcopy";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return HCCL_SUCCESS;
    }

    uint32_t usedBufNum = moConfig.memSlice / CcuRep::CCU_MS_SIZE;

    for (uint32_t index = 0; index < 2; index++) { // 需要实现化2个Loop
        CcuRep::LocalAddr src = CreateLocalAddr();
        CcuRep::LocalAddr dst = CreateLocalAddr();
        CcuRep::Variable  len = CreateVariable();
        CcuRep::LoopBlock lb(this, loopType + "_loop_" + std::to_string(index));
        lb(src, dst, len);

        CcuRep::CompletedEvent event = moRes.completedEvent[index];

        std::vector<CcuRep::CcuBuf> bufs = {moRes.ccuBuf.begin() + index * moConfig.msInterleave,
                                               moRes.ccuBuf.begin() + index * moConfig.msInterleave + usedBufNum};

        event.mask = 1;
        LocalCopyNb(bufs[0], src, len, event);
        WaitEvent(event);
        LocalCopyNb(dst, bufs[0], len, event);
        WaitEvent(event);
    }

    registeredLoop.insert(loopType);
    return HCCL_SUCCESS;
}

HcclResult CcuKernelAlgBase::GroupCopy(CcuRep::LocalAddr dst, CcuRep::LocalAddr src, GroupOpSize goSize)
{
    CHK_RET(CreateMultiOpCopy());
    CCU_IF(goSize.addrOffset != 0)
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam                  = GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize                  = moConfig.memSlice;
        auto lc                    = Loop("localcopy_loop_0")(src, dst, sliceSize);

        CcuRep::Variable paraCfg   = CreateVariable();
        paraCfg                    = GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg                  = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }

    CCU_IF(goSize.parallelParam != 0)
    {
        CcuRep::Condition cond(this, goSize.parallelParam != 0);

        src.addr += goSize.addrOffset;
        dst.addr += goSize.addrOffset;
        auto lc0 = Loop("localcopy_loop_0")(src, dst, goSize.residual);

        src.addr += goSize.residual;
        dst.addr += goSize.residual;
        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize                  = moConfig.memSlice;
        auto lc1                   = Loop("localcopy_loop_1")(src, dst, sliceSize);

        CcuRep::Variable loopCfg0  = CreateVariable();
        loopCfg0                   = GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1  = CreateVariable();
        loopCfg1                   = GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg                  = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
    }
    return HCCL_SUCCESS;
}

void CcuKernelAlgBase::LoopGroup(const std::vector<CcuRep::LoopCall> &loops, const std::vector<CcuRep::Variable> &loopCfg,
                           const CcuRep::Variable &paraCfg, const CcuRep::Variable &offsetCfg)
{
    auto                          lgc = CcuRep::LoopGroupCall(this);
    std::vector<CcuRep::Executor> executors;
    for (size_t i = 0; i < loops.size(); i++) {
        executors.push_back(moRes.executor[i]);
    }
    lgc.Run(loops, loopCfg, executors, paraCfg, offsetCfg);
}

CcuKernelAlgBase::GroupOpSize CcuKernelAlgBase::CreateGroupOpSize()
{
    return GroupOpSize{CreateVariable(), CreateVariable(), CreateVariable(), CreateVariable()};
}

std::vector<CcuRep::CcuBuf> CcuKernelAlgBase::CreateBlockCcuBuf(uint32_t count)
{
    std::vector<CcuRep::CcuBuf> res(count);
    // todo: 结果校验
    CcuKernel::CreateBlockCcuBuf(count, res.data());
    return res;
}

std::vector<CcuRep::Executor> CcuKernelAlgBase::CreateBlockExecutor(uint32_t count)
{
    std::vector<CcuRep::Executor> res(count);
    // todo: 结果校验
    CcuKernel::CreateBlockExecutor(count, res.data());
    return res;
}

std::vector<CcuRep::CompletedEvent> CcuKernelAlgBase::CreateBlockCompletedEvent(uint32_t count)
{
    std::vector<CcuRep::CompletedEvent> res(count);
    // todo: 结果校验
    CcuKernel::CreateBlockCompletedEvent(count, res.data());
    return res;
}

}