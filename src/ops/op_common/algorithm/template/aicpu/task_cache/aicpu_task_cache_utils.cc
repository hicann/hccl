/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include "aicpu_task_cache_utils.h"
#include "hccl_sym_win_dl.h"
namespace ops_hccl {

namespace {
    bool FillSymmetricMemForCache(
        const OpParam& param, uint32_t rankSize, uint32_t userRank, std::vector<void*>& addrs,
        std::vector<uint64_t>& sizes)
    {
        if (param.inputSymWindow == nullptr || param.outputSymWindow == nullptr
            || (!HcommIsSupportHcclSymWinGetRemoteAddr() && !HcommIsSupportHcclSymWinGetPeerPointer())) {
            HCCL_INFO("[%s] symmetric windows or peer address query are unavailable", __func__);
            return false;
        }
        for (uint32_t rank = 0; rank < rankSize; ++rank) {
            if (rank == userRank) {
                continue;
            }
            const uint32_t index = 2 + 2 * rank;
            const HcclResult inputRet
                = GetSymWinRemoteMem(param.inputSymWindow, param.inputOffset, rank, &addrs[index]);
            const HcclResult outputRet
                = GetSymWinRemoteMem(param.outputSymWindow, param.outputOffset, rank, &addrs[index + 1]);
            if (inputRet != HCCL_SUCCESS || outputRet != HCCL_SUCCESS) {
                HCCL_INFO(
                    "[%s] peer query failed, rank[%u] inputRet[%d] outputRet[%d]", __func__, rank, inputRet, outputRet);
                return false;
            }
            // 非V对称内存按相同窗口偏移访问各rank，同侧使用相同的算子字节范围。
            sizes[index] = sizes[0];
            sizes[index + 1] = sizes[1];
        }
        return true;
    }

    bool FillGraphMemForCache(
        const AlgResourceCtxSerializable& resCtx, std::vector<void*>& addrs, std::vector<uint64_t>& sizes,
        std::vector<bool>& rankSeen)
    {
        const uint32_t rankSize = resCtx.topoInfo.userRankSize;
        const uint32_t userRank = resCtx.topoInfo.userRank;
        // 图模式两层均遍历：外层是拓扑层级，同一个rank可能出现在多个channel中。
        for (const auto& levelChannels : resCtx.channels) {
            for (const auto& channel : levelChannels) {
                if (!channel.isValid || channel.remoteRank == userRank) {
                    continue;
                }
                const uint32_t rank = channel.remoteRank;
                CHK_PRT_RET(rank >= rankSize, HCCL_INFO("[%s] invalid remoteRank[%u]", __func__, rank), false);
                const HcclMem mems[] = {channel.remoteInputGraphMode, channel.remoteOutputGraphMode};
                for (uint32_t side = 0; side < 2; ++side) {
                    const uint32_t index = 2 + 2 * rank + side;
                    CHK_PRT_RET(
                        rankSeen[rank] && (addrs[index] != mems[side].addr || sizes[index] != mems[side].size),
                        HCCL_INFO("[%s] conflicting graph memory, rank[%u] side[%u]", __func__, rank, side), false);
                    addrs[index] = mems[side].addr;
                    sizes[index] = mems[side].size;
                }
                rankSeen[rank] = true;
            }
        }
        return true;
    }
} // namespace

HcclResult AicpuTaskCacheUtils::GetRemoteMemInfoForCache(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, std::vector<void*>& addrs,
    std::vector<uint64_t>& sizes, bool& complete)
{
    complete = false;
    addrs.clear();
    sizes.clear();
    const uint32_t rankSize = resCtx.topoInfo.userRankSize;
    const uint32_t userRank = resCtx.topoInfo.userRank;
    const uint64_t arrayCount = 2 + uint64_t{2} * rankSize;
    CHK_PRT_RET(
        rankSize == 0 || userRank >= rankSize || arrayCount > UINT32_MAX,
        HCCL_INFO("[%s] invalid rank memory layout", __func__), HCCL_SUCCESS);
    CHK_PRT_RET(
        (param.opMode != OpMode::OFFLOAD && !param.supportSymmetricMemory) || IsAlltoallvType(param.opType),
        HCCL_INFO("[%s] unsupported remote memory cache mode", __func__), HCCL_SUCCESS);

    addrs.resize(arrayCount, nullptr);
    sizes.resize(arrayCount, 0);
    const HcclResult sizeRet = GetInputOutputInfoForCache(param, rankSize, sizes[0], sizes[1]);
    if (sizeRet != HCCL_SUCCESS) {
        // Do not expose partial arrays or turn a cache preparation failure into an operation failure.
        addrs.clear();
        sizes.clear();
        HCCL_WARNING(
            "[AicpuTaskCacheUtils] cannot collect local memory sizes, fallback to normal unfold, comm[%s] "
            "opType[%u] ret[%d].",
            param.commName, static_cast<uint32_t>(param.opType), sizeRet);
        return HCCL_SUCCESS;
    }
    addrs[0] = param.inputPtr;
    addrs[1] = param.outputPtr;
    std::vector<bool> rankSeen(rankSize, false);
    rankSeen[userRank] = true;
    for (uint32_t side = 0; side < 2; ++side) {
        addrs[2 + 2 * userRank + side] = addrs[side];
        sizes[2 + 2 * userRank + side] = sizes[side];
    }
    if (param.supportSymmetricMemory) {
        // 命中会跳过executor；必须在Start/Execute之前按本次窗口和偏移重新取得远端地址。
        if (!FillSymmetricMemForCache(param, rankSize, userRank, addrs, sizes)) {
            return HCCL_SUCCESS;
        }
        std::fill(rankSeen.begin(), rankSeen.end(), true);
    } else if (!FillGraphMemForCache(resCtx, addrs, sizes, rankSeen)) {
        return HCCL_SUCCESS;
    }
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        CHK_PRT_RET(!rankSeen[rank], HCCL_INFO("[%s] missing remote memory, rank[%u]", __func__, rank), HCCL_SUCCESS);
    }
    for (size_t i = 0; i < addrs.size(); ++i) {
        const uint64_t base = reinterpret_cast<uint64_t>(addrs[i]);
        CHK_PRT_RET(
            base == 0 || sizes[i] == 0 || sizes[i] > UINT64_MAX - base,
            HCCL_INFO("[%s] unavailable user memory, index[%zu]", __func__, i), HCCL_SUCCESS);
    }
    complete = true;
    return HCCL_SUCCESS;
}

bool AicpuTaskCacheUtils::IsNonVariableOpType(HcclCMDType opType)
{
    // 注意: 当前hccl不支持HcclGather
    if (opType == HcclCMDType::HCCL_CMD_BROADCAST || opType == HcclCMDType::HCCL_CMD_ALLREDUCE
        || opType == HcclCMDType::HCCL_CMD_REDUCE || opType == HcclCMDType::HCCL_CMD_ALLGATHER
        || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER || opType == HcclCMDType::HCCL_CMD_ALLTOALL
        || opType == HcclCMDType::HCCL_CMD_SCATTER) {
        return true;
    }
    return false;
}

bool AicpuTaskCacheUtils::IsAlltoallvType(HcclCMDType opType)
{
    return opType == HcclCMDType::HCCL_CMD_ALLTOALLV || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC;
}

HcclResult AicpuTaskCacheUtils::AppendVOpInfoForCache(
    const OpParam& param, uint32_t rankSize, std::vector<void*>& addrs, std::vector<uint64_t>& sizes)
{
    CHK_PRT_RET(
        rankSize == 0 || addrs.size() != 2 || sizes.size() != 2,
        HCCL_ERROR("[%s] invalid rankSize or memory arrays", __func__), HCCL_E_PARA);
    const uint64_t arrayCount = 2 + uint64_t{2} * rankSize;
    CHK_PRT_RET(
        arrayCount > UINT32_MAX, HCCL_ERROR("[%s] V array count exceeds protocol limit", __func__), HCCL_E_PARA);
    addrs.reserve(arrayCount);
    sizes.reserve(arrayCount);
    const uint64_t* counts[]
        = {static_cast<const uint64_t*>(param.all2AllVDataDes.sendCounts),
           static_cast<const uint64_t*>(param.all2AllVDataDes.recvCounts)};
    const uint64_t* displs[]
        = {static_cast<const uint64_t*>(param.all2AllVDataDes.sdispls),
           static_cast<const uint64_t*>(param.all2AllVDataDes.rdispls)};
    const HcclDataType types[] = {param.all2AllVDataDes.sendType, param.all2AllVDataDes.recvType};
    // HCOMM不知道算子data type。传输协议中的counts和displs必须分别为字节数和字节偏移，不能传元素数。
    // 布局：[input, output, send rank 0..N-1, recv rank 0..N-1]；前两项仍为完整user buffer。
    for (uint32_t side = 0; side < 2; ++side) {
        CHK_PTR_NULL(counts[side]);
        CHK_PTR_NULL(displs[side]);
        const uint32_t type = static_cast<uint32_t>(types[side]);
        CHK_PRT_RET(
            type >= static_cast<uint32_t>(HCCL_DATA_TYPE_RESERVED), HCCL_ERROR("[%s] invalid data type", __func__),
            HCCL_E_PARA);
        const uint64_t typeSize = DATATYPE_SIZE_TABLE[type];
        CHK_PRT_RET(typeSize == 0, HCCL_ERROR("[%s] zero type size", __func__), HCCL_E_PARA);
        const uint64_t base = reinterpret_cast<uint64_t>(addrs[side]);
        for (uint32_t rank = 0; rank < rankSize; ++rank) {
            CHK_PRT_RET(
                counts[side][rank] > UINT64_MAX / typeSize || displs[side][rank] > UINT64_MAX / typeSize,
                HCCL_ERROR("[%s] byte conversion overflows", __func__), HCCL_E_PARA);
            const uint64_t bytes = counts[side][rank] * typeSize;
            const uint64_t offset = displs[side][rank] * typeSize;
            CHK_PRT_RET(
                offset > UINT64_MAX - base || bytes > UINT64_MAX - (base + offset)
                    || (bytes != 0 && (offset > sizes[side] || bytes > sizes[side] - offset)),
                HCCL_ERROR("[%s] V memory range is invalid", __func__), HCCL_E_PARA);
            // 零count的displ仍保留，但不要求对应实际可访问的内存。
            addrs.push_back(reinterpret_cast<void*>(base + offset));
            sizes.push_back(bytes);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheUtils::GetInputOutputInfoForCache(
    const OpParam& param, const uint32_t rankSize, uint64_t& inputSize, uint64_t& outputSize)
{
    const HcclCMDType opType = param.opType;

    // 准备data type和count
    // NOTE: 非V类算子 (DataRes), V类算子 (VDataDes), All2All类算子 (All2AllDataDes), batch类算子
    // (BatchSendRecvDataDes/BatchWriteDataDes)
    if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) { // alltoall算子
        // 注意: sendType和recvType一定相同
        HcclDataType sendType = param.all2AllDataDes.sendType;

        // 注意: 对于alltoall算子, inputSize和outputSize一定相同 (但不能直接使用param.input/outputSize,
        // alltoall算子不会设置这两个字段)
        CHK_PRT_RET(
            param.all2AllVDataDes.sendCounts == nullptr,
            HCCL_ERROR("[%s] all2AllVDataDes.sendCounts is nullptr, opType[%u]", __func__, opType), HCCL_E_PARA);
        const uint64_t sendCount = *(reinterpret_cast<const uint64_t*>(param.all2AllVDataDes.sendCounts));
        const uint32_t type = static_cast<uint32_t>(sendType);
        CHK_PRT_RET(
            type >= static_cast<uint32_t>(HCCL_DATA_TYPE_RESERVED), HCCL_ERROR("[%s] invalid data type", __func__),
            HCCL_E_PARA);
        const uint64_t typeSize = DATATYPE_SIZE_TABLE[type];
        CHK_PRT_RET(typeSize == 0, HCCL_ERROR("[%s] zero type size", __func__), HCCL_E_PARA);
        CHK_PRT_RET(sendCount > UINT64_MAX / typeSize, HCCL_ERROR("[%s] send bytes overflow", __func__), HCCL_E_PARA);
        const uint64_t sendBytes = sendCount * typeSize;
        CHK_PRT_RET(
            rankSize == 0 || sendBytes > UINT64_MAX / rankSize, HCCL_ERROR("[%s] input size overflow", __func__),
            HCCL_E_PARA);
        inputSize = sendBytes * rankSize;

        // 注意: 不能使用param.All2AllDataDes.recvCount * rankSize * SIZE_TABLE[recvType],
        // 因为alltoall使用sendCount来表示send/recvCount, 而recvCount本身为0
        outputSize = inputSize;

        HCCL_DEBUG(
            "[AicpuTaskCacheUtils][%s] opType[%u] rankSize[%u] sendType[%u] "
            "inputSize[%llu] outputSize[%llu] sendCount[%llu] dataTypeSize[%u]",
            __func__, opType, rankSize, sendType, inputSize, outputSize, sendCount, DATATYPE_SIZE_TABLE[sendType]);
    } else {
        inputSize = param.inputSize;
        outputSize = param.outputSize;

        HCCL_DEBUG(
            "[AicpuTaskCacheUtils][%s] opType[%u] rankSize[%u] inputSize[%llu] outputSize[%llu]", __func__, opType,
            rankSize, inputSize, outputSize);
    }

    return HCCL_SUCCESS;
}

void AicpuTaskCacheUtils::CalcInplaceDiff(
    const OpParam& param, const uint64_t inputSize, const uint64_t outputSize, int64_t& diff, uint64_t& overlapBytes)
{
    diff = 0;
    overlapBytes = 0;
    // 为tag计算实际重叠布局，不应用准入策略的Broadcast豁免；Broadcast记录完全重叠。
    // 空指针或任一长度为0时不计算重叠。
    if (param.inputPtr == nullptr || param.outputPtr == nullptr || inputSize == 0 || outputSize == 0) {
        return;
    }

    const uint64_t inputStart = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputStart = reinterpret_cast<uint64_t>(param.outputPtr);
    // 地址加法溢出防护(与HCOMM侧ParseTaskCacheArgs的校验对齐): 顶格地址视为无重叠
    if (inputSize > UINT64_MAX - inputStart || outputSize > UINT64_MAX - outputStart) {
        return;
    }
    const uint64_t inputEnd = inputStart + inputSize - 1;
    const uint64_t outputEnd = outputStart + outputSize - 1;

    // 区间不相交: input 在前 或 output 在前
    if (inputEnd < outputStart || outputEnd < inputStart) {
        return;
    }

    // 有符号差值用同向相减, 避免uint64到int64的超范围回绕转换(implementation-defined):
    // output 在后(含相同起点)为正, 在前为负
    diff = (outputStart >= inputStart) ? static_cast<int64_t>(outputStart - inputStart) :
                                         -static_cast<int64_t>(inputStart - outputStart);
    overlapBytes = std::min(inputEnd, outputEnd) - std::max(inputStart, outputStart) + 1;

    HCCL_INFO(
        "[AicpuTaskCacheUtils][%s] inputPtr[0x%016llx] inputSize[%llu] outputPtr[0x%016llx] "
        "outputSize[%llu] diff[%lld] overlapBytes[%llu]",
        __func__, inputStart, inputSize, outputStart, outputSize, diff, overlapBytes);
}

} // namespace ops_hccl
