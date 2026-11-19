/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_task_cache_policy.h"
#include <algorithm>
#include <cstring>
#include <unordered_set>

#include "alg_env_config.h"
#include "log.h"
#include "aicpu_task_cache_utils.h"

namespace ops_hccl {

namespace {
    constexpr uint32_t ALLTOALLV_CACHE_MAX_CONCURRENT_SIZE = 16;
    constexpr uint64_t ALLTOALLV_CACHE_SLICE_ALIGN = 128;
    constexpr char ALLTOALLV_CACHE_ALG_NAME[] = "AicpuAllToAllVSoleMesh";
    constexpr char ALLTOALLVC_CACHE_ALG_NAME[] = "AicpuAllToAllVCSoleMesh";
} // namespace

HcclResult AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, bool& isCacheEnable)
{
    isCacheEnable = false;
    if (!param.aicpuCacheEnable) {
        HCCL_INFO("[AicpuTaskCachePolicy][IsAicpuTaskCacheEnable] AICPU task cache is disabled");
        return HCCL_SUCCESS;
    }

    if (!EqualToOrGreaterThan950(param.deviceType)) {
        HCCL_INFO("[AicpuTaskCachePolicy][IsAicpuTaskCacheEnable] deviceType[%d] is not supported", param.deviceType);
        return HCCL_SUCCESS;
    }

    // MC2算子不支持
    if (param.isMc2) {
        HCCL_INFO("[AicpuTaskCachePolicy][IsAicpuTaskCacheEnable] mc2 is not supported");
        return HCCL_SUCCESS;
    }

    // isZeroCopy表示的独立零拷贝路径尚未验证task cache，维持屏蔽。
    // 图模式和对称内存分别由opMode、supportSymmetricMemory识别，不依赖此标志。
    if (param.isZeroCopy) {
        HCCL_INFO("[AicpuTaskCachePolicy][IsAicpuTaskCacheEnable] zero copy is not supported");
        return HCCL_SUCCESS;
    }

    // 普通ACLGraph仍走OPBASE流程：捕获AICPU kernel启动任务，图回放时在kernel内执行
    // cache miss的Start/End或cache hit的Execute，由HCOMM刷新并下发任务，无需再次捕获SQE/WQE。
    // 因此不因isCapture单独禁用cache；tag已包含isCapture，隔离捕获与非捕获调用。

    // 校验算子类型
    if (!IsOpTypeSupported(param)) {
        return HCCL_SUCCESS;
    }

    if (AicpuTaskCacheUtils::IsAlltoallvType(param.opType)) {
        bool isSupported = false;
        CHK_RET(IsAlltoallvCacheSupported(param, resCtx, isSupported));
        if (!isSupported) {
            return HCCL_SUCCESS;
        }
    }

    // inplace 场景放行(task cache 支持 inplace 准入): 重叠特征(位置+宽度)由 tag 区分(见 GetAicpuTaskCacheTag)
    bool isInplace = false;
    bool isEmptyInOut = false;
    const HcclResult inplaceRet = IsInplaceForCache(param, resCtx.topoInfo.userRankSize, isInplace, isEmptyInOut);
    if (inplaceRet != HCCL_SUCCESS) {
        // Cache admission is optional. Leave validation of the actual operation to normal unfolding.
        HCCL_WARNING(
            "[AicpuTaskCachePolicy] cannot determine input/output layout, disable cache and fallback to normal "
            "unfold, comm[%s] opType[%u] ret[%d].",
            param.commName, static_cast<uint32_t>(param.opType), inplaceRet);
        return HCCL_SUCCESS;
    }
    // A3 场景: tinySendRecvMem 顶替建链, cache 无法区分地址类型, 维持屏蔽
    if (isEmptyInOut) {
        HCCL_INFO("[AicpuTaskCachePolicy][IsAicpuTaskCacheEnable] empty input/output case is not supported for "
                  "operator unfolding cache.");
        return HCCL_SUCCESS;
    }
    // V 类算子(变长)不支持 inplace 重叠。V 类已支持非 inplace 进 cache, 此处拦截其 inplace 场景
    // (V 格式按对端子区间追加 addrs, 重叠特征判定与刷新均未适配)
    if (isInplace && AicpuTaskCacheUtils::IsAlltoallvType(param.opType)) {
        HCCL_INFO(
            "[AicpuTaskCachePolicy][IsAicpuTaskCacheEnable] inplace is not supported for variable ops, "
            "opType[%u].",
            static_cast<uint32_t>(param.opType));
        return HCCL_SUCCESS;
    }
    // 图模式下的 inplace 要求所有 rank 的重叠情况一致才可放行。
    // 当前无跨 rank 协商机制, 保守屏蔽; 图模式准入放开时需扩展为全局一致性判断
    if (isInplace && param.opMode == OpMode::OFFLOAD) {
        HCCL_INFO("[AicpuTaskCachePolicy][IsAicpuTaskCacheEnable] inplace with graph mode requires consistent "
                  "overlap across all ranks, not supported yet.");
        return HCCL_SUCCESS;
    }

    if (!IsTopoSupported(resCtx)) {
        return HCCL_SUCCESS;
    }

    // 所有使能判断都通过是返回true
    isCacheEnable = true;
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCachePolicy::IsInplaceForCache(
    const OpParam& param, const uint32_t rankSize, bool& isInplace, bool& isEmptyInOut)
{
    isInplace = false;
    isEmptyInOut = false;

    // 准备input/output size
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;

    CHK_RET(AicpuTaskCacheUtils::GetInputOutputInfoForCache(param, rankSize, inputSize, outputSize));

    // 注意: A3下alltoall/alltoallv/alltoallvc可能存在inputSize/outputSize为0的情况, 导致不分配user input/output
    //     但会使用tinySendRecvMem_更新algResource.paramInput/OutputMem用于建链, 导致cache无法区分给定地址字段的地址类型
    //     参考aicpu_communicator.cc中的SetAlltoAllInputAndOutPutMem
    // 注意: 这里继承A3, 不支持同时为0的场景
    if (inputSize == 0 && outputSize == 0) {
        isInplace = true;
        isEmptyInOut = true; // A3 场景: 标记由调用方维持屏蔽
        HCCL_INFO(
            "[AicpuTaskCachePolicy][IsInplace] inputSize[%llu] is overlapping with outputSize[%llu] -> isInplace[%d]",
            inputSize, outputSize, isInplace);
        return HCCL_SUCCESS;
    }

    // 注意: 如果inputSize和outputSize只有一个为0, 则一定是outplace场景
    if (inputSize == 0 || outputSize == 0) {
        isInplace = false;
        HCCL_INFO(
            "[AicpuTaskCachePolicy][IsInplace] inputSize[%llu] is not overlapping with outputSize[%llu] -> "
            "isInplace[%d]",
            inputSize, outputSize, isInplace);
        return HCCL_SUCCESS;
    }

    const uint64_t inputStart = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t inputEnd = inputStart + inputSize - 1;
    const uint64_t outputStart = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t outputEnd = outputStart + outputSize - 1;

    // Broadcast按接口语义共用同一块input/output buffer，布局固定，豁免通用inplace准入限制。
    // 此处isInplace=false仅表示准入豁免，并非内存无重叠；CalcInplaceDiff仍为tag记录真实布局
    // (diff=0, overlapBytes=inputSize)。不能直接按overlapBytes>0判定，否则会拦截图模式Broadcast。
    if (param.opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        CHK_PRT_RET(
            !(inputStart == outputStart && inputSize == outputSize),
            HCCL_ERROR(
                "[AicpuTaskCachePolicy][IsInplace] broadcast should input==output[0x%016llx, 0x%016llx] "
                "inputSize==outputSize[%llu,%llu]",
                inputStart, outputStart, inputSize, outputSize),
            HCCL_E_PARA);
        isInplace = false;
        HCCL_INFO(
            "[AicpuTaskCachePolicy][IsInplace] input==output[0x%016llx, 0x%016llx] for opType[%d] -> isInplace[%d]",
            inputStart, inputEnd, param.opType, isInplace);
        return HCCL_SUCCESS;
    }

    if (inputStart <= outputEnd && outputStart <= inputEnd) {
        isInplace = true;
        HCCL_INFO(
            "[AicpuTaskCachePolicy][IsInplace] input[0x%016llx, 0x%016llx] is overlapping with output[0x%016llx, "
            "0x%016llx] -> isInplace[%d]",
            inputStart, inputEnd, outputStart, outputEnd, isInplace);
    } else {
        isInplace = false;
        HCCL_INFO(
            "[AicpuTaskCachePolicy][IsInplace] input[0x%016llx, 0x%016llx] is not overlapping with "
            "output[0x%016llx, 0x%016llx] -> isInplace[%d]",
            inputStart, inputEnd, outputStart, outputEnd, isInplace);
    }

    return HCCL_SUCCESS;
}

bool AicpuTaskCachePolicy::IsTopoSupported(const AlgResourceCtxSerializable& resCtx)
{
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
    for (const auto& levelChannels : resCtx.channels) {
        for (const auto& channel : levelChannels) {
            if (!channel.isValid) {
                continue;
            }

            // 当前AICPU task cache仅支持基于UB_CTP/UBOE的跨卡通信。
            // UB_RTP虽是AICPU建链候选协议，但不支持AICPU task cache。
            // 不支持其他通信方式, 例如基于外置网卡的跨超RDMA, 基于PCIe的跨卡P2P等
            if (channel.protocol != CommProtocol::COMM_PROTOCOL_UB_CTP
                && channel.protocol != CommProtocol::COMM_PROTOCOL_UBOE) {
                HCCL_INFO(
                    "[AicpuTaskCachePolicy][IsTopoSupported] found channel protocol[%d] not supported",
                    channel.protocol);
                return false;
            }
        }
    }
    return true;
#else
    // 小于9.1.0版本，不支持aicpu task cache，直接返回false;
    (void)resCtx;
    return false;
#endif
}

bool AicpuTaskCachePolicy::IsOpTypeSupported(const OpParam& param)
{
    // batch类型算子以及send/recv不考虑动态缓存 (使用白名单而非黑名单管理, 避免非预期算子进入cache机制)
    // 注意: 当前hccl不支持HcclGather
    HcclCMDType opType = param.opType;
    if (opType == HcclCMDType::HCCL_CMD_BROADCAST || opType == HcclCMDType::HCCL_CMD_ALLGATHER
        || opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_SCATTER
        || AicpuTaskCacheUtils::IsAlltoallvType(opType)) {
        HCCL_INFO(
            "[AicpuTaskCachePolicy][IsOpTypeSupported] opType[%d] is supported for operator unfolding cache", opType);
        return true;
    }

    if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE || opType == HcclCMDType::HCCL_CMD_REDUCE
        || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        HcclDataType dataType = param.DataDes.dataType;
        // all_reduce/reduce/reduce_scatter在64位数据类型(INT64/UINT64/FP64)或PROD规约操作时,
        // 需要在aicpu上执行reduce操作，不支持aicpu task cache
        if (dataType == HcclDataType::HCCL_DATA_TYPE_INT64 || dataType == HcclDataType::HCCL_DATA_TYPE_UINT64
            || dataType == HcclDataType::HCCL_DATA_TYPE_FP64 || param.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
            HCCL_INFO(
                "[AicpuTaskCachePolicy][IsOpTypeSupported] opType[%d] is not supported, dataType[%d] "
                "reduceOp[%d]",
                opType, dataType, param.reduceType);
            return false;
        } else {
            HCCL_INFO(
                "[AicpuTaskCachePolicy][IsOpTypeSupported] opType[%d] is supported for operator unfolding cache",
                opType);
            return true;
        }
    }

    // 其它算子不支持
    HCCL_INFO("[AicpuTaskCachePolicy][IsOpTypeSupported] opType[%d] is not supported", param.opType);
    return false;
}

HcclResult AicpuTaskCachePolicy::IsAlltoallvCacheSupported(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, bool& isSupported)
{
    isSupported = false;
    // V类算子的图模式和对称内存暂不支持cache。
    if (param.opMode == OpMode::OFFLOAD || param.supportSymmetricMemory) {
        HCCL_INFO("[AicpuTaskCachePolicy][IsAlltoallvCacheSupported] graph or symmetric memory is not supported");
        return HCCL_SUCCESS;
    }
    if (param.isZeroCopy) {
        HCCL_INFO("[AicpuTaskCachePolicy][IsAlltoallvCacheSupported] zero copy is not supported");
        return HCCL_SUCCESS;
    }

    const char* expectedAlgName
        = param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV ? ALLTOALLV_CACHE_ALG_NAME : ALLTOALLVC_CACHE_ALG_NAME;
    if (std::strcmp(param.algName, expectedAlgName) != 0) {
        HCCL_INFO(
            "[AicpuTaskCachePolicy][IsAlltoallvCacheSupported] algName[%s] is not supported for opType[%d]",
            param.algName, param.opType);
        return HCCL_SUCCESS;
    }

    const uint32_t rankSize = resCtx.topoInfo.userRankSize;
    const uint32_t userRank = resCtx.topoInfo.userRank;
    CHK_PRT_RET(
        rankSize == 0 || userRank >= rankSize,
        HCCL_INFO("[%s] invalid rankSize[%u] or userRank[%u]", __func__, rankSize, userRank), HCCL_SUCCESS);

    // 多channel的分片可能随count变化，当前V类缓存仅支持每个远端peer一条有效channel。
    std::unordered_set<uint32_t> channelPeers;
    for (const auto& levelChannels : resCtx.channels) {
        for (const auto& channel : levelChannels) {
            if (!channel.isValid || channel.remoteRank == userRank) {
                continue;
            }
            if (!channelPeers.insert(channel.remoteRank).second) {
                HCCL_INFO(
                    "[AicpuTaskCachePolicy][IsAlltoallvCacheSupported] multiple channels for remoteRank[%u] "
                    "are not supported",
                    channel.remoteRank);
                return HCCL_SUCCESS;
            }
        }
    }

    CHK_PRT_RET(
        param.all2AllVDataDes.sendCounts == nullptr || param.all2AllVDataDes.recvCounts == nullptr,
        HCCL_INFO("[%s] counts are unavailable for cache", __func__), HCCL_SUCCESS);
    const auto* sendCounts = static_cast<const uint64_t*>(param.all2AllVDataDes.sendCounts);
    const auto* recvCounts = static_cast<const uint64_t*>(param.all2AllVDataDes.recvCounts);

    // 即使本轮只有self-copy，也需回放全peer预热缓存中的同步任务。
    uint64_t maxSendOrRecvCount = 0;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        maxSendOrRecvCount = std::max(maxSendOrRecvCount, sendCounts[rank]);
        maxSendOrRecvCount = std::max(maxSendOrRecvCount, recvCounts[rank]);
    }

    CHK_PRT_RET(
        resCtx.algHierarchyInfo.infos.empty() || resCtx.algHierarchyInfo.infos[0].empty(),
        HCCL_INFO("[%s] algHierarchyInfo is empty", __func__), HCCL_SUCCESS);
    const auto& subCommRanks = resCtx.algHierarchyInfo.infos[0];
    uint64_t templateRankSize = subCommRanks[0].size();
    if (subCommRanks.size() > 1) {
        templateRankSize *= subCommRanks[1].size();
    }
    CHK_PRT_RET(
        templateRankSize <= 1, HCCL_INFO("[%s] invalid templateRankSize[%llu]", __func__, templateRankSize),
        HCCL_SUCCESS);

    const uint64_t scratchMultiple = std::min<uint64_t>(ALLTOALLV_CACHE_MAX_CONCURRENT_SIZE, templateRankSize - 1);
    const uint64_t scratchBoundDataSize
        = resCtx.cclMem.size / scratchMultiple / ALLTOALLV_CACHE_SLICE_ALIGN * ALLTOALLV_CACHE_SLICE_ALIGN;
    const uint64_t maxDataSizePerLoop = std::min<uint64_t>(UB_MAX_DATA_SIZE, scratchBoundDataSize);
    const HcclDataType dataType = param.all2AllVDataDes.sendType;
    const uint32_t dataTypeValue = static_cast<uint32_t>(dataType);
    CHK_PRT_RET(
        dataTypeValue >= static_cast<uint32_t>(HcclDataType::HCCL_DATA_TYPE_RESERVED),
        HCCL_INFO("[%s] invalid dataType[%d]", __func__, dataType), HCCL_SUCCESS);
    const uint64_t dataTypeSize = DATATYPE_SIZE_TABLE[dataTypeValue];
    CHK_PRT_RET(dataTypeSize == 0, HCCL_INFO("[%s] dataTypeSize is zero", __func__), HCCL_SUCCESS);
    const uint64_t maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
    if (maxDataCountPerLoop == 0 || maxSendOrRecvCount > maxDataCountPerLoop) {
        HCCL_INFO(
            "[AicpuTaskCachePolicy][IsAlltoallvCacheSupported] split case is not supported: "
            "maxSendOrRecvCount[%llu] maxDataCountPerLoop[%llu]",
            maxSendOrRecvCount, maxDataCountPerLoop);
        return HCCL_SUCCESS;
    }

    isSupported = true;
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
