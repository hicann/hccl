/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cerrno>
#include <cstring>
#include <vector>

#include "channel.h"
#include "dev_type.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_primitives_impl_dl.h"
#include "ccu_launch_dl.h"
#include "hccl_rank_graph_dl.h"
#include "hccl_sym_win_dl.h"
#include "securec.h"

/* CCU 二级 AllToAll stage 模板 UT 桩：CalcRes/KernelRun 引用的外部符号，链接期兜底。 */

namespace ops_hccl {
// 桩按组序构造对端通道（真实链依赖 topo/host 重链）
namespace {
    // 场景状态 + launch 记录
    struct ChannelScenario {
        bool dualDie = false;
        uint32_t die0Bw = 1;
        uint32_t die1Bw = 1;
        std::vector<uint32_t> peers;            // 空 = 组序默认（排除自身）
        uint32_t die1MissingPeer = 0xFFFFFFFFu; // 该对端第二通道落 die0（die1 桶缺对端）
    };
    ChannelScenario& Scenario()
    {
        static ChannelScenario scenario;
        return scenario;
    }
    std::vector<std::vector<uint64_t>>& LaunchRecords()
    {
        static std::vector<std::vector<uint64_t>> records;
        return records;
    }
    // 端点属性编码于 localEndpoint.raws：[0..3]=dieId，[4..7]=bwCoeff
    void EncodeEndpoint(EndpointDesc& endpoint, uint32_t dieId, uint32_t bwCoeff)
    {
        uint32_t attrs[2] = {dieId, bwCoeff};
        (void)memcpy(endpoint.raws, attrs, sizeof(attrs));
    }
} // namespace

namespace ut_stub {
    void InjectChannelScenario(bool dualDie, uint32_t die0Bw, uint32_t die1Bw)
    {
        Scenario().dualDie = dualDie;
        Scenario().die0Bw = die0Bw;
        Scenario().die1Bw = die1Bw;
    }

    // 负向：组外 rank 逐位失配
    void InjectMisorderedPeers(const std::vector<uint32_t>& peers)
    {
        Scenario().dualDie = true;
        Scenario().peers = peers;
    }

    // 负向：die1 桶缺对端
    void InjectDie1MissingPeer(uint32_t peer)
    {
        Scenario().dualDie = true;
        Scenario().die1MissingPeer = peer;
    }

    void ResetChannelScenario() { Scenario() = ChannelScenario{}; }

    uint32_t ChannelDieOf(const HcclChannelDesc& channel)
    {
        uint32_t attrs[2] = {0, 0};
        (void)memcpy(attrs, channel.localEndpoint.raws, sizeof(attrs));
        return attrs[0];
    }

    uint32_t ChannelBwOf(const HcclChannelDesc& channel)
    {
        uint32_t attrs[2] = {0, 0};
        (void)memcpy(attrs, channel.localEndpoint.raws, sizeof(attrs));
        return attrs[1];
    }

    void ResetLaunchRecords() { LaunchRecords().clear(); }

    size_t LaunchRecordCount() { return LaunchRecords().size(); }

    const std::vector<uint64_t>& LaunchRecordArgs(size_t idx) { return LaunchRecords().at(idx); }

    // launch 记录桩（强定义覆盖 ccu_launch_dl 弱符号）
    extern "C" CcuResult HcommCcuKernelLaunch(
        ThreadHandle threadHandle, CcuKernelHandle kernelHandle, const void* taskArgs, uint32_t argSize)
    {
        (void)threadHandle;
        (void)kernelHandle;
        std::vector<uint64_t> record(argSize, 0);
        if (taskArgs != nullptr && argSize > 0) {
            (void)memcpy(record.data(), taskArgs, static_cast<size_t>(argSize) * sizeof(uint64_t));
        }
        LaunchRecords().push_back(std::move(record));
        return CCU_SUCCESS;
    }
} // namespace ut_stub

HcclResult CalcChannelRequestMesh1D(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
    (void)comm;
    (void)param;
    channels.clear();
    if (subcommInfo.empty() || subcommInfo[0].empty()) {
        return HcclResult::HCCL_SUCCESS;
    }
    u32 myRank = (topoInfo != nullptr) ? topoInfo->userRank : 0;
    const std::vector<u32>& peerList = Scenario().peers.empty() ? subcommInfo[0] : Scenario().peers;
    for (u32 rank : peerList) {
        if (rank == myRank) {
            continue;
        }
        HcclChannelDesc die0Desc{};
        die0Desc.remoteRank = rank;
        EncodeEndpoint(die0Desc.localEndpoint, 0, Scenario().die0Bw);
        channels.push_back(die0Desc);
        if (Scenario().dualDie) {
            u32 secondDie = (rank == Scenario().die1MissingPeer) ? 0 : 1;
            uint32_t secondBw = (secondDie == 1) ? Scenario().die1Bw : Scenario().die0Bw;
            HcclChannelDesc secondDesc{};
            secondDesc.remoteRank = rank;
            EncodeEndpoint(secondDesc.localEndpoint, secondDie, secondBw);
            channels.push_back(secondDesc);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

// hcomm 弱原语桩（KernelRun 路径）
HcclResult PreSyncInterThreads(
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxMainToSub)
{
    (void)mainThread;
    (void)subThreads;
    (void)notifyIdxMainToSub;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult PostSyncInterThreads(
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxSubToMain)
{
    (void)mainThread;
    (void)subThreads;
    (void)notifyIdxSubToMain;
    return HcclResult::HCCL_SUCCESS;
}

// ccu_res_dl.h 声明；GetToken 经此获取 token，强定义覆盖弱桩防 HCCL_COMPAT_ERROR 路径
extern "C" CcuResult HcommCcuGetMemToken(uint64_t srcVa, uint64_t size, uint64_t* token)
{
    (void)srcVa;
    (void)size;
    if (token != nullptr) {
        *token = 0;
    }
    return CCU_SUCCESS;
}
} // namespace ops_hccl

// dev_type.cc 实现，桩返回固定型号
HcclResult HcclGetDeviceType(HcclDevType& devType)
{
    devType = HcclDevType::DEV_TYPE_950;
    return HCCL_SUCCESS;
}

// 桩走 V1 路径
extern "C" bool HcommIsSupportCcuV2(void) { return false; }

// HCOMM 强符号，桩返回空链路表
HcclResult HcclRankGraphGetLinks(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize)
{
    (void)comm;
    (void)netLayer;
    (void)srcRank;
    (void)dstRank;
    if (linkList != nullptr) {
        *linkList = nullptr;
    }
    if (listSize != nullptr) {
        *listSize = 0;
    }
    return HCCL_SUCCESS;
}

// 端点属性桩：按 raws 编码返回 dieId/bwCoeff
extern "C" HcclResult HcclRankGraphGetEndpointInfo(
    HcclComm comm, uint32_t rankId, const EndpointDesc* endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen,
    void* info)
{
    (void)comm;
    (void)rankId;
    (void)infoLen;
    if (endpointDesc == nullptr || info == nullptr) {
        return HCCL_E_PTR;
    }
    uint32_t attrs[2] = {0, 0};
    (void)memcpy(attrs, endpointDesc->raws, sizeof(attrs));
    if (endpointAttr == ENDPOINT_ATTR_DIE_ID) {
        *static_cast<uint32_t*>(info) = attrs[0];
    } else if (endpointAttr == ENDPOINT_ATTR_BW_COEFF) {
        *static_cast<uint32_t*>(info) = attrs[1];
    } else {
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

// hccl_sym_win_dl.h 声明，桩返回空指针
HcclResult GetSymWinRemoteMem(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    (void)winHandle;
    (void)offset;
    (void)peerRank;
    if (ptr != nullptr) {
        *ptr = nullptr;
    }
    return HCCL_SUCCESS;
}

/* strcpy_s 桩（UT 不链接 libc_sec）：空指针/长度不足返回 EINVAL/ERANGE。 */
extern "C" errno_t strcpy_s(char* dest, size_t numberOfElements, const char* src)
{
    if (dest == nullptr || src == nullptr) {
        return EINVAL;
    }
    size_t len = strlen(src) + 1;
    if (numberOfElements < len) {
        return ERANGE;
    }
    (void)memcpy(dest, src, len);
    return EOK;
}
