/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志和对称窗口桩满足链接依赖，建链和通信原语桩用于验证资源及同步顺序。
#include <algorithm>
#include "log.h"
#include "hccl_sym_win_dl.h"
#include "channel.h"
#include "alg_data_trans_wrapper.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "barrier_rhd_sync_stub.h"

namespace barrier_rhd_sync_ut {
std::vector<std::string> calls;
int failSyncCall = 0;
int failSendCall = 0;
int failDpuSendCall = 0;
uint32_t responseMsgId = REQUEST_MSG_ID;
bool failBatchEnd = false;
bool failBatchStart = false;
bool useHostRemoteEndpoint = false;

void Reset()
{
    calls.clear();
    failSyncCall = 0;
    failSendCall = 0;
    failDpuSendCall = 0;
    responseMsgId = REQUEST_MSG_ID;
    failBatchEnd = false;
    failBatchStart = false;
    useHostRemoteEndpoint = false;
}
} // namespace barrier_rhd_sync_ut

extern "C" {
int32_t HcommBatchModeEnd(const char*)
{
    barrier_rhd_sync_ut::calls.push_back("end");
    return barrier_rhd_sync_ut::failBatchEnd ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}

int32_t HcommBatchModeStart(const char*)
{
    barrier_rhd_sync_ut::calls.push_back("start");
    return barrier_rhd_sync_ut::failBatchStart ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}

int32_t HcommThreadSynchronize(ThreadHandle)
{
    auto& calls = barrier_rhd_sync_ut::calls;
    calls.push_back("sync");
    const int syncCall = static_cast<int>(std::count(calls.begin(), calls.end(), "sync"));
    return syncCall == barrier_rhd_sync_ut::failSyncCall ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}

int32_t HcommSendRequest(uint64_t, const char*, const void*, size_t, uint32_t* msgId)
{
    barrier_rhd_sync_ut::calls.push_back("request");
    *msgId = barrier_rhd_sync_ut::REQUEST_MSG_ID;
    return HCCL_SUCCESS;
}

int32_t HcommWaitResponse(uint64_t, void*, size_t, uint32_t* msgId)
{
    barrier_rhd_sync_ut::calls.push_back("response");
    *msgId = barrier_rhd_sync_ut::responseMsgId;
    return HCCL_SUCCESS;
}
} // extern "C"

// ============ log 门控 ============

bool HcclCheckLogLevel(int logType, int moduleId)
{
    (void)logType;
    (void)moduleId;
    return false;
}

bool IsErrorToWarn() { return false; }

// ============ HCOMM 对称窗口（本 UT 不触达该路径） ============

HcclResult GetSymWinRemoteMem(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    (void)winHandle;
    (void)offset;
    (void)peerRank;
    (void)ptr;
    return HcclResult::HCCL_E_INTERNAL;
}

namespace ops_hccl {
HcclResult SendRecvWrite(const SendRecvInfo& info, const ThreadHandle&)
{
    auto& calls = barrier_rhd_sync_ut::calls;
    calls.push_back("send:" + std::to_string(info.sendRecvChannels_.txChannel_.remoteRank));
    const int sendCall = static_cast<int>(std::count_if(calls.begin(), calls.end(), [](const std::string& call) {
        return call.rfind("send:", 0) == 0;
    }));
    return sendCall == barrier_rhd_sync_ut::failSendCall ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}

HcclResult DpuSendRecvWrite(const SendRecvInfo& info, ThreadHandle thread, bool isDrainSupported)
{
    auto& calls = barrier_rhd_sync_ut::calls;
    calls.push_back(
        "dpu:" + std::to_string(info.sendRecvChannels_.txChannel_.remoteRank) + ":" + std::to_string(thread) + ":"
        + std::to_string(isDrainSupported));
    const int sendCall = static_cast<int>(std::count_if(calls.begin(), calls.end(), [](const std::string& call) {
        return call.rfind("dpu:", 0) == 0;
    }));
    return sendCall == barrier_rhd_sync_ut::failDpuSendCall ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}

HcclResult
CreateChannelRequestByRankId(HcclComm, const OpParam&, u32, u32 remoteRank, std::vector<HcclChannelDesc>& channels, u32)
{
    HcclChannelDesc channel{};
    channel.remoteRank = remoteRank;
    channel.localEndpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    channel.remoteEndpoint.loc.locType
        = barrier_rhd_sync_ut::useHostRemoteEndpoint ? ENDPOINT_LOC_TYPE_HOST : ENDPOINT_LOC_TYPE_DEVICE;
    channels = {channel};
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMesh1DHighestHostRoce(
    HcclComm, const OpParam&, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
    channels.clear();
    if (subcommInfo.size() != 1) {
        return HCCL_E_PARA;
    }
    const auto& group = subcommInfo.front();
    if (std::find(group.begin(), group.end(), topoInfo->userRank) == group.end()) {
        return HCCL_E_PARA;
    }
    for (u32 remoteRank : group) {
        if (remoteRank == topoInfo->userRank) {
            continue;
        }
        HcclChannelDesc channel{};
        channel.remoteRank = remoteRank;
        channel.localEndpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        channel.remoteEndpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
