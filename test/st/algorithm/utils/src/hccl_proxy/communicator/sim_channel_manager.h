/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_CHANNEL_MANAGER_H
#define SIM_CHANNEL_MANAGER_H

#include <memory>
#include <unordered_map>
#include "hccl_sim_pub.h"
#include "sim_channel.h"

// 解决与Hcomm仓合入问题，暂时定义为弱符号
#ifndef HCCL_CHANNEL_ABI
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
const uint32_t HCCL_CHANNEL_MAGIC_WORD = 0x0f0f0f0f;
const uint32_t HCCL_CHANNEL_VERSION = 1;

/**
 * @brief 兼容Abi字段结构体
 */
typedef struct {
    uint32_t version;
    uint32_t magicWord;
    uint32_t size;
    uint32_t reserved;
} HcclAbiHeader;

typedef struct {
    HcclAbiHeader header;
    uint32_t remoteRank;    ///< 远端rankId
    CommProtocol protocol;  ///< 通信协议
    uint32_t notifyNum;  ///< channel上使用的通知消息数量
    union {
        HccsAttr hccsAttr;
        RoCEAttr roceAttr;
        UbAttr ubAttr;
    };
} HcclChannelDesc;

inline void HcclChannelDescInit(HcclChannelDesc *channelDesc, uint32_t descNum)
{
    for (uint32_t idx = 0; idx < descNum; idx++) {
        if (channelDesc != nullptr) {
            // Abi字段初始化
            channelDesc->header.version     = HCCL_CHANNEL_VERSION;
            channelDesc->header.magicWord   = HCCL_CHANNEL_MAGIC_WORD;
            channelDesc->header.size        = sizeof(HcclChannelDesc);
            channelDesc->header.reserved    = 0;

            // HcclChannelDesc内容初始化
            channelDesc->remoteRank = ~0U;
            channelDesc->protocol   = COMM_PROTOCOL_RESERVED;
            channelDesc->notifyNum  = 0;
            (void)memset_s(&(channelDesc->hccsAttr), sizeof(HccsAttr), 0, sizeof(HccsAttr));
            (void)memset_s(&(channelDesc->roceAttr), sizeof(RoCEAttr), 0, sizeof(RoCEAttr));
            (void)memset_s(&(channelDesc->ubAttr), sizeof(UbAttr), 0, sizeof(UbAttr));
        }
    }
    return;
}

HcclResult HcclChannelAcquire(HcclComm comm, CommEngine engine, const HcclChannelDesc *channelDescList,
    uint32_t listNum, ChannelHandle *channelList) __attribute__((weak));

HcclResult HcommAclrtNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId) __attribute__((weak));
HcclResult HcommAclrtNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut) __attribute__((weak));

HcclResult HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx) __attribute__((weak));
HcclResult HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout) __attribute__((weak));

HcclResult HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx) __attribute__((weak));
HcclResult HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeout) __attribute__((weak));
#ifdef __cplusplus
}
#endif  // __cplusplus
#endif

namespace HcclSim {

class SimChannelMgr {
public:
    static std::string GetChannelKey(std::shared_ptr<SimChannel> channel);

    SimChannelMgr(std::string commId, uint32_t curRank) : commId_(commId), curRank_(curRank) {};
    ~SimChannelMgr() = default;

    HcclResult ChannelCommCreate(const std::string &commId, const std::string &tag, CommEngine engine, 
        const HcclChannelDesc *channelDescList, uint32_t listNum, ChannelHandle *channelList);

private:
    std::string commId_;
    uint32_t curRank_;

    std::unordered_map<std::string, std::shared_ptr<SimChannel>> channelMap_;
};

};
#endif  // SIM_CHANNEL_MANAGER_H