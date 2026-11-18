/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NDA_TEST_ACCESS_H
#define NDA_TEST_ACCESS_H

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

// 被测模板的私有方法通过 -fno-access-control 放开访问权限（对齐 reduce_scatter_nhr_dpu UT）
#include "ins_temp_dpu_alltoall_mesh_nda.h"

namespace nda_ut {
using namespace ops_hccl;

// LocalCopy 桩记录的一次本地拷贝
struct LocalCopyCall {
    ThreadHandle thread = 0;
    DataSlice src{nullptr, 0, 0, 0};
    DataSlice dst{nullptr, 0, 0, 0};
};

std::vector<LocalCopyCall>& LocalCopyCalls();

// 传输编排桩记录的一次 Send/Recv/SendRecv 下发（区分 NDA 与普通写语义）
enum class WriteKind {
    kSendRecvWrite,
    kSendRecvWriteNda,
    kSendWrite,
    kSendWriteNda,
    kRecvWrite,
    kRecvWriteNda,
};

struct WriteCall {
    WriteKind kind = WriteKind::kSendRecvWrite;
    ThreadHandle thread = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    std::vector<DataSlice> txSrc;
    std::vector<DataSlice> txDst;
    std::vector<DataSlice> rxSrc;
    std::vector<DataSlice> rxDst;
};

std::vector<WriteCall>& WriteCalls();

// NDA 通道级握手原语桩记录（NotifyRecord / NotifyWait / WriteWithNotify），用于断言握手时序
struct HandshakeCall {
    enum Kind { kNotifyRecord, kNotifyWait, kWriteWithNotify } kind = kNotifyRecord;
    ThreadHandle thread = 0;
    ChannelHandle channel = 0;
    u32 notifyIdx = 0;
    u32 timeout = 0;
    void* dst = nullptr;
    const void* src = nullptr;
    u64 len = 0;
};

std::vector<HandshakeCall>& HandshakeCalls();

// batch 模式窗口桩记录（Start/End 次数与 tag）
struct BatchCall {
    bool start = false;
    std::string tag;
};

std::vector<BatchCall>& BatchCalls();

// 可注入返回值：用于覆盖 SendRecvNda / SendRecvNdaMesh 的错误传播路径
void SetNotifyRecordResult(s32 result);
void SetNotifyWaitResult(s32 result);
void SetWriteWithNotifyResult(s32 result);
void SetSendRecvWriteResult(s32 result);
void SetBatchModeResult(s32 result);
void ResetStubState();

// 构造测试用 ChannelInfo
inline ChannelInfo MakeChannel(
    u32 remoteRank, CommProtocol protocol, EndpointLocType locationType, ChannelHandle handle = 0,
    void* remoteAddr = nullptr, u64 remoteSize = 0)
{
    ChannelInfo channel{};
    channel.remoteRank = remoteRank;
    channel.protocol = protocol;
    channel.locationType = locationType;
    channel.handle = handle;
    channel.remoteCclMem.addr = remoteAddr;
    channel.remoteCclMem.size = remoteSize;
    return channel;
}

// 用 -fno-access-control 直接设置模板私有状态，绕过 KernelRun 的线程同步与资源申请路径，
// 聚焦 SendRecvNda / SendRecvNdaMesh / PreCopyDataToCclInBuf / PostCopyDataToRecvBuf 的纯逻辑。
inline void SetupTemplate(
    InsTempDpuAlltoAllMeshNda& tpl, u32 myRank, u32 threadNum, u64 dataTypeSize, u64 hcclbuffBlockMemSize,
    u64 halfMaxTmpMemSize)
{
    tpl.myRank_ = myRank;
    tpl.threadNum_ = threadNum;
    tpl.dataTypeSize_ = dataTypeSize;
    tpl.hcclbuffBlockMemSize_ = hcclbuffBlockMemSize;
    tpl.halfMaxTmpMemSize_ = halfMaxTmpMemSize;
    tpl.threads_.assign(threadNum, static_cast<ThreadHandle>(myRank + 1));
}

inline std::vector<u32> MakeRankIds(u32 rankSize)
{
    std::vector<u32> rankIds(rankSize);
    for (u32 i = 0; i < rankSize; ++i) {
        rankIds[i] = i;
    }
    return rankIds;
}

// 构造最小可用的 TemplateDataParams：sendCounts/recvCounts/sdispls/rdispls 按 rankSize 展开，
// buffInfo 使用测试注入的 input/output/hcclBuff 地址。
inline TemplateDataParams MakeParams(
    u32 rankSize, const std::vector<u64>& sendCounts, const std::vector<u64>& recvCounts,
    const std::vector<u64>& sdispls, const std::vector<u64>& rdispls, void* inputPtr, void* outputPtr,
    void* hcclBuffAddr, u64 hcclBuffBaseOff)
{
    TemplateDataParams params;
    params.sendCounts = sendCounts;
    params.recvCounts = recvCounts;
    params.sdispls = sdispls;
    params.rdispls = rdispls;
    params.buffInfo.inputPtr = inputPtr;
    params.buffInfo.outputPtr = outputPtr;
    params.buffInfo.hcclBuff.addr = hcclBuffAddr;
    params.buffInfo.hcclBuff.size = 1ull << 20;
    params.buffInfo.hcclBuffBaseOff = hcclBuffBaseOff;
    (void)rankSize;
    return params;
}

} // namespace nda_ut

#endif // NDA_TEST_ACCESS_H
