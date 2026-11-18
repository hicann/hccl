/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nda_test_access.h"

#include "alg_v2_template_base.h"
#include "alg_v2_template_register.h"
#include "alg_data_trans_wrapper.h"
#include "channel.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "hccl_sym_win_dl.h"

namespace {

std::vector<nda_ut::LocalCopyCall> g_localCopyCalls;
std::vector<nda_ut::WriteCall> g_writeCalls;
std::vector<nda_ut::HandshakeCall> g_handshakeCalls;
std::vector<nda_ut::BatchCall> g_batchCalls;

s32 g_notifyRecordResult = 0;
s32 g_notifyWaitResult = 0;
s32 g_writeWithNotifyResult = 0;
s32 g_sendRecvWriteResult = 0;
s32 g_batchModeResult = 0;

} // namespace

namespace nda_ut {

std::vector<LocalCopyCall>& LocalCopyCalls() { return g_localCopyCalls; }
std::vector<WriteCall>& WriteCalls() { return g_writeCalls; }
std::vector<HandshakeCall>& HandshakeCalls() { return g_handshakeCalls; }
std::vector<BatchCall>& BatchCalls() { return g_batchCalls; }

void SetNotifyRecordResult(s32 result) { g_notifyRecordResult = result; }
void SetNotifyWaitResult(s32 result) { g_notifyWaitResult = result; }
void SetWriteWithNotifyResult(s32 result) { g_writeWithNotifyResult = result; }
void SetSendRecvWriteResult(s32 result) { g_sendRecvWriteResult = result; }
void SetBatchModeResult(s32 result) { g_batchModeResult = result; }

void ResetStubState()
{
    g_localCopyCalls.clear();
    g_writeCalls.clear();
    g_handshakeCalls.clear();
    g_batchCalls.clear();
    g_notifyRecordResult = 0;
    g_notifyWaitResult = 0;
    g_writeWithNotifyResult = 0;
    g_sendRecvWriteResult = 0;
    g_batchModeResult = 0;
}

} // namespace nda_ut

namespace ops_hccl {

HcclResult LocalCopy(const ThreadHandle& thread, const DataSlice& srcSlice, const DataSlice& dstSlice)
{
    nda_ut::LocalCopyCall call;
    call.thread = thread;
    call.src = srcSlice;
    call.dst = dstSlice;
    nda_ut::LocalCopyCalls().push_back(call);
    return HCCL_SUCCESS;
}

// 与生产 alg_data_trans_wrapper.cc::IsContinuousSlice 保持逐字一致，保证 AppendCopy 的合并判定真实。
bool IsContinuousSlice(const DataSlice& nxtSlice, const DataSlice& currSlice)
{
    if (nxtSlice.addr_ != currSlice.addr_) {
        return false;
    }
    if (nxtSlice.offset_ != currSlice.offset_ + currSlice.size_) {
        return false;
    }
    return true;
}

void RecordWrite(
    nda_ut::WriteKind kind, const ThreadHandle& thread, HcclDataType dataType, const std::vector<DataSlice>& txSrc,
    const std::vector<DataSlice>& txDst, const std::vector<DataSlice>& rxSrc, const std::vector<DataSlice>& rxDst)
{
    nda_ut::WriteCall call;
    call.kind = kind;
    call.thread = thread;
    call.dataType = dataType;
    call.txSrc = txSrc;
    call.txDst = txDst;
    call.rxSrc = rxSrc;
    call.rxDst = rxDst;
    nda_ut::WriteCalls().push_back(call);
}

HcclResult SendRecvWrite(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread)
{
    RecordWrite(
        nda_ut::WriteKind::kSendRecvWrite, thread, sendRecvInfo.dataType_,
        sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_, sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_,
        sendRecvInfo.sendRecvSlices_.rxSlicesList_.srcSlices_, sendRecvInfo.sendRecvSlices_.rxSlicesList_.dstSlices_);
    return static_cast<HcclResult>(g_sendRecvWriteResult);
}

HcclResult SendRecvWriteForNda(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread)
{
    RecordWrite(
        nda_ut::WriteKind::kSendRecvWriteNda, thread, sendRecvInfo.dataType_,
        sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_, sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_,
        sendRecvInfo.sendRecvSlices_.rxSlicesList_.srcSlices_, sendRecvInfo.sendRecvSlices_.rxSlicesList_.dstSlices_);
    return static_cast<HcclResult>(g_sendRecvWriteResult);
}

HcclResult SendWrite(const DataInfo& sendInfo, const ThreadHandle& thread)
{
    RecordWrite(
        nda_ut::WriteKind::kSendWrite, thread, sendInfo.dataType_, sendInfo.slices_.srcSlices_,
        sendInfo.slices_.dstSlices_, {}, {});
    return static_cast<HcclResult>(g_sendRecvWriteResult);
}

HcclResult SendWriteForNda(const DataInfo& sendInfo, const ThreadHandle& thread)
{
    RecordWrite(
        nda_ut::WriteKind::kSendWriteNda, thread, sendInfo.dataType_, sendInfo.slices_.srcSlices_,
        sendInfo.slices_.dstSlices_, {}, {});
    return static_cast<HcclResult>(g_sendRecvWriteResult);
}

HcclResult RecvWrite(const DataInfo& recvInfo, const ThreadHandle& thread)
{
    RecordWrite(
        nda_ut::WriteKind::kRecvWrite, thread, recvInfo.dataType_, {}, {}, recvInfo.slices_.srcSlices_,
        recvInfo.slices_.dstSlices_);
    return static_cast<HcclResult>(g_sendRecvWriteResult);
}

HcclResult RecvWriteForNda(const DataInfo& recvInfo, const ThreadHandle& thread)
{
    RecordWrite(
        nda_ut::WriteKind::kRecvWriteNda, thread, recvInfo.dataType_, {}, {}, recvInfo.slices_.srcSlices_,
        recvInfo.slices_.dstSlices_);
    return static_cast<HcclResult>(g_sendRecvWriteResult);
}

HcclResult PreSyncInterThreads(
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxMainToSub)
{
    (void)mainThread;
    (void)subThreads;
    (void)notifyIdxMainToSub;
    return HCCL_SUCCESS;
}

HcclResult PostSyncInterThreads(
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxSubToMain)
{
    (void)mainThread;
    (void)subThreads;
    (void)notifyIdxSubToMain;
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMesh1D(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMesh1DWithPriorityTopo(
    HcclComm comm, const OpParam& param, const TopoInfo* topoInfo, const std::vector<std::vector<u32>>& subcommInfo,
    std::vector<HcclChannelDesc>& channels, CommTopo priorityTopo)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    (void)priorityTopo;
    return HCCL_SUCCESS;
}

InsAlgTemplateRegistry& InsAlgTemplateRegistry::Instance()
{
    static InsAlgTemplateRegistry instance;
    return instance;
}

HcclResult InsAlgTemplateRegistry::Register(const std::string& name, const InsAlgTemplateCreator& creator)
{
    (void)name;
    (void)creator;
    return HCCL_SUCCESS;
}

std::unique_ptr<InsAlgTemplateBase> InsAlgTemplateRegistry::GetAlgTemplate(const std::string& name)
{
    (void)name;
    return nullptr;
}

} // namespace ops_hccl

// template_utils.cc 调用 GetSymWinRemoteMem；UT 独立编译时以桩提供定义。
HcclResult GetSymWinRemoteMem(HcclCommSymWindow, size_t, uint32_t, void**)
{
    ADD_FAILURE() << "unexpected symmetric window lookup";
    return HCCL_E_NOT_SUPPORT;
}

extern "C" {

int32_t HcommBatchModeStart(const char* batchTag)
{
    nda_ut::BatchCall call;
    call.start = true;
    call.tag = batchTag == nullptr ? "" : batchTag;
    nda_ut::BatchCalls().push_back(call);
    return g_batchModeResult;
}

int32_t HcommBatchModeEnd(const char* batchTag)
{
    nda_ut::BatchCall call;
    call.start = false;
    call.tag = batchTag == nullptr ? "" : batchTag;
    nda_ut::BatchCalls().push_back(call);
    return g_batchModeResult;
}

int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    nda_ut::HandshakeCall call;
    call.kind = nda_ut::HandshakeCall::kNotifyRecord;
    call.thread = thread;
    call.channel = channel;
    call.notifyIdx = remoteNotifyIdx;
    nda_ut::HandshakeCalls().push_back(call);
    return g_notifyRecordResult;
}

int32_t
HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
{
    nda_ut::HandshakeCall call;
    call.kind = nda_ut::HandshakeCall::kNotifyWait;
    call.thread = thread;
    call.channel = channel;
    call.notifyIdx = localNotifyIdx;
    call.timeout = timeOut;
    nda_ut::HandshakeCalls().push_back(call);
    return g_notifyWaitResult;
}

int32_t HcommWriteWithNotifyOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    nda_ut::HandshakeCall call;
    call.kind = nda_ut::HandshakeCall::kWriteWithNotify;
    call.thread = thread;
    call.channel = channel;
    call.dst = dst;
    call.src = src;
    call.len = len;
    call.notifyIdx = remoteNotifyIdx;
    nda_ut::HandshakeCalls().push_back(call);
    return g_writeWithNotifyResult;
}

} // extern "C"
