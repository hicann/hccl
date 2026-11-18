/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_gate_stub.h"

#include <cstdint>

namespace {
uint32_t gNotifyRecordCallCnt = 0;
uint64_t gLastNotifySrcThread = 0;
uint64_t gLastNotifyDstThread = 0;
uint32_t gLastNotifyIdx = 0;
} // namespace

// 强定义覆盖 hcomm_primitives_dl.cc 中 DEFINE_WEAK_FUNC 的弱实现（弱实现返回 -1）：
// 桩返回成功并记录调用次数与最近一次句柄，供守卫"恰好记录一次"断言使用。
// 注意参数用 uint64_t：符号绑定只看符号名，类型不参与；避免依赖 9.0.0+ 才有的 ThreadHandle 定义
extern "C" {
int32_t HcommThreadNotifyRecordOnThread(uint64_t thread, uint64_t dstThread, uint32_t dstNotifyIdx)
{
    gNotifyRecordCallCnt++;
    gLastNotifySrcThread = thread;
    gLastNotifyDstThread = dstThread;
    gLastNotifyIdx = dstNotifyIdx;
    return 0;
}

// 源包含 kernel_launch.cc 引入的其余 hcomm 外部符号（生产环境由 libhcomm.so 经 dlsym 提供）：
// 桩返回成功空操作，语义同 ST 的 hccl_proxy/hccl_stub.cc，仅供链接闭包
int32_t HcommAcquireComm(const char* commId)
{
    (void)commId;
    return 0;
}

int32_t HcommReleaseComm(const char* commId)
{
    (void)commId;
    return 0;
}

int32_t HcommBatchModeStart(const char* batchTag)
{
    (void)batchTag;
    return 0;
}

int32_t HcommBatchModeEnd(const char* batchTag)
{
    (void)batchTag;
    return 0;
}

int32_t HcommAclrtNotifyRecordOnThread(uint64_t thread, uint64_t dstNotifyId)
{
    (void)thread;
    (void)dstNotifyId;
    return 0;
}

int32_t HcommAclrtNotifyWaitOnThread(uint64_t thread, uint64_t notifyId, uint32_t timeOut)
{
    (void)thread;
    (void)notifyId;
    (void)timeOut;
    return 0;
}

int32_t HcommLocalCopyOnThread(uint64_t thread, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    return 0;
}

int32_t HcommWriteOnThread(uint64_t thread, uint64_t channel, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)len;
    return 0;
}

int32_t HcommReadOnThread(uint64_t thread, uint64_t channel, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)len;
    return 0;
}

int32_t HcommWriteReduceOnThread(
    uint64_t thread, uint64_t channel, void* dst, const void* src, uint64_t len, uint32_t reduceOp, uint32_t dataType)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)len;
    (void)reduceOp;
    (void)dataType;
    return 0;
}

int32_t HcommLocalReduceOnThread(
    uint64_t thread, void* dst, const void* src, uint64_t len, uint32_t reduceOp, uint32_t dataType)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    (void)reduceOp;
    (void)dataType;
    return 0;
}
}

namespace ops_hccl {
// topo.cc 依赖的外部输入配置查询（生产环境读全局环境配置），桩返回默认关闭
bool GetExternalInputIntraRoceSwitch() { return false; }

bool GetExternalInputInterHccsDisable() { return false; }
} // namespace ops_hccl

void GateStubResetNotifyRecordCnt()
{
    gNotifyRecordCallCnt = 0;
    gLastNotifySrcThread = 0;
    gLastNotifyDstThread = 0;
    gLastNotifyIdx = 0;
}

uint32_t GateStubGetNotifyRecordCnt() { return gNotifyRecordCallCnt; }

uint64_t GateStubGetLastNotifySrcThread() { return gLastNotifySrcThread; }

uint64_t GateStubGetLastNotifyDstThread() { return gLastNotifyDstThread; }

uint32_t GateStubGetLastNotifyIdx() { return gLastNotifyIdx; }
