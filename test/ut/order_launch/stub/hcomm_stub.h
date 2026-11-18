/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_ORDER_LAUNCH_HCOMM_STUB_H
#define UT_ORDER_LAUNCH_HCOMM_STUB_H

#include <cstdint>
#include <utility>
#include <vector>

// 桩状态：强符号覆写生产弱默认；旋钮在 SetUp 中 Reset，用例按需注入失败/句柄。
namespace OrderLaunchUtStub {
struct AddToModelCall {
    void* stream;
    void* model;
};

extern std::vector<AddToModelCall> g_addCalls;
extern int32_t g_notifyWaitCalls;
extern int32_t g_notifyRecordCalls;
extern int32_t g_acqRet;
extern uint64_t g_acqThread;
extern uint64_t g_deviceAcqThread; // DEVICE 类专用线程句柄（默认与 g_acqThread 同值）
extern bool g_supportDedicated;
extern int32_t g_captureRet;
extern int32_t g_captureStatus;
extern void* g_rtModel;
extern int32_t g_threadResRet;
extern void* g_threadResStream;
extern int32_t g_addRet;
extern int32_t g_notifyRecordRet;
extern int32_t g_notifyWaitRet;
extern int32_t g_exportRet;
// 顺序锚点：共享单调调用序号（自 1 起，0 = 未调用），固化「先入图、后握手」。
extern uint32_t g_callSeq;
extern uint32_t g_addSeq;               // 首次入图的序号
extern uint32_t g_firstNotifyRecordSeq; // 首次 notify record 的序号
extern uint32_t g_lastNotifyWaitSeq;    // 最近一次 notify wait 的序号

void Reset();
} // namespace OrderLaunchUtStub

#endif // UT_ORDER_LAUNCH_HCOMM_STUB_H
