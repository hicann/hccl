/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "hcomm_device_profiling_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针
HcclResult (*hcommProfilingReportMainStreamAndFirstTaskPtr)(ThreadHandle) = nullptr;
HcclResult (*hcommProfilingReportMainStreamAndLastTaskPtr)(ThreadHandle) = nullptr;
HcclResult (*hcommProfilingReportDeviceHcclOpInfoPtr)(HcomProInfoTmp) = nullptr;
HcclResult (*hcommProfilingInitPtr)(ThreadHandle*, uint32_t) = nullptr;
HcclResult (*hcommProfilingEndPtr)(ThreadHandle*, uint32_t) = nullptr;
HcclResult (*hcommProfilingReportDeviceOpPtr)(const char* groupname) = nullptr;
HcclResult (*hcommProfilingReportKernelStartTaskPtr)(uint64_t thread, const char* groupname) = nullptr;
HcclResult (*hcommProfilingReportKernelEndTaskPtr)(uint64_t thread, const char* groupname) = nullptr;

// 添加支持标志（静态，默认 false）
static bool g_hcommProfilingReportMainStreamAndFirstTaskSupported = false;
static bool g_hcommProfilingReportMainStreamAndLastTaskSupported = false;
static bool g_hcommProfilingReportDeviceHcclOpInfoSupported = false;
static bool g_hcommProfilingInitSupported = false;
static bool g_hcommProfilingEndSupported = false;

static bool g_hcommProfilingReportDeviceOpSupported = false;
static bool g_hcommProfilingReportKernelStartTaskSupported = false;
static bool g_hcommProfilingReportKernelEndTaskSupported = false;

// ---------- 桩函数定义 ----------
static HcclResult StubHcommProfilingReportMainStreamAndFirstTask(ThreadHandle thread) {
    (void)thread;
    HCCL_ERROR("[HcclWrapper] HcommProfilingReportMainStreamAndFirstTask not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingReportMainStreamAndLastTask(ThreadHandle thread) {
    (void)thread;
    HCCL_ERROR("[HcclWrapper] HcommProfilingReportMainStreamAndLastTask not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingReportDeviceHcclOpInfo(HcomProInfoTmp profInfo) {
    (void)profInfo;
    HCCL_ERROR("[HcclWrapper] HcommProfilingReportDeviceHcclOpInfo not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingInit(ThreadHandle* threads, uint32_t threadNum) {
    (void)threads; (void)threadNum;
    HCCL_ERROR("[HcclWrapper] HcommProfilingInit not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingEnd(ThreadHandle* threads, uint32_t threadNum) {
    (void)threads; (void)threadNum;
    HCCL_ERROR("[HcclWrapper] HcommProfilingEnd not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcommProfilingReportDeviceOp(const char* groupname)
{
    (void)groupname;
    HCCL_ERROR("[HcclWrapper] HcommProfilingReportDeviceOp not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingReportKernelStartTask(uint64_t thread, const char* groupname)
{
    (void)thread; (void)groupname;
    HCCL_ERROR("[HcclWrapper] HcommProfilingReportKernelStartTask not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingReportKernelEndTask(uint64_t thread, const char* groupname)
{
    (void)thread; (void)groupname;
    HCCL_ERROR("[HcclWrapper] HcommProfilingReportKernelEndTask not supported");
    return HCCL_E_NOT_SUPPORTED;
}


// 初始化
void HcommDeviceProfilingDlInit(void* libHcommHandle) {
    #define SET_PTR(ptr, handle, name, stub, support_flag) \
        do { \
            ptr = (decltype(ptr))dlsym(handle, name); \
            if (ptr == nullptr) { \
                ptr = stub; \
                support_flag = false; \
                HCCL_DEBUG("[HcclWrapper] %s not supported", name); \
            } else { \
                support_flag = true; \
            } \
        } while(0)

    SET_PTR(hcommProfilingReportMainStreamAndFirstTaskPtr, libHcommHandle, "HcommProfilingReportMainStreamAndFirstTask",
            StubHcommProfilingReportMainStreamAndFirstTask, g_hcommProfilingReportMainStreamAndFirstTaskSupported);
    SET_PTR(hcommProfilingReportMainStreamAndLastTaskPtr, libHcommHandle, "HcommProfilingReportMainStreamAndLastTask",
            StubHcommProfilingReportMainStreamAndLastTask, g_hcommProfilingReportMainStreamAndLastTaskSupported);
    SET_PTR(hcommProfilingReportDeviceHcclOpInfoPtr, libHcommHandle, "HcommProfilingReportDeviceHcclOpInfo",
            StubHcommProfilingReportDeviceHcclOpInfo, g_hcommProfilingReportDeviceHcclOpInfoSupported);
    SET_PTR(hcommProfilingInitPtr, libHcommHandle, "HcommProfilingInit",
            StubHcommProfilingInit, g_hcommProfilingInitSupported);
    SET_PTR(hcommProfilingEndPtr, libHcommHandle, "HcommProfilingEnd",
            StubHcommProfilingEnd, g_hcommProfilingEndSupported);
    SET_PTR(hcommProfilingReportDeviceOpPtr, libHcommHandle, "HcommProfilingReportDeviceOp",
        StubHcommProfilingReportDeviceOp, g_hcommProfilingReportDeviceOpSupported);
    SET_PTR(hcommProfilingReportKernelStartTaskPtr, libHcommHandle, "HcommProfilingReportKernelStartTask",
        StubHcommProfilingReportKernelStartTask, g_hcommProfilingReportKernelStartTaskSupported);
    SET_PTR(hcommProfilingReportKernelEndTaskPtr, libHcommHandle, "HcommProfilingReportKernelEndTask",
        StubHcommProfilingReportKernelEndTask, g_hcommProfilingReportKernelEndTaskSupported);

    #undef SET_PTR
}

void HcommDeviceProfilingDlFini(void) {
    hcommProfilingReportMainStreamAndFirstTaskPtr = StubHcommProfilingReportMainStreamAndFirstTask;
    g_hcommProfilingReportMainStreamAndFirstTaskSupported = false;
    hcommProfilingReportMainStreamAndLastTaskPtr = StubHcommProfilingReportMainStreamAndLastTask;
    g_hcommProfilingReportMainStreamAndLastTaskSupported = false;
    hcommProfilingReportDeviceHcclOpInfoPtr = StubHcommProfilingReportDeviceHcclOpInfo;
    g_hcommProfilingReportDeviceHcclOpInfoSupported = false;
    hcommProfilingInitPtr = StubHcommProfilingInit;
    g_hcommProfilingInitSupported = false;
    hcommProfilingEndPtr = StubHcommProfilingEnd;
    g_hcommProfilingEndSupported = false;
    hcommProfilingReportDeviceOpPtr = StubHcommProfilingReportDeviceOp;
    g_hcommProfilingReportDeviceOpSupported = false;
    hcommProfilingReportKernelStartTaskPtr = StubHcommProfilingReportKernelStartTask;
    g_hcommProfilingReportKernelStartTaskSupported = false;
    hcommProfilingReportKernelEndTaskPtr = StubHcommProfilingReportKernelEndTask;
    g_hcommProfilingReportKernelEndTaskSupported = false;
}

// ---------- 对外提供的查询接口 ----------
extern "C" bool HcommIsSupportHcommProfilingReportMainStreamAndFirstTask(void) {
    return g_hcommProfilingReportMainStreamAndFirstTaskSupported;
}
extern "C" bool HcommIsSupportHcommProfilingReportMainStreamAndLastTask(void) {
    return g_hcommProfilingReportMainStreamAndLastTaskSupported;
}
extern "C" bool HcommIsSupportHcommProfilingReportDeviceHcclOpInfo(void) {
    return g_hcommProfilingReportDeviceHcclOpInfoSupported;
}
extern "C" bool HcommIsSupportHcommProfilingInit(void) {
    return g_hcommProfilingInitSupported;
}
extern "C" bool HcommIsSupportHcommProfilingEnd(void) {
    return g_hcommProfilingEndSupported;
}
extern "C" bool HcommIsSupportHcommProfilingReportDeviceOp(void) {
    return g_hcommProfilingReportDeviceOpSupported;
}
extern "C" bool HcommIsSupportHcommProfilingReportKernelStartTask(void) {
    return g_hcommProfilingReportKernelStartTaskSupported;
}
extern "C" bool HcommIsSupportHcommProfilingReportKernelEndTaskSupported(void) {
    return g_hcommProfilingReportKernelEndTaskSupported;
}