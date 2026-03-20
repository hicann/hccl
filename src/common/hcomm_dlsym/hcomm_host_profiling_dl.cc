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
#include "hcomm_host_profiling_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针
HcclResult (*hcommProfilingRegThreadPtr)(HcomProInfoTmp, ThreadHandle*) = nullptr;
HcclResult (*hcommProfilingUnRegThreadPtr)(HcomProInfoTmp, ThreadHandle*) = nullptr;
HcclResult (*hcommProfilingReportKernelPtr)(uint64_t, const char*) = nullptr;
HcclResult (*hcommProfilingReportOpPtr)(HcomProInfoTmp) = nullptr;
uint64_t (*hcommGetProfilingSysCycleTimePtr)() = nullptr;
HcclResult (*hcclDfxRegOpInfoPtr)(HcclComm comm, void* dfxOpInfo) = nullptr;
HcclResult (*hcclProfilingReportOpPtr)(HcclComm comm, uint64_t beginTime) = nullptr;
HcclResult (*hcclReportAicpuKernelPtr)(HcclComm comm, uint64_t beginTime, char *kernelName) = nullptr;

// 添加支持标志（静态，默认 false）
static bool g_hcommProfilingRegThreadSupported = false;
static bool g_hcommProfilingUnRegThreadSupported = false;
static bool g_hcommProfilingReportKernelSupported = false;
static bool g_hcommProfilingReportOpSupported = false;
static bool g_hcommGetProfilingSysCycleTimeSupported = false;
static bool g_hcclDfxRegOpInfoSupported = false;
static bool g_hcclProfilingReportOpSupported = false;
static bool g_hcclReportAicpuKernelSupported = false;

// ---------- 桩函数定义 ----------
static HcclResult StubHcommProfilingRegThread(HcomProInfoTmp profInfo, ThreadHandle* threads) {
    (void)profInfo; (void)threads;
    HCCL_ERROR("[HcclWrapper] HcommProfilingRegThread not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingUnRegThread(HcomProInfoTmp profInfo, ThreadHandle* threads) {
    (void)profInfo; (void)threads;
    HCCL_ERROR("[HcclWrapper] HcommProfilingUnRegThread not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingReportKernel(uint64_t beginTime, const char* profName) {
    (void)beginTime; (void)profName;
    HCCL_ERROR("[HcclWrapper] HcommProfilingReportKernel not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingReportOp(HcomProInfoTmp profInfo) {
    (void)profInfo;
    HCCL_ERROR("[HcclWrapper] HcommProfilingReportOp not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static uint64_t StubHcommGetProfilingSysCycleTime() {
    HCCL_ERROR("[HcclWrapper] HcommGetProfilingSysCycleTime not supported");
    return 0;
}

static HcclResult StubHcclDfxRegOpInfo(HcclComm comm, void* dfxOpInfo)
{
    (void)comm; (void)dfxOpInfo;
    HCCL_ERROR("[HcclWrapper] StubHcclDfxRegOpInfo not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclProfilingReportOp(HcclComm comm, uint64_t beginTime)
{
    (void)comm; (void)beginTime;
    HCCL_ERROR("[HcclWrapper] HcclProfilingReportOp not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclReportAicpuKernel(HcclComm comm, uint64_t beginTime, char *kernelName)
{
    (void)comm; (void)beginTime; (void)kernelName;
    HCCL_ERROR("[HcclWrapper] HcclReportAicpuKernel not supported");
    return HCCL_E_NOT_SUPPORTED;
}

// 初始化
void HcommProfilingDlInit(void* libHcommHandle) {
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

    SET_PTR(hcommProfilingRegThreadPtr, libHcommHandle, "HcommProfilingRegThread", StubHcommProfilingRegThread, g_hcommProfilingRegThreadSupported);
    SET_PTR(hcommProfilingUnRegThreadPtr, libHcommHandle, "HcommProfilingUnRegThread", StubHcommProfilingUnRegThread, g_hcommProfilingUnRegThreadSupported);
    SET_PTR(hcommProfilingReportKernelPtr, libHcommHandle, "HcommProfilingReportKernel", StubHcommProfilingReportKernel, g_hcommProfilingReportKernelSupported);
    SET_PTR(hcommProfilingReportOpPtr, libHcommHandle, "HcommProfilingReportOp", StubHcommProfilingReportOp, g_hcommProfilingReportOpSupported);
    SET_PTR(hcommGetProfilingSysCycleTimePtr, libHcommHandle, "HcommGetProfilingSysCycleTime", StubHcommGetProfilingSysCycleTime, g_hcommGetProfilingSysCycleTimeSupported);
    SET_PTR(hcclDfxRegOpInfoPtr, libHcommHandle, "HcclDfxRegOpInfo", StubHcclDfxRegOpInfo, g_hcclDfxRegOpInfoSupported);
    SET_PTR(hcclProfilingReportOpPtr, libHcommHandle, "HcclProfilingReportOp", StubHcclProfilingReportOp, g_hcclProfilingReportOpSupported);
    SET_PTR(hcclReportAicpuKernelPtr, libHcommHandle, "HcclReportAicpuKernel", StubHcclReportAicpuKernel, g_hcclReportAicpuKernelSupported);

    #undef SET_PTR
}

void HcommProfilingDlFini(void) {
    hcommProfilingRegThreadPtr = StubHcommProfilingRegThread;
    g_hcommProfilingRegThreadSupported = false;
    hcommProfilingUnRegThreadPtr = StubHcommProfilingUnRegThread;
    g_hcommProfilingUnRegThreadSupported = false;
    hcommProfilingReportKernelPtr = StubHcommProfilingReportKernel;
    g_hcommProfilingReportKernelSupported = false;
    hcommProfilingReportOpPtr = StubHcommProfilingReportOp;
    g_hcommProfilingReportOpSupported = false;
    hcommGetProfilingSysCycleTimePtr = StubHcommGetProfilingSysCycleTime;
    g_hcommGetProfilingSysCycleTimeSupported = false;
    hcclDfxRegOpInfoPtr = StubHcclDfxRegOpInfo;
    g_hcclDfxRegOpInfoSupported = false;
    hcclProfilingReportOpPtr = StubHcclProfilingReportOp;
    g_hcclProfilingReportOpSupported = false;
    hcclReportAicpuKernelPtr = StubHcclReportAicpuKernel;
    g_hcclReportAicpuKernelSupported = false;
}

// ---------- 对外提供的查询接口 ----------
extern "C" bool HcommIsSupportHcommProfilingRegThread(void) {
    return g_hcommProfilingRegThreadSupported;
}
extern "C" bool HcommIsSupportHcommProfilingUnRegThread(void) {
    return g_hcommProfilingUnRegThreadSupported;
}
extern "C" bool HcommIsSupportHcommProfilingReportKernel(void) {
    return g_hcommProfilingReportKernelSupported;
}
extern "C" bool HcommIsSupportHcommProfilingReportOp(void) {
    return g_hcommProfilingReportOpSupported;
}
extern "C" bool HcommIsSupportHcommGetProfilingSysCycleTime(void) {
    return g_hcommGetProfilingSysCycleTimeSupported;
}
extern "C" bool HcommIsSupportHcclDfxRegOpInfo(void) {
    return g_hcclDfxRegOpInfoSupported;
}
extern "C" bool HcommIsSupportHcclProfilingReportOp(void) {
    return g_hcclProfilingReportOpSupported;
}
extern "C" bool HcommIsSupportHcclReportAicpuKernel(void) {
    return g_hcclReportAicpuKernelSupported;
}