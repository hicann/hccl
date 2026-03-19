/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_DEVICE_PROFILING_DL_H
#define HCOMM_DEVICE_PROFILING_DL_H

#include "hccl_res_dl.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

typedef struct HcomProInfoTmp {
#define MAX_LENGTH 128
    uint8_t dataType;
    uint8_t cmdType;
    uint64_t dataCount;
    uint32_t rankSize;
    uint32_t userRank;
    uint32_t blockDim = 0;
    uint64_t beginTime;
    uint32_t root;
    uint32_t slaveThreadNum;
    uint64_t commNameLen;
    uint64_t algTypeLen;
    char tag[MAX_LENGTH];
    char commName[MAX_LENGTH];
    char algType[MAX_LENGTH];
    bool isCapture = false;
    bool isAiv = false;
    uint8_t reserved[MAX_LENGTH];
}HcomProInfoTmp;

// 声明全局函数指针（小驼峰命名）
extern HcclResult (*hcommProfilingReportMainStreamAndFirstTaskPtr)(ThreadHandle);
extern HcclResult (*hcommProfilingReportMainStreamAndLastTaskPtr)(ThreadHandle);
extern HcclResult (*hcommProfilingReportDeviceHcclOpInfoPtr)(HcomProInfoTmp);
extern HcclResult (*hcommProfilingInitPtr)(ThreadHandle*, uint32_t);
extern HcclResult (*hcommProfilingEndPtr)(ThreadHandle*, uint32_t);
extern HcclResult (*hcommProfilingReportDeviceOpPtr)(const char* groupname);
extern HcclResult (*hcommProfilingReportKernelStartTaskPtr)(uint64_t thread, const char* groupname);
extern HcclResult (*hcommProfilingReportKernelEndTaskPtr)(uint64_t thread, const char* groupname);

// 宏：将原始API名映射为函数指针调用
#define HcommProfilingReportMainStreamAndFirstTask                (*hcommProfilingReportMainStreamAndFirstTaskPtr)
#define HcommProfilingReportMainStreamAndLastTask                 (*hcommProfilingReportMainStreamAndLastTaskPtr)
#define HcommProfilingReportDeviceHcclOpInfo                       (*hcommProfilingReportDeviceHcclOpInfoPtr)
#define HcommProfilingInit                                          (*hcommProfilingInitPtr)
#define HcommProfilingEnd                                           (*hcommProfilingEndPtr)
#define HcommProfilingReportDeviceOp                                (*hcommProfilingReportDeviceOpPtr)
#define HcommProfilingReportKernelStartTask                         (*hcommProfilingReportKernelStartTaskPtr)
#define HcommProfilingReportKernelEndTask                            (*hcommProfilingReportKernelEndTaskPtr)

// 查询函数声明
bool HcommIsSupportHcommProfilingReportMainStreamAndFirstTask(void);
bool HcommIsSupportHcommProfilingReportMainStreamAndLastTask(void);
bool HcommIsSupportHcommProfilingReportDeviceHcclOpInfo(void);
bool HcommIsSupportHcommProfilingInit(void);
bool HcommIsSupportHcommProfilingEnd(void);
bool HcommIsSupportHcommProfilingReportDeviceOp(void);
bool HcommIsSupportHcommProfilingReportKernelStartTask(void);
bool HcommIsSupportHcommProfilingReportKernelEndTask(void);

// 动态库管理接口
void HcommDeviceProfilingDlInit(void* libHcommHandle);
void HcommDeviceProfilingDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_DEVICE_PROFILING_DL_H