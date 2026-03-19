/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_HOST_PROFILING_DL_H
#define HCOMM_HOST_PROFILING_DL_H

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
extern HcclResult (*hcommProfilingRegThreadPtr)(HcomProInfoTmp, ThreadHandle*);
extern HcclResult (*hcommProfilingUnRegThreadPtr)(HcomProInfoTmp, ThreadHandle*);
extern HcclResult (*hcommProfilingReportKernelPtr)(uint64_t, const char*);
extern HcclResult (*hcommProfilingReportOpPtr)(HcomProInfoTmp);
extern uint64_t (*hcommGetProfilingSysCycleTimePtr)();
extern HcclResult (*hcclDfxRegOpInfoPtr)(HcclComm comm, void* dfxOpInfo);
extern HcclResult (*hcclProfilingReportOpPtr)(HcclComm comm, uint64_t beginTime);
extern HcclResult (*hcclReportAicpuKernelPtr)(HcclComm comm, uint64_t beginTime, char *kernelName);


// 宏：将原始API名映射为函数指针调用
#define HcommProfilingRegThread                (*hcommProfilingRegThreadPtr)
#define HcommProfilingUnRegThread               (*hcommProfilingUnRegThreadPtr)
#define HcommProfilingReportKernel               (*hcommProfilingReportKernelPtr)
#define HcommProfilingReportOp                    (*hcommProfilingReportOpPtr)
#define HcommGetProfilingSysCycleTime              (*hcommGetProfilingSysCycleTimePtr)
#define HcclDfxRegOpInfo                         (*hcclDfxRegOpInfoPtr)
#define HcclProfilingReportOp                         (*hcclProfilingReportOpPtr)
#define HcclReportAicpuKernel                         (*hcclReportAicpuKernelPtr)

// 查询函数声明
bool HcommIsSupportHcommProfilingRegThread(void);
bool HcommIsSupportHcommProfilingUnRegThread(void);
bool HcommIsSupportHcommProfilingReportKernel(void);
bool HcommIsSupportHcommProfilingReportOp(void);
bool HcommIsSupportHcommGetProfilingSysCycleTime(void);
bool HcommIsSupportHcclDfxRegOpInfo(void);
bool HcommIsSupportHcclProfilingReportOp(void);
bool HcommIsSupportHcclProfilingReportOp(void);

// 动态库管理接口
void HcommProfilingDlInit(void* libHcommHandle);
void HcommProfilingDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_PROFILING_DL_H