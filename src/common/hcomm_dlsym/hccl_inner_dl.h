/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INNER_DL_H
#define HCCL_INNER_DL_H

#include "hccl_inner.h"   // 原始头文件，包含所有类型和声明

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  (static_cast<HcclResult>(-2))
#endif

constexpr int VERSION_NUMBER = 90000000;

// 声明全局函数指针（小驼峰命名）
extern HcclResult (*hcclAllReduceInnerPtr)(void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
extern HcclResult (*hcclBroadcastInnerPtr)(void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
extern HcclResult (*hcclReduceScatterInnerPtr)(void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
extern HcclResult (*hcclReduceScatterVInnerPtr)(void*, const void*, const void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
extern HcclResult (*hcclScatterInnerPtr)(void*, void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
extern HcclResult (*hcclAllGatherInnerPtr)(void*, void*, uint64_t, HcclDataType, HcclComm, aclrtStream);
extern HcclResult (*hcclAllGatherVInnerPtr)(void*, uint64_t, void*, const void*, const void*, HcclDataType, HcclComm, aclrtStream);
extern HcclResult (*hcclSendInnerPtr)(void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
extern HcclResult (*hcclRecvInnerPtr)(void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
extern HcclResult (*hcclAlltoAllVCInnerPtr)(const void*, const void*, HcclDataType, const void*, HcclDataType, HcclComm, aclrtStream);
extern HcclResult (*hcclAlltoAllVInnerPtr)(const void*, const void*, const void*, HcclDataType, const void*, const void*, const void*, HcclDataType, HcclComm, aclrtStream);
extern HcclResult (*hcclAlltoAllInnerPtr)(const void*, uint64_t, HcclDataType, const void*, uint64_t, HcclDataType, HcclComm, aclrtStream);
extern HcclResult (*hcclReduceInnerPtr)(void*, void*, uint64_t, HcclDataType, HcclReduceOp, uint32_t, HcclComm, aclrtStream);
extern HcclResult (*hcclBatchSendRecvInnerPtr)(HcclSendRecvItem*, uint32_t, HcclComm, aclrtStream);
extern HcclResult (*hcclCreateOpResCtxInnerPtr)(HcclComm, uint8_t, HcclDataType, HcclDataType, HcclReduceOp, uint64_t, char*, uint32_t, void**);

// 宏：将原始API名映射为函数指针调用（保持API名大驼峰）
#define HcclAllReduceInner                (*hcclAllReduceInnerPtr)
#define HcclBroadcastInner                 (*hcclBroadcastInnerPtr)
#define HcclReduceScatterInner              (*hcclReduceScatterInnerPtr)
#define HcclReduceScatterVInner             (*hcclReduceScatterVInnerPtr)
#define HcclScatterInner                     (*hcclScatterInnerPtr)
#define HcclAllGatherInner                    (*hcclAllGatherInnerPtr)
#define HcclAllGatherVInner                   (*hcclAllGatherVInnerPtr)
#define HcclSendInner                          (*hcclSendInnerPtr)
#define HcclRecvInner                          (*hcclRecvInnerPtr)
#define HcclAlltoAllVCInner                    (*hcclAlltoAllVCInnerPtr)
#define HcclAlltoAllVInner                      (*hcclAlltoAllVInnerPtr)
#define HcclAlltoAllInner                        (*hcclAlltoAllInnerPtr)
#define HcclReduceInner                           (*hcclReduceInnerPtr)
#define HcclBatchSendRecvInner                     (*hcclBatchSendRecvInnerPtr)
#define HcclCreateOpResCtxInner                     (*hcclCreateOpResCtxInnerPtr)

// 查询函数声明
bool HcommIsSupportHcclAllReduceInner(void);
bool HcommIsSupportHcclBroadcastInner(void);
bool HcommIsSupportHcclReduceScatterInner(void);
bool HcommIsSupportHcclReduceScatterVInner(void);
bool HcommIsSupportHcclScatterInner(void);
bool HcommIsSupportHcclAllGatherInner(void);
bool HcommIsSupportHcclAllGatherVInner(void);
bool HcommIsSupportHcclSendInner(void);
bool HcommIsSupportHcclRecvInner(void);
bool HcommIsSupportHcclAlltoAllVCInner(void);
bool HcommIsSupportHcclAlltoAllVInner(void);
bool HcommIsSupportHcclAlltoAllInner(void);
bool HcommIsSupportHcclReduceInner(void);
bool HcommIsSupportHcclBatchSendRecvInner(void);
bool HcommIsSupportHcclCreateOpResCtxInner(void);

// 动态库管理接口
void HcclInnerDlInit(void* libHcommHandle);
void HcclInnerDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCCL_INNER_DL_H