/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_PRIMITIVES_DL_H
#define HCOMM_PRIMITIVES_DL_H

#include "hcomm_primitives.h"   // 原头文件，包含所有类型和定义
#include "hccl_res.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  (static_cast<HcclResult>(-2))
#endif

// 声明全局函数指针（小驼峰命名）
extern int32_t (*hcommLocalCopyOnThreadPtr)(ThreadHandle, void*, const void*, uint64_t);
extern int32_t (*hcommLocalReduceOnThreadPtr)(ThreadHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp);
extern int32_t (*hcommThreadNotifyRecordOnThreadPtr)(ThreadHandle, ThreadHandle, uint32_t);
extern int32_t (*hcommThreadNotifyWaitOnThreadPtr)(ThreadHandle, uint32_t, uint32_t);
extern int32_t (*hcommAclrtNotifyRecordOnThreadPtr)(ThreadHandle, uint64_t);
extern int32_t (*hcommAclrtNotifyWaitOnThreadPtr)(ThreadHandle, uint64_t, uint32_t);
extern int32_t (*hcommWriteOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t);
extern int32_t (*hcommWriteReduceOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp);
extern int32_t (*hcommWriteWithNotifyOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, uint32_t);
extern int32_t (*hcommWriteReduceWithNotifyOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp, uint32_t);
extern int32_t (*hcommReadOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t);
extern int32_t (*hcommReadReduceOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp);
extern int32_t (*hcommWriteNbiOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t);
extern int32_t (*hcommWriteWithNotifyNbiOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, uint32_t);
extern int32_t (*hcommReadNbiOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t);
extern int32_t (*hcommChannelNotifyRecordOnThreadPtr)(ThreadHandle, ChannelHandle, uint32_t);
extern int32_t (*hcommChannelNotifyWaitOnThreadPtr)(ThreadHandle, ChannelHandle, uint32_t, uint32_t);
extern int32_t (*hcommBatchModeStartPtr)(const char*);
extern int32_t (*hcommBatchModeEndPtr)(const char*);
extern int32_t (*hcommAcquireCommPtr)(const char*);
extern int32_t (*hcommReleaseCommPtr)(const char*);
extern HcclResult (*hcommSymWinGetPeerPointerPtr)(CommSymWindow, size_t, uint32_t, void**);
extern int32_t (*hcommThreadSynchronizePtr)(ThreadHandle);
extern int32_t (*hcommSendRequestPtr)(MsgHandle, const char*, const void*, size_t, uint32_t*);
extern int32_t (*hcommWaitResponsePtr)(MsgHandle, void*, size_t, uint32_t*);
extern int32_t (*hcommFenceOnThreadPtr)(ThreadHandle);
extern int32_t (*hcommChannelFenceOnThreadPtr)(ThreadHandle, ChannelHandle);

// 宏：将原始API名映射为函数指针调用（保持API名大驼峰）
#define HcommLocalCopyOnThread               (*hcommLocalCopyOnThreadPtr)
#define HcommLocalReduceOnThread              (*hcommLocalReduceOnThreadPtr)
#define HcommThreadNotifyRecordOnThread       (*hcommThreadNotifyRecordOnThreadPtr)
#define HcommThreadNotifyWaitOnThread         (*hcommThreadNotifyWaitOnThreadPtr)
#define HcommAclrtNotifyRecordOnThread        (*hcommAclrtNotifyRecordOnThreadPtr)
#define HcommAclrtNotifyWaitOnThread          (*hcommAclrtNotifyWaitOnThreadPtr)
#define HcommWriteOnThread                     (*hcommWriteOnThreadPtr)
#define HcommWriteReduceOnThread               (*hcommWriteReduceOnThreadPtr)
#define HcommWriteWithNotifyOnThread           (*hcommWriteWithNotifyOnThreadPtr)
#define HcommWriteReduceWithNotifyOnThread     (*hcommWriteReduceWithNotifyOnThreadPtr)
#define HcommReadOnThread                       (*hcommReadOnThreadPtr)
#define HcommReadReduceOnThread                 (*hcommReadReduceOnThreadPtr)
#define HcommWriteNbiOnThread                   (*hcommWriteNbiOnThreadPtr)
#define HcommWriteWithNotifyNbiOnThread         (*hcommWriteWithNotifyNbiOnThreadPtr)
#define HcommReadNbiOnThread                    (*hcommReadNbiOnThreadPtr)
#define HcommChannelNotifyRecordOnThread        (*hcommChannelNotifyRecordOnThreadPtr)
#define HcommChannelNotifyWaitOnThread          (*hcommChannelNotifyWaitOnThreadPtr)
#define HcommBatchModeStart                      (*hcommBatchModeStartPtr)
#define HcommBatchModeEnd                        (*hcommBatchModeEndPtr)
#define HcommAcquireComm                         (*hcommAcquireCommPtr)
#define HcommReleaseComm                         (*hcommReleaseCommPtr)
#define HcommSymWinGetPeerPointer                (*hcommSymWinGetPeerPointerPtr)
#define HcommThreadSynchronize                    (*hcommThreadSynchronizePtr)
#define HcommSendRequest                          (*hcommSendRequestPtr)
#define HcommWaitResponse                         (*hcommWaitResponsePtr)
#define HcommFenceOnThread                         (*hcommFenceOnThreadPtr)
#define HcommChannelFenceOnThread                  (*hcommChannelFenceOnThreadPtr)

void HcommPrimitivesDlInit(void* libHcommHandle);  // 本模块独立初始化
void HcommPrimitivesDlFini(void);                  // 本模块独立销毁

#ifdef __cplusplus
}
#endif

#endif // HCOMM_PRIMITIVES_DL_H