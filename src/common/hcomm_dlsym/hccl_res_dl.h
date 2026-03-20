/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RES_DL_H
#define HCCL_RES_DL_H

#include "hccl_res.h"
#include "hcomm_primitives.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  (static_cast<HcclResult>(-2))
#endif

// ---------- 对外API实现（通过函数指针转发）----------
HcclResult HcclGetHcclBuffer(HcclComm comm, void **buffer, uint64_t *size);
HcclResult HcclGetRemoteIpcHcclBuf(HcclComm comm, uint64_t remoteRank, void **addr, uint64_t *size);
HcclResult HcclThreadAcquire(HcclComm comm, CommEngine engine, uint32_t threadNum,
                              uint32_t notifyNumPerThread, ThreadHandle *threads);
HcclResult HcclThreadAcquireWithStream(HcclComm comm, CommEngine engine, aclrtStream stream,
                                       uint32_t notifyNum, ThreadHandle *thread);
HcclResult HcclChannelAcquire(HcclComm comm, CommEngine engine, const HcclChannelDesc *channelDescs,
                              uint32_t channelNum, ChannelHandle *channels);
HcclResult HcclChannelGetHcclBuffer(HcclComm comm, ChannelHandle channel, void **buffer, uint64_t *size);
HcclResult HcclEngineCtxCreate(HcclComm comm, const char *ctxTag, CommEngine engine,
                               uint64_t size, void **ctx);
HcclResult HcclEngineCtxGet(HcclComm comm, const char *ctxTag, CommEngine engine,
                            void **ctx, uint64_t *size);
HcclResult HcclEngineCtxCopy(HcclComm comm, CommEngine engine, const char *ctxTag,
                             const void *srcCtx, uint64_t size, uint64_t dstCtxOffset);
int32_t HcclTaskRegister(HcclComm comm, const char *msgTag, Callback cb);
int32_t HcclTaskUnRegister(HcclComm comm, const char *msgTag);
HcclResult HcclDevMemAcquire(HcclComm comm, const char *memTag, uint64_t *size,
                             void **addr, bool *newCreated);
HcclResult HcclThreadExportToCommEngine(HcclComm comm, uint32_t threadNum,
                                        const ThreadHandle *threads, CommEngine dstCommEngine,
                                        ThreadHandle *exportedThreads);
HcclResult HcclChannelGetRemoteMems(HcclComm comm, ChannelHandle channel,
                                   uint32_t *memNum, CommMem **remoteMems, char ***memTags);
HcclResult HcclCommMemReg(HcclComm comm, const char *memTag, const CommMem *mem,
                          HcclMemHandle *memHandle);
HcclResult HcclEngineCtxDestroy(HcclComm comm, const char *ctxTag, CommEngine engine);

// 动态库管理接口（大驼峰命名）
void HcclResDlInit(void* libHcommHandle);
void HcclResDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCCL_RES_DL_H