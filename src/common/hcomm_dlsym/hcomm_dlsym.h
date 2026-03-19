/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_DLSYM_H
#define HCOMM_DLSYM_H

#include "hccl_types.h"
#include "dtype_common.h"
#include "acl/acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

// 动态库管理接口（大驼峰命名）
void HcommDlInit(void);
void HcommDlFini(void);
int GetHcommVersion(void);

// 功能支持情况查询
bool HcommIsProfilingSupported();
bool HcommIsExportThreadSupported();

// 新增：查询函数声明
bool HcommIsSupportHcclGetRankId(void);
bool HcommIsSupportHcclGetRankSize(void);
bool HcommIsSupportHcclRankGraphGetLayers(void);
bool HcommIsSupportHcclRankGraphGetRanksByLayer(void);
bool HcommIsSupportHcclRankGraphGetRankSizeByLayer(void);
bool HcommIsSupportHcclRankGraphGetTopoTypeByLayer(void);
bool HcommIsSupportHcclRankGraphGetInstSizeListByLayer(void);
bool HcommIsSupportHcclRankGraphGetLinks(void);
bool HcommIsSupportHcclRankGraphGetTopoInstsByLayer(void);
bool HcommIsSupportHcclRankGraphGetTopoType(void);
bool HcommIsSupportHcclRankGraphGetRanksByTopoInst(void);
bool HcommIsSupportHcclGetHeterogMode(void);
bool HcommIsSupportHcclRankGraphGetEndpointNum(void);
bool HcommIsSupportHcclRankGraphGetEndpointDesc(void);
bool HcommIsSupportHcclRankGraphGetEndpointInfo(void);

bool HcommIsSupportHcclGetHcclBuffer(void);
bool HcommIsSupportHcclGetRemoteIpcHcclBuf(void);
bool HcommIsSupportHcclThreadAcquire(void);
bool HcommIsSupportHcclThreadAcquireWithStream(void);
bool HcommIsSupportHcclChannelAcquire(void);
bool HcommIsSupportHcclChannelGetHcclBuffer(void);
bool HcommIsSupportHcclEngineCtxCreate(void);
bool HcommIsSupportHcclEngineCtxGet(void);
bool HcommIsSupportHcclEngineCtxCopy(void);
bool HcommIsSupportHcclTaskRegister(void);
bool HcommIsSupportHcclTaskUnRegister(void);
bool HcommIsSupportHcclDevMemAcquire(void);
bool HcommIsSupportHcclThreadExportToCommEngine(void);
bool HcommIsSupportHcclChannelGetRemoteMems(void);
bool HcommIsSupportHcclCommMemReg(void);
bool HcommIsSupportHcclEngineCtxDestroy(void);

bool HcommIsSupportHcommLocalCopyOnThread(void);
bool HcommIsSupportHcommLocalReduceOnThread(void);
bool HcommIsSupportHcommThreadNotifyRecordOnThread(void);
bool HcommIsSupportHcommThreadNotifyWaitOnThread(void);
bool HcommIsSupportHcommAclrtNotifyRecordOnThread(void);
bool HcommIsSupportHcommAclrtNotifyWaitOnThread(void);
bool HcommIsSupportHcommWriteOnThread(void);
bool HcommIsSupportHcommWriteReduceOnThread(void);
bool HcommIsSupportHcommWriteWithNotifyOnThread(void);
bool HcommIsSupportHcommWriteReduceWithNotifyOnThread(void);
bool HcommIsSupportHcommReadOnThread(void);
bool HcommIsSupportHcommReadReduceOnThread(void);
bool HcommIsSupportHcommWriteNbi(void);
bool HcommIsSupportHcommWriteWithNotifyNbi(void);
bool HcommIsSupportHcommReadNbi(void);
bool HcommIsSupportHcommChannelNotifyRecordOnThread(void);
bool HcommIsSupportHcommChannelNotifyRecord(void);
bool HcommIsSupportHcommChannelNotifyWaitOnThread(void);
bool HcommIsSupportHcommChannelNotifyWait(void);
bool HcommIsSupportHcommBatchModeStart(void);
bool HcommIsSupportHcommBatchModeEnd(void);
bool HcommIsSupportHcommAcquireComm(void);
bool HcommIsSupportHcommReleaseComm(void);
bool HcommIsSupportHcommSymWinGetPeerPointer(void);
bool HcommIsSupportHcommThreadSynchronize(void);
bool HcommIsSupportHcommSendRequest(void);
bool HcommIsSupportHcommWaitResponse(void);
bool HcommIsSupportHcommFlush(void);
bool HcommIsSupportHcommChannelFence(void);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_DLSYM_H