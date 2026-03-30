/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exec_op.h"

namespace ops_hccl_p2p {
HcclResult ExecOp(OpParam &param, AlgResourceCtx* resCtx)
{
    uint64_t size = param.count * SIZE_TABLE[param.dataType];
    if (param.opType == HcclCMDType::HCCL_CMD_SEND) {
        // 拷贝到中转内存
        CHK_RET(HcommLocalCopyOnThread(resCtx->threadHandle, resCtx->localBuffer.addr, param.inputPtr, size));

        // 通知recv端，本端已经准备好数据
        CHK_RET(HcommChannelNotifyRecordOnThread(resCtx->threadHandle, resCtx->channelHandle, NOTIFY_IDX_ACK));

        // 等待recv端，告知已经读完本卡数据
        CHK_RET(HcommChannelNotifyWaitOnThread(resCtx->threadHandle, resCtx->channelHandle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    } else if (param.opType == HcclCMDType::HCCL_CMD_RECEIVE) {
        // 等待send端，告知本端可以开始读数据
        CHK_RET(HcommChannelNotifyWaitOnThread(resCtx->threadHandle, resCtx->channelHandle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

        // 单边读
        CHK_RET(HcommReadOnThread(resCtx->threadHandle, resCtx->channelHandle, param.outputPtr, resCtx->remoteBuffer.addr, size));

        // 通知send端，本端已经读完数据
        CHK_RET(HcommChannelNotifyRecordOnThread(resCtx->threadHandle, resCtx->channelHandle, NOTIFY_IDX_DATA_SIGNAL));
    }

    return HCCL_SUCCESS;
}
}
