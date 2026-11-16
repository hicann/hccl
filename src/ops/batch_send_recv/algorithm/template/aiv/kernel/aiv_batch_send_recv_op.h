/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_BATCH_SEND_RECV_OP_H
#define AIV_BATCH_SEND_RECV_OP_H

#include "aiv_defines.h"
#include "aiv_batch_send_recv_mesh_1D.h"

using namespace AscendC;

#define AIV_BATCH_SEND_RECV_KERNEL_BATCH_DEF(type)                  \
    extern "C" __global__ __aicore__ void aiv_batchsendrecv_##type( \
        KERNEL_ARGS_DEF, AivBatchSendRecv::ExtraArgsBsr extraArgs)  \
    {                                                               \
        AIV_INFO_HINT;                                              \
        AivBatchSendRecvV2Mesh1D(KERNEL_ARGS_CALL, extraArgs);      \
    }                                                               \
    EXPORT_AIV_META_INFO(aiv_batchsendrecv_##type)

AIV_BATCH_SEND_RECV_KERNEL_BATCH_DEF(int8_t);

#endif // AIV_BATCH_SEND_RECV_OP_H
