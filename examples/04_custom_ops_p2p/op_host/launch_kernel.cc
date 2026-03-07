/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "common.h"
#include "load_kernel.h"
#include "launch_kernel.h"

namespace ops_hccl_p2p {

thread_local aclrtNotify g_notifies[AICPU_CONTROL_NOTIFY_NUM];

HcclResult LaunchKernel(OpParam &param, aclrtStream stream, KernelLaunchMode mode)
{
    // Host stream通知Device主thread
    ACLCHECK(aclrtRecordNotify(g_notifies[0], stream));

    HCCL_INFO("[LAUNCH_KERNEL] Using BINARY mode with <<<>>> syntax");
    // 调用使用 ASC 编译的函数，支持 <<<>>> 语法
    HcclResult ret = LaunchKernelBinary(param, stream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[LAUNCH_KERNEL] Binary mode failed, ret[%d]", ret);
        return ret;
    }
    HCCL_INFO("[LAUNCH_KERNEL] Binary mode completed successfully");

    // Host stream等待Device的通知
    ACLCHECK(aclrtWaitAndResetNotify(g_notifies[1], stream, CUSTOM_TIMEOUT));

    return HCCL_SUCCESS;
}
}
