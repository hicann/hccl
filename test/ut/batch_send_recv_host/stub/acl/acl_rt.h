/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSR_HOST_TEST_ACL_RT_H
#define BSR_HOST_TEST_ACL_RT_H

// acl 运行时头的最小替身：host 侧被测代码只用到流句柄与异常信息类型，不做任何真实 acl 调用。

typedef void* rtStream_t;

typedef struct aclrtExceptionInfo {
    int32_t reserved;
} aclrtExceptionInfo;

#endif // BSR_HOST_TEST_ACL_RT_H
