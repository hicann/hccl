/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream_capture.h"
#include "acl_base.h"
#include "acl_rt.h"
#include "error_codes/rt_error_codes.h"
#include "log.h"

namespace ops_hccl {
bool IsStreamInCaptureMode(aclrtStream stream)
{
    // 无stream必不在捕获态（图编译期离线选路无runtime stream）；同时防未初始化脏值传入ACL
    if (stream == nullptr) {
        return false;
    }
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI rtModel = nullptr;
    aclError aclRet = aclmdlRICaptureGetInfo(stream, &captureStatus, &rtModel);
    if (aclRet == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        return false;
    }
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("[%s] aclmdlRICaptureGetInfo fail, ret[%d]", __func__, aclRet);
        return false;
    }
    return captureStatus == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
}
} // namespace ops_hccl
