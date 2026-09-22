/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter_error_manager_pub.h"
#include "base/err_mgr.h"

void RptInputErr(std::string error_code, std::vector<std::string> key, std::vector<std::string> value)
{
    // 将 std::vector<std::string> 转换为 std::vector<const char*>
    std::vector<const char*> key_cstr;
    for (const auto& k : key) {
        key_cstr.push_back(k.c_str());
    }

    std::vector<const char*> value_cstr;
    for (const auto& v : value) {
        value_cstr.push_back(v.c_str());
    }

    REPORT_PREDEFINED_ERR_MSG(error_code.c_str(), key_cstr, value_cstr);
    return;
}

void RptEnvErr(std::string error_code, std::vector<std::string> key, std::vector<std::string> value)
{
    // 将 std::vector<std::string> 转换为 std::vector<const char*>
    std::vector<const char*> key_cstr;
    for (const auto& k : key) {
        key_cstr.push_back(k.c_str());
    }

    std::vector<const char*> value_cstr;
    for (const auto& v : value) {
        value_cstr.push_back(v.c_str());
    }

    REPORT_PREDEFINED_ERR_MSG(error_code.c_str(), key_cstr, value_cstr);

    return;
}

// 设备侧（aicpu）环境无liberror_manager且加载链带-z now，强引用会使
// libscatter_aicpu_kernel.so加载失败（HcclLaunchAicpuKernel不可解析），故设备侧整体裁剪
#ifndef AICPU_COMPILE
ErrContext haclrtGetErrMgrContext(void)
{
    error_message::ErrorManagerContext sdk_ctx = error_message::GetErrMgrContext();

    ErrContext local_ctx;
    local_ctx.work_stream_id = sdk_ctx.work_stream_id;
    // 复制 reserved 数组
    errno_t ret = memcpy_s(local_ctx.reserved, sizeof(local_ctx.reserved), sdk_ctx.reserved, sizeof(sdk_ctx.reserved));

    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s]memcpy failed. errorno[%d]:", __func__, ret), local_ctx);

    return local_ctx;
}

void haclrtSetErrMgrContext(ErrContext error_context)
{
    error_message::ErrorManagerContext sdk_ctx;
    sdk_ctx.work_stream_id = error_context.work_stream_id;
    // 复制 reserved 数组
    errno_t ret
        = memcpy_s(sdk_ctx.reserved, sizeof(sdk_ctx.reserved), error_context.reserved, sizeof(error_context.reserved));

    if (ret != EOK) {
        HCCL_ERROR("[%s]memcpy failed. errorno[%d]:", __func__, ret);
        return;
    }

    error_message::SetErrMgrContext(sdk_ctx);
}
#endif
