/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tuner_setup.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

#include "alg_param.h"
#include "hccl_common.h"
#include "hccl_tuner_plugin.h"

namespace {
constexpr uint32_t TUNER_LOG_BUF_SIZE = 1024;
constexpr const char* TUNER_CTX_PREFIX = "__tuner_";

/* 为插件传入的 ctxTag 添加 "__tuner_" 前缀，避免与 HCCL 内部 context tag 冲突。 */
std::string PrefixCtxTag(const char* ctxTag)
{
    return std::string(TUNER_CTX_PREFIX) + ((ctxTag != nullptr) ? ctxTag : "");
}
} /* namespace */

HcclResult TunerCtxCreate(HcclComm comm, const char* ctxTag, uint64_t size, void** ctx)
{
    if (ctxTag == nullptr || ctx == nullptr) {
        return HCCL_E_PTR;
    }
    std::string prefixed = PrefixCtxTag(ctxTag);
    return HcclEngineCtxCreate(comm, prefixed.c_str(), COMM_ENGINE_CPU_TS, size, ctx);
}

HcclResult TunerCtxGet(HcclComm comm, const char* ctxTag, void** ctx, uint64_t* size)
{
    if (ctxTag == nullptr || ctx == nullptr) {
        return HCCL_E_PTR;
    }
    std::string prefixed = PrefixCtxTag(ctxTag);
    return HcclEngineCtxGet(comm, prefixed.c_str(), COMM_ENGINE_CPU_TS, ctx, size);
}

HcclResult TunerCtxDestroy(HcclComm comm, const char* ctxTag)
{
    if (ctxTag == nullptr) {
        return HCCL_E_PTR;
    }
    std::string prefixed = PrefixCtxTag(ctxTag);
    return HcclEngineCtxDestroy(comm, prefixed.c_str(), COMM_ENGINE_CPU_TS);
}

void TunerLogFunction(int level, const char* file, int line, const char* fmt, ...)
{
    char buf[TUNER_LOG_BUF_SIZE] = {};
    if (fmt != nullptr) {
        va_list args;
        va_start(args, fmt);
        if (vsnprintf_s(buf, sizeof(buf), sizeof(buf) - 1, fmt, args) < 0) {
            buf[0] = '\0';
        }
        va_end(args);
    }
    const char* srcFile = (file != nullptr) ? file : "?";
    switch (level) {
        case HCCL_TUNER_LOG_ERROR:
            HCCL_ERROR("[TunerPlugin][%s:%d] %s", srcFile, line, buf);
            break;
        case HCCL_TUNER_LOG_WARN:
            HCCL_WARNING("[TunerPlugin][%s:%d] %s", srcFile, line, buf);
            break;
        case HCCL_TUNER_LOG_INFO:
            HCCL_INFO("[TunerPlugin][%s:%d] %s", srcFile, line, buf);
            break;
        case HCCL_TUNER_LOG_DEBUG:
            HCCL_DEBUG("[TunerPlugin][%s:%d] %s", srcFile, line, buf);
            break;
        default:
            break;
    }
}
