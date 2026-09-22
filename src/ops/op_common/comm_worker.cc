/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_worker.h"
#include <exception>
#include <system_error>
#include "log.h"

namespace ops_hccl {

CommWorker::~CommWorker() { Shutdown(); }

HcclResult CommWorker::Start()
{
    try {
        thread_ = std::thread(&CommWorker::WorkerLoop, this);
    } catch (const std::system_error& e) {
        HCCL_ERROR("[%s] create thread failed, errCode[%d].", __func__, e.code().value());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult CommWorker::Submit(aclrtContext ctx, const ErrContext& errCtx, const std::function<HcclResult()>& task)
{
    std::unique_lock<std::mutex> lock(mtx_);
    if (stop_) {
        return HCCL_E_INTERNAL; // 已随comm销毁被回收（销毁与提交并发），本次直接失败
    }
    task_ = task;
    ctx_ = ctx;
    errCtx_ = errCtx;
    result_ = HCCL_E_INTERNAL;
    hasTask_ = true;
    cv_.notify_all();
    cv_.wait(lock, [this]() {
        return !hasTask_;
    });
    return result_;
}

void CommWorker::Shutdown()
{
    {
        std::unique_lock<std::mutex> lock(mtx_);
        stop_ = true;
        cv_.notify_all();
    }
    if (thread_.joinable()) {
        thread_.join();
    }
}

void CommWorker::WorkerLoop()
{
    std::unique_lock<std::mutex> lock(mtx_);
    while (true) {
        cv_.wait(lock, [this]() {
            return hasTask_ || stop_;
        });
        if (!hasTask_) {
            break; // 仅stop_触发且无待处理任务
        }
        aclrtContext ctx = ctx_;
        ErrContext errCtx = errCtx_;
        std::function<HcclResult()> task = task_;
        lock.unlock();

        // 恢复主线程ErrorMgr上下文：先于aclrtSetCurrentContext设置；ctx为null（ST仿真）时同样需要设置
        haclrtSetErrMgrContext(errCtx);

        // 每次任务重新绑定调用方context（调用方context可能变化，不能只在首次绑定时设置）
        HcclResult ret = HCCL_E_INTERNAL;
        bool ctxOk = true;
        if (ctx != nullptr) {
            aclError setRet = aclrtSetCurrentContext(ctx);
            if (setRet != ACL_SUCCESS) {
                HCCL_ERROR("[WorkerLoop] aclrtSetCurrentContext failed, ret[%d].", setRet);
                ctxOk = false;
                ret = HCCL_E_RUNTIME;
            }
        }
        if (ctxOk) {
            try {
                ret = task();
            } catch (const std::exception& e) {
                HCCL_ERROR("[WorkerLoop] task exception, what[%s].", e.what());
                ret = HCCL_E_INTERNAL;
            } catch (...) {
                HCCL_ERROR("[WorkerLoop] task unknown exception.");
                ret = HCCL_E_INTERNAL;
            }
        }

        lock.lock();
        result_ = ret;
        hasTask_ = false;
        cv_.notify_all();
    }
}

} // namespace ops_hccl
