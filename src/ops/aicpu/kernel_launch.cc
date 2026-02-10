/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <sstream>
#include <memory>
#include "alg_param.h"
#include "executor_base.h"
#include "coll_alg_exec_registry.h"
#include "hcomm_primitives.h"
#include "dfx/task_exception_fun.h"

using namespace ops_hccl;
using HcclGetOpInfoCallback = void (*)(const void *opInfo, char *outPut, size_t size);

#ifdef __cplusplus
extern "C" {
#endif
HcclResult __attribute__((weak)) HcommRegOpInfo(const char* commId, void* opInfo, size_t size);
HcclResult __attribute__((weak)) HcommRegOpTaskException(const char* commId, HcclGetOpInfoCallback callback);

HcclResult __attribute__((weak)) HcommProfilingReportMainStreamAndFirstTask(ThreadHandle thread);
HcclResult __attribute__((weak)) HcommProfilingReportMainStreamAndLastTask(ThreadHandle thread);
//device侧的OP
HcclResult __attribute__((weak)) HcommProfilingReportDeviceHcclOpInfo(HcomProInfo profInfo);
HcclResult __attribute__((weak)) HcommProfilingInit(ThreadHandle *threads, u32 threadNum);
HcclResult __attribute__((weak)) HcommProfilingEnd(ThreadHandle *threads, u32 threadNum);
#ifdef __cplusplus
}
#endif


extern "C" unsigned int HcclLaunchAicpuKernel(OpParam *param)
{
    HCCL_INFO("Entry-%s, commName[%s], tag[%s], algTag[%s]", __func__, param->commName, param->tag, param->algTag);
    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommAcquireComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }

    if (param->deviceType != DevType::DEV_TYPE_910_95) {
        if (HcommRegOpInfo != nullptr &&
            HcommRegOpInfo(param->commName, reinterpret_cast<void *>(param), sizeof(OpParam)) != HCCL_SUCCESS) {
            HCCL_ERROR("%s HcommRegOpInfo fail, commName[%s], algTag[%s], param[%p], size[%u]",
                __func__, param->commName, param->algTag, param, sizeof(OpParam));
            return 1;
        }

        if (HcommRegOpTaskException != nullptr &&
            HcommRegOpTaskException(param->commName, ops_hccl::GetScatterOpInfo) != HCCL_SUCCESS) {
            HCCL_ERROR("%s HcommRegOpTaskException fail, commName[%s], algTag[%s]", __func__, param->commName, param->algTag);
            return 1;
        }
    }

    // 根据算法名字获取executor
    std::string algName = std::string(param->algName);
    std::unique_ptr<ExecutorBase> executor = CollAlgExecRegistry::Instance().GetAlgExec(algName);
    if (executor.get() == nullptr) {
        HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str());
        return 1;
    }

    // 获取Device测主thread
    ThreadHandle* threadHandlePtr = reinterpret_cast<ThreadHandle *>(reinterpret_cast<u8 *>(param->resCtx) +
        sizeof(AlgResourceCtx));
    ThreadHandle thread = threadHandlePtr[0];
    ThreadHandle exportedAicpuTsThread = param->resCtx->opThread;
    u32 notifyNumOnMainThread = param->resCtx->notifyNumOnMainThread;
    if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set batch mode, tag is %s.", param->algTag);
        return 1;
    }

    if (HcommProfilingInit(threadHandlePtr, param->resCtx->slaveThreadNum+1) != HCCL_SUCCESS)
    {
        HCCL_ERROR("failed to init Profiling");
        return 1;
    }

    // 上报主流和第一个task  wait之前
    if (HcommProfilingReportMainStreamAndFirstTask(thread) != HCCL_SUCCESS) {
        HCCL_ERROR("failed to report MainStream And FirstTask");
        return 1;
    }

    // 主thread等待Host stream的通知
    HCCL_DEBUG("[%s]Notify wait on thread[%llu], notifyNumOnMainThread[%u], timeout[%u]", __func__, thread,
        notifyNumOnMainThread, CUSTOM_TIMEOUT);
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, notifyNumOnMainThread, CUSTOM_TIMEOUT)));

    // 执行算法编排
    if (executor->Orchestrate(*param, param->resCtx) != HCCL_SUCCESS) {
        HCCL_ERROR("orchestrate failed for alg:%s", param->algName);
        return 1;
    }

    // 上报device侧的op 附加信息
    HcomProInfo profInfo;
    std::string algTypeStr(param->algTypeStr);
    strcpy_s(profInfo.algType, sizeof(profInfo.algType), algTypeStr.c_str());
    strcpy_s(profInfo.commName, sizeof(profInfo.commName), param->commName);
    profInfo.commNameLen = strlen(param->commName);
    profInfo.dataCount = param->DataDes.count;
    profInfo.dataType = static_cast<uint8_t>(param->DataDes.dataType);
    profInfo.rankSize = param->resCtx->topoInfo.userRankSize;
    HcommProfilingReportDeviceHcclOpInfo(profInfo);
    // 主thread通知Host stream
    constexpr u32 DEFAULT_NOTIFY_IDX = 0;
    HCCL_DEBUG("[%s]Notify record on srcThread[%llu], dstThread[%llu], notifyIdx[%u]",__func__, thread, exportedAicpuTsThread,
        DEFAULT_NOTIFY_IDX);
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(thread, exportedAicpuTsThread,
        DEFAULT_NOTIFY_IDX)));
    
    // 上报主流和最后一个task 在notify之后
    if (HcommProfilingReportMainStreamAndLastTask(thread) != HCCL_SUCCESS) {
        HCCL_ERROR("failed to report MainStream And LastTask");
        return 1;
    }
    
    if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
        return 1;
    }

    if (HcommProfilingEnd(threadHandlePtr, param->resCtx->slaveThreadNum + 1) != HCCL_SUCCESS) {
        HCCL_ERROR("failed to End Profiling");
        return 1;
    }

    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommReleaseComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }
    HCCL_INFO("%s success, tag[%s], algTag[%s], commName[%s]", __func__, param->tag, param->algTag, param->commName);
    return 0;
}
