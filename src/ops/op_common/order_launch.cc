/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "order_launch.h"
#include "hcomm_primitives_dl.h"
#include "hccl_res_dl.h"
#include "dlhcomm_function.h"

namespace ops_hccl {

static HcclResult OpLaunchGetUnfoldStream(HcclComm comm, ThreadHandle unfoldThread, aclrtStream& resolvedStream)
{
    void* unfoldStream = nullptr;
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo) {
        resolvedStream = nullptr;
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcclThreadResGetInfo is invalid.");
        return HCCL_SUCCESS;
    }
    HcclResult ret
        = HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, unfoldThread, 0, sizeof(void*), &unfoldStream);
    if (ret == HCCL_E_NOT_SUPPORT) {
        resolvedStream = nullptr;
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcclThreadResGetInfo not support.");
        return HCCL_SUCCESS;
    } else if (ret != HCCL_SUCCESS) {
        resolvedStream = nullptr;
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcclThreadResGetInfo not success.");
        return HCCL_SUCCESS;
    }
    resolvedStream = unfoldStream;
    return HCCL_SUCCESS;
}

static HcclResult OpLaunchGetHostOrderStream(ThreadHandle hostOrderThread, aclrtStream& resolvedStream)
{
    void* hostOrderStream = nullptr;
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    if (!HcclThreadResGetInfoFunc.dlHcommThreadResGetInfo) {
        resolvedStream = nullptr;
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcommThreadResGetInfo is invalid.");
        return HCCL_SUCCESS;
    }
    HcclResult ret
        = HcclThreadResGetInfoFunc.dlHcommThreadResGetInfo(hostOrderThread, 0, sizeof(void*), &hostOrderStream);
    if (ret == HCCL_E_NOT_SUPPORT) {
        resolvedStream = nullptr;
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcommThreadResGetInfo not support.");
        return HCCL_SUCCESS;
    } else if (ret != HCCL_SUCCESS) {
        resolvedStream = nullptr;
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcommThreadResGetInfo not success.");
        return HCCL_SUCCESS;
    }
    resolvedStream = hostOrderStream;
    return HCCL_SUCCESS;
}

static const char* GetOrderLaunchModeName(OrderLaunchMode mode)
{
    if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {
        return "Aclgraph";
    } else if (mode == OrderLaunchMode::ORDER_LAUNCH_GE) {
        return "GE";
    }
    return "Opbase";
}

static HcclDedicatedThreadType GetOrderLaunchHostThreadType(OrderLaunchMode mode)
{
    if (mode == OrderLaunchMode::ORDER_LAUNCH_GE) {
        return HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE;
    } else if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {
        return HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH;
    }
    return HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE;
}

static HcclResult OpLaunchGetOrderStreams(
    HcclComm comm, ThreadHandle hostOrderThread, ThreadHandle unfoldThread, aclrtStream& hostOrderStream,
    aclrtStream& unfoldStream)
{
    CHK_RET(OpLaunchGetHostOrderStream(hostOrderThread, hostOrderStream));
    CHK_RET(OpLaunchGetUnfoldStream(comm, unfoldThread, unfoldStream));
    CHK_PRT_RET(
        hostOrderStream == nullptr || unfoldStream == nullptr,
        HCCL_ERROR(
            "[%s] failed to get hostOrderStream[%p] or unfoldStream[%p]", __func__, hostOrderStream, unfoldStream),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

static HcclResult AclgraphOrderLaunchEventToOrderStream(
    HcclComm comm, ThreadHandle hostOrderThread, ThreadHandle unfoldThread, HcclRtEvent event)
{
    aclrtStream hostOrderStream = nullptr;
    aclrtStream unfoldStream = nullptr;
    CHK_RET(OpLaunchGetOrderStreams(comm, hostOrderThread, unfoldThread, hostOrderStream, unfoldStream));

    aclError retEvent = aclrtRecordEvent(event, unfoldStream);
    CHK_PRT_RET(
        retEvent != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtRecordEvent failed, ret[%d]", __func__, retEvent),
        HCCL_E_RUNTIME);
    retEvent = aclrtStreamWaitEvent(hostOrderStream, event);
    CHK_PRT_RET(
        retEvent != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtStreamWaitEvent failed, ret[%d]", __func__, retEvent),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

static HcclResult AclgraphOrderLaunchEventToKernelStream(
    HcclComm comm, ThreadHandle hostOrderThread, ThreadHandle unfoldThread, HcclRtEvent event)
{
    aclrtStream hostOrderStream = nullptr;
    aclrtStream unfoldStream = nullptr;
    CHK_RET(OpLaunchGetOrderStreams(comm, hostOrderThread, unfoldThread, hostOrderStream, unfoldStream));

    aclError retEvent = aclrtRecordEvent(event, hostOrderStream);
    CHK_PRT_RET(
        retEvent != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtRecordEvent failed, ret[%d]", __func__, retEvent),
        HCCL_E_RUNTIME);
    retEvent = aclrtStreamWaitEvent(unfoldStream, event);
    CHK_PRT_RET(
        retEvent != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtStreamWaitEvent failed, ret[%d]", __func__, retEvent),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

/**
 * @brief OrderLaunch第一阶段资源准备：获取超时时间、确定启动模式、GE模式覆盖展开线程、ACLGRAPH模式创建event。
 * @param comm          通信域
 * @param param         算子参数
 * @param unfoldThread  展开线程句柄，GE模式下可能被覆盖
 * @return HcclResult
 */
HcclResult OrderLaunch::HcclOrderLaunchPrepare(HcclComm comm, const OpParam& param, ThreadHandle& unfoldThread)
{
    // 超时取自param；ExecTimeoutManager仅在AICPU侧被Set，host侧读取恒为默认值
    execTimeout_ = param.opConfig.execTimeout;

    launchMode_ = param.isCapture ? OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH :
                                    (param.opMode == OpMode::OFFLOAD ? OrderLaunchMode::ORDER_LAUNCH_GE :
                                                                       OrderLaunchMode::ORDER_LAUNCH_OPBASE);
    const char* modeName = GetOrderLaunchModeName(launchMode_);

    // GE模式通过HcclDedicatedThreadAcquire获取GE专用展开线程，覆盖unfoldThread；
    // 不支持时置0，跳过语义由Phase1的unfoldThread==0判空兜底，不提前return
    if (launchMode_ == OrderLaunchMode::ORDER_LAUNCH_GE) {
        if (!HcommIsSupportHcclDedicatedThreadAcquire()) {
            unfoldThread = 0;
            HCCL_WARNING("[%s]. HcclDedicatedThreadAcquire not supported, GE OrderLaunch is skipped.", __func__);
        } else {
            ThreadHandle geUnfoldThread = 0;
            CHK_RET(HcclDedicatedThreadAcquire(
                comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, ORDER_UNFOLD_THREAD_NOTIFY_NUM, &geUnfoldThread));
            unfoldThread = geUnfoldThread;
            HCCL_INFO("[%s]. GE unfoldThread[0x%llx]", __func__, unfoldThread);
        }
    }

    if (launchMode_ == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {
        CHK_RET(event0Guard_.Create());
        CHK_RET(event1Guard_.Create());
    }

    HCCL_INFO("[%s]. %s OrderLaunch Prepare success, unfoldThread[0x%llx]", __func__, modeName, unfoldThread);
    return HCCL_SUCCESS;
}

/**
 * @brief 按序launch第一阶段：将通信算子按序launch到OrderStream。
 * OPBASE / ACLGRAPH / HCOMM 三模式统一入口，流程如下：
 * 1. 获取并导出Host侧保序流（hostOrderThread），写入param.exportHostOrderThread；
 * 2. 仅ACLGRAPH模式：在unfoldStream上record event，hostOrderStream等待该event，建立流间依赖；
 * 3. 获取Device侧保序流（deviceOrderThread），写入param.deviceOrderThread；
 * 4. hostOrderThread在unfoldThread上record notify(idx)，unfoldThread等待该notify，完成第一阶段同步。
 * @param comm          通信域
 * @param param         算子参数，函数会写入exportHostOrderThread和deviceOrderThread
 * @param unfoldThread  展开线程句柄
 * @param notifyIdx     notify索引
 * @return HcclResult
 */
HcclResult
OrderLaunch::HcclOrderLaunchToOrderStream(HcclComm comm, OpParam& param, ThreadHandle unfoldThread, u32 notifyIdx)
{
    const char* modeName = GetOrderLaunchModeName(launchMode_);
    HcclDedicatedThreadType hostThreadType = GetOrderLaunchHostThreadType(launchMode_);
    HCCL_INFO("[%s]. %s OrderLaunch Phase1 Start, Comm[%p], timeout[%u].", __func__, modeName, comm, execTimeout_);

    // unfoldThread为0表示未走提前展开，或HcclDedicatedThreadAcquire不支持时，无需保序，直接跳过
    if (unfoldThread == 0 || !HcommIsSupportHcclDedicatedThreadAcquire()) {
        param.exportHostOrderThread = 0;
        param.deviceOrderThread = 0;
        orderLaunchRequired_ = false;
        HCCL_WARNING(
            "[%s]. unfoldThread is null or HcclDedicatedThreadAcquire not supported, %s OrderLaunch is skipped.",
            __func__, modeName);
        return HCCL_SUCCESS;
    }

    // 1.1、获取Host侧保序流
    ThreadHandle hostOrderThread;
    ThreadHandle exportHostOrderThread;
    CHK_RET(HcclDedicatedThreadAcquire(comm, hostThreadType, HOST_ORDER_THREAD_NOTIFY_NUM, &hostOrderThread));
    HCCL_INFO(
        "[%s]. %s After HcclDedicatedThreadAcquire hostOrderThread [0x%llx]", __func__, modeName, hostOrderThread);
    if (hostOrderThread == 0) {
        param.exportHostOrderThread = 0;
        param.deviceOrderThread = 0;
        orderLaunchRequired_ = false;
        HCCL_INFO(
            "[%s]. Communication domains Number is less than cores Number, %s OrderLaunch is not Required.", __func__,
            modeName);
        return HCCL_SUCCESS;
    }
    // 缓存句柄供Phase2复用，避免二次acquire结果漂移
    hostOrderThread_ = hostOrderThread;

    // 1.2、导出Host侧保序流
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &hostOrderThread, COMM_ENGINE_AICPU_TS, &exportHostOrderThread));
    param.exportHostOrderThread = exportHostOrderThread;
    HCCL_INFO(
        "[%s]. %s After HcclThreadExportToCommEngine hostOrderThread [0x%llx], exportHostOrderThread[0x%llx]", __func__,
        modeName, hostOrderThread, exportHostOrderThread);

    // 2、仅 ACLGRAPH 模式需要在 unfoldStream 上 record event，hostOrderStream 等待 event
    if (launchMode_ == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {
        CHK_RET(AclgraphOrderLaunchEventToOrderStream(comm, hostOrderThread, unfoldThread, event0Guard_.Get()));
    }

    // 3、获取Device侧保序流
    ThreadHandle deviceOrderThread;
    CHK_RET(HcclDedicatedThreadAcquire(
        comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, DEVICE_ORDER_THREAD_NOTIFY_NUM, &deviceOrderThread));
    if (deviceOrderThread == 0) {
        param.exportHostOrderThread = 0;
        param.deviceOrderThread = 0;
        orderLaunchRequired_ = false;
        HCCL_INFO(
            "[%s]. HcclDedicatedThreadAcquire unable to obtain deviceOrderThread, %s OrderLaunch is not Required.",
            __func__, modeName);
        return HCCL_SUCCESS;
    }
    param.deviceOrderThread = deviceOrderThread;
    HCCL_INFO(
        "[%s]. %s After HcclDedicatedThreadAcquire deviceOrderThread [0x%llx]", __func__, modeName, deviceOrderThread);

    // 4、notify0
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(hostOrderThread, unfoldThread, notifyIdx)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(unfoldThread, notifyIdx, execTimeout_)));
    HCCL_INFO("[%s]. %s OrderLaunch Phase1 Success, timeout[%u].", __func__, modeName, execTimeout_);
    return HCCL_SUCCESS;
}

/**
 * @brief 按序launch第二阶段：将通信算子按序launch到KernelStream。
 * OPBASE / ACLGRAPH / HCOMM 三模式统一入口，流程如下：
 * 1. 复用Phase1缓存的Host侧保序线程句柄（hostOrderThread），不再二次acquire；
 * 2. hostOrderThread等待第一阶段record的notify(idx)，完成第二阶段同步；
 * 3. 仅ACLGRAPH模式：在hostOrderStream上record event，unfoldStream等待该event，建立流间依赖。
 * @param comm          通信域
 * @param unfoldThread  展开线程句柄（仅ACLGRAPH模式使用）
 * @param notifyIdx     notify索引
 * @return HcclResult
 */
HcclResult OrderLaunch::HcclOrderLaunchToKernelStream(HcclComm comm, ThreadHandle unfoldThread, u32 notifyIdx)
{
    const char* modeName = GetOrderLaunchModeName(launchMode_);
    HCCL_INFO("[%s]. %s OrderLaunch Phase2 Start, Comm[%p], timeout[%u].", __func__, modeName, comm, execTimeout_);

    // 第一阶段已判定无需保序，直接跳过
    if (!orderLaunchRequired_) {
        HCCL_INFO("[%s]. %s OrderLaunch Phase2 is skipped, orderLaunchRequired is false.", __func__, modeName);
        return HCCL_SUCCESS;
    }

    // 1、复用Phase1缓存的Host侧保序线程句柄，避免二次acquire结果漂移
    ThreadHandle hostOrderThread = hostOrderThread_;
    CHK_PRT_RET(
        hostOrderThread == 0,
        HCCL_ERROR(
            "[%s]. %s hostOrderThread cached in Phase1 is null while orderLaunchRequired is true, internal state "
            "error.",
            __func__, modeName),
        HCCL_E_INTERNAL);
    HCCL_INFO("[%s]. %s reuse Phase1 hostOrderThread [0x%llx]", __func__, modeName, hostOrderThread);

    // 2、wait notify1
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(hostOrderThread, notifyIdx, execTimeout_)));

    // 3、仅 ACLGRAPH 模式需要在 hostOrderStream 上 record event，unfoldStream 等待 event
    if (launchMode_ == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {
        CHK_RET(AclgraphOrderLaunchEventToKernelStream(comm, hostOrderThread, unfoldThread, event1Guard_.Get()));
    }

    HCCL_INFO("[%s]. %s OrderLaunch Phase2 Success.timeout[%u].", __func__, modeName, execTimeout_);
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
