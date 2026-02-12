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
#include "coll_alg_v2_exec_registry.h"
#include "hcomm_primitives.h"
#include "dfx/task_exception_fun.h"
#include "kernel_launch.h"

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
    if (param->deviceType == DevType::DEV_TYPE_910_95) {
        AlgResourceCtxSerializable resCtx;

        char *ctx = static_cast<char*>(param->resCtx);
        std::vector<char> seq(ctx, ctx + param->ctxSize);
        resCtx.DeSerialize(seq);
        // 还原变长指针
        HcclResult ret = HCCL_SUCCESS;
        if (param->opType == HCCL_CMD_BATCH_SEND_RECV) {
            ret = RestoreVarDataBatchSendRecv(*param);
        } else if (param->opType == HCCL_CMD_ALLTOALLV || param->opType == HCCL_CMD_ALLTOALLVC ||
            param->opType == HCCL_CMD_ALLTOALL) {
            ret = RestoreVarDataAlltoAllV(*param, resCtx);
        } else if (param->opType == HCCL_CMD_REDUCE_SCATTER_V) {
            ret = RestoreVarDataReduceScatterV(*param, resCtx);
        } else if (param->opType == HCCL_CMD_ALLGATHER_V) {
            ret = RestoreVarDataAllGatherV(*param, resCtx);
        }
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("failed to restore optype [%d] data and counts.", param->opType);
            return 1;
        }
        // 获取Device测主thread
        ThreadHandle thread = resCtx.threads[0];
        if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set batch mode, tag is %s.", param->algTag);
            return 1;
        }

        // 主thread等待Host stream的通知
        if (HcommAclrtNotifyWaitOnThread(thread, resCtx.notifyIds[0], CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
            HCCL_ERROR("failed to wait notify[%d] from host main stream", resCtx.notifyIds[0]);
            return 1;
        }

        std::shared_ptr<InsCollAlgBase> executor =
                                            CollAlgExecRegistryV2::Instance().GetAlgExec(param->opType, algName);
        if (executor.get() == nullptr) {
            HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str());
            return 1;
        }

        // 执行算法编排
        if (executor->Orchestrate(*param, resCtx) != HCCL_SUCCESS) {
            HCCL_ERROR("orchestrate failed for alg:%s", param->algName);
            return 1;
        }
        // 主thread通知Host stream
        if (HcommAclrtNotifyRecordOnThread(thread, resCtx.notifyIds[1]) != HCCL_SUCCESS) {
            HCCL_ERROR("failed to record host main stream");
            return 1;
        }

        if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
            return 1;
        }
    } else {
        std::unique_ptr<ExecutorBase> executor = CollAlgExecRegistry::Instance().GetAlgExec(algName);
        if (executor.get() == nullptr) {
            HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str());
            return 1;
        }
        AlgResourceCtx* resCtx = reinterpret_cast<AlgResourceCtx*>(param->resCtx);
        // 获取Device测主thread
        ThreadHandle* threadHandlePtr = reinterpret_cast<ThreadHandle *>(reinterpret_cast<u8 *>(resCtx) +
            sizeof(AlgResourceCtx));
        ThreadHandle thread = threadHandlePtr[0];
        ThreadHandle exportedAicpuTsThread = resCtx->opThread;
    	u32 notifyNumOnMainThread = resCtx->notifyNumOnMainThread;
        if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set batch mode, tag is %s.", param->algTag);
            return 1;
        }

        if (HcommProfilingInit(threadHandlePtr, resCtx->slaveThreadNum+1) != HCCL_SUCCESS)
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
        if (executor->Orchestrate(*param, resCtx) != HCCL_SUCCESS) {
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
        profInfo.rankSize = resCtx->topoInfo.userRankSize;
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

        if (HcommProfilingEnd(threadHandlePtr, resCtx->slaveThreadNum + 1) != HCCL_SUCCESS) {
            HCCL_ERROR("failed to End Profiling");
            return 1;
        }
    }

    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommReleaseComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }
    HCCL_INFO("%s success, tag[%s], algTag[%s], commName[%s]", __func__, param->tag, param->algTag, param->commName);
    return 0;
}

HcclResult RestoreVarDataBatchSendRecv(OpParam &param)
{
    u64 sendRecvItemSize = static_cast<u64>(sizeof(HcclSendRecvItem));
    u64 itemNum = static_cast<u64>(param.batchSendRecvDataDes.itemNum);
    if (param.varMemSize != itemNum * sendRecvItemSize) {
        HCCL_ERROR("param.varMemSize[%lu] is not equal to itemNum[%lu] multiply [HcclSendRecvItem] size[%lu]."\
            "Failed to restore end recv info for BatchSendRecv!",
            param.varMemSize, itemNum, sendRecvItemSize);
        return HCCL_E_PARA;
    }
    param.batchSendRecvDataDes.sendRecvItemsPtr = reinterpret_cast<HcclSendRecvItem*>(param.varData);
    return HCCL_SUCCESS;
}

HcclResult RestoreVarDataAlltoAllV(OpParam &param, AlgResourceCtxSerializable &resCtx)
{
    u64 rankSize = resCtx.topoInfo.userRankSize;
    CHK_PRT_RET(param.varMemSize != ALL_TO_ALL_V_VECTOR_NUM * rankSize * sizeof(u64),
        HCCL_ERROR("[RestoreVarDataAlltoAllV] param.varMemSize [%llu] is invalid,"
            " ALL_TO_ALL_V_VECTOR_NUM is [%u], rankSize is [%u], sizeof(u64) is [%u],",
            param.varMemSize, ALL_TO_ALL_V_VECTOR_NUM, rankSize, sizeof(u64)), HCCL_E_PARA);

    u64* data = reinterpret_cast<u64*>(param.varData);
    param.all2AllVDataDes.sendCounts = data;
    param.all2AllVDataDes.recvCounts = data + rankSize;
    param.all2AllVDataDes.sdispls = data + 2 * rankSize;
    param.all2AllVDataDes.rdispls = data + 3 * rankSize;

    return HCCL_SUCCESS;
}

HcclResult RestoreVarDataReduceScatterV(OpParam &param, AlgResourceCtxSerializable &resCtx)
{
    u64 rankSize = resCtx.topoInfo.userRankSize;
    HCCL_INFO("rankSize:%u", rankSize);
    CHK_PRT_RET(param.varMemSize != REDUCE_SCATTER_V_VECTOR_NUM * rankSize * sizeof(u64),
        HCCL_ERROR("[RestoreVarDataReduceScatterV] param.varMemSize [%llu] is invalid,"
            "REDUCE_SCATTER_V_VECTOR_NUM is [%u], rankSize is [%u], sizeof(u64) is [%u],",
            param.varMemSize, REDUCE_SCATTER_V_VECTOR_NUM, rankSize, sizeof(u64)), HCCL_E_PARA);
 
    u64* data = reinterpret_cast<u64*>(param.varData);
    param.vDataDes.counts = data;
    param.vDataDes.displs = data + rankSize;
    return HCCL_SUCCESS;
}

HcclResult RestoreVarDataAllGatherV(OpParam &param, AlgResourceCtxSerializable &resCtx)
{
    u64 rankSize = resCtx.topoInfo.userRankSize;
    HCCL_INFO("rankSize:%u", rankSize);
    CHK_PRT_RET(param.varMemSize != ALL_GATHER_V_VECTOR_NUM * rankSize * sizeof(u64),
        HCCL_ERROR("[RestoreVarDataAllGatherV] param.varMemSize [%llu] is invalid,"
            "ALL_GATHER_V_VECTOR_NUM is [%u], rankSize is [%u], sizeof(u64) is [%u],",
            param.varMemSize, ALL_GATHER_V_VECTOR_NUM, rankSize, sizeof(u64)), HCCL_E_PARA);
    
    u64* data = reinterpret_cast<u64*>(param.varData);
    param.vDataDes.counts = data;
    for (u64 i=0;i<rankSize;i++){
        HCCL_INFO("param.vDataDes.counts[%u]:%u", i, reinterpret_cast<u64*>(param.vDataDes.counts)[i]);
    }
    param.vDataDes.displs = data + rankSize;
    for (u64 i=0;i<rankSize;i++){
        HCCL_INFO("param.vDataDes.displs[%u]:%u", i, reinterpret_cast<u64*>(param.vDataDes.displs)[i]);
    } 
    return HCCL_SUCCESS;
}