/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_op.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>
#include <cstring>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "sal.h"
#include "error_codes/rt_error_codes.h"
#include "mmpa_api.h"
#include "param_check.h"
#include "executor_base.h"
#include "coll_alg_exec_registry.h"
#include "alg_env_config.h"
#include "adapter_acl.h"
#include "topo.h"
#include "adapter_error_manager_pub.h"
#include "hccl_comm.h"
#include "config_log.h"
#include "workflow.h"
#include "load_kernel.h"
#include "alg_template_base.h"
#include "alg_template_register.h"
#include "hccl_inner.h"

using namespace std;
using namespace ops_hccl;

extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    if (!ops_hccl_reduce_scatter::CheckHCCLIndependentOp()) {
        return HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    }
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    // 非95设备转到老流程
    if (deviceType != DevType::DEV_TYPE_910_95) {
        return HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    }
    // 图模式引导到老的流程上面
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    }

    TopoInfo topoInfo;
    CHK_RET(GetSomeTopoInfo(comm, topoInfo));
    bool hostDPUOnly = false;
    if ((ops_hccl_reduce_scatter::CheckHostDPUOnly(comm, topoInfo, hostDPUOnly) != HCCL_SUCCESS) || !hostDPUOnly) {
        return HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    }
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    // A3是：export HCCL_OP_EXPANSION_MODE="AI_CPU"，A5的接口还没提供
    CHK_RET(InitEnvConfig());

    // 参数校验等工作
    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("input recvCount is 0, return reduce scatter success"), HCCL_SUCCESS);
    CHK_RET(ops_hccl_reduce_scatter::CheckReduceScatterInputPara(comm, sendBuf, recvBuf));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string tag = "ReduceScatter_" + string(commName);
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), recvCount, dataType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());
    CHK_RET(ops_hccl_reduce_scatter::CheckCount(recvCount));
    CHK_RET(ops_hccl_reduce_scatter::CheckDataType(dataType, true));

    // 执行ReduceScatter
    CHK_RET_AND_PRINT_IDE(ops_hccl_reduce_scatter::ReduceScatterOutPlace(sendBuf, recvBuf, recvCount, dataType, op, comm, stream, tag),
                          tag.c_str());

    return HCCL_SUCCESS;
}

namespace ops_hccl_reduce_scatter {
HcclResult CheckReduceScatterInputPara(HcclComm comm, void *sendBuf, void *recvBuf)
{
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatter", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatter", "sendBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatter", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);
    CHK_PRT_RET(sendBuf == recvBuf,
        HCCL_ERROR("[HcclReduceScatter] sendBuf and recvBuf cannot be same."), HCCL_E_PARA);

    return HCCL_SUCCESS;
}

HcclResult CheckCount(const u64 count)
{
    if (count > SYS_MAX_COUNT) {
        HCCL_ERROR("[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])",
            HCCL_ERROR_CODE(HCCL_E_PARA), count, SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CheckDataType(const HcclDataType dataType, bool needReduce)
{
    const std::vector<std::string> infoTitle({"ccl_op", "parameter", "value", "tips"});
    if (needReduce) {
        if ((dataType == HCCL_DATA_TYPE_UINT64) ||
            (dataType == HCCL_DATA_TYPE_UINT8)  || (dataType == HCCL_DATA_TYPE_UINT16) ||
            (dataType == HCCL_DATA_TYPE_UINT32) || (dataType == HCCL_DATA_TYPE_FP64) ||
            (dataType == HCCL_DATA_TYPE_RESERVED)) {
            RPT_INPUT_ERR(true, "EI0003", infoTitle, std::vector<std::string>({"CheckDataType", "dataType", GetDataTypeEnumStr(dataType), "please check dataType"}));
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                        HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                        GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else {
        if ((dataType >= HCCL_DATA_TYPE_RESERVED) || (dataType < HCCL_DATA_TYPE_INT8)) {
            RPT_INPUT_ERR(true, "EI0003", infoTitle, std::vector<std::string>({"CheckDataType", "dataType", GetDataTypeEnumStr(dataType), "please check dataType"}));
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                        HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                        GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

std::string GetSupportDataType(bool needReduce)
{
    std::vector<HcclDataType> supportList = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT32,
                                             HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32};
    if (needReduce) {
        supportList.insert(supportList.end(), {HCCL_DATA_TYPE_BFP16, HCCL_DATA_TYPE_INT64});
    } else {
        supportList.insert(supportList.end(), {HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT16,
                                               HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64});
        supportList.push_back(HCCL_DATA_TYPE_BFP16);
    }

    std::string supportInfo = "";
    for (u32 i = 0; i < supportList.size(); i++) {
        if (i != 0) {
            supportInfo += ", ";
        }
        supportInfo += GetDataTypeEnumStr(supportList[i]);
    }

    return supportInfo;
}

HcclResult ReduceScatterOutPlace(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, const std::string &tag)
{
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 outputSize = recvCount * perDataSize;
    u64 inputSize = outputSize * userRankSize;

    OpParam param;
    param.stream = stream;
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(param.deviceType));

    param.engine = CommEngine::COMM_ENGINE_CPU;
    CHK_RET(LoadAICPUKernel());

    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());  // topoInfo的tag，所有算子可以共享
    if (ret <= 0) {
        HCCL_ERROR("failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }
    CHK_RET(HcclGetCommName(comm, param.commName));

    param.inputPtr = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    param.reduceType = op;
    param.DataDes.count = recvCount;
    param.DataDes.dataType = dataType;
    // opType传下去的作用是什么
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    CHK_RET(ExecOp(comm, param));

    return HCCL_SUCCESS;
}

aclrtNotify g_notifies[AICPU_CONTROL_NOTIFY_NUM];
/* 执行通信算子 */
HcclResult ExecOp(HcclComm comm, OpParam &param)
{
    // 获取基础拓扑
    TopoInfo *topoInfo = nullptr;
    CHK_RET(CalcBaseTopoInfo(comm, param, &topoInfo));

    // 需要先解析环境变量
    AlgType algType;
    CHK_RET(GetAlgType(topoInfo, param.opType, algType));

    // 算法选择
    std::string algName;
    CHK_RET(SelectAlg(comm, param, topoInfo, algType, algName));
    std::unique_ptr<ExecutorBase> executor = CollAlgExecRegistry::Instance().GetAlgExec(algName);
    CHK_PRT_RET(executor.get() == nullptr, HCCL_ERROR("[ExecOp]Fail to find executor for algName[%s]",
        algName.c_str()), HCCL_E_PARA);

    // 获取资源
    AlgResourceCtx *resCtx;
    DPUAlgResourceCtx dpuResCtx;
    CHK_RET(GetAlgRes(comm, param, executor, topoInfo, algType, &resCtx, g_notifies, dpuResCtx));

    // 算法执行
    if (param.engine == COMM_ENGINE_CPU) {
        // 当前aicpu launch接口只能有一个输入参数，将Context指针放在param参数中
        param.resCtx = resCtx;
        // 将算法名字放在param参数中
        int result = sprintf_s(param.algName, sizeof(param.algName), "%s", algName.c_str());
        if (result <= 0) {
            HCCL_ERROR("faled to fill param.algName");
            return HCCL_E_INTERNAL;
        }

        CHK_RET(static_cast<HcclResult>(HcclTaskRegister(comm, param.algTag, HcclLaunchDpuKernel)));

        // Host stream通知Device主thread
        if (aclrtRecordNotify(g_notifies[0], param.stream) != ACL_SUCCESS) {
            HCCL_ERROR("failed to record aicpu stream");
            return HCCL_E_INTERNAL;
        }
        // 执行device测的算法编排
        std::string kernelName = "HcclLaunchAicpuKernel";
        aclrtFuncHandle funcHandle;
        aclrtArgsHandle argsHandle;

        aclError ret = aclrtBinaryGetFunction(g_binKernelHandle, kernelName.c_str(), &funcHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
            HCCL_ERROR("[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, kernelName:%s",
            ret, kernelName.c_str()), HCCL_E_RUNTIME);

        ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
            HCCL_ERROR("[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, kernelName:%s",
            ret, kernelName.c_str()), HCCL_E_RUNTIME);

        aclrtParamHandle paraHandle;
        ret = aclrtKernelArgsAppend(argsHandle, &param, sizeof(OpParam), &paraHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
            HCCL_ERROR("[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append size %u, kernelName:%s", ret,
            sizeof(OpParam), kernelName.c_str()), HCCL_E_RUNTIME);

        ret = aclrtKernelArgsFinalize(argsHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
            HCCL_ERROR("[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, kernelName:%s", ret,
            kernelName.c_str()), HCCL_E_RUNTIME);

        u16 NOTIFY_DEFAULT_WAIT_TIME = 27 * 68;   // notifywait默认1836等待时长
        aclrtLaunchKernelCfg cfg;
        aclrtLaunchKernelAttr attr;
        attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
        attr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME;
        cfg.numAttrs = 1;
        cfg.attrs = &attr;
        constexpr u32 blockDim = 1;
        aclError aclRet = aclrtLaunchKernelWithConfig(funcHandle, blockDim, param.stream, &cfg, argsHandle, nullptr);
        CHK_PRT_RET(aclRet != ACL_SUCCESS,
            HCCL_ERROR("[LoadCustomKernel][aclrtLaunchKernelWithConfig]errNo[0x%016llx] launch kernel failed", ret),
            HCCL_E_OPEN_FILE_FAILURE);
        // Host stream等待Device的通知
        if (aclrtWaitAndResetNotify(g_notifies[1], param.stream, CUSTOM_TIMEOUT) != ACL_SUCCESS) {
            HCCL_ERROR("failed to wait from aicpu stream");
            return HCCL_E_INTERNAL;
        }
        if (aclrtSynchronizeStream(param.stream) != 0) {
            HCCL_ERROR("Stream Synchronize Failed");
            return HCCL_E_INTERNAL;
        }
    }

    return HCCL_SUCCESS;
}

/* 算子级别基础拓扑解析，缓存在host上 */
HcclResult CalcBaseTopoInfo(HcclComm comm, OpParam &param, TopoInfo **topoInfo)
{
    void *topoInfoMemAddr;
    uint64_t topoInfoMemSize = sizeof(TopoInfo);
    if (HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, &topoInfoMemAddr, &topoInfoMemSize) != HCCL_SUCCESS) {
        // 创建新的Context
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, topoInfoMemSize, &topoInfoMemAddr));
        // 将Context内存地址强转为TopoInfo
        *topoInfo = static_cast<TopoInfo *>(topoInfoMemAddr);
        // 将对应拓扑信息填入到Context内存中
        DevType deviceType = DevType::DEV_TYPE_COUNT;
        CHK_RET(hrtGetDeviceType(deviceType));
        if (deviceType == DevType::DEV_TYPE_910_93) {
            CHK_RET(InitRankInfo(comm, *topoInfo));
        } else if (deviceType == DevType::DEV_TYPE_910_95) {
            CHK_RET(InitRankInfoForA5(comm, *topoInfo));
        }
        return HCCL_SUCCESS;
    }

    *topoInfo = static_cast<TopoInfo *>(topoInfoMemAddr);
    return HCCL_SUCCESS;
}

HcclResult GetAlgType(TopoInfo *topoInfo, HcclCMDType opType, AlgType &algType)
{
    std::vector<HcclAlgoType> ret;
    ret = GetExternalInputHcclAlgoConfig(opType);
    if (ret.size() != HCCL_ALGO_LEVEL_NUM) {
        HCCL_ERROR("alg type size is invalid");
        return HCCL_E_PARA;
    }

    AlgTypeLevel0 algType0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    AlgTypeLevel1 algType1 = AlgTypeLevel1::ALG_LEVEL1_RESERVED;
    AlgTypeLevel2 algType2 = AlgTypeLevel2::ALG_LEVEL2_RESERVED;

    algType.algoLevel0 = algType0;
    algType.algoLevel1 = algType1;
    algType.algoLevel2 = algType2;

    return HCCL_SUCCESS;
}

HcclResult SelectAlg(HcclComm comm, OpParam &param, TopoInfo *topoInfo, AlgType &algType, std::string &algName)
{
    if (topoInfo->deviceType == DevType::DEV_TYPE_910_95) {
        algName = "ReduceScatterMeshExecutor";
    }

    // 在原先的tag中添加算法名字，得到algTag
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        int ret = sprintf_s(param.algTag, sizeof(param.algTag), "%s_%s", param.tag, algName.c_str());
        if (ret <= 0) {
            HCCL_ERROR("faled to fill param.algTag");
            return HCCL_E_INTERNAL;
        }
    }

    // 在algTag中追加编排模式
    const char *launchMode = (param.engine == CommEngine::COMM_ENGINE_CPU_TS ? "_aicpu" : "_host");
    launchMode = "_dpu";
    int ret = strcat_s(param.algTag, sizeof(param.algTag), launchMode);
    if (ret != 0) {
        HCCL_ERROR("faled to fill param.algTag");
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[SelectAlg] ReduceScatter algTag is [%s] algName is [%s]", param.algTag, algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult GetAlgRes(HcclComm comm, OpParam &param, std::unique_ptr<ExecutorBase> &executor,
    TopoInfo *topoInfo, AlgType &algType, AlgResourceCtx **resCtx, aclrtNotify* notifies, DPUAlgResourceCtx &dpuResCtx)
{
    // 这种情况下资源已经有了
    void *resCtxMemAddr;
    uint64_t resCtxMemSize;
    if (HcclEngineCtxGet(comm, param.algTag, COMM_ENGINE_AICPU_TS, &resCtxMemAddr, &resCtxMemSize) == HCCL_SUCCESS) {
        *resCtx = static_cast<AlgResourceCtx *>(resCtxMemAddr);
        return HCCL_SUCCESS;
    }

    // 资源计算
    AlgHierarchyInfo algHierarchyInfo;
    AlgResourceRequest resRequest;
    CHK_RET(executor->CalcResRequest(comm, param, topoInfo, algHierarchyInfo, resRequest, algType));
    HCCL_INFO("GetAlgRes info : resRequest.slaveThreadNum : [%d], algHierarchyInfo.infos[0].size: [%zu]", resRequest.slaveThreadNum, algHierarchyInfo.infos[0].localRankSize);
    // 开始计算aicpu资源Context的长度
    // HcclMem resCtxMem;
    resCtxMemSize = sizeof(AlgResourceCtx);
    // 计算变长数据区中threads占用的空间
    resCtxMemSize += sizeof(ThreadHandle) * (resRequest.slaveThreadNum + 1);
    // 计算变长数据区中channels占用的空间
    resCtxMemSize += sizeof(ChannelInfo) * algHierarchyInfo.infos[0].localRankSize;
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, CommEngine::COMM_ENGINE_AICPU_TS, resCtxMemSize, &resCtxMemAddr));
    // 将内存强转为AlgResourceCtx结构体
    *resCtx = static_cast<AlgResourceCtx *>(resCtxMemAddr);

    AlgResourceCtx *resCtxHost;
    if (param.engine == COMM_ENGINE_CPU) {
        // AICPU模式下分配一块Host内存用于填充资源
        ACLCHECK(aclrtMallocHost(reinterpret_cast<void**>(&resCtxHost), resCtxMemSize));
    }

    uint64_t shmemSize = 100 * 1024 * 1024;
    void *shmemPtr = nullptr;
    bool newCreated;
    CHK_RET(HcclDevMemAcquire(comm, "DPUTAG", &shmemSize, &shmemPtr, &newCreated));

    resCtxHost->topoInfo = *topoInfo;
    resCtxHost->algType = algType;
    resCtxHost->algHierarchyInfo = algHierarchyInfo;
    resCtxHost->npu2DpuShmemPtr = shmemPtr;
    resCtxHost->dpu2NpuShmemPtr = static_cast<void*>(static_cast<uint8_t*>(shmemPtr) + shmemSize / 2);

    // 创建资源，并填充到Host内存上
    HcclResult ret = AllocAlgResource(comm, param, resRequest, resCtxHost, notifies);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("failed to alloc alg resource.");
        if (param.engine == COMM_ENGINE_CPU) {
            ACLCHECK(aclrtFreeHost(resCtxHost));
        }
        return ret;
    }

    dpuResCtx.tempIndex = static_cast<uint32_t>(TEMPLATE_REDUCE_SCATTER_HOST_DPU);
    dpuResCtx.algHierarchyInfo = algHierarchyInfo;
    dpuResCtx.cclInputMem = resCtxHost->cclInputMem;
    dpuResCtx.cclOutputMem = resCtxHost->cclOutputMem;
    ret = AllocDpuAlgResource(comm, param, resRequest, dpuResCtx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("failed to alloc alg resource.");
        return ret;
    }
    resCtxHost->dpuResCtx = dpuResCtx;

    if (param.engine == COMM_ENGINE_CPU) {
        // 从Host内存拷贝到Device Context内存上
        ACLCHECK(aclrtMemcpy(*resCtx, resCtxMemSize,
            resCtxHost, resCtxMemSize, ACL_MEMCPY_HOST_TO_DEVICE));
        ACLCHECK(aclrtFreeHost(resCtxHost));
    }

    return HCCL_SUCCESS;
}

HcclResult AllocAlgResource(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    AlgResourceCtx *resCtxHost, aclrtNotify* notifies)
{
    void* cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    u64 sizePerCcl = cclBufferSize / 2;
    // CCL IN使用CCL Buffer的前一半
    resCtxHost->cclInputMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, sizePerCcl};
    // CCL OUT使用CCL Buffer的后一半
    resCtxHost->cclOutputMem = HcclMem{HCCL_MEM_TYPE_DEVICE,
        static_cast<void*>(static_cast<s8 *>(cclBufferAddr) + sizePerCcl), sizePerCcl};
    resCtxHost->notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
    resCtxHost->slaveThreadNum = resRequest.slaveThreadNum;
    resCtxHost->notifyNumPerThread = resRequest.notifyNumPerThread;

    #define ACL_NOTIFY_DEFAULT          0x00000000U
    // 先使用acl接口来分配notify
    if (aclrtCreateNotify(&(notifies[0]), ACL_NOTIFY_DEFAULT) != ACL_SUCCESS) {
        HCCL_ERROR("failed to alloc notify");
        return HCCL_E_INTERNAL;
    }

    if (aclrtCreateNotify(&(notifies[1]), ACL_NOTIFY_DEFAULT) != ACL_SUCCESS) {
        HCCL_ERROR("failed to alloc notify");
        return HCCL_E_INTERNAL;
    }

    for (u32 idx = 0; idx < AICPU_CONTROL_NOTIFY_NUM; idx++) {
        uint32_t notifyId;
        // 获取notify Id，放入Context中
        if (aclrtGetNotifyId(notifies[idx], &notifyId) != ACL_SUCCESS) {
            HCCL_ERROR("failed to get notify id");
            return HCCL_E_INTERNAL;
        }
        resCtxHost->notifyIds[idx] = notifyId;
    }

    s8 *curPtr = reinterpret_cast<s8 *>(resCtxHost);
    curPtr += sizeof(AlgResourceCtx); // 偏移指针
    ThreadHandle *threads = reinterpret_cast<ThreadHandle *>(curPtr);
    if (param.engine == COMM_ENGINE_CPU) {
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, resRequest.notifyNumOnMainThread, threads));
    }
    curPtr += sizeof(ThreadHandle); // 指针向后偏移

    if (resRequest.slaveThreadNum > 0) {
        threads = reinterpret_cast<ThreadHandle *>(curPtr);
        // 创建从流thread及对应的notify
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, resRequest.slaveThreadNum,
            resRequest.notifyNumPerThread, threads));
        curPtr += sizeof(ThreadHandle) * resCtxHost->slaveThreadNum; // 指针向后偏移
    }
    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
        HCCL_DEBUG("[AllocAlgResource] slaveThreadNum[%u]", resRequest.slaveThreadNum);
        for (u32 i = 0; i < resRequest.slaveThreadNum; i++) {
            HCCL_DEBUG("[AllocAlgResource] threads[%u]=[%llu]", i, threads[i]);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllocDpuAlgResource(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    DPUAlgResourceCtx &dpuResCtx)
{
    AlgHierarchyInfo algHierarchyInfo = dpuResCtx.algHierarchyInfo;
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        // 暂时没有处理level0的场景
        if (level == 0) {
            continue;
        }
        dpuResCtx.channelNum += algHierarchyInfo.infos[level].localRankSize;
    }
    void *channelMemAddr;
    uint64_t channelMemSize = dpuResCtx.channelNum * sizeof(ChannelInfo);
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, CommEngine::COMM_ENGINE_CPU, channelMemSize, &channelMemAddr));
    dpuResCtx.channels = static_cast<ChannelInfo*>(channelMemAddr);

    // 迭代每个子通信域的建链请求，创建链路
    auto channels = dpuResCtx.channels;
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        // 暂时没有处理level0的场景
        if (level == 0) {
            continue;
        }
        // 获取子通信域的建链请求
        std::vector<HcclChannelDesc> &levelNChannelRequest = resRequest.channels[level];
        // 获取子通信域的建链数量
        u32 validChannelNum = levelNChannelRequest.size();
        std::vector<ChannelHandle> levelNChannels;
        levelNChannels.resize(validChannelNum);

        if (validChannelNum > 0) {
            // 调用控制面接口创建链路
            CHK_RET(HcclChannelAcquire(comm, param.engine, levelNChannelRequest.data(),
                validChannelNum, levelNChannels.data()));
        }

        SubCommInfo subCommInfo = dpuResCtx.algHierarchyInfo.infos[level];
        // 因为在创建context的时候，每一层预留的链路数量是子通信域rank数量，
        // 建链数量可能小于子通信域rank数量，所以先把链路全部设置为false，
        // 后再填充有效的链路
        for (u32 idx = 0; idx < subCommInfo.localRankSize; idx++) {
            // 先全部设置为false
            channels[idx].isValid = false;
        }
        for (u32 idx = 0; idx < validChannelNum; idx++) {
            // 对于真实建链的链路进行填充
            HcclChannelDesc &channelDesc = levelNChannelRequest[idx];
            u32 levelRank = channelDesc.remoteRank;
            channels[levelRank].isValid = true;
            channels[levelRank].remoteRank = channelDesc.remoteRank;
            channels[levelRank].protocol = channelDesc.localEndpoint.protocol;
            channels[levelRank].notifyNum = channelDesc.notifyNum;
            channels[levelRank].handle = levelNChannels[idx];

            void* remoteBufferAddr;
            uint64_t remoteBufferSize;
            CHK_RET(HcclChannelGetHcclBuffer(comm, levelNChannels[idx], &remoteBufferAddr, &remoteBufferSize));
            channels[levelRank].remoteInput = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteBufferAddr, remoteBufferSize / 2};
            channels[levelRank].remoteOutput = HcclMem{HCCL_MEM_TYPE_DEVICE, static_cast<void*>(static_cast<s8 *>(remoteBufferAddr) + remoteBufferSize / 2), remoteBufferSize / 2};
        }
        channels += subCommInfo.localRankSize;
    }

    return HCCL_SUCCESS;
}

int32_t HcclLaunchDpuKernel(uint64_t shmemPtr, int32_t DataSize)
{
    (void) DataSize;
    // 解析共享内存数据
    DPUAlgResourceCtx *dpuResCtx = reinterpret_cast<DPUAlgResourceCtx *>(shmemPtr);

    // 根据算法名字获取executor
    auto templateIns = AlgTemplateRegistry::Instance().GetAlgTemplate(static_cast<TemplateType>(dpuResCtx->tempIndex));
    if (templateIns.get() == nullptr) {
        return 4;
    }

    // 执行算法编排
    if (templateIns->RunAsync(dpuResCtx) != HCCL_SUCCESS) {
        return 4;
    }

    return 0;
}

HcclResult CheckHostDPUOnly(HcclComm comm, const TopoInfo &topoInfo, bool &hostDPUOnly)
{
    hostDPUOnly = false;

    // 只有一个server，不使用DPU
    if (topoInfo.serverNum == 1) {
        HCCL_INFO("Not using host dpu because there is only 1 server");
        return HCCL_SUCCESS;
    }

    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    if ((netLayers == nullptr) || (netLayerNum == 0)) {
        HCCL_WARNING("HcclGetNetLayers fail");
        return HCCL_E_INTERNAL;
    }

    bool hostDPU = false;
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; layerIdx++) {
        uint32_t netLayer = netLayers[layerIdx];
        // 只校验最后一个level
        if (netLayer < (topoInfo.topoLevelNums - 1)) {
            HCCL_INFO("Skip checking because it is not the last layer");
            continue;
        }
        uint32_t *topoInsts = nullptr;
        uint32_t topoInsNum = 0;
        CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &topoInsts, &topoInsNum));
        if ((topoInsts == nullptr) || (topoInsNum == 0)) {
            HCCL_WARNING("HcclRankGraphGetTopoInstsByLayer fail, netLayer[%u]", netLayer);
            return HCCL_E_INTERNAL;
        }
        for (uint32_t topoInsIdx = 0; topoInsIdx < topoInsNum; topoInsIdx++) {
            uint32_t topoInstId = topoInsts[topoInsIdx];
            CommTopo topoType;
            CHK_RET(HcclRankGraphGetTopoType(comm, netLayer, topoInstId, &topoType));
            if (topoType != COMM_TOPO_CLOS) {
                HCCL_INFO("Skip checking because the topo is not CLOS");
                continue;
            }
            uint32_t *ranks = nullptr;
            uint32_t rankNum = 0;
            CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInstId, &ranks, &rankNum));
            // 校验当前rank与其他所有rank连通
            if (rankNum != topoInfo.userRankSize) {
                HCCL_INFO("Skip checking because not all rank is linked");
                continue;
            }
            uint32_t endPointNums = 0;
            CHK_RET(HcclRankGraphGetEndpointNum(comm, netLayer, topoInstId, &endPointNums));
            EndpointDesc endPointDescs[endPointNums];
            CHK_RET(HcclRankGraphGetEndpointDesc(comm, netLayer, topoInstId, &endPointNums, endPointDescs));
            for (uint32_t endPointIdx = 0; endPointIdx < endPointNums; endPointIdx++) {
                EndpointDesc endPointDesc = endPointDescs[endPointIdx];
                if (endPointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
                    HCCL_INFO("Not using host dpu because there are links on device");
                    return HCCL_SUCCESS;
                } else if (endPointDesc.loc.locType == ENDPOINT_LOC_TYPE_HOST) {
                    HCCL_INFO("There are links on host");
                    hostDPU = true;
                }
            }
        }
    }
    if (hostDPU) {
        HCCL_INFO("Using host dpu to implement data trans");
        hostDPUOnly = true;
    }
    return HCCL_SUCCESS;
}

bool CheckHCCLIndependentOp()
{
    // 获取环境变量值
    const char* envValue = std::getenv("HCCL_INDEPENDENT_OP");
    
    // 检查环境变量是否存在且值为"1"
    if (envValue != nullptr && std::strcmp(envValue, "1") == 0) {
        return true;
    }
    
    return false;
}
}