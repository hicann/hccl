/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <future>
#include <map>
#include <string>
#include <memory>
#include <cstdlib>  // 包含getenv函数
#include <cstring>  // 包含strcmp函数
#include <stdexcept>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "sal.h"
#include "error_codes/rt_error_codes.h"
#include "mmpa_api.h"
#include "param_check.h"
#include "executor_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "alg_env_config.h"
#include "adapter_acl.h"
#include "topo_host.h"
#include "adapter_error_manager_pub.h"
#include "hccl_inner.h"
#include "hccl.h"
#include "config_log.h"
#include "workflow.h"
#include "load_kernel.h"
#include "alg_param.h"
#include "alg_type.h"
#include "op_common.h"
#include "hccl_aiv_utils.h"
#include "aiv_kernel_def.h"
#include "dpu/kernel_launch.h"
#include "hcomm_host_profiling_dl.h"
#include "rt.h"
#include "dlhcomm_function.h"

#ifdef __cplusplus
extern "C" {
#endif

// 兼容性处理
uint64_t __attribute__((weak)) HcommGetProfilingSysCycleTime();
HcclResult __attribute__((weak))  HcclDfxRegOpInfo(HcclComm comm, void* dfxOpInfo);
HcclResult __attribute__((weak)) HcclProfilingReportOp(HcclComm comm, uint64_t beginTime);
HcclResult __attribute__((weak)) HcclReportAicpuKernel(HcclComm comm, uint64_t beginTime, char *kernelName);
HcclResult __attribute__((weak)) HcclReportAivKernel(HcclComm comm, uint64_t beginTime);

struct HcclDfxOpInfo {
    CommAbiHeader       header;
    //DfxOpInfo_base
    uint64_t            beginTime = 0;
    uint64_t            endTime = 0;
    //baseCollOperator
    uint32_t            opMode = 0; // 单算子和图模式
    uint32_t            opType = 0; // 算子名称类型
    uint32_t            reduceOp = 0;
    uint32_t            dataType = 0;
    uint32_t            outputType = 0; //暂不删除，考虑后续算子使用
    uint64_t            dataCount = 0;
    uint32_t            root = INVALID_VALUE_RANKID;
    char                algTag[288]; // 算法名 = "算子类型 + 通信域id + 选择的算法"
    CommEngine          engine = COMM_ENGINE_RESERVED;
    //task_exception
    uint64_t            cpuTsThread = 0; // host侧算子主流的threadhandle
    uint32_t            cpuWaitAicpuNotifyIdx = INVALID_UINT; // host wait device notifyIdx
    uint32_t            cpuWaitAicpuNotifyId = INVALID_UINT; // host wait device notifyId
    int8_t              reserve[128]; // 预留扩展字段
};

#ifdef __cplusplus
}
#endif

namespace ops_hccl {
thread_local std::map<std::string, HcclMemHandle> g_memHandleCache; // 当前AIV存放注册内存的memHandle使用
// 用于维护增量建链算子的host ctx信息
thread_local std::map<std::string, std::unique_ptr<AlgResourceCtxSerializable>> g_hostCtx;
thread_local std::map<AivOpCacheArgs, std::shared_ptr<InsQueue>> g_hcclCacheMap;
constexpr u32 HOST_WAIT_AICPU_NOTIFYIDX = 0;// host主流wait aicpu流的notify idx

// 检查非对称拓扑支持情况
// 仅 AllGather, AllReduce, ReduceScatter 支持跨框非对称拓扑，其他算子拦截
HcclResult CheckAsymmetricTopoSupport(HcclCMDType opType, const TopoInfoWithNetLayerDetails* topoInfo)
{
    // 仅在跨框非对称场景下检查
    if (topoInfo->topoLevelNums > 1 && topoInfo->multiModuleDiffDeviceNumMode) {
        // 三个已适配非对称的算子：AllGather, AllReduce, ReduceScatter
        bool isSupportedOp = (opType == HcclCMDType::HCCL_CMD_ALLGATHER ||
                             opType == HcclCMDType::HCCL_CMD_ALLREDUCE ||
                             opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
        if (!isSupportedOp) {
            HCCL_ERROR("[CheckAsymmetricTopoSupport] OpType[%d] does not support asymmetric topology "
                "(multi-module diff device num mode), only ALLGATHER/ALLREDUCE/REDUCE_SCATTER are supported.",
                opType);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult Selector(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    std::string &algName)
{
    HCCL_INFO("Start to execute Selector.");
    param.hcclComm = comm;
    CHK_RET(HcclGetOpExpansionMode(comm, param));
    // 获取基础拓扑
    CHK_RET(HcclCalcTopoInfo(comm, param, topoInfo));

    // 检查非对称拓扑支持情况，非对称场景仅 AllGather/AllReduce/ReduceScatter 可用
    CHK_RET(CheckAsymmetricTopoSupport(param.opType, topoInfo.get()));

    // 算法选择，选择完后顺便param.algTag设置了，资源的保存是以算子+算法为单位
    std::shared_ptr<ExecuteSelector> collAlgSelector = std::make_shared<ExecuteSelector>(ExecuteSelector());
    CHK_RET(collAlgSelector->Run(param, topoInfo.get(), algName));
    if (algName == "") {
        HCCL_ERROR("[Selector] select algname fail!");
        return HCCL_E_PTR;
    }
    CHK_RET(SetCommEngine(param));
    // AIV_ONLY 模式下禁止回退到非 AIV 引擎，未选中 AIV 时直接返回不支持。
    if (GetExternalInputHcclAivOnlyMode() && param.engine != CommEngine::COMM_ENGINE_AIV) {
        HCCL_ERROR("[HcclExecOp] opType[%d] currently do not select aiv mode, aiv only not support.",
            static_cast<int>(param.opType));
        return HCCL_E_NOT_SUPPORT;
    }
    // 如果一开始读取到的Engine不是aicpu，经过算法选择后回退到aipcu，则需要重新LoadAICPUKernel
    if ((param.engine == CommEngine::COMM_ENGINE_AICPU_TS) || (param.engine == CommEngine::COMM_ENGINE_CPU)) {
        HCCL_DEBUG("[Selector] is aicpu mode");
        CHK_RET(LoadAICPUKernel()); // 该函数内部有防止重复加载的逻辑
    }
    CHK_RET(SetOpParamAlgTag(param, algName));
    HCCL_INFO("Success to execute Selector.");
    return HCCL_SUCCESS;
}

uint64_t GetHcclDfxOpInfoDataCount(const OpParam &param, const u32 &rankSize) {
    u64 sendCount = 0;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL
        || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        for (u64 i = 0; i < rankSize; i++) {
            sendCount += *(reinterpret_cast<const u64*>(param.all2AllVDataDes.sendCounts) + i);
        }
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        for (u64 i = 0; i < rankSize; i++) {
            sendCount += *(reinterpret_cast<const u64*>(param.varData) + i);
        }
    } else if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        for (u64 i = rankSize; i < 2*rankSize; i++) {
            sendCount += *(reinterpret_cast<const u64*>(param.varData) + i);
        }
    } else {
        sendCount = static_cast<u64>(param.DataDes.count);
    }
    return sendCount;
}

uint32_t GetHcclDfxOpInfoDataType(const OpParam &param) {
    uint32_t dataType = 0;
    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V
        || param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        dataType = static_cast<u32>(param.vDataDes.dataType);
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        dataType = static_cast<u32>(param.all2AllDataDes.sendType);
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        dataType = static_cast<u32>(param.all2AllVDataDes.sendType);
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        dataType = static_cast<u32>(param.all2AllVCDataDes.sendType);
    } else {
        dataType = static_cast<u32>(param.DataDes.dataType);
    }
    return dataType;
}

HcclResult SetOpParamFastLaunchTag(OpParam &param)
{
    HcclDataType tmpDataType;
    if(param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
        param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        tmpDataType = param.all2AllVDataDes.sendType;
    } else {
        tmpDataType = param.DataDes.dataType;
    }
    
    const std::string dataType = HCOM_DATA_TYPE_STR_MAP.at(tmpDataType);
    // 1.通信域tag + 数据类型，得到基础FastLaunchTag
    std::string tagBuilder = std::string(param.tag) + "_" + dataType;
    // 2.reduceType
    if (param.opType == HcclCMDType::HCCL_CMD_ALLREDUCE || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ||
        param.opType == HcclCMDType::HCCL_CMD_REDUCE || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        const std::string reduceType = HCOM_REDUCE_OP_STR_MAP.at(param.reduceType);
        tagBuilder += "_" + reduceType;
    }
    // 3.count
    if (param.opType != HcclCMDType::HCCL_CMD_ALLTOALLV) {
        std::string count = std::to_string(param.DataDes.count); //todo: alltoall 的count不是从这里取
        tagBuilder += "_" + count;
    }
    // 4.root
    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE || param.opType == HcclCMDType::HCCL_CMD_SCATTER ||
        param.opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        std::string root = std::to_string(param.root);
        tagBuilder += "_r" + root;
    }
    CHK_PRT_RET((tagBuilder.length() >= sizeof(param.fastLaunchTag)), 
        "failed to fill fastLaunchTag, tag too long", HcclResult::HCCL_E_INTERNAL);
    snprintf_s(param.fastLaunchTag, sizeof(param.fastLaunchTag), sizeof(param.fastLaunchTag), "%s", tagBuilder.c_str());

    HCCL_INFO("[SetOpParamFastLaunchTag] fastLaunchTag: [%s]", param.fastLaunchTag);
    return HcclResult::HCCL_SUCCESS;
}

bool ShouldGoCcuFastLaunch(HcclComm comm, OpParam &param, CcuFastLaunchCtx **ccuFastLaunchCtx)
{
    param.hcclComm = comm;
    
    // 1. 是ccu模式
    if (GetExternalInputHcclCcuMSMode()) {
        HCCL_DEBUG("[HcclExecOp] is ccu ms mode");
        param.opExecuteConfig = OpExecuteConfig::CCU_MS;
        param.engine = CommEngine::COMM_ENGINE_CCU;
    } else if (GetExternalInputHcclCcuSchedMode()) {
        HCCL_DEBUG("[HcclExecOp] is ccu sched mode");
        param.opExecuteConfig = OpExecuteConfig::CCU_SCHED;
        param.engine = CommEngine::COMM_ENGINE_CCU;
    } else {
        // 非CCU模式，返回走正常流程
        return false;
    }

    CHK_RET(SetOpParamFastLaunchTag(param));
    
    // 2. 查到engineCtx
    uint64_t size = 0;
    void *fastLaunchCtxPtr = nullptr;
    if (HcclEngineCtxGet(comm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, &fastLaunchCtxPtr, &size) == HCCL_SUCCESS) {
        HCCL_INFO("[ShouldGoCcuFastLaunch] get fastLaunchCtx success, size is %u", size);
        *ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(fastLaunchCtxPtr);
        return true;
    }
    return false;
}

HcclResult HcclExecOpCcuFastLaunch(HcclComm comm, OpParam &param, const CcuFastLaunchCtx *ccuFastLaunchCtx)
{
    HCCL_INFO("[HcclExecOpCcuFastLaunch] HcclExecOpCcuFastLaunch start");
    std::string algName = ccuFastLaunchCtx->algName;
    HCCL_DEBUG("[HcclExecOpCcuFastLaunch] algName: [%s]", algName.c_str());
    std::unique_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param.opType, algName);
    CHK_PRT_RET(
        executor.get() == nullptr, HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str()), HCCL_E_PARA);
    
    void *cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    // CCL IN使用所有的CCL Buffer，这个其实就是scratch buffer
    param.hcclBuff = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};
    
    HCCL_INFO("[HcclExecOpCcuFastLaunch] FastLaunch start");
    CHK_RET(executor->FastLaunch(param, ccuFastLaunchCtx));
    
    HCCL_INFO("[HcclExecOpCcuFastLaunch] HcclExecOpCcuFastLaunch end");
    return HCCL_SUCCESS;
}

HcclResult ExecuteAivCacheLogic(OpParam &param, const std::string &algName,
                                std::unique_ptr<InsCollAlgBase> &executor,
                                AlgResourceCtxSerializable &resCtxHost)
{
    // Cache Logic
    bool useCache = (param.opType != HCCL_CMD_ALLTOALLV && param.opType != HCCL_CMD_ALLTOALLVC && 
                     param.opType != HCCL_CMD_ALLGATHER_V && param.opType != HCCL_CMD_REDUCE_SCATTER_V);
    
    AivOpCacheArgs cacheKey = {};
    if (useCache) {
        cacheKey.commName = param.commName;
        cacheKey.algName = algName;
        cacheKey.opType = param.opType;
        cacheKey.root = param.root;
        cacheKey.reduceOp = param.reduceType;

        if (param.opType == HCCL_CMD_ALLTOALL) {
            cacheKey.sendType = param.all2AllDataDes.sendType;
            cacheKey.recvType = param.all2AllDataDes.recvType;
            cacheKey.sendCount = param.all2AllDataDes.sendCount;
            cacheKey.recvCount = param.all2AllDataDes.recvCount;
        } else {
            cacheKey.count = param.DataDes.count;
            cacheKey.dataType = param.DataDes.dataType;
        }
    }

    if (useCache && g_hcclCacheMap.find(cacheKey) != g_hcclCacheMap.end()) {
        // Hit
        auto queue = g_hcclCacheMap[cacheKey];
        for (auto& ins : *queue) {
            AivOpArgs newArgs = ins.opArgs;
            newArgs.stream = param.stream;
            
            // Update addresses
            newArgs.input = (u64)param.inputPtr + ins.inputOffset;
            newArgs.output = (u64)param.outputPtr + ins.outputOffset;
            
            CHK_RET(ExecuteKernelLaunch(newArgs));
        }
    } else {
        // Miss
        if (useCache) {
            g_recordingQueue = std::make_shared<InsQueue>();
            g_baseInputAddr = (u64)param.inputPtr;
            g_baseOutputAddr = (u64)param.outputPtr;
        }

        CHK_RET(executor->Orchestrate(param, resCtxHost));

        if (useCache && g_recordingQueue) {
            g_hcclCacheMap[cacheKey] = g_recordingQueue;
            g_recordingQueue = nullptr;
            g_baseInputAddr = 0;
            g_baseOutputAddr = 0;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclExecOp(HcclComm comm, OpParam &param,
                      std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo, std::string &algName, const ResPackGraphMode &resPack)
{
    uint64_t beginTime = HcommGetProfilingSysCycleTime();
    HCCL_INFO("[HcclExecOp]Start to execute HcclExecOp.HcommGetProfilingSysCycleTime.%llu", beginTime);
    // 在原先的commName中添加执行模式，得到commModeTag
    param.hcclComm = comm;
    bool isOpBase = true;
    const char* opModeStr = isOpBase ? "_opbase" : "_offload";
    auto ret = sprintf_s(param.commModeTag, sizeof(param.commModeTag), "%s_%s", param.commName, opModeStr);
    if (ret <= 0) {
        HCCL_ERROR("[%s] failed to fill param.commModeTag", __func__);
        return HCCL_E_INTERNAL;
    }

    std::unique_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param.opType, algName);
    CHK_PRT_RET(
        executor.get() == nullptr, HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str()), HCCL_E_PARA);

    // 资源结构体
    std::unique_ptr<AlgResourceCtxSerializable> resCtxHost = std::make_unique<AlgResourceCtxSerializable>();
    // 资源序列化结果
    void *resCtxSequence = nullptr;
    bool isResourceReused = false;

    ThreadHandle cpuTsThread{0};
    ThreadHandle exportedAicpuTsThread{0};
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, 1, &cpuTsThread));
        // Export cpuTsThread
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &cpuTsThread, COMM_ENGINE_AICPU_TS, &exportedAicpuTsThread));
    }

    CHK_RET(HcclGetAlgRes(comm, param, executor, topoInfo.get(), resCtxHost, &resCtxSequence, isResourceReused));

    // Op注册
    HcclDfxOpInfo hcclDfxOpInfo{};
    hcclDfxOpInfo.opMode = static_cast<u32>(param.opMode);
    hcclDfxOpInfo.opType = static_cast<u32>(param.opType);
    hcclDfxOpInfo.reduceOp = static_cast<u32>(param.reduceType);
    hcclDfxOpInfo.dataType = GetHcclDfxOpInfoDataType(param);

    // rankSize获取指定算子的dataCount
    u32 userRankSize{0};
    CHK_RET(HcclGetRankSize(comm, &userRankSize));
    hcclDfxOpInfo.dataCount = GetHcclDfxOpInfoDataCount(param, userRankSize);
    hcclDfxOpInfo.root = param.root;
    hcclDfxOpInfo.engine = param.engine;
    hcclDfxOpInfo.cpuTsThread = cpuTsThread;
    hcclDfxOpInfo.cpuWaitAicpuNotifyIdx = HOST_WAIT_AICPU_NOTIFYIDX;
    s32 sRet = strncpy_s(hcclDfxOpInfo.algTag, ALG_TAG_LENGTH, param.algTag, ALG_TAG_LENGTH);
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("%s call strncpy_s failed, param.algTag %s,  return %d.", __func__, param.algTag, sRet), HCCL_E_MEMORY);
    HcclDfxOpInfo *tempOp = &hcclDfxOpInfo;

    CHK_RET(HcclDfxRegOpInfo(comm, static_cast<void*>(tempOp)));
    ThreadHandle exportedCpuTsThread;
    ThreadHandle mainThread;
    u32 notifyNumOnMainThread;
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        // 获取主流信息
        CHK_RET(GetMainThreadInfo(comm, param, mainThread, notifyNumOnMainThread));
        // Export mainThread
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &mainThread, COMM_ENGINE_CPU_TS, &exportedCpuTsThread));
        // cpuTsThread 添加到param里
        param.opThread = exportedAicpuTsThread;
    }

    // 算法执行
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        ThreadHandle unfoldThread;
        CHK_RET(GetUnfoldThreadInfo(comm, param, unfoldThread));
        // 根据主流的捕获状态决定展开流的状态
        CHK_RET(CaptureSlaveStreams(comm, param.stream, {mainThread, unfoldThread}));
        CHK_RET(HcclAicpuKernelEntranceLaunch(comm, param, cpuTsThread, exportedCpuTsThread, notifyNumOnMainThread,
            resCtxSequence, algName, unfoldThread));
    } else if (param.engine == COMM_ENGINE_AIV) {
        uint64_t aivBeginTime = HcommGetProfilingSysCycleTime();
        param.resCtx = resCtxSequence;
        AlgResourceCtxSerializable &aivResCtxHost = *static_cast<AlgResourceCtxSerializable *>(resCtxSequence);
        CHK_RET(HcclAivKernelEntranceLaunch(param, topoInfo, aivResCtxHost));
        CHK_RET(ExecuteAivCacheLogic(param, algName, executor, aivResCtxHost));
        CHK_RET(HcclReportAivKernel(comm, aivBeginTime));
    } else if (param.engine == COMM_ENGINE_CCU) {
        if (isResourceReused) {
            // 复用资源，则需从engineCtx取得res，进行反序列化
            char *ctx = static_cast<char*>(resCtxSequence);
            std::vector<char> seq(ctx, ctx + param.ctxSize);
            resCtxHost->DeSerialize(seq);
        }
        int result = sprintf_s(param.algName, sizeof(param.algName), "%s", algName.c_str());
        if (result <= 0) {
            HCCL_ERROR("faled to fill param.algName");
            return HCCL_E_INTERNAL;
        }
        if (resCtxHost->slaveThreadNum > 0) {
            CHK_RET(CaptureSlaveStreams(comm, param.stream, resCtxHost->threads));
        }
        CHK_RET(executor->Orchestrate(param, *resCtxHost));
    } else {
        if (isResourceReused) {
            // 复用资源，则需从engineCtx取得res，进行反序列化
            char *ctx = static_cast<char*>(resCtxSequence);
            std::vector<char> seq(ctx, ctx + param.ctxSize);
            resCtxHost->DeSerialize(seq);
        }
        CHK_RET(executor->Orchestrate(param, *resCtxHost));
    }
    // op上报
    CHK_RET(HcclProfilingReportOp(comm, beginTime));
    HCCL_INFO("Execute HcclExecOp success.");
    return HCCL_SUCCESS;
}

HcclResult HcclAicpuKernelEntranceLaunch(HcclComm comm, OpParam &param, ThreadHandle cpuTsThread,
    ThreadHandle exportedCpuTsThread, u32 notifyNumOnMainThread, void *resCtxSequence, std::string &algName, ThreadHandle unfoldThread)
{
    HCCL_DEBUG("[HcclAicpuKernelEntranceLaunch]start to run aicpu kernel");
    // 当前aicpu launch接口只能有一个输入参数，将Context指针放在param参数中
    param.resCtx = resCtxSequence;
    // 将算法名字放在param参数中
    int result = sprintf_s(param.algName, sizeof(param.algName), "%s", algName.c_str());
    if (result <= 0) {
        HCCL_ERROR("failed to fill param.algName");
        return HCCL_E_INTERNAL;
    }

    if (param.engine == COMM_ENGINE_CPU) {
        // 注册dpu回调函数
        CHK_RET(static_cast<HcclResult>(HcclTaskRegister(comm, param.algTag, HcclLaunchDPUKernel)));
    }

    // Host stream通知Device主thread，使用主流上idx最大的notify
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(cpuTsThread, exportedCpuTsThread,
        notifyNumOnMainThread - 1)));
    // AicpuKernel report
    uint64_t beginTime = HcommGetProfilingSysCycleTime();
    CHK_RET(AicpuKernelLaunch(comm, param, unfoldThread));
    CHK_PTR_NULL(comm);
    std::string kernelName = "HcclLaunchAicpuKernel";
    char* kernelNameCStr = const_cast<char*>(kernelName.c_str());
    HcclResult ret = HcclReportAicpuKernel(comm, beginTime, kernelNameCStr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclAicpuKernelEntranceLaunch] HcclReportAicpuKernel failed, beginTime %lu, kernelNameCStr %s, ret %d ", beginTime, kernelNameCStr, ret);
        return ret;
    }
    // Host stream等待Device的通知
    u16 NOTIFY_WAIT_TIME = 27 * 68;
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(cpuTsThread, HOST_WAIT_AICPU_NOTIFYIDX, NOTIFY_WAIT_TIME)));

    return HCCL_SUCCESS;
}

HcclResult AicpuKernelLaunch(HcclComm comm, OpParam &param, ThreadHandle unfoldThread)
{
    std::string kernelName = "HcclLaunchAicpuKernel";
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;
    // 注意，目前开源HCCL加载AICPU kernel使用的是从json文件加载
    // 详见load_kernel.cc中的LoadAICPUKernel函数，且只实现了scatter的，先共用scatter的
    aclError ret = aclrtBinaryGetFunction(g_binKernelHandle, kernelName.c_str(), &funcHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS,
        HCCL_ERROR("[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, kernelName:%s",
            ret, kernelName.c_str()), HCCL_E_RUNTIME);
    ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, kernelName:%s", ret, kernelName.c_str()),
        HCCL_E_RUNTIME);
    aclrtParamHandle paraHandle;
    size_t paramSize = sizeof(OpParam) + param.varMemSize;
    ret = aclrtKernelArgsAppend(argsHandle, &param, paramSize, &paraHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS,
        HCCL_ERROR("[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append size %u,"
                    "kernelName:%s", ret, paramSize, kernelName.c_str()), HCCL_E_RUNTIME);
    ret = aclrtKernelArgsFinalize(argsHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS,
        HCCL_ERROR("[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, kernelName:%s",
            ret, kernelName.c_str()), HCCL_E_RUNTIME);
    // notifywait默认1836等待时长
    u16 NOTIFY_DEFAULT_WAIT_TIME = 27 * 68;
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr u32 numBlocks = 1;
    // 通过Thread获取展开流stream
    HCCL_INFO("[AicpuKernelLaunch] unfoldThread [%lu]", unfoldThread);
    void* unfoldStream = nullptr;
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    // 如果不支持这个接口则不走提前展开
    if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo || param.opMode == OpMode::OFFLOAD) {
        ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &cfg, argsHandle, nullptr);
    } else {
        CHK_RET(HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, unfoldThread, 0, sizeof(void*), &unfoldStream));
        ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, unfoldStream, &cfg, argsHandle, nullptr); // 提前展开，传入展开流
    }
    CHK_PRT_RET(ret != ACL_SUCCESS,
        HCCL_ERROR("[LoadCustomKernel][aclrtLaunchKernelWithConfig]errNo[0x%016llx] launch kernel failed", ret),
        HCCL_E_OPEN_FILE_FAILURE);
    return HCCL_SUCCESS;
}

HcclResult HcclAivKernelEntranceLaunch(OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    AlgResourceCtxSerializable &resCtxHost)
{
    HCCL_INFO("[%s] algTag[%s] commModeTag[%s] resCtx(Host)[%p] aivCommInfoPtr(Device)[%p]", __func__,
        param.algTag, param.commModeTag, param.resCtx, resCtxHost.aivCommInfoPtr);
    u32 numBlocksLimit = MAX_NUM_BLOCKS;
    ACLCHECK(aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &numBlocksLimit));
    CHK_PRT_RET(numBlocksLimit < 1,
        HCCL_ERROR("[%s] block num less than 1, block num[%d]", __func__, numBlocksLimit), HCCL_E_PARA);
    param.numBlocksLimit = numBlocksLimit;
    HCCL_INFO("[%s] Aiv core limit is [%d].", __func__, numBlocksLimit);
    return HCCL_SUCCESS;
}

HcclResult CaptureSlaveStreams(HcclComm comm, aclrtStream mainStream, const std::vector<ThreadHandle>& threads)
{
    aclmdlRI rtModel = nullptr;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclError ret = aclmdlRICaptureGetInfo(mainStream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[%s]Stream capture not support.", __func__);
        return HCCL_SUCCESS;
    } else {
        CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[%s]aclmdlRICaptureGetInfo fail. return[%d].", __func__, ret),
            HCCL_E_RUNTIME);
    }
    if (captureStatus != aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
        HCCL_INFO("[%s]captureStatus is not active, captureStatus[%d]", __func__, captureStatus);
        return HCCL_SUCCESS;
    }
    //thread[0] is main thread
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    for (size_t i = 1; i < threads.size(); ++i) {
        void* stream = nullptr;
        CHK_PRT_RET(!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo, HCCL_ERROR("AclGraph is not support."),
            HCCL_E_NOT_SUPPORT);
        CHK_RET(HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, threads[i], 0, sizeof(void*), &stream));
        rtError_t addRet = rtStreamAddToModel(stream, rtModel);
        CHK_PRT_RET(addRet != RT_ERROR_NONE, HCCL_ERROR("[%s]rtStreamAddToModel fail. return[%d].", __func__, addRet),
            HCCL_E_RUNTIME);
        HCCL_DEBUG("[%s]add slaveStream to model success, idx[%zu], stream[%p], rtModel[%p]", __func__, i, stream, rtModel);
    }
    HCCL_INFO("[%s]success, captured streams to rtmodel:[%p], slaveStreamNum:[%zu]", __func__, rtModel, threads.size() > 0 ? threads.size() - 1 : 0);
    return HCCL_SUCCESS;
}

HcclResult HcclCalcTopoInfo(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo)
{
    HCCL_INFO("[%s] HcclCalcTopoInfo start.", __func__);
    uint64_t size = 0;
    void *ctx = nullptr;
    // 若获取Context失败，表示对应Context尚未缓存
    if (HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size) != HCCL_SUCCESS) {
        // 初始化topoInfo
        CHK_RET(InitRankInfo(comm, topoInfo.get()));
        // 序列化
        std::vector<char> seq = topoInfo->Serialize();
        size = seq.size();
        // 创建新的Context保存
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
        CHK_SAFETY_FUNC_RET(memcpy_s(ctx, size, seq.data(), size));
        return HCCL_SUCCESS;
    }
    char *ctxTemp = reinterpret_cast<char*>(ctx);
    std::vector<char> seq(ctxTemp, ctxTemp + size);
    TopoInfoWithNetLayerDetails topoInfoTemp;
    topoInfoTemp.DeSerialize(seq);
    topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>(std::move(topoInfoTemp));
    HCCL_INFO("[%s] HcclCalcTopoInfo end.", __func__);
    return HCCL_SUCCESS;
}

void CompReqChannelWithExistChannel(const std::vector<std::vector<ChannelInfo>>& existChannels,
    AlgResourceRequest &resRequest)
{
    std::set<u32> existRemoteRankSet = {};
    std::vector<HcclChannelDesc> needAllocChannelDesc;
    // 先把所有已存在的channel的remoteRank整理成集合
    for (const ChannelInfo& channel: existChannels[0]) {
        existRemoteRankSet.insert(channel.remoteRank);
    }
    // 在集合中查找有没有request的channel
    for (const HcclChannelDesc& channelDesc : resRequest.channels[0]) {
        if (existRemoteRankSet.find(channelDesc.remoteRank) == existRemoteRankSet.end()) {
            needAllocChannelDesc.push_back(channelDesc);
        }
    }
    resRequest.channels = {needAllocChannelDesc};
    return;
}

HcclResult HcclGetAlgRes(HcclComm comm, OpParam& param, std::unique_ptr<InsCollAlgBase>& executor, TopoInfoWithNetLayerDetails* topoInfo,
                         std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, void** resCtxSequence, bool &isResourceReused)
{
    HCCL_INFO("Start to execute HcclGetAlgRes.");

    void *ctx = nullptr;
    bool increCreateChannelFlag = false;
    if (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && param.opMode == OpMode::OPBASE) {
        // 增量建链模式
        increCreateChannelFlag = true;
    }
    uint64_t size = 0;
    // 图模式不支持资源复用，且不存在增量建链场景
    if (!increCreateChannelFlag && param.opMode == OpMode::OPBASE) {
        void *ctx = nullptr;
        // 这种情况下资源已经有了
        CommEngine ctxEngine = param.engine;
        if (param.engine == CommEngine::COMM_ENGINE_AIV) {
            // AIV模式固定利用利用algTag申请1块host内存resCtx
            ctxEngine = COMM_ENGINE_CPU_TS;
        } else if (param.engine == COMM_ENGINE_CPU) {
            // host dpu申请device内存用于存放resctx
            ctxEngine = COMM_ENGINE_AICPU_TS;
        }
        if (HcclEngineCtxGet(comm, param.algTag, ctxEngine, &ctx, &size) == HCCL_SUCCESS) {
            HCCL_DEBUG("Already have context, skip create, ctxSize is %u", param.ctxSize);
            isResourceReused = true;
            *resCtxSequence = ctx;
            param.ctxSize = size;
            return HCCL_SUCCESS;
        }
    }

    // 计算AlgHierarchyInfo
    AlgHierarchyInfoForAllLevel algHierarchyInfo;  // 分级通信域信息{localRankId, localRankSize}
    CHK_RET(executor->CalcAlgHierarchyInfo(comm, topoInfo, algHierarchyInfo));
    // 资源计算
    AlgResourceRequest resRequest;
    CHK_RET(executor->CalcRes(comm, param, topoInfo, algHierarchyInfo, resRequest));

    // host侧资源
    if (param.engine == COMM_ENGINE_RESERVED) {
        // COMM_ENGINE_RESERVED
    } else if (param.engine == COMM_ENGINE_CPU) {
        CHK_RET(GetAlgResDPU(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence,
            size, increCreateChannelFlag));
    } else if (param.engine == COMM_ENGINE_CPU_TS) {
        // COMM_ENGINE_CPU_TS
    } else if (param.engine == COMM_ENGINE_AICPU) {
        // COMM_ENGINE_AICPU
    } else if (param.engine == COMM_ENGINE_AICPU_TS) {
        CHK_RET(GetAlgResAICPU(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence,
                               size, increCreateChannelFlag));
    } else if (param.engine == COMM_ENGINE_AIV) {
        CHK_RET(GetAlgResAiv(comm, param, resRequest, topoInfo, algHierarchyInfo, resCtxSequence));
    } else if (param.engine == COMM_ENGINE_CCU) {
        CHK_RET(GetAlgResCcu(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size));
    } else {
        HCCL_ERROR("fail to get engine.", HCCL_E_PARA);
    }
    param.ctxSize = size;
    return HCCL_SUCCESS;
}

HcclResult GetAlgResAICPU(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag)
{
    std::string tagStr = param.algTag;
    if (!increCreateChannelFlag || g_hostCtx.find(tagStr) == g_hostCtx.end()) {
        // 非增量建链流程，直接创建host侧Ctx
        resCtxHost->commInfoPtr = static_cast<void *>(comm);
        resCtxHost->topoInfo = *topoInfo;
        resCtxHost->algHierarchyInfo = algHierarchyInfo;
        // 创建资源，并填充到Host内存上
        HcclResult ret = HcclAllocAlgResourceAICPU(comm, param, resRequest, resCtxHost);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("failed to alloc alg resource."), ret);
        // 在device侧创建Ctx，并将host资源拷贝到device侧
        ret = HcclMemcpyCtxHostToDevice(comm, param, resCtxHost, resCtxSequence, ctxSize);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("failed to memcpy hostCtx to device."), ret);
        // 如果是增量建链模式，转移hostCtx的所有权
        if (increCreateChannelFlag) {
            g_hostCtx[tagStr] = std::move(resCtxHost);
        }
    } else {
        // 先比对需要的channel和已建链的channel
        CompReqChannelWithExistChannel(g_hostCtx.at(tagStr)->channels, resRequest);
        if (resRequest.channels[0].size() == 0) {
            // 资源可以直接复用，直接获取到device的ctx资源
            void *ctx = nullptr;
            uint64_t size = 0;
            HcclResult ret = HcclEngineCtxGet(comm, param.algTag, param.engine, &ctx, &size);
            if (ret == HCCL_SUCCESS) {
                *resCtxSequence = ctx;
                ctxSize = size;
            } else {
                HCCL_ERROR("failed to get device ctx.");
            }
            return ret;
        }
        // 资源不能直接复用，需要增量建链(会直接在已有的hostCtx中填充)
        HcclResult ret = HcclGetChannel(comm, param, resRequest, g_hostCtx.at(tagStr));
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("failed to incrementally create channel."), ret);
        // 把device侧此tag的ctx销毁
        ret = HcclEngineCtxDestroy(comm, param.algTag, param.engine);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("failed to destroy device Ctx."), ret);
        ret = HcclMemcpyCtxHostToDevice(comm, param, g_hostCtx.at(tagStr), resCtxSequence, ctxSize);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("failed to memcpy hostCtx to device."), ret);
        HCCL_INFO("Incrementally add channel success");
    }

    HCCL_INFO("Execute GetAlgResAICPU success.");
    return HCCL_SUCCESS;
}

HcclResult HcclMemcpyCtxHostToDevice(HcclComm comm, const OpParam &param,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, void **resCtxSequence, uint64_t& ctxSize)
{
    // 序列化
    std::vector<char> seq = resCtxHost->Serialize();
    uint64_t size = seq.size();
    void *ctx = nullptr;
    // 创建Context, aicpu和host dpu申请device内存
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, COMM_ENGINE_AICPU_TS, size, &ctx));
    // 从Host内存拷贝到Device Context内存上
    CHK_RET(HcclEngineCtxCopy(comm, COMM_ENGINE_AICPU_TS, param.algTag, seq.data(), size, 0));
    // 将内存强转为AlgResourceCtx结构体
    *resCtxSequence = ctx;
    ctxSize = size;
    HCCL_INFO("Memcpy hostCtx to device success.");
    return HCCL_SUCCESS;
}

HcclResult HcclAllocAlgResourceAICPU(
    HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    HCCL_INFO("Start to execute AllocAlgResource.");
    void *cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    // CCL IN使用所有的CCL Buffer，这个其实就是scratch buffer
    resCtxHost->cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};
    resCtxHost->notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
    resCtxHost->slaveThreadNum = resRequest.slaveThreadNum;
    resCtxHost->notifyNumPerThread = resRequest.notifyNumPerThread;
    CHK_RET(HcclGetThread(comm, param, resRequest, resCtxHost));
    CHK_RET(HcclGetChannel(comm, param, resRequest, resCtxHost));
    return HCCL_SUCCESS;
}

HcclResult HcclGetThread(
    HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        u32 maxNotifyNum = resRequest.notifyNumOnMainThread;
        for (u32 i = 0; i < resRequest.notifyNumPerThread.size(); i++) {
            if (resRequest.notifyNumPerThread[i] > maxNotifyNum) {
                maxNotifyNum = resRequest.notifyNumPerThread[i];
            }
        }
        u32 threadNum = resRequest.slaveThreadNum + 1;
        std::vector<ThreadHandle> threads(threadNum);
        // maxNotifyNum需要再增加一个用于host-device同步
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, threadNum, maxNotifyNum + 1, threads.data()));
        CHK_RET(SaveMainThreadInfo(comm, param, threads[0], maxNotifyNum + 1));
        // 申请展开流对应的Thread
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_CPU, 1, 0, &resCtxHost->unfoldThread));
        CHK_RET(SaveUnfoldThreadInfo(comm, param, resCtxHost->unfoldThread));
        HCCL_INFO("[HcclGetThread] unfoldThread [%lu]", resCtxHost->unfoldThread);
        HCCL_DEBUG("threads ptr is %p\n", threads.data());
        for (u32 i = 0; i < threadNum; i++) {
            resCtxHost->threads.push_back(threads[i]);
        }
    } else {
        ThreadHandle thread;
        // host模式下，将主流封装为thread，并创建主流上的notify
        CHK_RET(HcclThreadAcquireWithStream(comm, param.engine, param.stream,
            resRequest.notifyNumOnMainThread, &thread));
        resCtxHost->threads.push_back(thread);
        u32 maxNotifyNum = 0;
        for (u32 i = 0; i < resRequest.notifyNumPerThread.size(); i++) {
            if (resRequest.notifyNumPerThread[i] > maxNotifyNum) {
                maxNotifyNum = resRequest.notifyNumPerThread[i];
            }
        }
        u32 threadNum = resRequest.slaveThreadNum;
        if (threadNum > 0) {
            std::vector<ThreadHandle> threads(threadNum);
            CHK_RET(HcclThreadAcquire(comm, param.engine, threadNum, maxNotifyNum, threads.data()));
            for (u32 i = 0; i < threadNum; i++) {
                resCtxHost->threads.push_back(threads[i]);
            }
        }
    }

    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
        HCCL_DEBUG("[HcclGetThread] slaveThreadNum[%u]", resRequest.slaveThreadNum);
        for (u32 i = 0; i < resRequest.slaveThreadNum + 1; i++) {
            HCCL_DEBUG("[HcclGetThread] threads[%u]=[%llu]", i, resCtxHost->threads[i]);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult SaveMainThreadInfo(HcclComm comm, const OpParam &param, ThreadHandle thread, u32 notifyNum)
{
    uint64_t size = sizeof(ThreadHandle) + sizeof(u32);
    void *ctx = nullptr;
    // 申请一块host类型内存，保存主流信息
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
    // 填充主流handle信息
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle *>(ctx);
    *threadPtr = thread;
    // 填充主流notify数量信息
    char* curPtr = reinterpret_cast<char *>(ctx);
    curPtr += sizeof(ThreadHandle);
    u32 *notifyNumPtr = reinterpret_cast<u32 *>(curPtr);
    *notifyNumPtr = notifyNum;
    HCCL_INFO("[SaveMainThreadInfo]threadPtr[%p], thread[%lu], notifyNumPtr[%p], notifyNum[%lu]",
        threadPtr, thread, notifyNumPtr, notifyNum);
    return HCCL_SUCCESS;
}

HcclResult SaveUnfoldThreadInfo(HcclComm comm, const OpParam &param, ThreadHandle unfoldThread)
{
    uint64_t size = sizeof(ThreadHandle);
    void *ctx = nullptr;
    // 申请一块host类型内存，保存展开流信息
    char unfoldAlgTag[ALG_TAG_LENGTH] = {0};
    int ret = snprintf_s(unfoldAlgTag, sizeof(unfoldAlgTag), sizeof(unfoldAlgTag) - 1, "%s_unfold", param.algTag);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[%s] failed to fill unfoldAlgTag", __func__), HCCL_E_INTERNAL);
    CHK_RET(HcclEngineCtxCreate(comm, unfoldAlgTag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
    // 填充主流handle信息
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle *>(ctx);
    *threadPtr = unfoldThread;
    HCCL_INFO("[SaveUnfoldThreadInfo]unfoldAlgTag[%s], threadPtr[%p], unfoldThread[%lu]",
        unfoldAlgTag, threadPtr, unfoldThread);
    return HCCL_SUCCESS;
}

HcclResult GetUnfoldThreadInfo(HcclComm comm, const OpParam &param, ThreadHandle& unfoldThread)
{
    uint64_t size = sizeof(ThreadHandle);
    void *ctx = nullptr;
    char unfoldAlgTag[ALG_TAG_LENGTH] = {0};
    int ret = snprintf_s(unfoldAlgTag, sizeof(unfoldAlgTag), sizeof(unfoldAlgTag) - 1, "%s_unfold", param.algTag);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[%s] failed to fill unfoldAlgTag", __func__), HCCL_E_INTERNAL);
    CHK_RET(HcclEngineCtxGet(comm, unfoldAlgTag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size));
    // 获取展开流handle信息
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle *>(ctx);
    unfoldThread = *threadPtr;
    HCCL_INFO("[GetUnfoldThreadInfo]unfoldAlgTag[%s], threadPtr[%p], unfoldThread[%lu]",
        unfoldAlgTag, threadPtr, unfoldThread);
    return HCCL_SUCCESS;
}

HcclResult GetMainThreadInfo(HcclComm comm, const OpParam &param, ThreadHandle &thread, u32 &notifyNum)
{
    uint64_t size = sizeof(ThreadHandle) + sizeof(u32);
    void *ctx = nullptr;
    CHK_RET(HcclEngineCtxGet(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size));

    // 获取主流handle信息
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle *>(ctx);
    thread = *threadPtr;
    // 获取主流notify数量信息
    char* curPtr = reinterpret_cast<char *>(ctx);
    curPtr += sizeof(ThreadHandle);
    u32 *notifyNumPtr = reinterpret_cast<u32 *>(curPtr);
    notifyNum = *notifyNumPtr;
    HCCL_INFO("[GetMainThreadInfo]threadPtr[%p], thread[%lu], notifyNumPtr[%p], notifyNum[%lu]",
        threadPtr, thread, notifyNumPtr, notifyNum);
    return HCCL_SUCCESS;
}

HcclResult HcclGetChannel(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{

    resCtxHost->channels.resize(resRequest.channels.size());
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        // 获取子通信域的建链请求
        std::vector<HcclChannelDesc> &levelNChannelRequest = resRequest.channels[level];
        std::vector<HcclChannelDesc> deviceChannelRequest;
        std::vector<HcclChannelDesc> hostChannelRequest;
        for (auto &channelRequest : levelNChannelRequest) {
            if (channelRequest.remoteEndpoint.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
                deviceChannelRequest.emplace_back(channelRequest);
            } else if (channelRequest.remoteEndpoint.loc.locType == ENDPOINT_LOC_TYPE_HOST) {
                hostChannelRequest.emplace_back(channelRequest);
            }
        }
        // device建链
        CHK_RET(HcclGetChannelImpl(level, comm, param, deviceChannelRequest, COMM_ENGINE_AICPU_TS, resCtxHost));
        // host建链
        CHK_RET(HcclGetChannelImpl(level, comm, param, hostChannelRequest, COMM_ENGINE_CPU, resCtxHost));

    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetChannelImpl(const u32 level, HcclComm comm, const OpParam &param, std::vector<HcclChannelDesc>& channelRequest,
                              const CommEngine commEngine, std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost) {
    // 获取子通信域的建链数量
    if (channelRequest.empty()) {
        HCCL_INFO("[HcclGetChannelImpl] channelRequest is empty");
        return HCCL_SUCCESS;
    }
    u32 channelNum = channelRequest.size();
    std::vector<ChannelHandle> levelNChannels;
    levelNChannels.resize(channelNum);
    char inputBuffTag[MAX_MEM_TAG_LENGTH];
    char outputBuffTag[MAX_MEM_TAG_LENGTH];
    std::vector<HcclMemHandle> memHandles;
    if (param.opMode == OpMode::OFFLOAD) {
        HCCL_INFO("[HcclGetChannelImpl] start to RegGraphModeBuffers");
        CHK_RET(RegGraphModeBuffers(comm, param, channelRequest, inputBuffTag, outputBuffTag, memHandles));
        for (auto &channelDesc : channelRequest) {
            channelDesc.memHandles = memHandles.data();
            channelDesc.memHandleNum = memHandles.size();
        }
    }
    if (channelNum > 0) {
        CHK_RET(HcclChannelAcquire(comm, commEngine, channelRequest.data(),
            channelNum, levelNChannels.data()));
    }

    for (u32 idx = 0; idx < channelNum; idx++) {
        ChannelInfo channel;
        // 对于真实建链的链路进行填充
        const HcclChannelDesc &channelDescNew = channelRequest[idx];
        channel.isValid = true;
        channel.remoteRank = channelDescNew.remoteRank;
        channel.protocol = channelDescNew.channelProtocol;
        channel.locationType = channelDescNew.remoteEndpoint.loc.locType;
        channel.notifyNum = channelDescNew.notifyNum;
        channel.handle = levelNChannels[idx];

        void* remoteCclBufferAddr;
        uint64_t remoteCclBufferSize;
        CHK_RET(HcclChannelGetHcclBuffer(comm, levelNChannels[idx], &remoteCclBufferAddr, &remoteCclBufferSize));
        channel.remoteCclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteCclBufferAddr, remoteCclBufferSize};

        if (param.opMode == OpMode::OFFLOAD) {
            CHK_RET(GetGraphModeBuffers(comm, levelNChannels[idx], inputBuffTag, outputBuffTag, channel));
        }


        resCtxHost->channels[level].push_back(channel);
    }
    return HCCL_SUCCESS;
}


HcclResult RegGraphModeBuffers(HcclComm comm, const OpParam &param, std::vector<HcclChannelDesc>& channelRequest, char* inputBuffTag, char* outputBuffTag, std::vector<HcclMemHandle>& memHandles) {
    HCCL_INFO("[RegGraphModeBuffers] param.tag[%s]", param.tag);
    if (channelRequest.empty()) {
        HCCL_INFO("[RegGraphModeBuffers]channelRequest is empty");
        return HCCL_SUCCESS;
    }

    auto retIn = sprintf_s(inputBuffTag, MAX_MEM_TAG_LENGTH, "%s_%s", param.tag, "InputBuffer");
    auto retOut =  sprintf_s(outputBuffTag, MAX_MEM_TAG_LENGTH, "%s_%s", param.tag, "OutputBuffer");
    if (retIn <= 0 || retOut <= 0){
        HCCL_ERROR("[RegGraphModeBuffers]failed to fill BuffTag");
        return HcclResult::HCCL_E_INTERNAL;
    }

    HCCL_INFO("[RegGraphModeBuffers] graph mode regstry remote buuffer");
    if (param.inputPtr != nullptr && param.inputSize != 0) {
        HcclMemHandle inputHandle = nullptr;
        CHK_RET(HcclRegstryBuff(comm, inputBuffTag, param.inputPtr, param.inputSize, &inputHandle));
        CHK_PTR_NULL(inputHandle);
        memHandles.emplace_back(inputHandle);
    }
    if (param.outputPtr != nullptr && param.outputSize != 0) {
        HcclMemHandle outputHandle = nullptr;
        CHK_RET(HcclRegstryBuff(comm, outputBuffTag, param.outputPtr, param.outputSize, &outputHandle));
        CHK_PTR_NULL(outputHandle);
        memHandles.emplace_back(outputHandle);
    }
    HCCL_INFO("[RegGraphModeBuffers]memHandles size[%d]", memHandles.size());
    return HCCL_SUCCESS;
}

HcclResult GetGraphModeBuffers(HcclComm comm, ChannelHandle channelHandle, const char* inputBuffTag, const char* outputBuffTag, ChannelInfo& channel) {
    void* remoteInputBufferAddr = nullptr;
    uint64_t remoteInputBufferSize = 0;
    CHK_RET(HcclGetRemoteBuff(comm, channelHandle, inputBuffTag, &remoteInputBufferAddr, &remoteInputBufferSize));
    if (remoteInputBufferAddr != nullptr && remoteInputBufferSize > 0) {
        channel.remoteInputGraphMode = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteInputBufferAddr, remoteInputBufferSize};
    }

    void* remoteOutputBufferAddr = nullptr;
    uint64_t remoteOutputBufferSize = 0;
    CHK_RET(HcclGetRemoteBuff(comm, channelHandle, outputBuffTag, &remoteOutputBufferAddr, &remoteOutputBufferSize));
    if (remoteOutputBufferAddr != nullptr && remoteOutputBufferSize > 0) {
        channel.remoteOutputGraphMode = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteOutputBufferAddr, remoteOutputBufferSize};
    }
    return HCCL_SUCCESS;
}

HcclResult GetAlgResCcu(HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
                        std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails* topoInfo,
                        AlgHierarchyInfoForAllLevel& algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize)
{
    resCtxHost->topoInfo = *topoInfo;
    resCtxHost->algHierarchyInfo = algHierarchyInfo;

    // 创建资源，并填充到Host内存上
    HcclResult ret = HcclAllocAlgResourceCcu(comm, param, resRequest, resCtxHost);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("failed to alloc alg resource.");
        return ret;
    }
    // 序列化
    std::vector<char> seq = resCtxHost->Serialize();
    uint64_t size = seq.size();

    void *ctx = nullptr;
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, param.engine, size, &ctx));
    memcpy_s(ctx, size, seq.data(), size);
    *resCtxSequence = ctx;
    ctxSize = size;
    HCCL_INFO("Execute GetAlgResCCU success.");
    return HCCL_SUCCESS;
}

HcclResult HcclAllocAlgResourceCcu(HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
                                   std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    HCCL_INFO("Start to execute AllocAlgResource.");
    void *cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    // CCL IN使用所有的CCL Buffer，这个其实就是scratch buffer
    resCtxHost->cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};
    resCtxHost->notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
    resCtxHost->slaveThreadNum = resRequest.slaveThreadNum;
    resCtxHost->notifyNumPerThread = resRequest.notifyNumPerThread;
    CHK_RET(HcclGetThread(comm, param, resRequest, resCtxHost));
    CHK_RET(HcclGetChannelForCcu(comm, param, resRequest));
    CHK_RET(HcclGetCcuKernel(comm, resRequest, resCtxHost));
    return HCCL_SUCCESS;
}

HcclResult HcclGetChannelForCcu(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest)
{
    // 以kernel为粒度申请channel
    for (CcuKernelInfo& kernelInfo: resRequest.ccuKernelInfos) {
        std::vector<HcclChannelDesc> &kernelChannelRequest = kernelInfo.channels;

        u32 channelNum = kernelChannelRequest.size();
        std::vector<ChannelHandle> kernelChannels;
        kernelChannels.resize(channelNum);

        if (channelNum > 0) {
            CHK_RET(HcclChannelAcquire(comm, param.engine, kernelChannelRequest.data(),
                channelNum, kernelChannels.data()));
        }
        kernelInfo.kernelArg->channels = kernelChannels;
        HCCL_INFO("[HcclGetChannelForCcu] Get [%lu] channels", channelNum);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetCcuKernel(HcclComm comm, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    u32 totalKernelNum = 0;
    for (auto t: resRequest.ccuKernelNum) {
        totalKernelNum += t;
    }
    CHK_PRT_RET(totalKernelNum != resRequest.ccuKernelInfos.size(),
        HCCL_ERROR("[HcclGetCcuKernel]ccuKernel num not match!"),
        HCCL_E_INTERNAL);

    // 按照resgroup进行注册
    u32 currentResGroup = 0;
    u32 maxResGroup = 0;
    resCtxHost->ccuKernels.resize(totalKernelNum);
    while (currentResGroup <= maxResGroup) {
        for (u32 i = 0; i < totalKernelNum; i++) {
            CcuKernelInfo& kernelInfo = resRequest.ccuKernelInfos[i];
            if (kernelInfo.resGroup > maxResGroup) {
                maxResGroup = kernelInfo.resGroup;
            }
            if (kernelInfo.resGroup != currentResGroup) {
                continue;
            }
            void* kernelArgPtr = static_cast<void*>(kernelInfo.kernelArg.get()); // 保证没有释放
            void* creatorPtr = static_cast<void*>(&kernelInfo.creator);

            HCCL_DEBUG("[AllocAlgResource] kernelArgPtr[%p], creator[%p]", kernelArgPtr, &(kernelInfo.creator));
            CcuKernelHandle handle;
            CHK_RET(HcclCcuKernelRegister(comm, &handle, creatorPtr, kernelArgPtr));
            resCtxHost->ccuKernels[i] = handle;
        }
        CHK_RET(HcclCcuKernelRegisterFinish(comm));
        currentResGroup++;
    }
    resCtxHost->ccuKernelNum = resRequest.ccuKernelNum;
    return HCCL_SUCCESS;
}

HcclResult GetAlgResAiv(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence)
{
    uint64_t size = sizeof(AlgResourceCtxSerializable);
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, size, resCtxSequence));

    AlgResourceCtxSerializable* resCtxHost = static_cast<AlgResourceCtxSerializable *>(*resCtxSequence);
    resCtxHost->topoInfo = *topoInfo;
    resCtxHost->algHierarchyInfo = algHierarchyInfo;

    CHK_RET(HcclAllocAlgResourceAiv(comm, param, resRequest, resCtxHost));
    return HCCL_SUCCESS;
}

HcclResult HcclAllocAlgResourceAiv(
    HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest, AlgResourceCtxSerializable* resCtxHost)
{
    HCCL_INFO("[%s]Start to execute.", __func__);
    HcclMemHandle memHandle; // 注册到通信域内存的handle，用于建链
    // 获取存放AIV对端信息和标记区的空间
    uint64_t commInfoSize = 0;
    if (HcclEngineCtxGet(comm, param.commModeTag, param.engine, &(resCtxHost->aivCommInfoPtr), &commInfoSize) != HCCL_SUCCESS) {
        CHK_RET(HcclEngineCtxCreate(comm, param.commModeTag, param.engine, AIV_TAG_BUFF_LEN, &(resCtxHost->aivCommInfoPtr)));
        // 清零
        ACLCHECK(aclrtMemset(resCtxHost->aivCommInfoPtr, AIV_TAG_BUFF_LEN, 0, AIV_TAG_BUFF_LEN));
        // 注册到通信域，支持建链时交换
        CommMem regMem{COMM_MEM_TYPE_DEVICE, resCtxHost->aivCommInfoPtr, AIV_TAG_BUFF_LEN};
        CHK_RET(HcclCommMemReg(comm, param.commModeTag, &regMem, &memHandle));
        g_memHandleCache[param.commModeTag] = memHandle;
    } else {
        if (g_memHandleCache.find(param.commModeTag) == g_memHandleCache.end()) {
            HCCL_ERROR("[%s]aiv memHandle not found in map", __func__);
            return HCCL_E_INTERNAL;
        }
        memHandle = g_memHandleCache[param.commModeTag];
    }
    HCCL_INFO("[%s]commModeTag[%s] regMemAddr[%p] memHandle[%p]", __func__, param.commModeTag, resCtxHost->aivCommInfoPtr,
        memHandle);

    void* cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    HCCL_INFO("[%s]local cclBufferAddr[%p] cclBufferSize[%llu]", __func__, cclBufferAddr, cclBufferSize);
    resCtxHost->cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};

    void* buffersIn[MAX_RANK_SIZE] = {};
    void* buffersOut[MAX_RANK_SIZE] = {};
    buffersIn[resCtxHost->topoInfo.userRank] = cclBufferAddr;
    buffersOut[resCtxHost->topoInfo.userRank] = resCtxHost->aivCommInfoPtr;

    // 迭代每个子通信域的建链请求，创建链路
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        // 获取子通信域的建链请求
        std::vector<HcclChannelDesc> &levelNChannelRequest = resRequest.channels[level];
        for (auto &channelDesc : levelNChannelRequest) {
            channelDesc.memHandles = &memHandle;
            channelDesc.memHandleNum = 1;
        }
        // 获取子通信域的建链数量
        u32 validChannelNum = levelNChannelRequest.size();
        std::vector<ChannelHandle> levelNChannels;
        levelNChannels.resize(validChannelNum);
        HCCL_INFO("[%s]level[%u] validChannelNum[%u]", __func__, level, validChannelNum);

        if (validChannelNum > 0) {
            CHK_RET(HcclChannelAcquire(comm, param.engine, levelNChannelRequest.data(),
                validChannelNum, levelNChannels.data()));
        }

        for (u32 idx = 0; idx < validChannelNum; idx++) {
            HcclChannelDesc &channelDesc = levelNChannelRequest[idx];
            void* remoteBufferAddr;
            uint64_t remoteBufferSize;
            CHK_RET(HcclChannelGetHcclBuffer(comm, levelNChannels[idx], &remoteBufferAddr, &remoteBufferSize));
            HCCL_INFO("[%s]remoteRank[%u] cclBufferAddr[%p] cclBufferSize[%llu]", __func__, channelDesc.remoteRank,
                remoteBufferAddr, remoteBufferSize);
            buffersIn[channelDesc.remoteRank] = remoteBufferAddr;

            u32 memNum;
            CommMem* remoteMems;
            char** memTags;
            CHK_RET(HcclChannelGetRemoteMems(comm, levelNChannels[idx], &memNum, &remoteMems, &memTags));
            CHK_PRT_RET(memNum != 1,
                HCCL_ERROR("[%s] HcclChannelGetRemoteMems memNum[%u] not equal to 1", __func__, memNum), HCCL_E_PARA);
            HCCL_INFO("[%s]remoteRank[%u] memNum[%u] regMemAddr[%p] regMemSize[%llu] memTag[%s]", __func__,
                channelDesc.remoteRank, memNum, remoteMems[0].addr, remoteMems[0].size, memTags[0]);
            buffersOut[channelDesc.remoteRank] = remoteMems[0].addr;
        }
    }

    CHK_RET(haclrtMemcpy(resCtxHost->aivCommInfoPtr, MAX_RANK_SIZE * sizeof(void*), buffersIn, MAX_RANK_SIZE * sizeof(void*),
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHK_RET(haclrtMemcpy(static_cast<u8*>(resCtxHost->aivCommInfoPtr) + AIV_TAG_ADDR_OFFSET, MAX_RANK_SIZE * sizeof(void*),
        buffersOut, MAX_RANK_SIZE * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE));

    HCCL_INFO("[%s] Alloc res success.", __func__);
    return HCCL_SUCCESS;
}

HcclResult GetAlgResDPU(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag)
{
    // 申请共享内存
    uint64_t shmemSize = 100 * 1024 * 1024;
    void *shmemPtr = nullptr;
    bool newCreated;
    CHK_RET(HcclDevMemAcquire(comm, "DPUTAG", &shmemSize, &shmemPtr, &newCreated));
    resCtxHost->npu2DpuShmemPtr = shmemPtr;
    constexpr uint64_t DPU2NPU_SHMEM_RATIO = 2;
    resCtxHost->dpu2NpuShmemPtr = static_cast<void*>(static_cast<uint8_t*>(shmemPtr) + shmemSize / DPU2NPU_SHMEM_RATIO);

    CHK_RET(GetAlgResAICPU(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence,
                           ctxSize, increCreateChannelFlag));

    HCCL_INFO("Execute GetAlgResAICPU success.");
    return HCCL_SUCCESS;
}

HcclResult CheckCount(const u64 count)
{
    if (UNLIKELY(count > SYS_MAX_COUNT)) {
        HCCL_ERROR("[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])",
                    HCCL_ERROR_CODE(HCCL_E_PARA), count, SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CheckDataType(const HcclDataType dataType, bool needReduce)
{
    const std::vector<std::string> infoTitle({"ccl_op", "value", "parameter", "expect"});
    if (needReduce) {
        if ((dataType == HCCL_DATA_TYPE_UINT8)   || (dataType == HCCL_DATA_TYPE_UINT16)  ||
            (dataType == HCCL_DATA_TYPE_UINT32)  || (dataType == HCCL_DATA_TYPE_INT128)  ||
            (dataType == HCCL_DATA_TYPE_HIF8)    || (dataType == HCCL_DATA_TYPE_FP8E4M3) ||
            (dataType == HCCL_DATA_TYPE_FP8E5M2) || (dataType == HCCL_DATA_TYPE_FP8E8M0) ||
            (dataType == HCCL_DATA_TYPE_MXFP8) || (dataType == HCCL_DATA_TYPE_RESERVED)) {
            RPT_INPUT_ERR(true, "EI0003", infoTitle, std::vector<std::string>({"CheckDataType", GetDataTypeEnumStr(dataType), "dataType",
                GetSupportDataType(needReduce)}));
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                        HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                        GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else {
        if ((dataType >= HCCL_DATA_TYPE_RESERVED) || (dataType < HCCL_DATA_TYPE_INT8) ||
            (dataType == HCCL_DATA_TYPE_INT128)) {
            RPT_INPUT_ERR(true, "EI0003", infoTitle, std::vector<std::string>({"CheckDataType", GetDataTypeEnumStr(dataType), "dataType",
                GetSupportDataType(needReduce).c_str()}));
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                        HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                        GetSupportDataType(needReduce));
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
        supportList.insert(supportList.end(), {HCCL_DATA_TYPE_BFP16, HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64,
                                               HCCL_DATA_TYPE_FP64});
    } else {
        supportList.insert(supportList.end(), {HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT16,
                                               HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64,
                                               HCCL_DATA_TYPE_HIF8, HCCL_DATA_TYPE_FP8E4M3,  HCCL_DATA_TYPE_FP8E5M2,
                                               HCCL_DATA_TYPE_FP8E8M0, HCCL_DATA_TYPE_MXFP8});
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

HcclResult SetCommEngine(OpParam &param)
{
    // 使用一个静态的映射表来关联配置和引擎值
    static const std::unordered_map<OpExecuteConfig, CommEngine> ConfigToEngineMap = {
        {OpExecuteConfig::HOSTCPU_TS, COMM_ENGINE_CPU_TS},
        {OpExecuteConfig::AICPU_TS,   COMM_ENGINE_AICPU_TS},
        {OpExecuteConfig::AIV,        COMM_ENGINE_AIV},
        {OpExecuteConfig::AIV_ONLY,  COMM_ENGINE_AIV}, // AIV_ONLY 和 AIV 映射到同一引擎
        {OpExecuteConfig::CCU_MS,     COMM_ENGINE_CCU},
        {OpExecuteConfig::CCU_SCHED,  COMM_ENGINE_CCU},
        {OpExecuteConfig::AICPU,      COMM_ENGINE_AICPU},
        {OpExecuteConfig::HOSTCPU,    COMM_ENGINE_CPU},
    };

    auto it = ConfigToEngineMap.find(param.opExecuteConfig);
    if (it != ConfigToEngineMap.end()) {
        param.engine = it->second;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[op_common][SetCommEngine] Unsupported or unknown opExecuteConfig: {%d}", static_cast<int>(param.opExecuteConfig));
    return HCCL_E_NOT_SUPPORT;
}

HcclResult SingleRankProc(const OpParam &param)
{
    if (GetExternalInputHcclAivOnlyMode()) {
        HCCL_ERROR("[SingleRankProc] opType[%d] currently do not select aiv mode, aiv only not support, "
            "please ensure rankNum is greater than one", static_cast<int>(param.opType));
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.opType == HcclCMDType::HCCL_CMD_SEND || param.opType == HcclCMDType::HCCL_CMD_RECEIVE) {
        HCCL_WARNING("[%s] ranksize == 1 is not support BATCHSENDRECV SEND RECV", __func__);
        return HcclResult::HCCL_SUCCESS;
    }
    if (param.inputPtr == param.outputPtr) {
        HCCL_WARNING("[%s] sendBuf == recvBuf, return success", __func__);
        return HcclResult::HCCL_SUCCESS;
    }
    u64 len{0};
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
        param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        len = DATATYPE_SIZE_TABLE[param.all2AllVDataDes.sendType] * *(static_cast<const u64 *>(param.all2AllVDataDes.sendCounts));
    } else if (param.opType == HCCL_CMD_ALLGATHER_V || param.opType == HCCL_CMD_REDUCE_SCATTER_V) {
        len = DATATYPE_SIZE_TABLE[param.vDataDes.dataType] * *(static_cast<const u64 *>(param.vDataDes.counts));
    } else {
        len = DATATYPE_SIZE_TABLE[param.DataDes.dataType] * param.DataDes.count;
    }

    HCCL_INFO("[CommunicatorImpl][%s] sendBuf[%p], recvBuf[%p], len[%llu]", __func__,
              param.inputPtr, param.outputPtr, len);
    if (len > 0) {
        HcclResult ret = haclrtMemcpy(param.outputPtr, len, param.inputPtr, len, ACL_MEMCPY_DEVICE_TO_DEVICE);
        HCCL_DEBUG("Call haclrtMemcpy, return value[%d], para: dstAddr[%p], destMax[%llu], "
                "srcAddr[%p], count[%llu], rtKind[%d]", ret, param.outputPtr, len, param.inputPtr,
                len, ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[SingleRankProc][AsyncCopy][Mem]errNo[0x%016llx] rt memory async copy failed, "
                    "return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], kind[%d].",
                    HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, param.outputPtr, len, param.inputPtr,
                    len, ACL_MEMCPY_DEVICE_TO_DEVICE);
            return HcclResult::HCCL_E_RUNTIME;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCheckTag(const char *tag)
{
    CHK_PTR_NULL(tag);

    u32 tagLen = strnlen(tag, TAG_MAX_LEN + 1);
    if (UNLIKELY((tagLen == (TAG_MAX_LEN + 1) || tagLen == 0))) {
        HCCL_ERROR("[Check][Tag]errNo[0x%016llx] tag is too long", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult SetOpParamAlgTag(OpParam &param, const std::string &algName)
{
    std::string temp = algName; // 创建algName的副本

    const char* launchMode = (((param.engine == CommEngine::COMM_ENGINE_AICPU) ||
                                (param.engine == CommEngine::COMM_ENGINE_AICPU_TS)) ? "device" : "host");
    // 原有tag + algName + 编排模式，得到基础algTag
    int len = snprintf_s(param.algTag, sizeof(param.algTag), sizeof(param.algTag), "%s_%s_%s", param.tag, temp.c_str(), launchMode);
    if (len < 0|| len >= sizeof(param.algTag)) {
        HCCL_ERROR("failed to fill param.algTag");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // ccu模式，考虑kernel是否能复用，需要添加dataType和reduceType
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        try{
            HcclDataType tmpDataType;
            if(param.opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
               param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
               param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
                tmpDataType = param.all2AllVDataDes.sendType;
            } else if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V ||
                       param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
                tmpDataType = param.vDataDes.dataType;
            } else {
                tmpDataType = param.DataDes.dataType;
            }
            std::string ccuExtraTag = "_" + HCOM_DATA_TYPE_STR_MAP.at(tmpDataType);

            if (param.opType == HcclCMDType::HCCL_CMD_ALLREDUCE ||
                param.opType == HcclCMDType::HCCL_CMD_REDUCE ||
                param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ||
                param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
                ccuExtraTag += "_" + HCOM_REDUCE_OP_STR_MAP.at(param.reduceType);
            }
            size_t remainBytes = sizeof(param.algTag) - len;

            int len_ccu = snprintf_s(param.algTag + len, remainBytes, remainBytes, "%s", ccuExtraTag.c_str());
            if (len_ccu < 0 || len_ccu >= sizeof(param.algTag) - len) {
                HCCL_ERROR("failed to fill alg tag with ccu dataType");
                return HcclResult::HCCL_E_INTERNAL;
            }
        }
        catch (const std::out_of_range& e) {
            HCCL_ERROR("[SetOpParamAlgTag] dataType or reduceType out of range: %s", e.what());
            return HCCL_E_PARA;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclGetOpExpansionMode(HcclComm comm, OpParam &param)
{
    HcclOpExpansionMode finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    // 第一步：决定使用哪种模式
    HcclResult ret = DecideHcclOpExpansionMode(comm, finalMode);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("DecideHcclOpExpansionMode failed, ret: %d", ret);
        return ret;
    }

    // 第二步：应用选择的模式到param
    ret = ApplyOpExpansionMode(param, finalMode);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("ApplyOpExpansionMode failed, ret: %d", ret);
        return ret;
    }
    return HCCL_SUCCESS;
}

static constexpr uint32_t opExpansionModeCcuSched = 5;
static constexpr uint32_t opExpansionModeCcuMs = 4;

HcclResult DecideHcclOpExpansionMode(HcclComm comm, HcclOpExpansionMode &finalMode)
{
    HcclOpExpansionMode configOpExpansionMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    uint32_t infoLen = sizeof(HcclOpExpansionMode);
    CHK_RET(HcclConfigGetInfo(comm, HcclConfigType::HCCL_CONFIG_TYPE_OP_EXPANSION_MODE, infoLen, &configOpExpansionMode));
    finalMode = configOpExpansionMode;
    if (GetExternalInputHcclAicpuUnfold() == true) {
        finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AI_CPU;
    } else if (GetExternalInputHcclAivOnlyMode() == true) {
        finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY;
    } else if (GetExternalInputHcclAivMode() == true) {
        finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AIV;
    } else if (GetExternalInputHcclCcuMSMode()) {
        finalMode = static_cast<HcclOpExpansionMode>(opExpansionModeCcuMs);
    } else if (GetExternalInputHcclCcuSchedMode()) {
        finalMode = static_cast<HcclOpExpansionMode>(opExpansionModeCcuSched);
    }

    if (configOpExpansionMode != finalMode) {
        HCCL_DEBUG("[DecideHcclOpExpansionMode] configOpExpansionMode: %d, environment mode: %d, conflict, use environment mode.",
            configOpExpansionMode, finalMode);
    }
    return HCCL_SUCCESS;
}

HcclResult ApplyOpExpansionMode(OpParam &param, HcclOpExpansionMode finalMode)
{
    switch (finalMode) {
        case HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AI_CPU:
            param.opExecuteConfig = OpExecuteConfig::AICPU_TS;
            param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
            CHK_RET(LoadAICPUKernel());
            HCCL_DEBUG("[ApplyOpExpansionMode] AICPU mode selected.");
            break;
        case HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AIV:
            param.opExecuteConfig = OpExecuteConfig::AIV;
            param.engine = CommEngine::COMM_ENGINE_AIV;
            CHK_RET(RegisterKernel(param.opType, g_aivKernelInfoMap[param.opType].first, g_aivKernelInfoMap[param.opType].second));
            HCCL_DEBUG("[ApplyOpExpansionMode] AIV mode selected.");
            break;
        case HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY:
            param.opExecuteConfig = OpExecuteConfig::AIV_ONLY;
            param.engine = CommEngine::COMM_ENGINE_AIV;
            CHK_RET(RegisterKernel(param.opType, g_aivKernelInfoMap[param.opType].first, g_aivKernelInfoMap[param.opType].second));
            HCCL_DEBUG("[ApplyOpExpansionMode] AIV_ONLY mode selected.");
            break;
        case static_cast<HcclOpExpansionMode>(opExpansionModeCcuMs):
            param.opExecuteConfig = OpExecuteConfig::CCU_MS;
            param.engine = CommEngine::COMM_ENGINE_CCU;
            HCCL_DEBUG("[ApplyOpExpansionMode] CCU_MS mode selected.");
            break;
        case static_cast<HcclOpExpansionMode>(opExpansionModeCcuSched):
            param.opExecuteConfig = OpExecuteConfig::CCU_SCHED;
            param.engine = CommEngine::COMM_ENGINE_CCU;
            HCCL_DEBUG("[ApplyOpExpansionMode] CCU_SCHED mode selected.");
            break;
        default:
            // 回退到aicpu
            HCCL_WARNING("[ApplyOpExpansionMode] Invalid HcclOpExpansionMode: %d, fallback to AICPU_TS.", finalMode);
            param.opExecuteConfig = OpExecuteConfig::AICPU_TS;
            param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
            CHK_RET(LoadAICPUKernel());
            break;
    }
    return HcclResult::HCCL_SUCCESS;
}

bool HcclCheckAicpuEnableOpen()
{
    const char* envValue = std::getenv("HCCL_ENABLE_OPEN_AICPU");

    if (envValue != nullptr && std::strcmp(envValue, "1") == 0) {
        return true;
    }

    return false;
}

HcclResult HcclRegstryBuff(HcclComm comm, const char *memTag, void *bufferPtr, uint64_t bufferSize, HcclMemHandle *memHandle)
{
    CHK_PTR_NULL(memHandle);
    CommMem regMem{COMM_MEM_TYPE_DEVICE, bufferPtr, bufferSize};
    CHK_RET(HcclCommMemReg(comm, memTag, &regMem, memHandle));
    HCCL_INFO("[%s] regMemAddr[%p] regMemSize[%llu]", __func__, regMem.addr, regMem.size);
    CHK_PTR_NULL(*memHandle);
    return HCCL_SUCCESS;
}

HcclResult HcclGetRemoteBuff(HcclComm comm, ChannelHandle channel, const char *memTag, void **bufferPtr, uint64_t *bufferSize)
{
    CHK_PTR_NULL(bufferPtr);
    CHK_PTR_NULL(bufferSize);

    u32 memNum;
    CommMem *remoteMemList;
    char **memTags;
    CHK_RET(HcclChannelGetRemoteMems(comm, channel, &memNum, &remoteMemList, &memTags));
    HCCL_INFO("[%s] HcclChannelGetRemoteMems memNum[%u]", __func__, memNum);
    for (u32 i=0; i< memNum; i++) {
        HCCL_INFO("[%s] memNum[%u/%u] memTags[%s]", __func__, i + 1, memNum, memTags[i]);
        if (strcmp(memTags[i], memTag) == 0) {
            *bufferPtr = remoteMemList[i].addr;
            *bufferSize = remoteMemList[i].size;
            HCCL_INFO("[%s] Found %u memNum[%u/%u] is %u at index %u: addr=%p, size=%llu", __func__, *memTag,
                i + 1, memNum, remoteMemList[i].addr, remoteMemList[i].size);
            break;
        }
    }
    if (*bufferPtr == nullptr) {
        HCCL_WARNING("[%s] Failed to find %s in remote mem list", __func__, memTag);
    }
    return HCCL_SUCCESS;
}


bool HcclCheckCcuEnableOpen()
{
    const char* envValue = std::getenv("HCCL_ENABLE_OPEN_CCU");

    if (envValue != nullptr && std::strcmp(envValue, "1") == 0) {
        return true;
    }

    return false;
}

bool HcclCheckAivEnableOpen()
{
    const char* envValue = std::getenv("HCCL_ENABLE_OPEN_AIV");

    if (envValue != nullptr && std::strcmp(envValue, "1") == 0) {
        return true;
    }

    return false;
}

bool ShouldUseInnerOp(OpExecuteConfig opExecuteConfig)
{
    bool isAicpuOrHostMode = (opExecuteConfig == OpExecuteConfig::AICPU_TS ||
                              opExecuteConfig == OpExecuteConfig::HOSTCPU);
    bool isCcuMode = (opExecuteConfig == OpExecuteConfig::CCU_MS ||
                      opExecuteConfig == OpExecuteConfig::CCU_SCHED);
    bool isAivMode = (opExecuteConfig == OpExecuteConfig::AIV ||
                      opExecuteConfig == OpExecuteConfig::AIV_ONLY);

    if (isAicpuOrHostMode) {
        return !HcclCheckAicpuEnableOpen();
    } else if (isCcuMode) {
        return !HcclCheckCcuEnableOpen();
    } else if (isAivMode) {
        return !HcclCheckAivEnableOpen();
    }

    return false;
}

HcclResult LogHcclExit(const std::string &opName, const char *tag, HcclUs startut)
{
    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        std::string endInfo = opName + ":success,take time: " +
            std::to_string(DURATION_US(endut - startut).count()) + " us, tag: " + tag;
        HCCL_RUN_INFO("%s", endInfo.c_str());
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
