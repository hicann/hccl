/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_op.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>
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
#include "hccl_inner.h"
#include "hccl.h"
#include "config_log.h"
#include "workflow.h"
#include "load_kernel.h"

using namespace std;
using namespace ops_hccl;
constexpr uint32_t ROOTINFO_INDENTIFIER_MAX_LENGTH = 128;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount,
    HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    // 获取设备类型拦截混合组网
    HcclHeterogMode allDeviceType;
    CHK_RET(HcclGetHeterogMode(comm, &allDeviceType));
    if(allDeviceType != HcclHeterogMode::HCCL_HETEROG_MODE_HOMOGENEOUS) {
        HCCL_ERROR("[HcclScatter] Scatter only support singleDeviceType");
        return HCCL_E_NOT_SUPPORT;
    }

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (!RunIndependentOpExpansion(deviceType)) {
        return HcclScatterInner(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    }

    // 入口的地方先解析环境变量
    CHK_RET(InitEnvConfig());
    
    // AclGraph引导到老的流程上面
    if (IsStreamCapture(stream)) {
        return HcclScatterInner(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    }
    // 重执行引导到老的流程上面
    if (deviceType == DevType::DEV_TYPE_910_93 && (GetExternalInputIntraServerRetryEnable()
        || GetExternalInputInterServerRetryEnable() || GetExternalInputInterSuperPodRetryEnable())) {
        return HcclScatterInner(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    }

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));

    // 图模式引导到老的流程上面
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return HcclScatterInner(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    }

    // Attention! 图模式、zeroCopy模式、recompute等先不支持，且当前不引导到老的流程上

    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内

    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("input recvCount is 0, return scatter success"), HCCL_SUCCESS);
    CHK_RET(CheckScatterInputPara(comm, recvBuf));

    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    if (userRank == root) {     // 本rank为root节点，send_buff不可以为空
        RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
            std::vector<std::string>({"HcclScatter", "sendBuf", "nullptr", "please check sendBuf"}));
        CHK_PTR_NULL(sendBuf);
    }

    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string tag = "Scatter_" + string(commName);

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), recvCount, dataType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, root), tag.c_str());
    CHK_RET(CheckCount(recvCount));
    CHK_RET(CheckDataType(dataType, false));

    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], recvCount[%llu], data_type[%s], root[%u]",
        tag.c_str(), sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(), root);

    /* 接口交互信息日志 */
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], recvCount[%llu], dataType[%s], root[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(), root, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclScatter:" + std::string(stackLogBuffer); // capture的entry信息待补充
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }

    CHK_RET_AND_PRINT_IDE(ScatterOutPlace(sendBuf, recvBuf, recvCount, dataType, root, comm, stream, tag),
                          tag.c_str());

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo = "HcclScatter:success,take time: " +
            std::to_string(DURATION_US(endut - startut).count()) + " us, tag: " + tag;
        HCCL_RUN_INFO("%s", endInfo.c_str());
    }
    return HCCL_SUCCESS;
}

namespace ops_hccl {

constexpr u32 DEVICE_EIGHT = 8;
constexpr u32 DEVICE_FOUR = 4;
constexpr u32 DEVICE_TWO = 2;
constexpr u32 DEVICE_ONE = 1;
constexpr u32 HCCL_INTER_SERVER_RING_ALGO_MAX_SUPPORT_SERVER_NUM = 8; // server 间 ring 算法支持的最大server数: 8

HcclResult CheckScatterInputPara(HcclComm comm, void *recvBuf)
{
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclScatter", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclScatter", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);

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

HcclResult CheckDataType(const HcclDataType dataType, bool needReduce)
{
    const vector<string> infoTitle({"ccl_op", "parameter", "value", "tips"});
    if (needReduce) {
        if ((dataType == HCCL_DATA_TYPE_UINT64) ||
            (dataType == HCCL_DATA_TYPE_UINT8) || (dataType == HCCL_DATA_TYPE_UINT16) ||
            (dataType == HCCL_DATA_TYPE_UINT32) || (dataType == HCCL_DATA_TYPE_FP64) ||
            (dataType == HCCL_DATA_TYPE_RESERVED)) {
            RPT_INPUT_ERR(true, "EI0003", infoTitle, vector<string>({"CheckDataType", "dataType", GetDataTypeEnumStr(dataType), "please check dataType"}));
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                        HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                        GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else {
        if ((dataType >= HCCL_DATA_TYPE_RESERVED) || (dataType < HCCL_DATA_TYPE_INT8)) {
            RPT_INPUT_ERR(true, "EI0003", infoTitle, vector<string>({"CheckDataType", "dataType", GetDataTypeEnumStr(dataType), "please check dataType"}));
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                        HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                        GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

bool IsStreamCapture(aclrtStream stream)
{
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    u64 modelId = 0xFFFFFFFF;
    CHK_PRT(haclrtGetCaptureInfo(stream, captureStatus, modelId, isCapture));
    return isCapture;
}

bool IsAiCpuMode(DevType deviceType, u32 rankSize)
{
    if (GetExternalInputHcclAicpuUnfold() == true && deviceType == DevType::DEV_TYPE_910_93 && (rankSize != 1)) {
        return true;
    }
    return false;
}

HcclResult ScatterOutPlace(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream, const std::string &tag)
{
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 outputSize = recvCount * perDataSize;
    u64 inputSize = outputSize * userRankSize;

    OpParam param;
    param.stream = stream;

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (IsAiCpuMode(deviceType, userRankSize)) {
        HCCL_DEBUG("is aicpu mode");
        CHK_RET(LoadAICPUKernel());
        param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    } else {
        HCCL_DEBUG("is host mode");
        param.engine = CommEngine::COMM_ENGINE_CPU_TS;
    }

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
    param.DataDes.count = recvCount;
    param.DataDes.dataType = dataType;
    param.root = root;
    // opType传下去的作用是什么
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

    CHK_RET(ExecOp(comm, param));

    return HCCL_SUCCESS;
}

/* 执行通信算子 */
HcclResult ExecOp(HcclComm comm, OpParam &param)
{
    // 获取基础拓扑
    TopoInfo* topoInfo = nullptr;
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
    AlgResourceCtx* resCtx;
    ThreadHandle cpuTsThread;
    ThreadHandle exportedAicpuTsThread;
    CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, 1, &cpuTsThread));
    // Export cpuTsThread
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &cpuTsThread, COMM_ENGINE_AICPU_TS, &exportedAicpuTsThread));
    CHK_RET(GetAlgRes(comm, param, executor, topoInfo, algType, &resCtx));
    
    // Export aicpu ts thread
    ThreadHandle mainThread = topoInfo->mainThread;
    ThreadHandle exportedCpuTsThread;
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &mainThread, COMM_ENGINE_CPU_TS, &exportedCpuTsThread));
    // cpuTsThread 添加到ctx里
    char* curPtr = reinterpret_cast<char *>(resCtx);
    curPtr = curPtr + sizeof(AlgResourceCtx) - sizeof(TopoInfo) - sizeof(ThreadHandle); // 偏移指针
    ACLCHECK(aclrtMemcpy(curPtr, sizeof(ThreadHandle), &exportedAicpuTsThread, sizeof(ThreadHandle),
        ACL_MEMCPY_HOST_TO_DEVICE));

    // 算法执行
    if (param.engine == COMM_ENGINE_AICPU_TS) {
        // 当前aicpu launch接口只能有一个输入参数，将Context指针放在param参数中
        param.resCtx = resCtx;
        // 将算法名字放在param参数中
        int result = sprintf_s(param.algName, sizeof(param.algName), "%s", algName.c_str());
        if (result <= 0) {
            HCCL_ERROR("faled to fill param.algName");
            return HCCL_E_INTERNAL;
        }

        // Host stream通知Device主thread，使用主流上idx最大的notify
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(cpuTsThread, exportedCpuTsThread,
            topoInfo->notifyNumOnMainThread)));

        // 执行device测的算法编排
        std::string kernelName = "HcclLaunchAicpuKernel";
        aclrtFuncHandle funcHandle;
        aclrtArgsHandle argsHandle;

        aclError ret = aclrtBinaryGetFunction(g_binKernelHandle, kernelName.c_str(), &funcHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
                    HCCL_ERROR("[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, kernelName:%s",
                                ret, kernelName.c_str()),
                    HCCL_E_RUNTIME);

        ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
                    HCCL_ERROR("[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, kernelName:%s", ret, kernelName.c_str()),
                    HCCL_E_RUNTIME);

        aclrtParamHandle paraHandle;
        ret = aclrtKernelArgsAppend(argsHandle, &param, sizeof(OpParam), &paraHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
                    HCCL_ERROR("[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append size %u, kernelName:%s", ret,
                                sizeof(OpParam), kernelName.c_str()),
                    HCCL_E_RUNTIME);

        ret = aclrtKernelArgsFinalize(argsHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
                    HCCL_ERROR("[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, kernelName:%s", ret,
                                kernelName.c_str()),
                    HCCL_E_RUNTIME);

        u16 NOTIFY_DEFAULT_WAIT_TIME = 27 * 68;   // notifywait默认1836等待时长
        aclrtLaunchKernelCfg cfg;
        aclrtLaunchKernelAttr attr;
        attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
        attr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME;
        cfg.numAttrs = 1;
        cfg.attrs = &attr;
        constexpr u32 numBlocks = 1;
        aclError aclRet = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &cfg, argsHandle, nullptr);
        CHK_PRT_RET(aclRet != ACL_SUCCESS,
                    HCCL_ERROR("[LoadCustomKernel][aclrtLaunchKernelWithConfig]errNo[0x%016llx] launch kernel failed", ret), HCCL_E_OPEN_FILE_FAILURE);

        // Host stream等待Device的通知
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(cpuTsThread, 0, NOTIFY_DEFAULT_WAIT_TIME)));
    } else {
        CHK_RET(executor->Orchestrate(param, resCtx));
    }

    return HCCL_SUCCESS;
}

/* 算子级别基础拓扑解析，缓存在host上 */
HcclResult CalcBaseTopoInfo(HcclComm comm, OpParam &param, TopoInfo** topoInfo)
{
    uint64_t size = sizeof(TopoInfo);
    void *ctx = nullptr;
    // 若获取Context失败，表示对应Context尚未缓存
    if (HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size) != HCCL_SUCCESS) {
        // 创建新的Context
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
        // 将Context内存地址强转为TopoInfo
        *topoInfo = static_cast<TopoInfo *>(ctx);
        // 将对应拓扑信息填入到Context内存中
        CHK_RET(InitRankInfo(comm, *topoInfo));
        return HCCL_SUCCESS;
    }

    *topoInfo = static_cast<TopoInfo *>(ctx);
    return HCCL_SUCCESS;
}

HcclResult SetAlgoLevel0(TopoInfo* topoInfo, HcclAlgoType algoConfig, AlgTypeLevel0 &algType)
{
    if (algoConfig == HcclAlgoType::HCCL_ALGO_TYPE_NULL) {
        algType = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
        return HCCL_SUCCESS;
    }

    if (algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT && algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_NA) {
        HCCL_WARNING("level0:%d algo is not supported. the config is ignored.", algoConfig);
    }

    CHK_RET(GetDefaultAlgoLevel0Module(topoInfo, algType));
    return HCCL_SUCCESS;
}

HcclResult GetDefaultAlgoLevel0Module(TopoInfo* topoInfo, AlgTypeLevel0 &algType)
{
    u32 deviceNumPerAggregation = topoInfo->deviceNumPerModule;
    if (deviceNumPerAggregation == DEVICE_EIGHT) {
        algType = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    } else if (deviceNumPerAggregation == DEVICE_FOUR) {
        algType = AlgTypeLevel0::ALG_LEVEL0_4P_MESH;
    } else if (deviceNumPerAggregation == DEVICE_TWO) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else if (deviceNumPerAggregation == DEVICE_ONE) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else {
        algType = AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING;
    }

    if (!topoInfo->multiModuleDiffDeviceNumMode && topoInfo->deviceType == DevType::DEV_TYPE_910B) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
        HCCL_DEBUG("[GetDefaultAlgoLevel0Module] AlgTypeLevel0 is set to ALG_LEVEL0_NP_MESH (HCCS links is enabled).");
    }

    if (topoInfo->deviceType == DevType::DEV_TYPE_910_93) {
        algType = topoInfo->isHCCSSWNumEqualToTwiceSIONum ? AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING :
                                                            AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
        HCCL_DEBUG("[GetDefaultAlgoLevel0Module] AlgTypeLevel0 is set to [%u].", algType);
    }
    return HCCL_SUCCESS;
}

HcclResult SetAlgoLevel1(TopoInfo* topoInfo, HcclAlgoType algoConfig, AlgTypeLevel1 &algType,
    HcclCMDType opType)
{
    HcclAlgoType algoConfigShadow = algoConfig;
    switch (algoConfig) {
        case HcclAlgoType::HCCL_ALGO_TYPE_HDR:
            algType = AlgTypeLevel1::ALG_LEVEL1_HD;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_RING:
            algType = AlgTypeLevel1::ALG_LEVEL1_RING;
            HCCL_INFO("server num[%u]: level1:ring algo is set.", topoInfo->moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_INFO("server num[%u]: level1:nhr algo is set.", topoInfo->moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR_V1;
            HCCL_INFO("server num[%u]: level1:nhr_v1 algo is set.", topoInfo->moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_AHC:
            if (opType < HcclCMDType::HCCL_CMD_ALL) {
                algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
                HCCL_INFO("server num[%u]: level1:ahc algo is not support, set default.", topoInfo->moduleNum);
                break;
            } else {
                algType = AlgTypeLevel1::ALG_LEVEL1_AHC;
                return HCCL_SUCCESS;
            }
        case HcclAlgoType::HCCL_ALGO_TYPE_AHC_BROKE:
            if (opType < HcclCMDType::HCCL_CMD_ALL) {
                algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
                HCCL_INFO("server num[%u]: level1:ahc broke algo is not support, set default.", topoInfo->moduleNum);
                break;
            } else {
                algType = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
                return HCCL_SUCCESS;
            }
        case HcclAlgoType::HCCL_ALGO_TYPE_NB:
            algType = AlgTypeLevel1::ALG_LEVEL1_NB;
            HCCL_INFO("server num[%u]: level1:nb algo is set.", topoInfo->moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE:
            algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
            HCCL_INFO("server num[%u]: level1:pipeline algo is set.", topoInfo->moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH:
        case HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE:
            HCCL_WARNING("level1:fullmesh algo is not supported. the config is ignored.");
        default:
            algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
            break;
    }

    HCCL_DEBUG("[AlgConfigurator][SetAlgoLevel1] algType[%u], deviceType_[%u], workflowmode[%u]", algType,
        topoInfo->deviceType, GetWorkflowMode());

    if (algoConfigShadow == HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) {
        CHK_RET(GetDefaultAlgoLevel1V1(topoInfo, algType));
    }
    return HCCL_SUCCESS;
}

HcclResult GetDefaultAlgoLevel1V1(TopoInfo* topoInfo, AlgTypeLevel1 &algType)
{
    u32 moduleNum = topoInfo->moduleNum;
    if (moduleNum >=  HCCL_INTER_SERVER_RING_ALGO_MAX_SUPPORT_SERVER_NUM) {
        // server 数为 8 以上：使用 HD 算法
        algType = AlgTypeLevel1::ALG_LEVEL1_HD;
    } else {
        // server 数为 2 的非整数次幂：使用 RING 算法
        // server 数为 2 的整数次幂：使用 HD 算法
        algType = (((moduleNum & (moduleNum - 1)) != 0) || (moduleNum == 1)) ?
            AlgTypeLevel1::ALG_LEVEL1_RING :
            AlgTypeLevel1::ALG_LEVEL1_HD;
    }
    if (algType == AlgTypeLevel1::ALG_LEVEL1_HD && topoInfo->deviceType == DevType::DEV_TYPE_910_93) {
        algType = AlgTypeLevel1::ALG_LEVEL1_NHR;
    }
    HCCL_INFO("[AlgConfigurator][GetDefaultAlgoLevel1V1] algType[%u], moduleNum[%u]", algType, moduleNum);
    return HCCL_SUCCESS;
}

HcclResult SetAlgoLevel2(TopoInfo* topoInfo, HcclAlgoType algoConfig, AlgTypeLevel2 &algType)
{
    u32 superPodNum = topoInfo->superPodNum;
    switch (algoConfig) {
        case HcclAlgoType::HCCL_ALGO_TYPE_HDR:
            algType = AlgTypeLevel2::ALG_LEVEL2_HD;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_RING:
            algType = AlgTypeLevel2::ALG_LEVEL2_RING;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR:
            algType = AlgTypeLevel2::ALG_LEVEL2_NHR;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NB:
            algType = AlgTypeLevel2::ALG_LEVEL2_NB;
            break;
        default: {
            // Level2默认选择NHR算法
            algType = AlgTypeLevel2::ALG_LEVEL2_NHR;
            break;
        }
    }
    HCCL_DEBUG("[AlgConfigurator][SetAlgoLevel2]algType[%u], deviceType_[%u], superPodNum_[%u]",
        algType, topoInfo->deviceType, superPodNum);
    return HCCL_SUCCESS;
}

HcclResult GetAlgType(TopoInfo* topoInfo, HcclCMDType opType, AlgType& algType)
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

    CHK_RET(SetAlgoLevel0(topoInfo, ret[HCCL_ALGO_LEVEL_0], algType0));
    CHK_RET(SetAlgoLevel1(topoInfo, ret[HCCL_ALGO_LEVEL_1], algType1, opType));
    CHK_RET(SetAlgoLevel2(topoInfo, ret[HCCL_ALGO_LEVEL_2], algType2));

    algType.algoLevel0 = algType0;
    algType.algoLevel1 = algType1;
    algType.algoLevel2 = algType2;

    return HCCL_SUCCESS;
}

std::string SetLaunchMode(CommEngine engine)
{
    std::string launchMode = "UNKNOWN";
    if (engine == CommEngine::COMM_ENGINE_CPU) {
        launchMode = "HOST";
    } else if (engine == CommEngine::COMM_ENGINE_CPU_TS) {
        launchMode = "HOST_TS";
    } else if ((engine == CommEngine::COMM_ENGINE_AICPU) ||
               (engine == CommEngine::COMM_ENGINE_AICPU_TS)) {
        launchMode = "AI_CPU";
    } else if (engine == CommEngine::COMM_ENGINE_AIV) {
        launchMode = "AIV";
    }
    return launchMode;
}

HcclResult SelectAlg(HcclComm comm, OpParam &param, TopoInfo* topoInfo, AlgType& algType, std::string &algName)
{
    (void) comm;
    // 由于scatter只支持server间ring,nb和NHR，如果不是上述算法，需要重定向到ring；910_93仅支持server间ring
    if (!(algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) &&
        !(algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) &&
        !(algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING)) {
        HCCL_INFO("[ScatterOperator][Scatter] algType[%s] is not supported, reset algType=ring",
            AlgTypeToStr(algType).c_str());
        algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    }

    if (topoInfo->userRankSize == 1 && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        algName = "ScatterSingleExecutor";
    } else if (topoInfo->multiModuleDiffDeviceNumMode || topoInfo->multiSuperPodDiffServerNumMode) {
        algName = "ScatterCommExecutor";
    } else if (topoInfo->deviceType == DevType::DEV_TYPE_910B) {
        algName = "ScatterMeshExecutor";
    } else if (topoInfo->deviceType == DevType::DEV_TYPE_910_93) {
        algName = "ScatterRingFor91093Executor";
    }

    // 在原先的tag中添加算法名字，得到algTag
    bool isOpBase = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    if (isOpBase) {
        int ret = sprintf_s(param.algTag, sizeof(param.algTag), "%s_%s_%d", param.tag, algName.c_str(), param.root);
        if (ret <= 0) {
            HCCL_ERROR("faled to fill param.algTag");
            return HCCL_E_INTERNAL;
        }
    }

    if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
        // 在algTag中追加编排模式
        std::string opExpansionStr = SetLaunchMode(param.engine);

        const char* launchMode = (((param.engine == CommEngine::COMM_ENGINE_AICPU) ||
                                   (param.engine == CommEngine::COMM_ENGINE_AICPU_TS)) ? "_device" : "_host");
        int ret = strcat_s(param.algTag, sizeof(param.algTag), launchMode);
        if (ret != 0) {
            HCCL_ERROR("faled to fill param.algTag");
            return HCCL_E_INTERNAL;
        }

        HCCL_INFO("[SelectAlg] Scatter algTag is [%s] algName is [%s]", param.algTag, algName.c_str());
        HCCL_CONFIG_INFO(HCCL_ALG,
                         "[%s] algTag[%s] algName[%s] userRank[%u] algType[%s] "\
                         "userRankSize[%u] level0Size[%u] moduleNum[%u] "\
                         "level2Size[%u] opExpansionMode[%s] isZeroCopy[%u] isOpBase[%u].",
                         __func__, param.algTag, algName.c_str(), topoInfo->userRank, AlgTypeToStr(algType).c_str(),
                         topoInfo->userRankSize, topoInfo->deviceNumPerModule, topoInfo->moduleNum,
                         topoInfo->superPodNum, opExpansionStr.c_str(), param.isZeroCopy, isOpBase);
    }
    return HCCL_SUCCESS;
}

HcclResult GetAlgRes(HcclComm comm, OpParam &param, std::unique_ptr<ExecutorBase> &executor,
    TopoInfo* topoInfo, AlgType& algType, AlgResourceCtx** resCtx)
{
    // 这种情况下资源已经有了
    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.algTag, param.engine, &ctx, &size) == HCCL_SUCCESS) {
        *resCtx = static_cast<AlgResourceCtx *>(ctx);
        return HCCL_SUCCESS;
    }

    // 资源计算
    AlgHierarchyInfo algHierarchyInfo;
    AlgResourceRequest resRequest;
    CHK_RET(executor->CalcResRequest(comm, param, topoInfo, algHierarchyInfo, resRequest, algType));

    // 开始计算资源Context的长度
    size = sizeof(AlgResourceCtx);
    // 计算变长数据区中threads占用的空间
    size += sizeof(ThreadHandle) * (resRequest.slaveThreadNum + 1);
    // 计算变长数据区中channels占用的空间
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        size += sizeof(ChannelInfo) * algHierarchyInfo.infos[level].localRankSize;
    }
    // 创建Context
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, param.engine, size, &ctx));
    // 将内存强转为AlgResourceCtx结构体
    *resCtx = static_cast<AlgResourceCtx *>(ctx);

    AlgResourceCtx* resCtxHost;
    if (param.engine == COMM_ENGINE_AICPU_TS) {
        // AICPU模式下分配一块Host内存用于填充资源
        ACLCHECK(aclrtMallocHost(reinterpret_cast<void**>(&resCtxHost), size));
    } else {
        resCtxHost = *resCtx;
    }

    topoInfo->notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
    resCtxHost->topoInfo = *topoInfo;
    resCtxHost->algType = algType;
    resCtxHost->algHierarchyInfo = algHierarchyInfo;

    // 创建资源，并填充到Host内存上
    HcclResult ret = AllocAlgResource(comm, param, resRequest, resCtxHost);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("failed to alloc alg resource.");
        if (param.engine == COMM_ENGINE_AICPU_TS) {
            ACLCHECK(aclrtFreeHost(resCtxHost));
        }
        return ret;
    }
    topoInfo->mainThread = resCtxHost->topoInfo.mainThread;
 
    CHK_RET(HcclEngineCtxCopy(comm, param.engine, param.algTag, resCtxHost, size, 0));
    if (param.engine == COMM_ENGINE_AICPU_TS) {
        // 从Host内存拷贝到Device Context内存上
        ACLCHECK(aclrtFreeHost(resCtxHost));
    }
    return HCCL_SUCCESS;
}

HcclResult AllocAlgResource(HcclComm comm, const OpParam& param, AlgResourceRequest &resRequest,
    AlgResourceCtx* resCtxHost)
{
    void* cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer的地址和大小
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    u64 sizePerCcl = cclBufferSize / 2;
    // CCL IN使用CCL Buffer的前一半
    resCtxHost->cclInputMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, sizePerCcl};
    // CCL OUT使用CCL Buffer的后一半
    resCtxHost->cclOutputMem = HcclMem{HCCL_MEM_TYPE_DEVICE,
        static_cast<void*>(static_cast<char*>(cclBufferAddr) + sizePerCcl), sizePerCcl};
    resCtxHost->notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
    resCtxHost->slaveThreadNum = resRequest.slaveThreadNum;
    resCtxHost->notifyNumPerThread = resRequest.notifyNumPerThread;

    char* curPtr = reinterpret_cast<char *>(resCtxHost);
    curPtr += sizeof(AlgResourceCtx); // 偏移指针
    ThreadHandle* threads = reinterpret_cast<ThreadHandle *>(curPtr);
    if (param.engine == COMM_ENGINE_AICPU_TS) {
        // 主流多申请1个notify用于host和device同步
        CHK_RET(HcclThreadAcquire(comm, param.engine, 1,
            resRequest.notifyNumOnMainThread + 1, threads));
        resCtxHost->topoInfo.mainThread = *threads;
        HCCL_DEBUG("threads ptr is %p\n", *threads);
    } else {
        // host模式下，将主流封装为thread，并创建主流上的notify
        CHK_RET(HcclThreadAcquireWithStream(comm, param.engine, param.stream,
            resRequest.notifyNumOnMainThread, threads));
    }
    curPtr += sizeof(ThreadHandle); // 指针向后偏移

    if (resRequest.slaveThreadNum > 0) {
        threads = reinterpret_cast<ThreadHandle *>(curPtr);
        // 创建从流thread及对应的notify
        CHK_RET(HcclThreadAcquire(comm, param.engine, resRequest.slaveThreadNum,
            resRequest.notifyNumPerThread, threads));
        curPtr += sizeof(ThreadHandle) * resCtxHost->slaveThreadNum; // 指针向后偏移
    }
    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
        HCCL_DEBUG("[AllocAlgResource] slaveThreadNum[%u]", resRequest.slaveThreadNum);
        for (u32 i = 0; i < resRequest.slaveThreadNum; i++) {
            HCCL_DEBUG("[AllocAlgResource] threads[%u]=[%llu]", i, threads[i]);
        }
    }

    // 迭代每个子通信域的建链请求，创建链路
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
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

        SubCommInfo &subCommInfo = resCtxHost->algHierarchyInfo.infos[level];
        // 因为在创建context的时候，每一层预留的链路数量是子通信域rank数量，
        // 建链数量可能小于子通信域rank数量，所以先把链路全部设置为false，
        // 后再填充有效的链路
        ChannelInfo* channels = reinterpret_cast<ChannelInfo *>(curPtr);
        for (u32 rank = 0; rank < subCommInfo.localRankSize; rank++) {
            // 先全部设置为false
            channels[rank].isValid = false;
        }
        for (u32 idx = 0; idx < validChannelNum; idx++) {
            // 对于真实建链的链路进行填充
            HcclChannelDesc &channelDesc = levelNChannelRequest[idx];
            u32 levelRank;
            CHK_RET(GetSubCommRankByUserRank(channelDesc.remoteRank, level, resCtxHost->algHierarchyInfo, levelRank));
            channels[levelRank].isValid = true;
            channels[levelRank].remoteRank = channelDesc.remoteRank;
            channels[levelRank].protocol = channelDesc.channelProtocol;
            channels[levelRank].notifyNum = channelDesc.notifyNum;
            channels[levelRank].handle = levelNChannels[idx];
            void* remoteBufferAddr;
            uint64_t remoteBufferSize;
            CHK_RET(HcclChannelGetHcclBuffer(comm, levelNChannels[idx], &remoteBufferAddr, &remoteBufferSize));
            channels[levelRank].remoteInput = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteBufferAddr, remoteBufferSize};
            channels[levelRank].remoteOutput = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteBufferAddr, remoteBufferSize};
        }
        curPtr += sizeof(ChannelInfo) * subCommInfo.localRankSize; // 偏移指针
    }

    HCCL_INFO("[%s] Alloc res success.", __func__);
    return HCCL_SUCCESS;
}
}