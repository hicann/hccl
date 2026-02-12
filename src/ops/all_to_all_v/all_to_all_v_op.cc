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
#include "all_to_all_v_op.h"
#include "op_common.h"

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
    uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclAlltoAll");
    if (!CheckHCCLIndependentOp()) {
        return HcclAlltoAllInner(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream);
    }

    // 穿刺的时候只考虑A5
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    // 非95设备转到老流程
    if (deviceType != DevType::DEV_TYPE_910_95) {
        return HcclAlltoAllInner(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream);
    }
    // 图模式引导到老的流程上面
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return HcclAlltoAllInner(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream);
    }
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    // A3是：export HCCL_OP_EXPANSION_MODE="AI_CPU"，A5的接口还没提供
    CHK_RET(InitEnvConfig());

    // 参数校验等工作
    CHK_RET(CheckAlltoAllInputPara(comm, sendBuf, sendCount, sendType, recvBuf, recvCount, recvType));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string tag =  "ALLTOALL_" + string(commName);
    CHK_RET(HcclCheckTag(tag.c_str()));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());
    CHK_RET(CheckCount(recvCount));
    CHK_RET(CheckDataType(recvType, false));

    // 构造四个矩阵，适配alltoallV的逻辑
    u64 dataCountPerRank = recvCount;
    u64 dataCountOffset = 0;
    vector<u64> sendCounts(rankSize, dataCountPerRank);
    vector<u64> recvCounts(rankSize, dataCountPerRank);
    vector<u64> sdispls(rankSize, dataCountOffset);
    vector<u64> rdispls(rankSize, dataCountOffset);
    for (u64 i = 0; i < rankSize; i++) {
        sdispls[i] = dataCountOffset;
        rdispls[i] = dataCountOffset;
        dataCountOffset += dataCountPerRank;
    }

    // 底层走AlltoAllV
    CHK_RET_AND_PRINT_IDE(AlltoAllVOutPlace(sendBuf, sendCounts.data(), sdispls.data(),
        recvBuf, recvCounts.data(), rdispls.data(), recvType, comm, stream, tag, HcclCMDType::HCCL_CMD_ALLTOALL, rankSize),
        tag.c_str());

    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
    const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclAlltoAllV");
    if (!CheckHCCLIndependentOp()) {
        return HcclAlltoAllVInner(sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, comm, stream);
    }

    // 穿刺的时候只考虑A5
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    // 非95设备转到老流程
    if (deviceType != DevType::DEV_TYPE_910_95) {
        return HcclAlltoAllVInner(sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, comm, stream);
    }
    // 图模式引导到老的流程上面
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return HcclAlltoAllVInner(sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, comm, stream);
    }
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    // A3是：export HCCL_OP_EXPANSION_MODE="AI_CPU"，A5的接口还没提供
    CHK_RET(InitEnvConfig());

    // 参数校验等工作
    CHK_RET(CheckAlltoAllVInputPara(comm, sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string tag =  "ALLTOALLV_" + string(commName);
    CHK_RET(HcclCheckTag(tag.c_str()));

    u64 maxSendRecvCount = 0;
    for (u64 i = 0; i < rankSize; i++) {
        maxSendRecvCount = max(maxSendRecvCount, static_cast<const u64 *>(sendCounts)[i]);
        maxSendRecvCount = max(maxSendRecvCount, static_cast<const u64 *>(recvCounts)[i]);
    }

    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());
    CHK_RET(CheckCount(maxSendRecvCount));
    CHK_RET(CheckDataType(recvType, false));

    // 底层走AlltoAllV
    CHK_RET_AND_PRINT_IDE(AlltoAllVOutPlace(sendBuf, sendCounts, sdispls, recvBuf, recvCounts, rdispls, recvType, comm, stream,
        tag, HcclCMDType::HCCL_CMD_ALLTOALLV, rankSize), tag.c_str());

    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclAlltoAllVC");
    if (!CheckHCCLIndependentOp()) {
        return HcclAlltoAllVCInner(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, comm, stream);
    }

    // 穿刺的时候只考虑A5
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    // 非95设备转到老流程
    if (deviceType != DevType::DEV_TYPE_910_95) {
        return HcclAlltoAllVCInner(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, comm, stream);
    }
    // 图模式引导到老的流程上面
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return HcclAlltoAllVCInner(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, comm, stream);
    }
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    // A3是：export HCCL_OP_EXPANSION_MODE="AI_CPU"，A5的接口还没提供
    CHK_RET(InitEnvConfig());

    // 参数校验等工作
    CHK_RET(CheckAlltoAllVCInputPara(comm, sendBuf, sendCountMatrix, sendType, recvBuf, recvType));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string tag =  "ALLTOALLVC_" + string(commName);
    CHK_RET(HcclCheckTag(tag.c_str()));

    // 取出sendCountMatrix的数据
    std::vector<std::vector<u64>> outputMatrix;
    const u64* data = static_cast<const u64*>(sendCountMatrix);
    outputMatrix.resize(rankSize);
    for (u64 i = 0; i < rankSize; ++i) {
        // 计算当前行的起始指针位置（行优先顺序）
        const u64* rowStart = data + i * rankSize;
        // 直接通过指针初始化当前行的vector
        outputMatrix[i].assign(rowStart, rowStart + rankSize);
    }

    // 构造四个矩阵，适配alltoallV的逻辑
    std::vector<u64> sendCounts(rankSize, 0);
    std::vector<u64> recvCounts(rankSize, 0);
    std::vector<u64> sdispls(rankSize, 0);
    std::vector<u64> rdispls(rankSize, 0);

    u64 dataCountOffset = 0;
    for (u64 i = 0; i < rankSize; i++) {
        sendCounts[i] = outputMatrix[userRank][i];
        sdispls[i] = dataCountOffset;
        dataCountOffset += sendCounts[i];
    }

    dataCountOffset = 0;
    for (u64 i = 0; i < rankSize; i++) {
        recvCounts[i] = outputMatrix[i][userRank];
        rdispls[i] = dataCountOffset;
        dataCountOffset += recvCounts[i];
    }

    u64 maxSendRecvCount = 0;
    for (u64 i = 0; i < rankSize * rankSize; i++) {
        maxSendRecvCount = max(maxSendRecvCount, data[i]);
    }

    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());
    CHK_RET(CheckCount(maxSendRecvCount));
    CHK_RET(CheckDataType(recvType, false));

    // 底层走AlltoAllV
    CHK_RET_AND_PRINT_IDE(AlltoAllVOutPlace(sendBuf, sendCounts.data(), sdispls.data(),
        recvBuf, recvCounts.data(), rdispls.data(), recvType, comm, stream, tag, HcclCMDType::HCCL_CMD_ALLTOALLVC, rankSize),
        tag.c_str());

    return HCCL_SUCCESS;
}

namespace ops_hccl {

HcclResult CheckAlltoAllInputPara(HcclComm comm, const void *sendBuf, uint64_t sendCount, HcclDataType sendType,
    const void *recvBuf, uint64_t recvCount, HcclDataType recvType)
{
    // 入参合法性校验
    CHK_PRT_RET(sendCount == 0 && recvCount == 0,
        HCCL_WARNING("sendCount and recvCount are both 0, return AllToAll success"), HCCL_SUCCESS);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAll", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAll", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAll", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);
    CHK_PRT_RET(sendCount != recvCount,
        HCCL_ERROR("sendCount[%lu] and recvCount[%lu] are not equal, please check params",
            sendCount, recvCount), HCCL_E_PARA);
    CHK_PRT_RET(sendType != recvType,
        HCCL_ERROR("sendType[%s] and recvType[%s] are not equal, please check params",
            GetDataTypeEnumStr(sendType).c_str(), GetDataTypeEnumStr(recvType).c_str()), HCCL_E_PARA);
    CHK_PRT_RET(sendBuf == recvBuf,
        HCCL_ERROR("[HcclAlltoAll] sendBuf and recvBuf cannot be same."), HCCL_E_PARA);

    return HCCL_SUCCESS;
}

HcclResult CheckAlltoAllVInputPara(HcclComm comm, const void *sendBuf, const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType)
{
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);
    RPT_INPUT_ERR(sendCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "sendCounts", "nullptr", "please check sendCounts"}));
    CHK_PTR_NULL(sendCounts);
    RPT_INPUT_ERR(sdispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "sdispls", "nullptr", "please check sdispls"}));
    CHK_PTR_NULL(sdispls);
    RPT_INPUT_ERR(recvCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "recvCounts", "nullptr", "please check recvCounts"}));
    CHK_PTR_NULL(recvCounts);
    RPT_INPUT_ERR(rdispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "rdispls", "nullptr", "please check rdispls"}));
    CHK_PTR_NULL(rdispls);
    CHK_PRT_RET(sendType != recvType,
        HCCL_ERROR("sendType[%s] and recvType[%s] are not equal, please check params",
            GetDataTypeEnumStr(sendType).c_str(), GetDataTypeEnumStr(recvType).c_str()), HCCL_E_PARA);
    CHK_PRT_RET(sendBuf == recvBuf,
        HCCL_ERROR("[HcclAlltoAllV] sendBuf and recvBuf cannot be same."), HCCL_E_PARA);

    return HCCL_SUCCESS;
}

HcclResult CheckAlltoAllVCInputPara(HcclComm comm, const void *sendBuf, const void *sendCountMatrix,
    HcclDataType sendType, const void *recvBuf, HcclDataType recvType)
{
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllVC", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendCountMatrix == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllVC", "sendCountMatrix", "nullptr", "please check sendCountMatrix"}));
    CHK_PTR_NULL(sendCountMatrix);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllVC", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllVC", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);
    CHK_PRT_RET(sendType != recvType,
        HCCL_ERROR("sendType[%s] and recvType[%s] are not equal, please check params",
            GetDataTypeEnumStr(sendType).c_str(), GetDataTypeEnumStr(recvType).c_str()), HCCL_E_PARA);
    CHK_PRT_RET(sendBuf == recvBuf,
        HCCL_ERROR("[HcclAlltoAllVC] sendBuf and recvBuf cannot be same."), HCCL_E_PARA);

    return HCCL_SUCCESS;
}

// alltoall/alltoallv/alltoallvc 统一，当前只支持outPlace
HcclResult AlltoAllVOutPlace(const void *sendBuf, const void *sendCounts, const void *sdispls, const void *recvBuf,
    const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclComm comm, aclrtStream stream,
    const std::string &tag, HcclCMDType opType, u32 rankSize)
{
    HCCL_INFO("Start to execute AlltoAllVOutPlace");

    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    u64 varMemSize = ALL_TO_ALL_V_VECTOR_NUM * userRankSize * sizeof(u64);
    void *paramMem = malloc(sizeof(OpParam) + varMemSize);
    if (!paramMem) {
        // 内存分配失败
        HCCL_ERROR("malloc OpParam failed!");
        return HCCL_E_INTERNAL;
    }
    OpParam *paramPtr = new (paramMem)OpParam();
    OpParam &param = *paramPtr;

    CHK_RET(HcclGetCommName(comm, param.commName));
    param.stream = stream;
    param.opMode = OpMode::OPBASE;
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    param.deviceType = deviceType;

    // topoInfo的tag，所有相同的算子可以共享
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
    if (ret <= 0) {
        HCCL_ERROR("failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }

    param.inputPtr = const_cast<void*>(sendBuf);
    param.outputPtr = const_cast<void*>(recvBuf);
    param.varMemSize = varMemSize;
    param.all2AllVDataDes.sendType = dataType;
    param.all2AllVDataDes.recvType = dataType;
    param.all2AllVDataDes.sendCounts = const_cast<void *>(sendCounts);
    param.all2AllVDataDes.recvCounts = const_cast<void *>(recvCounts);
    param.all2AllVDataDes.sdispls = const_cast<void *>(sdispls);
    param.all2AllVDataDes.rdispls = const_cast<void *>(rdispls);
    // 后面结构体改了之后，这里该赋值的地方要赋值，不然后续别人校验可能会有问题的
    u64 inputSize = 0;
    u64 outputSize = 0;
    for (u64 i = 0; i < userRankSize; i++) {
        inputSize += static_cast<const u64*>(sendCounts)[i] * SIZE_TABLE[dataType];
        outputSize += static_cast<const u64*>(recvCounts)[i] * SIZE_TABLE[dataType];
    }

    param.inputSize = inputSize;
    param.outputSize = outputSize;
    param.enableDetour = false;
    param.opType = opType;

    u64* data = reinterpret_cast<u64*>(param.varData);
    for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * userRankSize; i++) {
        u64 val = i / rankSize;
        switch(val) {
            case 0:
                data[i] = static_cast<const u64*>(sendCounts)[i % rankSize];
                break;
            case 1:
                data[i] = static_cast<const u64*>(recvCounts)[i % rankSize];
                break;
            case 2:
                data[i] = static_cast<const u64*>(sdispls)[i % rankSize];
                break;
            case 3:
                data[i] = static_cast<const u64*>(rdispls)[i % rankSize];
                break;
            default:
                break;
        }
    }

    for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * userRankSize; i++) {
        HCCL_INFO("[AlltoAllVOutPlace] varData[%u] is [%u]", i, data[i]);
    }
    HCCL_INFO("[AlltoAllVOutPlace] SIZE_TABLE[dataType] is [%u]", SIZE_TABLE[dataType]);

    if (userRankSize == 1) {
        HCCL_WARNING("[%s] ranksize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(param));
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_RET(HcclExecOp(comm, param));
    paramPtr->~OpParam();
    free(paramMem);
    HCCL_INFO("Execute AlltoAllVOutPlace success.");
    return HCCL_SUCCESS;
}
}