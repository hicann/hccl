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
#include "reduce_scatter_v_op.h"
#include "op_common.h"

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

// 代码入口
HcclResult HcclReduceScatterV(void *sendBuf,  const void *sendCounts, const void *sendDispls, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclReduceScatterV");
    if (!CheckHCCLIndependentOp()) {
        return HcclReduceScatterVInner(sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    }
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    // 非95设备转到老流程
    if (deviceType != DevType::DEV_TYPE_910_95) {
        return HcclReduceScatterVInner(sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    }
    // 图模式引导到老的流程上面
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return HcclReduceScatterVInner(sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    }
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    // A3是：export HCCL_OP_EXPANSION_MODE="AI_CPU"，A5的接口还没提供
    CHK_RET(InitEnvConfig());

    // 参数校验等工作;
    // 校验入参
    CHK_RET(CheckReduceScatterVInputPara(comm, sendBuf, recvBuf, sendCounts, sendDispls, stream));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    // 校验sendCounts全部为0的情况
    const u64* sendCountsAddr = reinterpret_cast<const u64*>(sendCounts);
    CHK_PRT_RET(std::all_of(sendCountsAddr, sendCountsAddr + rankSize, [](auto count) { return count == 0; }), 
            HCCL_WARNING("input all %u elements in sendCounts are 0, return success", rankSize), 
            HCCL_SUCCESS);  
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string tag = "ReduceScatterV_" + string(commName);
    CHK_RET(HcclCheckTag(tag.c_str()));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());
    CHK_RET(CheckCount(recvCount));
    CHK_RET(CheckDataType(dataType, true));

    // 初始化参数
    CHK_RET_AND_PRINT_IDE(ReduceScatterVOutPlace(sendBuf, sendDispls, sendCounts, recvBuf, recvCount, dataType, op, comm, stream, tag),
                          tag.c_str());

    return HCCL_SUCCESS;
}

namespace ops_hccl {
HcclResult CheckReduceScatterVInputPara(HcclComm comm, void *sendBuf, void *recvBuf, const void *sendCounts, const void *sendDispls, aclrtStream stream)
{
    // 入参合法性校验
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatter", "stream", "nullptr", "please check stream"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatterV", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);

    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatterV", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);

    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatterV", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);

    RPT_INPUT_ERR(sendCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatterV", "sendCounts", "nullptr", "please check sendCounts"}));
    CHK_PTR_NULL(sendCounts);

    RPT_INPUT_ERR(sendDispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatterV", "sendDispls", "nullptr", "please check sendDispls"}));
    CHK_PTR_NULL(sendDispls);

    CHK_PRT_RET(sendBuf == recvBuf,
        HCCL_ERROR("[HcclReduceScatterV] sendBuf and recvBuf cannot be same."), HCCL_E_PARA);

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVOutPlace(void *sendBuf, const void *sendDispls, const void *sendCounts, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, const std::string &tag)
{
    HCCL_INFO("Start to execute ReduceScatterVOutPlace");
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 outputSize = recvCount * perDataSize;
    u64 inputSize = outputSize * userRankSize;

    // 申请OpParam参数结构体内存
    u64 varMemSize = 2 * userRankSize * sizeof(u64);

    void* paramMem = malloc(sizeof(OpParam) + varMemSize);

if (!paramMem) {
        // 内存分配失败
        HCCL_ERROR("malloc OpParam failed!");
        return HCCL_E_INTERNAL;
    }
    OpParam* paramPtr = new (paramMem) OpParam();
    OpParam& param = *paramPtr;

    CHK_RET(HcclGetCommName(comm, param.commName));
    param.stream = stream;
    param.reduceType = op;
    param.opMode = OpMode::OPBASE;

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    // topoInfo的tag，所有相同的算子可以共享
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
    if (ret <= 0) {
        HCCL_ERROR("failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }

    // 参数准备
    param.inputPtr = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    param.vDataDes.dataType = dataType;

    // 参数准备
    std::vector<u64> merged(userRankSize*2);
    const u64* sendDisplsAddr = reinterpret_cast<const u64*>(sendDispls);
    const u64* sendCountsAddr = reinterpret_cast<const u64*>(sendCounts);
    std::copy(sendCountsAddr, sendCountsAddr + userRankSize, merged.begin());
    std::copy(sendDisplsAddr, sendDisplsAddr + userRankSize, merged.begin() + userRankSize);
    param.varMemSize = varMemSize;

    // 从源内存地址按字节直接拷贝数据到目标地址
    memcpy(param.varData, merged.data(), varMemSize);
    const u64* varData = reinterpret_cast<const u64*>(param.varData);

    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
    param.enableDetour = false;
    param.deviceType = deviceType;

    if (userRankSize == 1) {
        HCCL_WARNING("[%s] ranksize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(param));
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_RET(HcclExecOp(comm, param));
    paramPtr->~OpParam();
    free(paramMem);
    HCCL_INFO("Execute ReduceScatterVOutPlace success.");
    return HCCL_SUCCESS;
}
}