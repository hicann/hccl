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
#include "adapter_error_manager_pub.h"
#include "hccl_inner.h"
#include "hccl.h"
#include "config_log.h"
#include "workflow.h"
#include "load_kernel.h"
#include "all_gather_v_op.h"
#include "op_common.h"

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts,
    const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclAllGatherV");
 
    if (!CheckHCCLIndependentOp()) {
        return HcclAllGatherVInner(sendBuf, sendCount, recvBuf, recvCounts, recvDispls, dataType, comm, stream);
    }
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_910_95) {
        return HcclAllGatherVInner(sendBuf, sendCount, recvBuf, recvCounts, recvDispls, dataType, comm, stream);
    }
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return HcclAllGatherVInner(sendBuf, sendCount, recvBuf, recvCounts, recvDispls, dataType, comm, stream);
    }
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    CHK_RET(InitEnvConfig());
    // 参数校验等工作
    CHK_PRT_RET(sendCount == 0, HCCL_WARNING("input recvCount is 0, return all gather success"), HCCL_SUCCESS);
    CHK_RET(CheckAllGatherVInputPara(comm, sendBuf, recvBuf));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string tag = "AllGatherV_" + string(commName);
    // CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), sendCount, dataType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());
    CHK_RET(CheckCount(sendCount));
    CHK_RET(CheckDataType(dataType, false));
 
    // 执行AllGatherV
    CHK_RET_AND_PRINT_IDE(AllGatherVOutPlace(sendBuf, recvBuf, sendCount, recvCounts, recvDispls, dataType, comm, stream, tag), tag.c_str());
 
    return HCCL_SUCCESS;
}
 
namespace ops_hccl {
HcclResult CheckAllGatherVInputPara(HcclComm comm, void *sendBuf, void *recvBuf)
{
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),
                  std::vector<std::string>({"HcclAllGatherV", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),
                  std::vector<std::string>({"HcclAllGatherV", "sendBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),
                  std::vector<std::string>({"HcclAllGatherV", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);
 
    return HCCL_SUCCESS;
}
 
HcclResult AllGatherVOutPlace(void *sendBuf, void *recvBuf, uint64_t sendCount,const void *recvCounts,const void *recvDispls,
    HcclDataType dataType, HcclComm comm, aclrtStream stream, const std::string &tag)
{
    HCCL_INFO("Start to execute AllGatherVOutPlace");
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));
    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 inputSize = sendCount * perDataSize;    // all gather v 每个rank上一份数据
    u64 outputSize = 0;  
    const u64 *u64RecvCount = reinterpret_cast<const u64 *>(recvCounts);
    for (u64 i = 0; i < userRankSize; i++) {
        outputSize += u64RecvCount[i] * perDataSize;
    }  // 结果为recvCount中的数据之和

    // 申请OpParam参数结构体内存
    u64 varMemSize = userRankSize * 2 * sizeof(u64);
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
    param.DataDes.count = sendCount;
    // param.DataDes.dataType = dataType;
    param.vDataDes.dataType = dataType;


    // 带V算子的参数
    param.varMemSize = varMemSize;
    // 从源内存地址按字节直接拷贝数据到目标地址
    std::vector<u64> merged(userRankSize * 2);
    const uint64_t *countsPtr = reinterpret_cast<const uint64_t *>(recvCounts);
    const uint64_t *displsPtr = reinterpret_cast<const uint64_t *>(recvDispls);
    std::copy(countsPtr, countsPtr + userRankSize, merged.begin());
    std::copy(displsPtr, displsPtr + userRankSize, merged.begin() + userRankSize);
    memcpy(param.varData, merged.data(), varMemSize);
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER_V;
    param.enableDetour = false;
    param.deviceType = deviceType;
    CHK_RET(HcclExecOp(comm, param));
    paramPtr->~OpParam();
    free(paramMem);
    HCCL_INFO("Execute AllGatherVOutPlace success.");
    return HCCL_SUCCESS;
}
 
}  // namespace ops_hccl