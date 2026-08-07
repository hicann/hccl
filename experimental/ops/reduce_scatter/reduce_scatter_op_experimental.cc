/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_op_experimental.h"
#include "topo_host.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>
#include "op_common_experimental.h"
#include "param_check.h"
#include "reduce_scatter_op.h"
#include "load_kernel.h"
#include "hcomm_dlsym.h"
#include "hcomm_host_profiling_dl.h"

using namespace std;

extern "C" HcclResult HcclReduceScatterInner(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream);

namespace ops_hccl_experimental {
using ops_hccl::CheckCount;
using ops_hccl::CheckDataType;
using ops_hccl::CheckReduceOp;
using ops_hccl::CheckReduceScatterInputPara;
using ops_hccl::DATATYPE_SIZE_TABLE;
using ops_hccl::HcclCheckTag;
using ops_hccl::HcomCheckUserRank;
using ops_hccl::InitEnvConfig;
using ops_hccl::LoadAICPUKernel;
using ops_hccl::LogHcclExit;
using ops_hccl::OpMode;
using ops_hccl::ReduceScatterEntryLog;

HcclResult ReduceScatterExperimental(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    if (!MatchBIRS()) {
        return HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    }

    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内
    // 入口的地方先解析环境变量
    CHK_RET(InitEnvConfig());

    OpParam param;
    // 参数校验等工作
    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("input recvCount is 0, return reduce scatter success"), HCCL_SUCCESS);
    CHK_RET(CheckReduceScatterInputPara(comm, sendBuf, recvBuf, stream));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    CHK_RET(HcomCheckUserRank(rankSize, userRank));
    CHK_RET(CheckCount(recvCount));
    CHK_RET(CheckDataType(dataType, true));
    CHK_RET(HcclGetCommName(comm, param.commName));
    // topoInfo的tag，所有相同的算子可以共享
    int ret = sprintf_s(param.tag, sizeof(param.tag), "ReduceScatter_%s", param.commName);
    CHK_PRT_RET((ret <= 0), HCCL_ERROR("failed to fill param.tag"), HCCL_E_INTERNAL);
    CHK_RET(HcclCheckTag(param.tag));
    CHK_RET(CheckReduceOp(dataType, op));

    /* 接口交互信息日志 */
    CHK_RET(ReduceScatterEntryLog(sendBuf, recvBuf, recvCount, dataType, op, stream, param.tag, "HcclReduceScatter"));

    CHK_RET(ReduceScatterOutPlaceCustom(param, sendBuf, recvBuf, recvCount, dataType, op, comm, stream, rankSize));

    CHK_RET(LogHcclExit("HcclReduceScatter", param.tag, startut));

    return HCCL_SUCCESS;
}

static HcclResult PrepareReduceScatterParam(
    OpParam& param, void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream, u32 userRankSize, OpMode opMode)
{
    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 outputSize = recvCount * perDataSize;
    u64 inputSize = outputSize * userRankSize;

    param.stream = stream;
    param.reduceType = op;
    param.opMode = opMode;

    if (param.commName[0] == '\0') {
        CHK_RET(HcclGetCommName(comm, param.commName));
    }
    HcclDevType deviceType = HcclDevType::DEV_TYPE_COUNT;
    CHK_RET(HcclGetDeviceType(deviceType));

    param.inputPtr = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    param.DataDes.count = recvCount;
    param.DataDes.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    param.enableDetour = false;
    param.deviceType = deviceType;

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOutPlaceCustom(
    OpParam& param, void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream, u32 userRankSize)
{
    HCCL_INFO("Start to execute ReduceScatterOutPlaceCustom");
    CHK_RET(PrepareReduceScatterParam(
        param, sendBuf, recvBuf, recvCount, dataType, op, comm, stream, userRankSize, OpMode::OPBASE));

    if (IsAiCpuMode(param.deviceType, userRankSize)) {
        HCCL_DEBUG("is aicpu mode");
        CHK_RET(LoadAICPUKernel());
        param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    } else {
        HCCL_DEBUG("is host mode");
        param.engine = CommEngine::COMM_ENGINE_CPU_TS;
    }

    uint64_t beginTime;
    if (HcommIsProfilingSupported()) {
        beginTime = HcommGetProfilingSysCycleTime();
    }

    CHK_RET(ProcessA3(comm, param, beginTime));

    HCCL_INFO("Execute ReduceScatterOutPlaceCustom success.");
    return HCCL_SUCCESS;
}

bool MatchBIRS()
{
    const char* val = std::getenv("HCCL_BIRS_ENABLE");
    if (val == nullptr)
        return false;
    std::string str = val;
    if (str == "TRUE") {
        return true;
    }
    return false;
}

} // namespace ops_hccl_experimental
