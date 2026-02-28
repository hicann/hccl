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
#include "workflow.h"
#include "send_op.h"
#include "op_common.h"

using namespace std;
using namespace ops_hccl;

extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclSend(
    void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("[HcclSend] Start.");
    if (!CheckHCCLIndependentOp()) {
        return HcclSendInner(sendBuf, count, dataType, destRank, comm, stream);
    }
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_910_95) {
        return HcclSendInner(sendBuf, count, dataType, destRank, comm, stream);
    }
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return HcclSendInner(sendBuf, count, dataType, destRank, comm, stream);
    }

    CHK_RET(InitEnvConfig());

    // 参数校验
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return send success"), HcclResult::HCCL_SUCCESS);
    CHK_RET(CheckSendInputPara(comm, sendBuf));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string tag = "Send_" + string(commName) + "_" + std::to_string(userRank) + "_" + std::to_string(destRank);
    CHK_RET(HcclCheckTag(tag.c_str()));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, destRank), tag.c_str());
    CHK_PRT_RET(userRank == destRank, HCCL_ERROR("[HcclSend] destRank cannot be equal to self."), HcclResult::HCCL_E_NOT_SUPPORT);
    CHK_RET(CheckCount(count));
    CHK_RET(CheckDataType(dataType, false));

    CHK_RET_AND_PRINT_IDE(SendExec(sendBuf, count, dataType, destRank, comm, stream, tag), tag.c_str());

    HCCL_INFO("[HcclSend][%d]->[%d] Success.", userRank, destRank);
    return HcclResult::HCCL_SUCCESS;
}

namespace ops_hccl {
    HcclResult CheckSendInputPara(HcclComm comm, const void *sendBuf) {
        // 入参合法性校验
        RPT_INPUT_ERR(
            comm == nullptr,
            "EI0003",
            std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),
            std::vector<std::string>({"HcclSend", "comm", "nullptr", "please check comm"}));
        CHK_PTR_NULL(comm);
        RPT_INPUT_ERR(
            sendBuf == nullptr,
            "EI0003",
            std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),
            std::vector<std::string>({"HcclSend", "sendBuf", "nullptr", "please check sendBuf"}));
        CHK_PTR_NULL(sendBuf);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult SendExec(
        void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm,
        aclrtStream stream, const std::string &tag)
    {
        HCCL_DEBUG("[SendExec][%s] Start.", tag);
        u32 userRankSize;
        CHK_RET(HcclGetRankSize(comm, &userRankSize));

        u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
        u64 dataSize = count * dataTypeSize;

        // 参数构建
        OpParam param;
        param.opType = HcclCMDType::HCCL_CMD_SEND;
        param.enableDetour = false;

        DevType deviceType = DevType::DEV_TYPE_COUNT;
        CHK_RET(hrtGetDeviceType(deviceType));
        param.deviceType = deviceType;

        // 获取通信域名称
        CHK_RET(HcclGetCommName(comm, param.commName));

        // topoInfo的tag，所有相同的算子可以共享
        int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
        if (ret <= 0) {
            HCCL_ERROR("failed to fill param.tag");
            return HcclResult::HCCL_E_INTERNAL;
        }

        param.stream = stream;
        param.opMode = OpMode::OPBASE;
        param.inputPtr = sendBuf;
        param.inputSize = dataSize;
        param.sendRecvRemoteRank = destRank;
        param.outputPtr = nullptr;
        param.outputSize = dataSize;
        param.DataDes.count = count;
        param.DataDes.dataType = dataType;
        if (userRankSize == 1) {
            HCCL_WARNING("[%s] ranksize == 1, enter SingleRankProc", __func__);
            CHK_RET(SingleRankProc(param));
            return HcclResult::HCCL_SUCCESS;
        }

        CHK_RET(HcclExecOp(comm, param));

        return HcclResult::HCCL_SUCCESS;
    }

} // namespace ops_hccl
