/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_op.h"
#include "op_common_ops.h"
#include "topo_host.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclBroadcast");
    if (!CheckHCCLIndependentOp()) {
        return HcclBroadcastInner(buf, count, dataType, root, comm, stream);
    }
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    // 非95设备转到老流程
    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95) {
    #endif
        return HcclBroadcastInner(buf, count, dataType, root, comm, stream);
    }
    // 图模式引导到老的流程上面
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return HcclBroadcastInner(buf, count, dataType, root, comm, stream);
    }
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    // A3是：export HCCL_OP_EXPANSION_MODE="AI_CPU"，A5的接口还没提供
    CHK_RET(InitEnvConfig());

    // 参数校验等工作
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return broadcast success"), HCCL_SUCCESS);
    CHK_RET(CheckBroadcastInputPara(comm, buf));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string tag = "Broadcast_" + string(commName);
    CHK_RET(HcclCheckTag(tag.c_str()));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());
    CHK_RET(CheckCount(count));
    CHK_RET(CheckDataType(dataType, false));

    // 执行Broadcast
    CHK_RET_AND_PRINT_IDE(BroadcastOutPlace(buf, count, dataType, root, comm, stream, tag),
                          tag.c_str());

    return HCCL_SUCCESS;
}

namespace ops_hccl {
HcclResult CheckBroadcastInputPara(const HcclComm comm, const void *buf)
{
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclBroadcast", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(buf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclBroadcast", "buf", "nullptr", "please check buf"}));
    CHK_PTR_NULL(buf);

    return HCCL_SUCCESS;
}

HcclResult BroadcastOutPlace(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream, const std::string &tag)
{
    HCCL_INFO("Start to execute BroadcastOutPlace");
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 inputSize = count * perDataSize;
    u64 outputSize = inputSize;

    OpParam param;
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
    param.inputPtr = buf;
    param.inputSize = inputSize;
    param.outputPtr = buf;
    param.outputSize = outputSize;
    param.DataDes.count = count;
    param.DataDes.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;
    param.enableDetour = false;
    param.deviceType = deviceType;
    if (userRankSize == 1) {
        HCCL_WARNING("[%s] ranksize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(param));
        return HcclResult::HCCL_SUCCESS;
    }

    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(Selector(comm, param, topoInfo, algName));
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName));
    HCCL_INFO("Execute BroadcastOutPlace success.");
    return HCCL_SUCCESS;
}
}
