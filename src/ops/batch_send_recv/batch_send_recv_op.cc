/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batch_send_recv_op.h"
#include "op_common_ops.h"
#include "topo_host.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclBatchSendRecv(HcclSendRecvItem *sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclBatchSendRecv.");
    if (!CheckHCCLIndependentOp()) {
        return HcclBatchSendRecvInner(sendRecvInfo, itemNum, comm, stream);
    }
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95) {
    #endif
        return HcclBatchSendRecvInner(sendRecvInfo, itemNum, comm, stream);
    }
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return HcclBatchSendRecvInner(sendRecvInfo, itemNum, comm, stream);
    }
    CHK_RET(InitEnvConfig());

    // 参数校验等工作
    CHK_PRT_RET(itemNum == 0, HCCL_WARNING("taskList itemNum is zero."), HCCL_SUCCESS);
    CHK_RET(CheckBatchSendRecvInputPara(comm, sendRecvInfo, stream));
    for (u32 i = 0; i < itemNum; i++) {
        CHK_RET(CheckCount(sendRecvInfo[i].count));
        CHK_RET(CheckDataType(sendRecvInfo[i].dataType, false));
    }
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string tag = "BatchSendRecv_" + string(commName);
    CHK_RET(HcclCheckTag(tag.c_str()));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());

    // 执行BatchSendRecv
    CHK_RET_AND_PRINT_IDE(BatchSendRecvOutPlace(sendRecvInfo, itemNum, comm, stream, tag),
                          tag.c_str());

    return HCCL_SUCCESS;
}

namespace ops_hccl {
HcclResult CheckBatchSendRecvInputPara(const HcclComm &comm, const HcclSendRecvItem *sendRecvInfo, const aclrtStream stream)
{
    // 入参合法性校验
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclBatchSendRecv", "stream", "nullptr", "please check stream"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclBatchSendRecv", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendRecvInfo == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclBatchSendRecv", "sendRecvInfo", "nullptr", "please check sendRecvInfo"}));
    CHK_PTR_NULL(sendRecvInfo);

    return HCCL_SUCCESS;
}

HcclResult BatchSendRecvOutPlace(HcclSendRecvItem *sendRecvInfo, uint32_t itemNum,
    HcclComm comm, aclrtStream stream, const std::string &tag)
{
    HCCL_INFO("Start to execute BatchSendRecvOutPlace.");
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    // 申请OpParam参数结构体内存
    u64 varMemSize = itemNum * sizeof(HcclSendRecvItem);
    void* paramMem = malloc(sizeof(OpParam) + varMemSize);
    if (!paramMem) {
        HCCL_ERROR("[BatchSendRecvOutPlace] malloc OpParam failed!");
        return HCCL_E_INTERNAL;
    }
    OpParam* batchSendRecvParamPtr = new (paramMem) OpParam();
    auto deleter = [](OpParam* tmp) {
        if (tmp) {
            tmp->~OpParam();
            free(tmp);
        }
    };
    std::unique_ptr<OpParam, decltype(deleter)> paramPtr(batchSendRecvParamPtr, deleter);
    OpParam& param = *paramPtr;
    CHK_RET(HcclGetCommName(comm, param.commName));
    param.stream = stream;
    param.opMode = OpMode::OPBASE;

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
    if (ret <= 0) {
        HCCL_ERROR("failed to fill param.tag!");
        return HCCL_E_INTERNAL;
    }

    // 参数准备
    param.varMemSize = varMemSize;
    memcpy_s(param.varData, varMemSize, sendRecvInfo, varMemSize);
    param.batchSendRecvDataDes.itemNum = itemNum;
    param.batchSendRecvDataDes.sendRecvItemsPtr = 
        reinterpret_cast<HcclSendRecvItem*>(param.varData);
    param.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    param.deviceType = deviceType;

    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(Selector(comm, param, topoInfo, algName));
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName));
    HCCL_INFO("Execute BatchSendRecvOutPlace success.");
    return HCCL_SUCCESS;
}

}