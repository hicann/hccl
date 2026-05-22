/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_op.h"
#include "op_common_ops.h"
#include "topo_host.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                         HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclAllReduce");
    if (GetHcommVersion() < 90000000) { // compat handle
        return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);
    }

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95) {
    #endif
        return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);
    }
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return all reduce success"), HCCL_SUCCESS);
    
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    OpParam param;
    CHK_RET(AllReduceInitAndCheck(comm, sendBuf, recvBuf, count, dataType, op, stream, param));

    /* 接口交互信息日志 */
    CHK_RET(AllReduceEntryLog(sendBuf, recvBuf, count, dataType, op, stream, param.tag, "HcclAllReduce"));

    // 执行AllReduce
    CHK_RET_AND_PRINT_IDE(AllReduceOutPlace(sendBuf, recvBuf, count, dataType, op, comm, stream, param),
                          param.tag);

    CHK_RET(LogHcclExit("HcclAllReduce", param.tag, startut));

    return HCCL_SUCCESS;
}

HcclResult HcclAllReduceGraphMode(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclReduceOp op, const char* group, 
                                  aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_INFO("Start to run execute HcclAllReduceGraphMode");
    // 根据group获取通信域
    HcclComm comm = nullptr;
    HCCL_INFO("[HcclAllReduceGraphMode] get group name: %s", group);
    CHK_RET(HcomGetCommHandleByGroup(group, &comm));
    CHK_PRT_RET(sendCount == 0, HCCL_WARNING("input sendCount is 0, return all reduce success"), HCCL_SUCCESS);

    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    OpParam param;
    CHK_RET(AllReduceInitAndCheck(comm, sendBuf, recvBuf, sendCount, dataType, op, stream, param));

    // 检查tag有效性
    CHK_RET(HcclCheckTag(tag));
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag);
    CHK_PRT_RET((ret <= 0), HCCL_ERROR("failed to fill param.tag"), HCCL_E_INTERNAL);
    
    // 拼装ResPackGraphMode
    ResPackGraphMode resPack;
    // 设置tag
    if (strncpy_s(resPack.tag, sizeof(resPack.tag), tag, sizeof(resPack.tag) - 1) != 0) {
        HCCL_ERROR("failed to fill resPack.tag");
        return HCCL_E_INTERNAL;
    }
    // 设置streams
    if (streams != nullptr && streamCount > 0) {
        for (size_t i = 0; i < streamCount; i++) {
            resPack.streams.push_back(static_cast<aclrtStream>(streams[i]));
        }
    }
    // 设置scratchMem
    resPack.scratchMemAddr = scratchMemAddr;
    resPack.scratchMemSize = scratchMemSize;
    std::string tagStr = tag;

    /* 接口交互信息日志 */
    CHK_RET(AllReduceEntryLog(sendBuf, recvBuf, sendCount, dataType, op, stream, param.tag, "HcclAllReduceGraphMode"));
    // 执行AllReduce
    CHK_RET_AND_PRINT_IDE(AllReduceOutPlaceGraphMode(sendBuf, recvBuf, sendCount, dataType, op, comm, stream, resPack, param), tagStr.c_str());
    CHK_RET(LogHcclExit("HcclAllReduceGraphMode", param.tag, startut));

    return HCCL_SUCCESS;
}

namespace ops_hccl {
HcclResult AllReduceInitAndCheck(HcclComm comm, void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, aclrtStream stream, OpParam &param)
{
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    CHK_RET(InitEnvConfig());
    
    // 参数校验等工作
    CHK_RET(CheckAllReduceInputPara(comm, sendBuf, recvBuf, stream));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    CHK_RET(HcclGetCommName(comm, param.commName));

    // topoInfo的tag，所有相同的算子可以共享
    int ret = sprintf_s(param.tag, sizeof(param.tag), "AllReduce_%s", param.commName);
    CHK_PRT_RET((ret <= 0), "failed to fill param.tag", HCCL_E_INTERNAL);

    CHK_RET(HcclCheckTag(param.tag));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), param.tag);
    CHK_RET(CheckCount(count));
    CHK_RET(CheckDataType(dataType, true));
    CHK_RET(CheckReduceOp(dataType, op));

    return HCCL_SUCCESS;
}

HcclResult CheckAllReduceInputPara(const HcclComm comm, const void* sendBuf, const void* recvBuf, const aclrtStream stream)
{
    // 入参合法性校验
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAllReduce", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAllReduce", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAllReduce", "nullptr", "sendBuf", "non-null pointer"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAllReduce", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);

    return HCCL_SUCCESS;
}

HcclResult FillAllReduceOpParam(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                                HcclReduceOp op, HcclComm comm, aclrtStream stream, OpMode opMode, OpParam &param)
{
    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 outputSize = count * perDataSize;
    u64 inputSize = outputSize;

    param.stream = stream;
    param.reduceType = op;
    param.opMode = opMode;

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    // 参数准备
    param.inputPtr = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    param.DataDes.count = count;
    param.DataDes.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    param.enableDetour = false;
    param.deviceType = deviceType;
    param.reduceType = op;
    return HCCL_SUCCESS;
}

HcclResult AllReduceOutPlaceCommon(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                                   HcclReduceOp op, HcclComm comm, aclrtStream stream, OpMode opMode, const ResPackGraphMode &resPack, OpParam &param)
{
    HCCL_INFO("Start to execute AllReduceOutPlace");

    CHK_RET(FillAllReduceOpParam(sendBuf, recvBuf, count, dataType, op, comm, stream, opMode, param));
    
    CHK_RET(HcclGetOpExpansionMode(comm, param));
    
    CcuFastLaunchCtx *ccuFastLaunchCtx = nullptr;
    if ((opMode == OpMode::OPBASE) && ShouldGoCcuFastLaunch(comm, param, &ccuFastLaunchCtx)) {
        return HcclExecOpCcuFastLaunch(comm, param, ccuFastLaunchCtx);
    }

    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(Selector(comm, param, topoInfo, algName));
    if (ShouldUseInnerOp(param.opExecuteConfig) && param.opMode == OpMode::OPBASE) {
        return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);
    }
    // 单卡校验
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));
    if (userRankSize == 1) {
        HCCL_WARNING("[%s] ranksize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));
    HCCL_INFO("Execute AllReduceOutPlace success.");
    return HCCL_SUCCESS;
}

HcclResult AllReduceEntryLog(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    aclrtStream stream, const char *tag, const std::string &opName)
{
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], reduceOp[%s], streamId[%d], deviceLogicId[%d]",
            tag, sendBuf, recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(), streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceOutPlaceGraphMode(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
                                      aclrtStream stream, const ResPackGraphMode &resPack, OpParam &param)
{
    HCCL_INFO("Start to execute AllReduceOutPlaceGraphMode");
    CHK_RET(AllReduceOutPlaceCommon(sendBuf, recvBuf, count, dataType, op, comm, stream, OpMode::OFFLOAD, resPack, param));
    HCCL_INFO("Execute AllReduceOutPlaceGraphMode success.");
    return HCCL_SUCCESS;
}


HcclResult AllReduceOutPlace(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
                                      aclrtStream stream, OpParam &param)
{
    HCCL_INFO("Start to execute AllReduceOutPlace");
    CHK_RET(AllReduceOutPlaceCommon(sendBuf, recvBuf, count, dataType, op, comm, stream, OpMode::OPBASE, ResPackGraphMode(), param));
    HCCL_INFO("Execute AllReduceOutPlace success.");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
