/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_op.h"
#include "op_common_ops.h"
#include "topo_host.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, HcclComm comm, aclrtStream stream)
{
    if (!HcclCheckCcuEnableOpen() && !HcclCheckAicpuEnableOpen() && !HcclCheckAivEnableOpen()) {
        return HcclReduceInner(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
    }
    HCCL_INFO("Start to run execute HcclReduce");
    if (GetHcommVersion() < 90000000) { // compat handle
        return HcclReduceInner(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
    }

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    // 非95设备转到老流程
    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95) {
    #endif
        return HcclReduceInner(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
    }
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return reduce success"), HCCL_SUCCESS);

    std::string opTag;
    CHK_RET(ReduceInitAndCheck(comm, sendBuf, recvBuf, count, dataType, op, stream, opTag));

    CHK_RET(ReduceEntryLog(sendBuf, recvBuf, count, dataType, op, root, stream, opTag, "HcclReduce"));

    // 执行Reduce
    CHK_RET_AND_PRINT_IDE(ReduceOutPlace(sendBuf, recvBuf, count, dataType, op, root, comm, stream, opTag),
        opTag.c_str());

    CHK_RET(LogHcclExit("HcclReduce", opTag.c_str(), startut));

    return HCCL_SUCCESS;
}

HcclResult HcclReduceGraphMode(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount,
    void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_INFO("Start to run execute HcclReduceGraphMode");
    // 根据group获取通信域
    HcclComm comm = nullptr;
    HCCL_INFO("[HcclReduceGraphMode] get group name: %s", group);
    HcomGetCommHandleByGroup(group, &comm);

    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return reduce success"), HCCL_SUCCESS);

    std::string opTag;
    CHK_RET(ReduceInitAndCheck(comm, sendBuf, recvBuf, count, dataType, op, stream, opTag));

    CHK_RET(HcclCheckTag(tag));

    // 拼装ResPackGraphMode
    ResPackGraphMode resPack;
    // 设置tag
    strncpy_s(resPack.tag, sizeof(resPack.tag), tag, sizeof(resPack.tag) - 1);
    // 设置streams
    if (streams != nullptr && streamCount > 0) {
        for (size_t i = 0; i < streamCount; i++) {
            resPack.streams.push_back(static_cast<aclrtStream>(streams[i]));
        }
    }
    // 设置scratchMem
    resPack.scratchMemAddr = scratchMemAddr;
    resPack.scratchMemSize = scratchMemSize;

    CHK_RET(ReduceEntryLog(sendBuf, recvBuf, count, dataType, op, root, stream, opTag, "HcclReduceGraphMode"));

    // 执行
    CHK_RET_AND_PRINT_IDE(ReduceOutPlaceGraphMode(sendBuf, recvBuf, count, dataType, op, root, comm, stream,
        std::string(tag), resPack), tag);

    CHK_RET(LogHcclExit("HcclReduceGraphMode", opTag.c_str(), startut));

    return HCCL_SUCCESS;
}

namespace ops_hccl {
// 除了错误都是公共的
HcclResult CheckReduceInputPara(const HcclComm comm, const void* sendBuf, const void* recvBuf, const aclrtStream stream)
{
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr,
        "EI0003",
        std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclReduce", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr,
        "EI0003",
        std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclReduce", "nullptr", "sendBuf", "non-null pointer"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr,
        "EI0003",
        std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclReduce", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclReduce", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    return HCCL_SUCCESS;
}

HcclResult ReduceInitAndCheck(HcclComm comm, void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
    HcclReduceOp op, aclrtStream stream, std::string &opTag)
{
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    CHK_RET(InitEnvConfig());
    // 检查入参指针有效性
    CHK_RET(CheckReduceInputPara(comm, sendBuf, recvBuf, stream));
    // tag有效性，是否过长
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    opTag = "Reduce_" + string(commName);
    CHK_RET(HcclCheckTag(opTag.c_str()));
    // 检查count是否合法（超出系统上限）
    CHK_RET(CheckCount(count));
    // 检查数据类型是否支持
    CHK_RET(CheckDataType(dataType, true));
    CHK_RET(CheckReduceOp(dataType, op));
    // 检查rank有效性，是否超出rankSize
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), opTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceOutPlace(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, HcclComm comm, aclrtStream stream, const std::string &tag)
{
    HCCL_INFO("Start to execute ReduceOutPlace");
    CHK_RET(ReduceOutPlaceCommon(sendBuf, recvBuf, count, dataType, op, root, comm, stream, tag, OpMode::OPBASE,
        ResPackGraphMode()));
    HCCL_INFO("Execute ReduceOutPlace success.");
    return HCCL_SUCCESS;
}

HcclResult ReduceOutPlaceGraphMode(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, HcclComm comm, aclrtStream stream, const std::string &tag, const ResPackGraphMode &resPack)
{
    HCCL_INFO("Start to execute ReduceOutPlaceGraphMode");
    CHK_RET(ReduceOutPlaceCommon(sendBuf, recvBuf, count, dataType, op, root, comm, stream, tag, OpMode::OFFLOAD,
        resPack));
    HCCL_INFO("Execute ReduceOutPlaceGraphMode success.");
    return HCCL_SUCCESS;
}

HcclResult ReduceConstructOpParam(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, HcclComm comm, aclrtStream stream, const std::string &tag, OpParam &param, OpMode opMode)
{
    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;

    CHK_RET(HcclGetCommName(comm, param.commName));
    param.stream = stream;
    param.reduceType = op;
    param.opMode = opMode;

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
    param.inputSize = totalSize;
    param.outputPtr = recvBuf;
    param.outputSize = totalSize;
    param.DataDes.count = count;
    param.DataDes.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE;
    param.enableDetour = false;
    param.deviceType = deviceType;
    param.root = root;
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    param.userRank = userRank;

    return HCCL_SUCCESS;
}

HcclResult ReduceOutPlaceCommon(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, HcclComm comm, aclrtStream stream, const std::string &tag, OpMode opMode, const ResPackGraphMode &resPack)
{
    HCCL_INFO("Start to execute ReduceOutPlaceCommon");
    u32 userRankSize = 0;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    OpParam param;
    CHK_RET(ReduceConstructOpParam(sendBuf, recvBuf, count, dataType, op, root, comm, stream, tag, param, opMode));
    
    CHK_RET(HcclGetOpExpansionMode(comm, param));

    CcuFastLaunchCtx *ccuFastLaunchCtx = nullptr;
    if ((opMode == OpMode::OPBASE) && ShouldGoCcuFastLaunch(comm, param, &ccuFastLaunchCtx)) {
        return HcclExecOpCcuFastLaunch(comm, param, ccuFastLaunchCtx);
    }

    if (userRankSize == 1) {
        HCCL_WARNING("[%s] ranksize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
        return HcclResult::HCCL_SUCCESS;
    }

    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(Selector(comm, param, topoInfo, algName));
    if (ShouldUseInnerOp(param.opExecuteConfig) && param.opMode == OpMode::OPBASE) {
        return HcclReduceInner(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
    }
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));
    HCCL_INFO("Execute ReduceOutPlaceCommon success.");
    return HCCL_SUCCESS;
}

HcclResult ReduceEntryLog(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, aclrtStream stream, const std::string &tag, const std::string &opName)
{
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], reduceOp[%s], root[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(), root, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl