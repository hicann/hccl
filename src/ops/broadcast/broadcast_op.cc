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
    if (GetHcommVersion() < 90000000) { // compat handle
        return HcclBroadcastInner(buf, count, dataType, root, comm, stream);
    }

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95) {
    #endif
        return HcclBroadcastInner(buf, count, dataType, root, comm, stream);
    }
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return broadcast success"), HCCL_SUCCESS);
    OpParam param;
    CHK_RET(BroadcastInitAndCheck(comm, buf, count, dataType, root, stream, param));
    CHK_RET(BroadcastEntryLog(buf, count, dataType, root, stream, param.tag, "HcclBroadcast"));

    // 执行Broadcast
    CHK_RET_AND_PRINT_IDE(BroadcastOutPlace(param, buf, count, dataType, root, comm, stream),
                          param.tag);

    CHK_RET(LogHcclExit("HcclBroadcast", param.tag, startut));

    return HCCL_SUCCESS;
}
HcclResult HcclBroadcastGraphMode(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_INFO("Start to run execute HcclBroadcastGraphMode");
    // 根据group获取通信域
    OpParam param;
    HcclComm comm = nullptr;
    HCCL_INFO("[HcclBroadcastGraphMode] get group name: %s", group);
    HcomGetCommHandleByGroup(group, &comm);
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return broadcast success"), HCCL_SUCCESS);
    CHK_RET(BroadcastInitAndCheck(comm, buf, count, dataType, root, stream, param));
    
    // 检查tag有效性
    CHK_RET(HcclCheckTag(tag));
    
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

    CHK_RET(BroadcastEntryLog(buf, count, dataType, root, stream, param.tag, "HcclBroadcastGraphMode"));

    // 执行Broadcast
    CHK_RET_AND_PRINT_IDE(BroadcastOutPlaceGraphMode(buf, count, dataType, root, comm, stream, tagStr, resPack), tagStr.c_str());

    CHK_RET(LogHcclExit("HcclBroadcastGraphMode", param.tag, startut));

    return HCCL_SUCCESS;
}

namespace ops_hccl {
HcclResult BroadcastInitAndCheck(HcclComm comm, void *buf, uint64_t count, HcclDataType dataType, uint32_t root, aclrtStream stream, OpParam &param)
{
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    CHK_RET(InitEnvConfig());

    // 参数校验等工作
    CHK_RET(CheckBroadcastInputPara(comm, buf));
    CHK_RET(HcclGetCommName(comm, param.commName));
    int ret = sprintf_s(param.tag, sizeof(param.tag), "Broadcast_%s", param.commName);
    CHK_PRT_RET((ret <= 0), "failed to fill param.tag", HCCL_E_INTERNAL);
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    CHK_RET(HcclCheckTag(param.tag));
    CHK_RET(HcomCheckUserRank(rankSize, userRank));
    CHK_RET(CheckCount(count));
    CHK_RET(CheckDataType(dataType, false));

    return HCCL_SUCCESS;
}

HcclResult CheckBroadcastInputPara(const HcclComm comm, const void *buf)
{
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclBroadcast", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(buf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclBroadcast", "nullptr", "buf", "non-null pointer"}));
    CHK_PTR_NULL(buf);

    return HCCL_SUCCESS;
}

HcclResult BroadcastOutPlaceCommon(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream, const std::string &tag,
                                   OpMode opMode, const ResPackGraphMode &resPack)
{
    HCCL_INFO("Start to execute BroadcastOutPlaceCommon");
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 inputSize = count * perDataSize;
    u64 outputSize = inputSize;

    OpParam param;
    CHK_RET(HcclGetCommName(comm, param.commName));
    param.stream = stream;
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

    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(HcclGetOpExpansionMode(comm, param));
    CHK_RET(Selector(comm, param, topoInfo, algName));
    if (ShouldUseInnerOp(param.opExecuteConfig) && param.opMode == OpMode::OPBASE) {
        return HcclBroadcastInner(buf, count, dataType, root, comm, stream);
    }
    if (userRankSize == 1) {
        HCCL_WARNING("[%s] ranksize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));
    HCCL_INFO("Execute BroadcastOutPlaceCommon success.");
    return HCCL_SUCCESS;
}

HcclResult BroadcastOutPlaceGraphMode(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
                                      aclrtStream stream, const std::string &tag, const ResPackGraphMode &resPack)
{
    HCCL_INFO("Start to execute BroadcastOutPlaceGraphMode");
    CHK_RET(BroadcastOutPlaceCommon(buf, count, dataType, root, comm, stream, tag, OpMode::OFFLOAD, resPack));
    HCCL_INFO("Execute BroadcastOutPlaceGraphMode success.");
    return HCCL_SUCCESS;
}


HcclResult BroadcastOutPlace(OpParam &param, void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
                                      aclrtStream stream)
{
    HCCL_INFO("Start to execute BroadcastOutPlace");
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 inputSize = count * perDataSize;
    u64 outputSize = inputSize;

    param.stream = stream;
    param.opMode = OpMode::OPBASE;

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

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
    
    CHK_RET(HcclGetOpExpansionMode(comm, param));

    CcuFastLaunchCtx *ccuFastLaunchCtx = nullptr;
    if (ShouldGoCcuFastLaunch(comm, param, &ccuFastLaunchCtx)) {
        return HcclExecOpCcuFastLaunch(comm, param, ccuFastLaunchCtx);
    }

    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(Selector(comm, param, topoInfo, algName));
    if (ShouldUseInnerOp(param.opExecuteConfig) && param.opMode == OpMode::OPBASE) {
        return HcclBroadcastInner(buf, count, dataType, root, comm, stream);
    }
    if (userRankSize == 1) {
        HCCL_WARNING("[%s] ranksize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName));

    HCCL_INFO("Execute BroadcastOutPlace success.");
    return HCCL_SUCCESS;
}

HcclResult BroadcastEntryLog(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, aclrtStream stream, const char *tag, const std::string &opName)
{
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], buf[%p], count[%llu], dataType[%s], root[%u], streamId[%d], deviceLogicId[%d]",
            tag, buf, count, GetDataTypeEnumStr(dataType).c_str(), root, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}

}
