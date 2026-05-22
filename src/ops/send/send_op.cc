/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "send_op.h"
#include "op_common_ops.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>

using namespace std;
using namespace ops_hccl;

extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclSendNext(
    void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("[HcclSend] Start.");
    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内
    CHK_RET(InitEnvConfig());
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    u32 userRank = INVALID_VALUE_RANKID;
    std::string tag;
    // 参数获取和校验
    CHK_PRT_RET(count == 0, HCCL_WARNING("[HcclSend] input count is 0, return send success"), HcclResult::HCCL_SUCCESS);
    CHK_RET(GetAndCheckSendPara(comm, sendBuf, count, dataType, destRank, rankSize, userRank, tag));

    /* 接口交互信息日志 */
    CHK_RET(SendEntryLog(sendBuf, count, dataType, destRank, stream, tag, "HcclSend"));
        
    CHK_RET_AND_PRINT_IDE(SendExec(sendBuf, count, dataType, destRank, comm, stream, rankSize, OpMode::OPBASE, tag),
        tag.c_str());

    CHK_RET(LogHcclExit("HcclSend", tag.c_str(), startut));

    HCCL_INFO("[HcclSend][%d]->[%d] Success.", userRank, destRank);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclSend(
    void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("[HcclSend] Start.");

    if (IsHostDpu(comm)) {
        return HcclSendNext(sendBuf, count, dataType, destRank, comm, stream);
    }

    if (GetHcommVersion() < 90000000) {
        return HcclSendInner(sendBuf, count, dataType, destRank, comm, stream);
    }

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95) {
    #endif
        return HcclSendInner(sendBuf, count, dataType, destRank, comm, stream);
    }
    
    return HcclSendNext(sendBuf, count, dataType, destRank, comm, stream);
}

HcclResult HcclSendGraphMode(
    void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, const char* group, aclrtStream stream,
    const char *tag, void **streams, size_t streamCount, void *scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_INFO("[HcclSendGraphMode] Start.");
    // 根据group获取通信域
    HcclComm comm = nullptr;
    HCCL_INFO("[HcclSendGraphMode] get group name: %s", group);
    HcomGetCommHandleByGroup(group, &comm);
    
    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内
    CHK_RET(InitEnvConfig());
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    u32 userRank = INVALID_VALUE_RANKID;
    std::string opTag(tag ? tag : "");
    // 参数获取和校验
    CHK_PRT_RET(count == 0, HCCL_WARNING("[HcclSendGraphMode] input count is 0, return send success"), HcclResult::HCCL_SUCCESS);
    CHK_RET(GetAndCheckSendPara(comm, sendBuf, count, dataType, destRank, rankSize, userRank, opTag));

    // 拼装ResPackGraphMode
    ResPackGraphMode resPack;
    // 设置tag
    s32 fillTagRet = strncpy_s(resPack.tag, sizeof(resPack.tag), tag, sizeof(resPack.tag) - 1);
    CHK_PRT_RET(
        fillTagRet != EOK,
        HCCL_ERROR("[HcclSendGraphMode] failed to fill resPack.tag, tag %s, return %d.", tag, fillTagRet),
        HcclResult::HCCL_E_INTERNAL);
    // 设置streams
    if (streams != nullptr && streamCount > 0) {
        for (size_t i = 0; i < streamCount; i++) {
            resPack.streams.push_back(static_cast<aclrtStream>(streams[i]));
        }
    }
    // 设置scratchMem
    resPack.scratchMemAddr = scratchMemAddr;
    resPack.scratchMemSize = scratchMemSize;

    /* 接口交互信息日志 */
    CHK_RET(SendEntryLog(sendBuf, count, dataType, destRank, stream, opTag, "HcclSendGraphMode"));

    // 执行Send
    CHK_RET_AND_PRINT_IDE(SendExec(sendBuf, count, dataType, destRank, comm, stream, rankSize, OpMode::OFFLOAD, opTag, resPack), opTag.c_str());

    CHK_RET(LogHcclExit("HcclSendGraphMode", opTag.c_str(), startut));

    HCCL_INFO("[HcclSendGraphMode][%d]->[%d] Success.", userRank, destRank);
    return HcclResult::HCCL_SUCCESS;
}

namespace ops_hccl {
    HcclResult GetAndCheckSendPara(
        const HcclComm comm, const void *sendBuf, const uint64_t count, const HcclDataType dataType,
        const uint32_t destRank, u32 &rankSize, u32 &userRank, std::string &tag)
    {
        // 参数校验
        RPT_INPUT_ERR(
            comm == nullptr,
            "EI0003",
            std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
            std::vector<std::string>({"HcclSend", "nullptr", "comm", "non-null pointer"}));
        CHK_PTR_NULL(comm);
        RPT_INPUT_ERR(
            sendBuf == nullptr,
            "EI0003",
            std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
            std::vector<std::string>({"HcclSend", "nullptr", "sendBuf", "non-null pointer"}));
        CHK_PTR_NULL(sendBuf);
        CHK_RET(CheckCount(count));
        CHK_RET(CheckDataType(dataType, false));
        CHK_RET(HcclGetRankSize(comm, &rankSize));
        CHK_RET(HcclGetRankId(comm, &userRank));
        CHK_PRT_RET(userRank == destRank, HCCL_ERROR("[HcclSend] destRank cannot be equal to self."), HcclResult::HCCL_E_NOT_SUPPORT);
        if (tag.empty()) {
            char commName[COMM_INDENTIFIER_MAX_LENGTH];
            CHK_RET(HcclGetCommName(comm, commName));
            tag = "SendRecv_" + string(commName) + "_" + std::to_string(userRank) + "_" + std::to_string(destRank);
        }
        CHK_RET(HcclCheckTag(tag.c_str()));
        CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());
        CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, destRank), tag.c_str());
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult GenerateSendOpParam(
        OpParam &param, void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
        const HcclComm comm, const aclrtStream stream, const std::string &tag)
    {
        // 获取通信域名称
        CHK_RET(HcclGetCommName(comm, param.commName));
        u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
        u64 dataSize = count * dataTypeSize;
        param.opType = HcclCMDType::HCCL_CMD_SEND;
        param.enableDetour = false;

        DevType deviceType = DevType::DEV_TYPE_COUNT;
        CHK_RET(hrtGetDeviceType(deviceType));
        param.deviceType = deviceType;

        // topoInfo的tag，所有相同的算子可以共享
        auto fillTagRet = strncpy_s(param.tag, sizeof(param.tag), tag.c_str(), sizeof(param.tag) - 1);
        CHK_PRT_RET(
            fillTagRet != EOK,
            HCCL_ERROR("[GenerateSendOpParam] failed to fill param.tag, tag %s, return %d.", tag.c_str(), fillTagRet),
            HcclResult::HCCL_E_INTERNAL);

        param.stream = stream;
        param.inputPtr = sendBuf;
        param.inputSize = dataSize;
        param.sendRecvRemoteRank = destRank;
        param.outputPtr = nullptr;
        param.outputSize = 0;
        param.DataDes.count = count;
        param.DataDes.dataType = dataType;

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult SendExec(
        void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
        const HcclComm comm, const aclrtStream stream, const u32 &rankSize,
        const OpMode &opMode, const std::string &tag, const ResPackGraphMode &resPack)
    {
        HCCL_DEBUG("[SendExec][%s][%s] Start.", tag.c_str(), opMode == OpMode::OPBASE ? "OPBASE" : "OFFLOAD");

        // 参数构建
        OpParam param;
        CHK_RET(GenerateSendOpParam(param, sendBuf, count, dataType, destRank, comm, stream, tag));
        param.opMode = opMode;

        std::string algName;
        std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
        CHK_RET(HcclGetOpExpansionMode(comm, param));
        CHK_RET(Selector(comm, param, topoInfo, algName));

        if (ShouldUseInnerOp(param.opExecuteConfig) && param.opMode == OpMode::OPBASE) {
            return HcclSendInner(sendBuf, count, dataType, destRank, comm, stream);
        }
        if (rankSize == 1) {
            HCCL_WARNING("[SendExec][%s][%s] ranksize == 1, enter SingleRankProc", tag.c_str(),
                opMode == OpMode::OPBASE ? "OPBASE" : "OFFLOAD");
            CHK_RET(SingleRankProc(comm, param));
            return HcclResult::HCCL_SUCCESS;
        }

        CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));

        return HcclResult::HCCL_SUCCESS;
    }
    
HcclResult SendEntryLog(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
    aclrtStream stream, const std::string &tag, const std::string &opName)
{
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], count[%llu], dataType[%s], destRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, count, GetDataTypeEnumStr(dataType).c_str(), destRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
