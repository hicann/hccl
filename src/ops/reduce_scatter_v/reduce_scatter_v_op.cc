/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_v_op.h"
#include "op_common_ops.h"
#include "topo_host.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

// 代码入口
HcclResult HcclReduceScatterV(void *sendBuf,  const void *sendCounts, const void *sendDispls, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclReduceScatterV");
    if (GetHcommVersion() < 90000000) { // compat handle
        return HcclReduceScatterVInner(sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    }

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    // 非95设备转到老流程
    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95) {
    #endif
        return HcclReduceScatterVInner(sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    }
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    // A3是：export HCCL_OP_EXPANSION_MODE="AI_CPU"，A5的接口还没提供
    CHK_RET(InitEnvConfig());

    // 参数校验等工作;
    // 校验入参
    CHK_RET(CheckReduceScatterVInputParam(comm, sendBuf, recvBuf, recvCount, sendCounts, sendDispls, stream));
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

    /* 接口交互信息日志 */
    CHK_RET(ReduceScatterVEntryLog(sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, stream, tag, "HcclReduceScatterV"));

    // 初始化参数
    CHK_RET_AND_PRINT_IDE(ReduceScatterVOutPlace(sendBuf, sendDispls, sendCounts, recvBuf, recvCount, dataType, op, comm, stream, tag),
                          tag.c_str());

    CHK_RET(LogHcclExit("ReduceScatterV", tag.c_str(), startut));

    return HCCL_SUCCESS;
}


HcclResult HcclReduceScatterVGraphMode(void *sendBuf,  const void *sendCounts, const void *sendDispls, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_INFO("Start to run execute HcclReduceScatterVGraphMode");
    // 根据group获取通信域
    HcclComm comm = nullptr;
    HCCL_INFO("[HcclReduceScatterVGraphMode] get group name: %s", group);
    HcomGetCommHandleByGroup(group, &comm);
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    CHK_RET(InitEnvConfig());
    // 参数校验等工作;
    CHK_RET(CheckReduceScatterVInputParam(comm, sendBuf, recvBuf, recvCount, sendCounts, sendDispls, stream));

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
    const string opTag = "ReduceScatterV_" + string(commName);
    CHK_RET(HcclCheckTag(opTag.c_str()));
    CHK_RET(HcclCheckTag(tag));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), opTag.c_str());
    CHK_RET(CheckCount(recvCount));
    CHK_RET(CheckDataType(dataType, true));

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

    /* 接口交互信息日志 */
    CHK_RET(ReduceScatterVEntryLog(sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, stream, opTag, "HcclReduceScatterVGraphMode"));

    // 执行
    CHK_RET_AND_PRINT_IDE(ReduceScatterVOutPlaceGraphMode(sendBuf, sendDispls, sendCounts, recvBuf, recvCount, dataType, op, comm, stream, tag, resPack), opTag);

    CHK_RET(LogHcclExit("HcclReduceScatterVGraphMode", opTag.c_str(), startut));

    return HCCL_SUCCESS;
}



namespace ops_hccl {
HcclResult CheckReduceScatterVInputParam(
    const HcclComm comm, const void *sendBuf, const void *recvBuf, uint64_t recvCount,
    const void *sendCounts, const void *sendDispls, const aclrtStream stream)
{
    // 入参合法性校验
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclReduceScatterV", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclReduceScatterV", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);

    RPT_INPUT_ERR(sendCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclReduceScatterV", "nullptr", "sendCounts", "non-null pointer"}));
    CHK_PTR_NULL(sendCounts);

    RPT_INPUT_ERR(sendDispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclReduceScatterV", "nullptr", "sendDispls", "non-null pointer"}));
    CHK_PTR_NULL(sendDispls);
    if (UNLIKELY(recvCount > 0 && recvBuf == nullptr)) {
        RPT_INPUT_ERR(true, "EI0003",\
        std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclReduceScatterV", "nullptr", "recvBuf", "non-null pointer"}));
        CHK_PTR_NULL(recvBuf);
    }

    return HCCL_SUCCESS;
}

HcclResult PrepareReduceScatterVParam(void *sendBuf, const void *sendDispls, const void *sendCounts, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, const std::string &tag, OpMode opMode, u32 userRankSize, u64 varMemSize, OpParam &param) 
{
    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 outputSize = recvCount * perDataSize;
    HCCL_INFO("PrepareReduceScatterVParam[outputSize]:[%u]", outputSize);

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
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    param.vDataDes.dataType = dataType;
    const void *temp = sendCounts;
    param.vDataDes.counts = const_cast<void*>(temp);

    HCCL_INFO("PrepareReduceScatterVParam: sendBuf:[%u]", sendBuf);
    HCCL_INFO("PrepareReduceScatterVParam: recvBuf:[%u]", recvBuf);
    HCCL_INFO("PrepareReduceScatterVParam: recvCount:[%u]", recvCount);

    // 参数准备
    std::vector<u64> countsAndDispls(userRankSize*2);
    const u64* sendDisplsAddr = reinterpret_cast<const u64*>(sendDispls);
    const u64* sendCountsAddr = reinterpret_cast<const u64*>(sendCounts);

    param.inputSize = (sendDisplsAddr[userRankSize-1] + sendCountsAddr[userRankSize-1]) * perDataSize;

    std::copy(sendCountsAddr, sendCountsAddr + userRankSize, countsAndDispls.begin());
    std::copy(sendDisplsAddr, sendDisplsAddr + userRankSize, countsAndDispls.begin() + userRankSize);
    param.varMemSize = varMemSize;

    for (u64 i=0; i < countsAndDispls.size();++i) {
        HCCL_INFO("PrepareReduceScatterVParam: countsAndDispls[%u]:[%u]", i, countsAndDispls[i]);
    }

    // 从源内存地址按字节直接拷贝数据到目标地址
    memcpy_s(param.varData, varMemSize, countsAndDispls.data(), varMemSize);
    const u64* varData = reinterpret_cast<const u64*>(param.varData);

    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
    param.enableDetour = false;
    param.deviceType = deviceType;
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVOutPlace(void *sendBuf, const void *sendDispls, const void *sendCounts, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, const std::string &tag)
{
    HCCL_INFO("Start to execute ReduceScatterVOutPlace");
    CHK_RET(ReduceScatterVOutPlaceCommon(sendBuf, sendDispls, sendCounts, recvBuf, recvCount, dataType, op, comm, stream, tag, OpMode::OPBASE, ResPackGraphMode()));
    HCCL_INFO("Execute ReduceScatterVOutPlace success.");
    return HCCL_SUCCESS;
}


HcclResult ReduceScatterVOutPlaceGraphMode(void *sendBuf, const void *sendDispls, const void *sendCounts, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, const std::string &tag, const ResPackGraphMode &resPack)
{
    HCCL_INFO("Start to execute ReduceScatterVOutPlaceGraphMode");
    CHK_RET(ReduceScatterVOutPlaceCommon(sendBuf, sendDispls, sendCounts, recvBuf, recvCount, dataType, op, comm, stream, tag, OpMode::OFFLOAD, resPack));
    HCCL_INFO("Execute ReduceScatterVOutPlaceGraphMode success.");
    return HCCL_SUCCESS;
}


HcclResult ReduceScatterVOutPlaceCommon(void *sendBuf, const void *sendDispls, const void *sendCounts, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, const std::string &tag, OpMode opMode, const ResPackGraphMode &resPack)
{
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    // 申请OpParam参数结构体内存
    u64 varMemSize = 2 * userRankSize * sizeof(u64);

    void* paramMem = malloc(sizeof(OpParam) + varMemSize);

    if (!paramMem) {
        // 内存分配失败
        HCCL_ERROR("[ReduceScatterVOutPlaceCommon] malloc OpParam failed!");
        return HCCL_E_INTERNAL;
    }
    OpParam* tmpParamPtr = new (paramMem) OpParam();
    auto deleter = [](OpParam* p) {
        if (p) {
            p->~OpParam();
            free(p);
        }
     };
    std::unique_ptr<OpParam, decltype(deleter)> paramPtr(tmpParamPtr, deleter);
    OpParam& param = *paramPtr;

    CHK_RET(PrepareReduceScatterVParam(sendBuf, sendDispls, sendCounts, recvBuf, recvCount, dataType, op, comm, stream, tag, opMode, userRankSize, varMemSize, param));
    
    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(HcclGetOpExpansionMode(comm, param));
    CHK_RET(Selector(comm, param, topoInfo, algName));

    if (ShouldUseInnerOp(param.opExecuteConfig) && param.opMode == OpMode::OPBASE) {
        return HcclReduceScatterVInner(sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    }
    
    if (userRankSize == 1) {
        HCCL_WARNING("[%s] ranksize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVEntryLog(void *sendBuf, const void *sendCounts, const void *sendDispls, void *recvBuf,
    uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, aclrtStream stream, const std::string &tag, const std::string &opName)
{
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], sendCounts[%p], sendDispls[%p], recvBuf[%p], recvCount[%llu], dataType[%s], reduceOp[%s], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, sendCounts, sendDispls, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(), streamId, deviceLogicId);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}
}
