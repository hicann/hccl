/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_to_all_v_op.h"
#include "op_common_ops.h"
#include "topo_host.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
    uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclAlltoAll");
    if (GetHcommVersion() < 90000000) { // compat handle
        return HcclAlltoAllInner(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream);
    }

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95) {
    #endif
        return HcclAlltoAllInner(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream);
    }
    CHK_PRT_RET(sendCount == 0 && recvCount == 0,
        HCCL_WARNING("sendCount and recvCount are both 0, return AllToAll success"), HCCL_SUCCESS);
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    CHK_RET(InitEnvConfig());

    // 参数校验等工作
    CHK_RET(CheckAlltoAllInputPara(comm, sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, stream));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    const string tag =  "ALLTOALL_" + string(commName);
    CHK_RET(HcclCheckTag(tag.c_str()));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());
    CHK_RET(CheckCount(recvCount));
    CHK_RET(CheckDataType(recvType, false));

    // 构造四个矩阵，适配alltoallV的逻辑
    std::vector<u64> sdispls(rankSize, 0);
    std::vector<u64> rdispls(rankSize, 0);
    std::vector<u64> sendCounts(rankSize, recvCount);
    std::vector<u64> recvCounts(rankSize, recvCount);
    CHK_RET(ConvertAlltoAllParam(recvCount, rankSize, sdispls, rdispls));

    /* 接口交互信息日志 */
    CHK_RET(AlltoAllEntryLog(sendBuf, recvBuf, sendCount, recvCount, sendType, recvType, stream, tag, "HcclAlltoAll"));

    // 底层走AlltoAllV
    bool useInnerOp = false;
    CHK_RET_AND_PRINT_IDE(AlltoAllVOutPlace(sendBuf, sendCounts.data(), sdispls.data(),
        recvBuf, recvCounts.data(), rdispls.data(), recvType, comm, stream, tag,
        HcclCMDType::HCCL_CMD_ALLTOALL, rankSize, useInnerOp), tag.c_str());
    
    CHK_RET(LogHcclExit("HcclAlltoAll", tag.c_str(), startut));

    if (useInnerOp) {
        return HcclAlltoAllInner(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
    const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclAlltoAllV");
    if (GetHcommVersion() < 90000000) { // compat handle
        return HcclAlltoAllVInner(sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, comm, stream);
    }

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95) {
    #endif
        return HcclAlltoAllVInner(sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, comm, stream);
    }
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    CHK_RET(InitEnvConfig());

    // 参数校验等工作
    CHK_RET(CheckAlltoAllVInputPara(comm, sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, stream));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string tag =  "ALLTOALLV_" + string(commName);
    CHK_RET(HcclCheckTag(tag.c_str()));
    CHK_RET(CheckBufNullptr(reinterpret_cast<const u64*>(sendCounts), rankSize, sendBuf, std::string(__func__), "sendBuf"));
    CHK_RET(CheckBufNullptr(reinterpret_cast<const u64*>(recvCounts), rankSize, recvBuf, std::string(__func__), "recvBuf"));

    u64 maxSendRecvCount = 0;
    for (u64 i = 0; i < rankSize; i++) {
        maxSendRecvCount = max(maxSendRecvCount, static_cast<const u64 *>(sendCounts)[i]);
        maxSendRecvCount = max(maxSendRecvCount, static_cast<const u64 *>(recvCounts)[i]);
    }

    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());
    CHK_RET(CheckCount(maxSendRecvCount));
    CHK_RET(CheckDataType(recvType, false));

    /* 接口交互信息日志 */
    CHK_RET(AlltoAllVEntryLog(sendBuf, recvBuf, sendCounts, recvCounts, sdispls, rdispls, sendType, recvType, stream, tag, "HcclAlltoAllV"));

    // 底层走AlltoAllV
    bool useInnerOp = false;
    CHK_RET_AND_PRINT_IDE(AlltoAllVOutPlace(sendBuf, sendCounts, sdispls, recvBuf, recvCounts, rdispls, recvType, comm, stream,
        tag, HcclCMDType::HCCL_CMD_ALLTOALLV, rankSize, useInnerOp), tag.c_str());
    
    CHK_RET(LogHcclExit("HcclAlltoAllV", tag.c_str(), startut));

    if (useInnerOp) {
        return HcclAlltoAllVInner(sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, comm, stream);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclAlltoAllVC");

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95) {
    #endif
        return HcclAlltoAllVCInner(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, comm, stream);
    }
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    CHK_RET(InitEnvConfig());

    // 参数校验等工作
    CHK_RET(CheckAlltoAllVCInputPara(comm, sendBuf, sendCountMatrix, sendType, recvBuf, recvType, stream));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string tag =  "ALLTOALLVC_" + string(commName);
    CHK_RET(HcclCheckTag(tag.c_str()));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());

    // 构造四个矩阵，适配alltoallV的逻辑
    std::vector<u64> sendCounts(rankSize, 0);
    std::vector<u64> recvCounts(rankSize, 0);
    std::vector<u64> sdispls(rankSize, 0);
    std::vector<u64> rdispls(rankSize, 0);
    CHK_RET(ConvertAlltoAllVCParam(rankSize, userRank, sendCountMatrix, sendCounts, recvCounts, sdispls, rdispls));
    CHK_RET(CheckBufNullptr(sendCounts.data(), rankSize, sendBuf, std::string(__func__), "sendBuf"));
    CHK_RET(CheckBufNullptr(recvCounts.data(), rankSize, recvBuf, std::string(__func__), "recvBuf"));

    CHK_RET(CheckDataType(recvType, false));

    /* 接口交互信息日志 */
    CHK_RET(AlltoAllVCEntryLog(sendBuf, recvBuf, sendCountMatrix, sendType, recvType, stream, tag, "HcclAlltoAllVC"));

    // 底层走AlltoAllV
    bool useInnerOp = false;
    CHK_RET_AND_PRINT_IDE(AlltoAllVOutPlace(sendBuf, sendCounts.data(), sdispls.data(),
        recvBuf, recvCounts.data(), rdispls.data(), recvType, comm, stream, tag,
        HcclCMDType::HCCL_CMD_ALLTOALLVC, rankSize, useInnerOp), tag.c_str());

    CHK_RET(LogHcclExit("HcclAlltoAllVC", tag.c_str(), startut));    

    if (useInnerOp) {
        return HcclAlltoAllVCInner(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, comm, stream);
    }
    return HCCL_SUCCESS;
}

// 图模式对外接口
HcclResult HcclAlltoAllGraphMode(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
    uint64_t recvCount, HcclDataType recvType, const char* group, aclrtStream stream, const char* tag,
    void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_INFO("Start to run execute HcclAlltoAllGraphMode");
    // 根据group获取通信域
    HcclComm comm = nullptr;
    HCCL_INFO("[HcclAlltoAllGraphMode] get group name: %s", group);
    HcomGetCommHandleByGroup(group, &comm);
    CHK_PRT_RET(sendCount == 0 && recvCount == 0,
        HCCL_WARNING("sendCount and recvCount are both 0, return AllToAll success"), HCCL_SUCCESS);
    HcclUs startut = TIME_NOW();
    CHK_RET(InitEnvConfig());

    // 参数校验等工作
    CHK_RET(CheckAlltoAllInputPara(comm, sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, stream));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string opTag = "AlltoAll_" + string(commName);
    CHK_RET(HcclCheckTag(opTag.c_str()));
    CHK_RET(HcclCheckTag(tag));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), opTag.c_str());
    CHK_RET(CheckCount(recvCount));
    CHK_RET(CheckDataType(recvType, false));

    // 构造四个矩阵，适配alltoallV的逻辑
    std::vector<u64> sendCounts(rankSize, recvCount);
    std::vector<u64> recvCounts(rankSize, recvCount);
    std::vector<u64> sdispls(rankSize, 0);
    std::vector<u64> rdispls(rankSize, 0);
    CHK_RET(ConvertAlltoAllParam(recvCount, rankSize, sdispls, rdispls));

    // 拼装ResPackGraphMode
    ResPackGraphMode resPack;
    CHK_RET(GenResPack(tag, streams, streamCount, scratchMemAddr, scratchMemSize, resPack));

    /* 接口交互信息日志 */
    CHK_RET(AlltoAllEntryLog(sendBuf, recvBuf, sendCount, recvCount, sendType, recvType, stream, opTag, "HcclAlltoAllGraphMode"));

    // 执行AlltoAllV
    CHK_RET_AND_PRINT_IDE(AlltoAllVOutPlaceGraphMode(sendBuf, sendCounts.data(), sdispls.data(),
        recvBuf, recvCounts.data(), rdispls.data(), recvType, comm, stream, tag,
        HcclCMDType::HCCL_CMD_ALLTOALL, rankSize, resPack), opTag);

    CHK_RET(LogHcclExit("HcclAlltoAllGraphMode", opTag.c_str(), startut));

    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllVGraphMode(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
    const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType, const char* group, aclrtStream stream, const char* tag,
    void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_INFO("Start to run execute HcclAlltoAllVGraphMode");
    // 根据group获取通信域
    HcclComm comm = nullptr;
    HCCL_INFO("[HcclAlltoAllVGraphMode] get group name: %s", group);
    HcomGetCommHandleByGroup(group, &comm);
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    CHK_RET(InitEnvConfig());

    // 参数校验等工作
    CHK_RET(CheckAlltoAllVInputPara(comm, sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, stream));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));

    const string opTag = "AlltoAllV_" + string(commName);
    CHK_RET(HcclCheckTag(opTag.c_str()));
    CHK_RET(HcclCheckTag(tag));

    u64 maxSendRecvCount = 0;
    for (u64 i = 0; i < rankSize; i++) {
        maxSendRecvCount = max(maxSendRecvCount, static_cast<const u64 *>(sendCounts)[i]);
        maxSendRecvCount = max(maxSendRecvCount, static_cast<const u64 *>(recvCounts)[i]);
    }

    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), opTag.c_str());
    CHK_RET(CheckCount(maxSendRecvCount));
    CHK_RET(CheckDataType(recvType, false));

    // 拼装ResPackGraphMode
    ResPackGraphMode resPack;
    CHK_RET(GenResPack(tag, streams, streamCount, scratchMemAddr, scratchMemSize, resPack));

    /* 接口交互信息日志 */
    CHK_RET(AlltoAllVEntryLog(sendBuf, recvBuf, sendCounts, recvCounts, sdispls, rdispls, sendType, recvType, stream, opTag, "HcclAlltoAllVGraphMode"));

    // 执行AlltoAllV
    CHK_RET_AND_PRINT_IDE(AlltoAllVOutPlaceGraphMode(sendBuf, sendCounts, sdispls,
        recvBuf, recvCounts, rdispls, recvType, comm, stream, tag,
        HcclCMDType::HCCL_CMD_ALLTOALLV, rankSize, resPack), opTag);

    CHK_RET(LogHcclExit("HcclAlltoAllVGraphMode", opTag.c_str(), startut));

    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllVCGraphMode(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, const char* group, aclrtStream stream, const char* tag,
    void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_INFO("Start to run execute HcclAlltoAllVCGraphMode");
    // 根据group获取通信域
    HcclComm comm = nullptr;
    HCCL_INFO("[HcclAlltoAllVCGraphMode] get group name: %s", group);
    HcomGetCommHandleByGroup(group, &comm);
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    CHK_RET(InitEnvConfig());

    // 参数校验等工作
    CHK_RET(CheckAlltoAllVCInputPara(comm, sendBuf, sendCountMatrix, sendType, recvBuf, recvType, stream));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));

    const string opTag = "AlltoAllVC_" + string(commName);
    CHK_RET(HcclCheckTag(opTag.c_str()));
    CHK_RET(HcclCheckTag(tag));

    // 构造四个矩阵，适配alltoallV的逻辑
    std::vector<u64> sendCounts(rankSize, 0);
    std::vector<u64> recvCounts(rankSize, 0);
    std::vector<u64> sdispls(rankSize, 0);
    std::vector<u64> rdispls(rankSize, 0);
    CHK_RET(ConvertAlltoAllVCParam(rankSize, userRank, sendCountMatrix, sendCounts,
        recvCounts, sdispls, rdispls));

    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), opTag.c_str());
    CHK_RET(CheckDataType(recvType, false));

    // 拼装ResPackGraphMode
    ResPackGraphMode resPack;
    CHK_RET(GenResPack(tag, streams, streamCount, scratchMemAddr, scratchMemSize, resPack));

    /* 接口交互信息日志 */
    CHK_RET(AlltoAllVCEntryLog(sendBuf, recvBuf, sendCountMatrix, sendType, recvType, stream, opTag, "HcclAlltoAllVCGraphMode"));

    // 执行AlltoAllV
    CHK_RET_AND_PRINT_IDE(AlltoAllVOutPlaceGraphMode(sendBuf, sendCounts.data(), sdispls.data(),
        recvBuf, recvCounts.data(), rdispls.data(), recvType, comm, stream, tag,
        HcclCMDType::HCCL_CMD_ALLTOALLVC, rankSize, resPack), opTag);

    CHK_RET(LogHcclExit("HcclAlltoAllVCGraphMode", opTag.c_str(), startut));

    return HCCL_SUCCESS;
}

namespace ops_hccl {

HcclResult GenResPack(const char* tag, void** streams, const size_t streamCount,
    void* scratchMemAddr, const uint64_t scratchMemSize, ResPackGraphMode &resPack)
{
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
    return HCCL_SUCCESS;
}

HcclResult ConvertAlltoAllParam(const u64 recvCount, const u32 rankSize, std::vector<u64> &sdispls, std::vector<u64> &rdispls)
{
    u64 dataCountOffset = 0;
    for (u64 i = 0; i < rankSize; i++) {
        sdispls[i] = dataCountOffset;
        rdispls[i] = dataCountOffset;
        dataCountOffset += recvCount;
    }
    return HCCL_SUCCESS;
}

HcclResult ConvertAlltoAllVCParam(const u32 rankSize, const u32 userRank, const void *sendCountMatrix,
    std::vector<u64> &sendCounts, std::vector<u64> &recvCounts, std::vector<u64> &sdispls, std::vector<u64> &rdispls)
{
    // 取出sendCountMatrix的数据
    const u64* data = static_cast<const u64*>(sendCountMatrix);
    u64 maxSendRecvCount = 0;
    for (u64 i = 0; i < static_cast<u64>(rankSize) * rankSize; i++) {
        maxSendRecvCount = max(maxSendRecvCount, data[i]);
    }
    CHK_RET(CheckCount(maxSendRecvCount));

    std::vector<std::vector<u64>> outputMatrix;
    outputMatrix.resize(rankSize);
    for (u64 i = 0; i < rankSize; ++i) {
        // 计算当前行的起始指针位置（行优先顺序）
        const u64* rowStart = data + i * rankSize;
        // 直接通过指针初始化当前行的vector
        outputMatrix[i].assign(rowStart, rowStart + rankSize);
    }

    u64 dataCountOffset = 0;
    for (u64 i = 0; i < rankSize; i++) {
        sendCounts[i] = outputMatrix[userRank][i];
        sdispls[i] = dataCountOffset;
        dataCountOffset += sendCounts[i];
    }

    dataCountOffset = 0;
    for (u64 i = 0; i < rankSize; i++) {
        recvCounts[i] = outputMatrix[i][userRank];
        rdispls[i] = dataCountOffset;
        dataCountOffset += recvCounts[i];
    }
    return HCCL_SUCCESS;
}

HcclResult CheckAlltoAllInputPara(const HcclComm comm, const void *sendBuf, const uint64_t sendCount,
    const HcclDataType sendType, const void *recvBuf, const uint64_t recvCount,
    const HcclDataType recvType, const aclrtStream stream)
{
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAlltoAll", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAlltoAll", "nullptr", "sendBuf", "non-null pointer"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAlltoAll", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);
    CHK_PRT_RET(sendCount != recvCount,
        HCCL_ERROR("sendCount[%lu] and recvCount[%lu] are not equal, please check params",
            sendCount, recvCount), HCCL_E_PARA);
    CHK_PRT_RET(sendType != recvType,
        HCCL_ERROR("sendType[%s] and recvType[%s] are not equal, please check params",
            GetDataTypeEnumStr(sendType).c_str(), GetDataTypeEnumStr(recvType).c_str()), HCCL_E_PARA);
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAlltoAllV", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(sendBuf == recvBuf,
        HCCL_ERROR("[HcclAlltoAll] sendBuf and recvBuf cannot be same."), HCCL_E_PARA);

    return HCCL_SUCCESS;
}

HcclResult CheckAlltoAllVInputPara(const HcclComm comm, const void *sendBuf, const void *sendCounts, const void *sdispls,
    const HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls,
    const HcclDataType recvType, const aclrtStream stream)
{
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAlltoAllV", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAlltoAllV", "nullptr", "sendCounts", "non-null pointer"}));
    CHK_PTR_NULL(sendCounts);
    RPT_INPUT_ERR(sdispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAlltoAllV", "nullptr", "sdispls", "non-null pointer"}));
    CHK_PTR_NULL(sdispls);
    RPT_INPUT_ERR(recvCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAlltoAllV", "nullptr", "recvCounts", "non-null pointer"}));
    CHK_PTR_NULL(recvCounts);
    RPT_INPUT_ERR(rdispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAlltoAllV", "nullptr", "rdispls", "non-null pointer"}));
    CHK_PTR_NULL(rdispls);
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAlltoAllV", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    return HCCL_SUCCESS;
}

HcclResult CheckAlltoAllVCInputPara(const HcclComm comm, const void *sendBuf, const void *sendCountMatrix,
    const HcclDataType sendType, const void *recvBuf, const HcclDataType recvType, const aclrtStream stream)
{
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAlltoAllVC", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendCountMatrix == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAlltoAllVC", "nullptr", "sendCountMatrix", "non-null pointer"}));
    CHK_PTR_NULL(sendCountMatrix);
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAlltoAllVC", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    return HCCL_SUCCESS;
}

HcclResult CheckBufNullptr(const u64* countsData, u32 rankSize, const void* buf, const std::string funcName,
    const std::string bufName)
{
    bool zeroFlag = true;
    for (u32 i = 0; i < rankSize; i++) {
        if (countsData[i] != 0) {
            zeroFlag = false;
            break;
        }
    }
    if (zeroFlag) {
        return HCCL_SUCCESS;
    }
    RPT_INPUT_ERR(buf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({funcName, "nullptr", bufName, "non-null pointer"}));
    CHK_PTR_NULL(buf);
    return HCCL_SUCCESS;
}

HcclResult CalcInputOutputSize(const u64* sendCountsData, const u64* recvCountsData, const u64* sdisplsData,
    const u64* rdisplsData, const u32 userRankSize, u64 &inputSize, u64 &outputSize)
{
    for (u64 i = 0; i < userRankSize; i++) {
        u64 tmpInputSize = sdisplsData[i] + sendCountsData[i];
        u64 tmpOutputSize = rdisplsData[i] + recvCountsData[i];
        if (tmpInputSize > inputSize) {
            inputSize = tmpInputSize;
        }
        if (tmpOutputSize > outputSize) {
            outputSize = tmpOutputSize;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ContructVarData(const u64* sendCountsData, const u64* recvCountsData, const u64* sdisplsData,
    const u64* rdisplsData, const u32 userRankSize, const u32 rankSize, OpParam &param)
{
    u64* data = reinterpret_cast<u64*>(param.varData);
    for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * userRankSize; i++) {
        u64 val = i / rankSize;
        switch(val) {
            case SEND_COUNT_IDX:
                data[i] = sendCountsData[i % rankSize];
                break;
            case RECV_COUNT_IDX:
                data[i] = recvCountsData[i % rankSize];
                break;
            case SEND_DISPL_IDX:
                data[i] = sdisplsData[i % rankSize];
                break;
            case RECV_DISPL_IDX:
                data[i] = rdisplsData[i % rankSize];
                break;
            default:
                break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVConstructOpParam(const void *sendBuf, const void *sendCounts, const void *sdispls, const void *recvBuf,
    const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclComm comm, aclrtStream stream,
    const std::string &tag, HcclCMDType opType, u32 rankSize, OpMode opMode, u64 varMemSize, OpParam &param)
{
    CHK_RET(HcclGetCommName(comm, param.commName));
    param.stream = stream;
    param.opMode = opMode;
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    param.deviceType = deviceType;

    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
    if (ret <= 0) {
        HCCL_ERROR("failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }

    param.inputPtr = const_cast<void*>(sendBuf);
    param.outputPtr = const_cast<void*>(recvBuf);
    param.varMemSize = varMemSize;
    param.all2AllVDataDes.sendType = dataType;
    param.all2AllVDataDes.recvType = dataType;

    const u64* sendCountsData = static_cast<const u64*>(sendCounts);
    const u64* recvCountsData = static_cast<const u64*>(recvCounts);
    const u64* sdisplsData = static_cast<const u64*>(sdispls);
    const u64* rdisplsData = static_cast<const u64*>(rdispls);
    // 计算整片数据包含中间间隔的大小，防止图模式注册内存踩踏
    u64 inputSize = 0;
    u64 outputSize = 0;
    CHK_RET(CalcInputOutputSize(sendCountsData, recvCountsData, sdisplsData, rdisplsData,
        rankSize, inputSize, outputSize));
    param.inputSize = inputSize;
    param.outputSize = outputSize;

    param.enableDetour = false;
    param.opType = opType;

    CHK_RET(ContructVarData(sendCountsData, recvCountsData, sdisplsData, rdisplsData, rankSize, rankSize, param));
    u64* data = reinterpret_cast<u64*>(param.varData);
    param.all2AllVDataDes.sendCounts = data;
    param.all2AllVDataDes.recvCounts = data + RECV_COUNT_IDX * rankSize;
    param.all2AllVDataDes.sdispls = data + SEND_DISPL_IDX * rankSize;
    param.all2AllVDataDes.rdispls = data + RECV_DISPL_IDX * rankSize;

    for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * rankSize; i++) {
        HCCL_INFO("[AlltoAllVConstructOpParam] varData[%u] is [%u]", i, data[i]);
    }
    HCCL_INFO("[AlltoAllVConstructOpParam] SIZE_TABLE[dataType] is [%u]", SIZE_TABLE[dataType]);
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVOutPlaceCommon(const void *sendBuf, const void *sendCounts, const void *sdispls, const void *recvBuf,
    const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclComm comm, aclrtStream stream,
    const std::string &tag, HcclCMDType opType, u32 rankSize, bool &useInnerOp, OpMode opMode, const ResPackGraphMode &resPack)
{
    u64 varMemSize = ALL_TO_ALL_V_VECTOR_NUM * rankSize * sizeof(u64);
    void *paramMem = malloc(sizeof(OpParam) + varMemSize);
    if (!paramMem) {
        // 内存分配失败
        HCCL_ERROR("[AlltoAllVOutPlaceCommon] malloc OpParam failed!");
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
    OpParam &param = *paramPtr;

    CHK_RET(AlltoAllVConstructOpParam(sendBuf, sendCounts, sdispls, recvBuf, recvCounts, rdispls, dataType,
        comm, stream, tag, opType, rankSize, opMode, varMemSize, param));
    
    CHK_RET(HcclGetOpExpansionMode(comm, param));

    CcuFastLaunchCtx *ccuFastLaunchCtx = nullptr;
    if (ShouldGoCcuFastLaunch(comm, param, &ccuFastLaunchCtx)) {
        return HcclExecOpCcuFastLaunch(comm, param, ccuFastLaunchCtx);
    }

    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(Selector(comm, param, topoInfo, algName));
    if (ShouldUseInnerOp(param.opExecuteConfig) && param.opMode == OpMode::OPBASE) {
        useInnerOp = true;
        return HCCL_SUCCESS;
    }
    if (rankSize == 1) {
        HCCL_WARNING("[%s] rankSize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVOutPlaceGraphMode(const void *sendBuf, const void *sendCounts, const void *sdispls, const void *recvBuf,
    const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclComm comm, aclrtStream stream,
    const std::string &tag, HcclCMDType opType, u32 rankSize, const ResPackGraphMode &resPack)
{
    HCCL_INFO("Start to execute AlltoAllVOutPlaceGraphMode");
    bool useInnerOp = false;
    CHK_RET(AlltoAllVOutPlaceCommon(sendBuf, sendCounts, sdispls, recvBuf, recvCounts, rdispls, dataType, comm, stream,
        tag, opType, rankSize, useInnerOp, OpMode::OFFLOAD, resPack));
    if (useInnerOp) {
        HCCL_ERROR("should use inner op!");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("Execute AlltoAllVOutPlaceGraphMode success.");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVOutPlace(const void *sendBuf, const void *sendCounts, const void *sdispls, const void *recvBuf,
    const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclComm comm, aclrtStream stream,
    const std::string &tag, HcclCMDType opType, u32 rankSize, bool &useInnerOp)
{
    HCCL_INFO("Start to execute AlltoAllVOutPlace");
    CHK_RET(AlltoAllVOutPlaceCommon(sendBuf, sendCounts, sdispls, recvBuf, recvCounts, rdispls, dataType, comm, stream,
        tag, opType, rankSize, useInnerOp, OpMode::OPBASE, ResPackGraphMode()));
    HCCL_INFO("Execute AlltoAllVOutPlace success.");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllEntryLog(const void *sendBuf, const void *recvBuf, uint64_t sendCount, uint64_t recvCount,
    HcclDataType sendType, HcclDataType recvType, aclrtStream stream, const std::string &tag, const std::string &opName)
{
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], recvCount[%llu], sendType[%s], recvType[%s], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, sendCount, recvCount, GetDataTypeEnumStr(sendType).c_str(), GetDataTypeEnumStr(recvType).c_str(),
            streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVEntryLog(const void *sendBuf, const void *recvBuf, const void *sendCounts, const void *recvCounts,
    const void *sdispls, const void *rdispls, HcclDataType sendType, HcclDataType recvType, aclrtStream stream,
    const std::string &tag, const std::string &opName)
{
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], sendCounts[%p], recvCounts[%p], sdispls[%p], rdispls[%p], sendType[%s], recvType[%s], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, sendCounts, recvCounts, sdispls, rdispls, GetDataTypeEnumStr(sendType).c_str(),
            GetDataTypeEnumStr(recvType).c_str(), streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVCEntryLog(const void *sendBuf, const void *recvBuf, const void *sendCountMatrix,
    HcclDataType sendType, HcclDataType recvType, aclrtStream stream, const std::string &tag, const std::string &opName)
{
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], sendCountMatrix[%p], sendType[%s], recvType[%s], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, sendCountMatrix, GetDataTypeEnumStr(sendType).c_str(), GetDataTypeEnumStr(recvType).c_str(),
            streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}

}