/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_v_op.h"
#include "op_common_ops.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts,
    const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclAllGatherV");
    if (GetHcommVersion() < 90000000) { // compat handle
        return HcclAllGatherVInner(sendBuf, sendCount, recvBuf, recvCounts, recvDispls, dataType, comm, stream);
    }
 
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95) {
    #endif
        return HcclAllGatherVInner(sendBuf, sendCount, recvBuf, recvCounts, recvDispls, dataType, comm, stream);
    }
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    CHK_RET(InitEnvConfig());
    // 参数校验等工作
 	CHK_RET(CheckAllGatherVInputPara(comm, recvBuf, recvCounts, recvDispls, stream));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    const u64* recvCountsAddr = reinterpret_cast<const u64*>(recvCounts);
    CHK_PRT_RET(std::all_of(recvCountsAddr, recvCountsAddr + rankSize, [](auto count) { return count == 0; }),
            HCCL_WARNING("input all %u elements in recvCounts are 0, return success", rankSize),
            HCCL_SUCCESS);
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string tag = "AllGatherV_" + string(commName);
    // CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), sendCount, dataType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());
    CHK_RET(CheckCount(sendCount));
    CHK_RET(CheckDataType(dataType, false));
            
    CHK_RET(AllGatherVEntryLog(sendBuf, recvBuf, sendCount, recvCounts, recvDispls, dataType, stream, tag, "HcclAllGatherV"));

    // 执行AllGatherV
    CHK_RET_AND_PRINT_IDE(AllGatherVOutPlace(sendBuf, recvBuf, sendCount, recvCounts, recvDispls, dataType, comm, stream, tag), tag.c_str());

    CHK_RET(LogHcclExit("HcclAllGatherV", tag.c_str(), startut));

    return HCCL_SUCCESS;
}
 
HcclResult HcclAllGatherVGraphMode(void *sendBuf, void *recvBuf, uint64_t sendCount, const void *recvCounts, const void *recvDispls, 
 	HcclDataType dataType, const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
 	HCCL_INFO("Start to run execute HcclAllGatherVGraphMode");
 	// 根据group获取通信域
 	HcclComm comm = nullptr;
    HCCL_INFO("[HcclAllGatherVGraphMode] get group name: %s", group);
 	CHK_RET(HcomGetCommHandleByGroup(group, &comm));
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
 	// 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    CHK_RET(InitEnvConfig());
 	// 检查入参指针有效性
 	CHK_RET(CheckAllGatherVInputPara(comm, recvBuf, recvCounts, recvDispls, stream));
 	// tag有效性,是否过长
 	char commName[COMM_INDENTIFIER_MAX_LENGTH];
 	CHK_RET(HcclGetCommName(comm, commName));
 	const string opTag = "AllGatherV_" + string(commName);
 	CHK_RET(HcclCheckTag(opTag.c_str()));
 	CHK_RET(HcclCheckTag(tag));
 	// 检查sendCount是否合法(超出系统上限)
 	CHK_RET(CheckCount(sendCount));
 	// 检查数据类型是否支持
 	CHK_RET(CheckDataType(dataType, false));
 	// 检查rank有效性，是否超出rankSize
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    const u64* recvCountsAddr = reinterpret_cast<const u64*>(recvCounts);
    CHK_PRT_RET(std::all_of(recvCountsAddr, recvCountsAddr + rankSize, [](auto count) { return count == 0; }),
        HCCL_WARNING("input all %u elements in recvCounts are 0, return success", rankSize), 
        HCCL_SUCCESS);
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), opTag.c_str());
 	  	 
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

 	CHK_RET(AllGatherVEntryLog(sendBuf, recvBuf, sendCount, recvCounts, recvDispls, dataType, stream, opTag, "HcclAllGatherVGraphMode"));  	 
 	// 执行AllGatherV
 	CHK_RET_AND_PRINT_IDE(AllGatherVOutPlaceGraphMode(sendBuf, recvBuf, sendCount, recvCounts, recvDispls, dataType, comm, stream, tag, resPack), opTag);
 	CHK_RET(LogHcclExit("HcclAllGatherVGraphMode", opTag.c_str(), startut));

 	return HCCL_SUCCESS;
}
 	
namespace ops_hccl {
HcclResult CheckAllGatherVInputPara(const HcclComm comm, const void* recvBuf, const void *recvCounts, const void *recvDispls, const aclrtStream stream)
{
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
                  std::vector<std::string>({"HcclAllGatherV", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
                  std::vector<std::string>({"HcclAllGatherV", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);
    RPT_INPUT_ERR(recvCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAllGatherV", "nullptr", "recvCounts", "non-null pointer"}));
    CHK_PTR_NULL(recvCounts);
    RPT_INPUT_ERR(recvDispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAllGatherV", "nullptr", "recvDispls", "non-null pointer"}));
    CHK_PTR_NULL(recvDispls);
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAllGatherV", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    return HCCL_SUCCESS;
}
 
HcclResult AllGatherVOutPlace(void *sendBuf, void *recvBuf, uint64_t sendCount,const void *recvCounts,const void *recvDispls,
    HcclDataType dataType, HcclComm comm, aclrtStream stream, const std::string &tag)
{
    HCCL_INFO("Start to execute AllGatherVOutPlace");
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));
    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 inputSize = sendCount * perDataSize;    // all gather v 每个rank上一份数据
    u64 outputSize = 0;  
    const u64 *u64RecvCount = reinterpret_cast<const u64 *>(recvCounts);
    for (u64 i = 0; i < userRankSize; i++) {
        outputSize += u64RecvCount[i] * perDataSize;
    }  // 结果为recvCount中的数据之和

    // 申请OpParam参数结构体内存
    u64 varMemSize = (userRankSize + userRankSize) * sizeof(u64);
    void* paramMem = malloc(sizeof(OpParam) + varMemSize);
    if (!paramMem) {
        // 内存分配失败
        HCCL_ERROR("malloc OpParam failed!");
        return HCCL_E_INTERNAL;
    }
    OpParam* paramPtr = new (paramMem) OpParam();
    OpParam& param = *paramPtr;
    CHK_RET(HcclGetCommName(comm, param.commName));
    param.opMode = OpMode::OPBASE;
    param.stream = stream;
 
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
 
    // topoInfo的tag，所有相同的算子可以共享
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
    if (ret <= 0) {
        HCCL_ERROR("failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }
 
    // 参数准备
    param.outputSize = outputSize;
    param.DataDes.count = sendCount;
    param.vDataDes.dataType = dataType;
    param.inputPtr = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    const void *temp = recvCounts;
    param.vDataDes.counts = const_cast<void*>(temp);

    // 带V算子的参数
    param.varMemSize = varMemSize;
    // 从源内存地址按字节直接拷贝数据到目标地址
    std::vector<u64> merged(userRankSize + userRankSize); 
    const uint64_t *countsPtr = reinterpret_cast<const uint64_t *>(recvCounts);
    const uint64_t *displsPtr = reinterpret_cast<const uint64_t *>(recvDispls);
    std::copy(countsPtr, countsPtr + userRankSize, merged.begin());
    std::copy(displsPtr, displsPtr + userRankSize, merged.begin() + userRankSize);
    memcpy_s(param.varData, varMemSize, merged.data(), varMemSize);
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER_V;
    param.enableDetour = false;
    param.deviceType = deviceType;
    if (userRankSize == 1) {
 	  	HCCL_WARNING("[%s] rankSize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
 	  	return HcclResult::HCCL_SUCCESS;
 	}

    CHK_RET(HcclGetOpExpansionMode(comm, param));
    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(Selector(comm, param, topoInfo, algName));
    if (ShouldUseInnerOp(param.opExecuteConfig) && param.opMode == OpMode::OPBASE) {
        return HcclAllGatherVInner(sendBuf, sendCount, recvBuf, recvCounts, recvDispls, dataType, comm, stream);
    }
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName));
    paramPtr->~OpParam();
    free(paramMem);
    HCCL_INFO("Execute AllGatherVOutPlace success.");
    return HCCL_SUCCESS;
}

HcclResult AllGatherVEntryLog(void *sendBuf, void *recvBuf, uint64_t sendCount, const void *recvCounts, const void *recvDispls,
    HcclDataType dataType, aclrtStream stream, const std::string &tag, const std::string &opName)
{
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], recvCounts[%p], recvDispls[%p], dataType[%s], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, sendCount, recvCounts, recvDispls, GetDataTypeEnumStr(dataType).c_str(), streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}
HcclResult AllGatherVOutPlaceGraphMode(void *sendBuf, void *recvBuf, uint64_t sendCount, const void *recvCounts,const void *recvDispls, HcclDataType dataType, HcclComm comm, 
    aclrtStream stream, const std::string &tag, const ResPackGraphMode &resPack)
{
 	HCCL_INFO("Start to execute AllGatherVOutPlaceGraphMode");
 	u32 userRankSize;
 	CHK_RET(HcclGetRankSize(comm, &userRankSize));
 	  	 
 	u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
 	u64 inputSize = sendCount * perDataSize;    // all gather v 每个rank上一份数据
 	u64 outputSize = 0;  
    const u64 *u64RecvCount = reinterpret_cast<const u64 *>(recvCounts);
    const u64 *u64RecvDispls = reinterpret_cast<const u64 *>(recvDispls);
    for (u64 i = 0; i < userRankSize; i++) {
        outputSize = (outputSize > (u64RecvDispls[i] + u64RecvCount[i]) * perDataSize) ? outputSize : (u64RecvDispls[i] + u64RecvCount[i]) * perDataSize;
    }// 结果为最大的displs加recvcount 	 
 	// 申请OpParam参数结构体内存
 	u64 varMemSize = (userRankSize + userRankSize) * sizeof(u64);
 	void* paramMem = malloc(sizeof(OpParam) + varMemSize);
 	if (!paramMem) {
 	    // 内存分配失败
 	    HCCL_ERROR("malloc OpParam failed!");
 	    return HCCL_E_INTERNAL;
 	} 
 	OpParam* paramPtr = new (paramMem) OpParam();
 	OpParam& param = *paramPtr; 
    CHK_RET(HcclGetCommName(comm, param.commName));
 	  	 
    DevType deviceType = DevType::DEV_TYPE_COUNT;
 	CHK_RET(hrtGetDeviceType(deviceType));
 	  	 
 	// topoInfo的tag，所有相同的算子可以共享
 	int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
 	if (ret <= 0) {
 	  	HCCL_ERROR("failed to fill param.tag");
 	  	return HCCL_E_INTERNAL;
 	}
 	  	 
 	// 参数准备
    const void *temp = recvCounts;
 	param.stream = stream, param.opMode = OpMode::OFFLOAD, param.inputPtr = sendBuf, param.inputSize = inputSize, param.outputPtr = recvBuf, param.outputSize = outputSize, param.DataDes.count = sendCount, param.vDataDes.dataType = dataType, param.varMemSize = varMemSize, param.vDataDes.counts = const_cast<void*>(temp);
    // 从源内存地址按字节直接拷贝数据到目标地址
    std::vector<u64> merged(userRankSize + userRankSize); 
    const uint64_t *countsPtr = reinterpret_cast<const uint64_t *>(recvCounts);
    const uint64_t *displsPtr = reinterpret_cast<const uint64_t *>(recvDispls);
    std::copy(countsPtr, countsPtr + userRankSize, merged.begin());
    std::copy(displsPtr, displsPtr + userRankSize, merged.begin() + userRankSize);
    memcpy_s(param.varData, varMemSize, merged.data(), varMemSize);
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER_V, param.enableDetour = false, param.deviceType = deviceType;
 	if (userRankSize == 1) {
 	  	HCCL_WARNING("[%s] rankSize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
 	  	return HcclResult::HCCL_SUCCESS;
 	}
 	std::string algName;
 	std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(HcclGetOpExpansionMode(comm, param));
 	CHK_RET(Selector(comm, param, topoInfo, algName));
 	CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));
 	HCCL_INFO("Execute AllGatherVOutPlaceGraphMode success.");
 	return HCCL_SUCCESS;
}

}  // namespace ops_hccl