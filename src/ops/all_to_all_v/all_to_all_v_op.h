/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_ALL_TO_ALL_V_OP_H
#define OPS_HCCL_SRC_OPS_ALL_TO_ALL_V_OP_H

#include <string>
#include <memory>
#include "hccl.h"

#include "alg_param.h"
#include "executor_v2_base.h"
#include "alg_type.h"
#include "execute_selector.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
    uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream);
HcclResult HcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
    const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType, HcclComm comm, aclrtStream stream);
HcclResult HcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, HcclComm comm, aclrtStream stream);
HcclResult HcclAlltoAllGraphMode(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
    uint64_t recvCount, HcclDataType recvType, const char* group, aclrtStream stream, const char* tag,
    void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize);
HcclResult HcclAlltoAllVGraphMode(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
    const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType, const char* group, aclrtStream stream, const char* tag,
    void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize);
HcclResult HcclAlltoAllVCGraphMode(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, const char* group, aclrtStream stream, const char* tag,
    void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize);
#ifdef __cplusplus
}
#endif

namespace ops_hccl {
constexpr u64 SEND_COUNT_IDX = 0;
constexpr u64 RECV_COUNT_IDX = 1;
constexpr u64 SEND_DISPL_IDX = 2;
constexpr u64 RECV_DISPL_IDX = 3;
HcclResult ConvertAlltoAllParam(const u64 recvCount, const u32 rankSize, std::vector<u64> &sdispls, std::vector<u64> &rdispls);
HcclResult ConvertAlltoAllVCParam(const u32 rankSize, const u32 userRank, const void *sendCountMatrix,
    std::vector<u64> &sendCounts, std::vector<u64> &recvCounts, std::vector<u64> &sdispls, std::vector<u64> &rdispls);
HcclResult GenResPack(const char* tag, void** streams, const size_t streamCount,
    void* scratchMemAddr, const uint64_t scratchMemSize, ResPackGraphMode &resPack);
HcclResult CalcInputOutputSize(const u64* sendCountsData, const u64* recvCountsData,
    const u64* sdisplsData, const u64* rdisplsData, const u32 userRankSize, u64 &inputSize, u64 &outputSize);
HcclResult ContructVarData(const u64* sendCountsData, const u64* recvCountsData, const u64* sdisplsData,
    const u64* rdisplsData, const u32 userRankSize, const u32 rankSize, OpParam &param);
HcclResult CheckAlltoAllInputPara(const HcclComm comm, const void *sendBuf, const uint64_t sendCount,
    const HcclDataType sendType, const void *recvBuf, const uint64_t recvCount,
    const HcclDataType recvType, const aclrtStream stream);
HcclResult CheckAlltoAllVInputPara(const HcclComm comm, const void *sendBuf, const void *sendCounts, const void *sdispls,
    const HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls,
    const HcclDataType recvType, const aclrtStream stream);
HcclResult CheckAlltoAllVCInputPara(const HcclComm comm, const void *sendBuf, const void *sendCountMatrix,
    const HcclDataType sendType, const void *recvBuf, const HcclDataType recvType, const aclrtStream stream);
HcclResult CheckBufNullptr(const u64* countsData, u32 rankSize, const void* buf, const std::string funcName,
    const std::string bufName);
HcclResult AlltoAllVConstructOpParam(const void *sendBuf, const void *sendCounts, const void *sdispls, const void *recvBuf,
    const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclComm comm, aclrtStream stream,
    const std::string &tag, HcclCMDType opType, u32 rankSize, OpMode opMode, u64 varMemSize, OpParam &param);
HcclResult AlltoAllVOutPlaceCommon(const void *sendBuf, const void *sendCounts, const void *sdispls, const void *recvBuf,
    const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclComm comm, aclrtStream stream,
    const std::string &tag, HcclCMDType opType, u32 rankSize, bool &useInnerOp, OpMode opMode, const ResPackGraphMode &resPack);
HcclResult AlltoAllVOutPlace(const void *sendBuf, const void *sendCounts, const void *sdispls, const void *recvBuf,
    const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclComm comm, aclrtStream stream,
    const std::string &tag, HcclCMDType opType, u32 rankSize, bool &useInnerOp);
HcclResult AlltoAllVOutPlaceGraphMode(const void *sendBuf, const void *sendCounts, const void *sdispls, const void *recvBuf,
    const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclComm comm, aclrtStream stream,
    const std::string &tag, HcclCMDType opType, u32 rankSize, const ResPackGraphMode &resPack);
HcclResult AlltoAllEntryLog(const void *sendBuf, const void *recvBuf, uint64_t sendCount, uint64_t recvCount,
    HcclDataType sendType, HcclDataType recvType, aclrtStream stream, const std::string &tag, const std::string &opName);
HcclResult AlltoAllVEntryLog(const void *sendBuf, const void *recvBuf, const void *sendCounts, const void *recvCounts,
    const void *sdispls, const void *rdispls, HcclDataType sendType, HcclDataType recvType, aclrtStream stream,
    const std::string &tag, const u32 totalRanks, const std::string &opName);
HcclResult AlltoAllVCEntryLog(const void *sendBuf, const void *recvBuf, const void *sendCountMatrix,
    HcclDataType sendType, HcclDataType recvType, aclrtStream stream, const std::string &tag, const std::string &opName);

}

#endif