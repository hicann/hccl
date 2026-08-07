/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_ALLGATHER_COMMON_H
#define OPS_HCCL_ALLGATHER_COMMON_H

#include <vector>

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>
#include <hccl/hcomm_primitives.h>
#include <acl/acl_rt.h>
#include "binary_stream.h"

constexpr uint32_t NOTIFY_IDX_ACK = 0;
constexpr uint32_t NOTIFY_IDX_DATA_SIGNAL = 1;
constexpr uint32_t CUSTOM_TIMEOUT = 1836;

constexpr uint32_t COMM_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 32;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_INDENTIFIER_MAX_LENGTH;
constexpr uint32_t INVALID_VALUE_RANKID = 0xFFFFFFFF;
constexpr uint32_t AICPU_CONTROL_NOTIFY_NUM = 2;
using namespace ops_hccl_allgather;
// 设备类型
enum DeviceType {
    DEVICE_TYPE_A2 = 0,
    DEVICE_TYPE_A3 = 1,
    DEVICE_TYPE_A5 = 2,
};

typedef struct {
    void* addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
};

struct AlgResourceCtx {
    CommBuffer cclMem;
    uint32_t notifyNumOnMainThread;
    uint32_t slaveThreadNum;
    std::vector<uint32_t> notifyNumPerThread;
    std::vector<ThreadHandle> threads;
    std::vector<ChannelInfo> channels;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;

        binaryStream << cclMem;
        binaryStream << notifyNumOnMainThread;
        binaryStream << slaveThreadNum;
        binaryStream << notifyNumPerThread;
        binaryStream << threads;
        binaryStream << channels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char>& data)
    {
        BinaryStream binaryStream(data);

        binaryStream >> cclMem;
        binaryStream >> notifyNumOnMainThread;
        binaryStream >> slaveThreadNum;
        binaryStream >> notifyNumPerThread;
        binaryStream >> threads;
        binaryStream >> channels;
    }
};

struct OpParam {
    char tag[TAG_LENGTH];
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    void* inputPtr = nullptr;
    void* outputPtr = nullptr;
    uint64_t count = 0;
    DeviceType devType;
    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    ThreadHandle cpuThread;
    ThreadHandle aicpuThreadOnCpu;
    // host cpu thread导出到aicpu引擎的句柄，与stream绑定，每次调用刷新，不随context复用
    ThreadHandle cpuThreadOnAicpu = 0;
    uint32_t aicpuRecordCpuIdx = 0;
    void* resCtxDevice = nullptr;
    uint64_t ctxSize = 0;
};

constexpr uint32_t SIZE_TABLE[HCCL_DATA_TYPE_RESERVED]
    = {sizeof(int8_t),
       sizeof(int16_t),
       sizeof(int32_t),
       2,
       sizeof(float),
       sizeof(int64_t),
       sizeof(uint64_t),
       sizeof(uint8_t),
       sizeof(uint16_t),
       sizeof(uint32_t),
       8,
       2,
       16,
       2,
       1,
       1,
       1,
       1};

#endif // OPS_HCCL_ALLGATHER_COMMON_H
