/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_P2P_COMMON_H
#define OPS_HCCL_P2P_COMMON_H

#include <vector>
#include <memory>
#include <unordered_map>

#include <acl/acl_rt.h>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>
#include <hccl/hcomm_primitives.h>
#include <hccl/hccl_rank_graph.h>
#include <ccu/ccu_types.h>
#include <ccu/ccu_res_defs.h>
#include <ccu/ccu_variable.hpp>
#include <ccu/ccu_event.hpp>
#include <ccu/ccu_primitives.hpp>

#include "binary_stream.h"

using CcuKernelHandle = uint64_t;

constexpr uint32_t INVALID_VALUE_RANKID = 0xFFFFFFFF; // rank id非法值
constexpr uint32_t NOTIFY_IDX_ACK = 0;
constexpr uint32_t NOTIFY_IDX_DATA_SIGNAL = 1;
constexpr uint32_t CUSTOM_TIMEOUT = 1800;

constexpr uint32_t COMM_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 32;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_INDENTIFIER_MAX_LENGTH;
constexpr uint32_t ALG_TAG_LENGTH = TAG_LENGTH + 128;
constexpr uint32_t OP_ALG_LENGTH = 128;
constexpr uint64_t CCU_MAX_RANK_SIZE = 16;

constexpr uint32_t CCU_DIE_NUM = 2; // CCU IO Die 数量

// 算子执行模式
enum class OpMode {
    OPBASE = 0,
    OFFLOAD = 1,
    ACLGRAPH = 2
};

// 设备类型
enum DeviceType {
    DEVICE_TYPE_A2 = 0,
    DEVICE_TYPE_A3 = 1,
    DEVICE_TYPE_A5 = 2,
};

// 算法类型
enum class AlgType {
    ALG_TYPE_MESH_1D = 0,
    ALG_TYPE_NHR = 1,
    ALG_TYPE_RESERVED
};

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
};

struct CcuKernelArgBase {
    ChannelHandle channels[CCU_MAX_RANK_SIZE];
    uint32_t      channelCount;
};

// ccu kernel register所需信息
struct CcuKernelInfo {
    // kernel名称
    char kernelFuncName[64];
    // kernel函数
    void* kernelFunc;
    // KernelArg实例指针
    void *kernelArg;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template<typename T>
    void setKernelArg(std::shared_ptr<T> arg) {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void*>(arg.get());
    }
};

struct AlgResourceCtxSerializable {
    CommBuffer cclMem;
    uint32_t notifyNumOnMainThread;
    std::vector<ThreadHandle> threads;
    // ccu
    std::vector<uint32_t> ccuKernelNum;
    std::vector<CcuKernelHandle> ccuKernels;

     std::vector<char> Serialize()
    {
        BinaryStream binaryStream;

        binaryStream << cclMem;
        binaryStream << notifyNumOnMainThread;
        binaryStream << threads;
        binaryStream << ccuKernelNum;
        binaryStream << ccuKernels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);

        binaryStream >> cclMem;
        binaryStream >> notifyNumOnMainThread;
        binaryStream >> threads;
        binaryStream >> ccuKernelNum;
        binaryStream >> ccuKernels;
    }

};

// 算子参数
struct OpParam {
    void* hcclComm;
    char tag[TAG_LENGTH];
    char algTag[ALG_TAG_LENGTH];
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    aclrtStream stream;
    void* inputPtr = nullptr;
    uint64_t inputSize = 0;
    void* outputPtr = nullptr;
    uint64_t outputSize = 0;
    OpMode opMode;
    CommEngine engine = CommEngine::COMM_ENGINE_RESERVED;
    AlgType algType;
    char algName[OP_ALG_LENGTH];
    uint64_t count;
    HcclDataType dataType;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    uint64_t ctxSize = 0;
    void* resCtx = nullptr;
    uint32_t myRank = 0;
    uint32_t rankSize = 0;
    DeviceType devType = DEVICE_TYPE_A5;
};

constexpr uint32_t SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

// CCU返回码转换为HCCL返回码
inline HcclResult ConvertCcuToHccl(CcuResult ccuResult) {
    switch (ccuResult) {
        case CCU_SUCCESS: return HCCL_SUCCESS;
        case CCU_E_PARA: return HCCL_E_PARA;
        case CCU_E_PTR: return HCCL_E_PTR;
        case CCU_E_INTERNAL: return HCCL_E_INTERNAL;
        case CCU_E_NOT_SUPPORT: return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND: return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL: return HCCL_E_UNAVAIL;
        default:
            return HCCL_E_INTERNAL;
    }
}

// 以总资源量的1/5预设默认资源数量
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_ADDRESS = 400;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_LOOP = 16;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_BUF = 128;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_VARIABLE = 400;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_EVENT = 48;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_THREAD = 2;

#endif // OPS_HCCL_P2P_COMMON_H