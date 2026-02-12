/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef HCCL_AIV_UTILS_H
#define HCCL_AIV_UTILS_H
 
#include "string"
#include <array>
#include "hccl_types.h"
#include "acl/acl_rt.h"
#include "alg_param.h"

namespace ops_hccl {
constexpr u32 MAX_RANK_SIZE = 8; // 注意要和device侧的一致
constexpr u32 MAX_NUM_BLOCKS = 56; // 56-72
 
constexpr s32 TAG_INIT_VALUE = 1;
constexpr s32 TAG_RESET_COUNT = 1000;
constexpr s32 TOPO_LEN = 32;

constexpr u32 AIV_TAG_MOVE_LEFT_BITS = 16;
constexpr u32 AIV_TAG_ADDR_OFFSET = 16 * 1024;
constexpr u32 AIV_TOPO_ADDR_OFFSET = 32 * 1024;
constexpr u32 AIV_TOPO_BUFF_LEN = 8 * 1024;
constexpr u32 AIV_FLAG_ADDR_OFFSET = 40 * 1024;
constexpr u32 AIV_FLAG_AREA_SIZE = 1000 * 1024;
constexpr u32 AIV_FLAG_CLEAR_OFFSET = 1040 * 1024;
constexpr u32 AIV_TAG_BUFF_LEN = 2 * 1024 * 1024;
constexpr u32 AIV_LOW_16_BITS = 0xFFFF;

constexpr u32 AIV_ATTRNUM_THREE = 3;

using AivCountTagArray = std::array<s32, MAX_RANK_SIZE>;

enum class KernelArgsType {
    ARGS_TYPE_SERVER = 0, // kernel参数为单机内
    ARGS_TYPE_TWO_SHOT = 1,
    ARGS_TYPE_DEFAULT
};

using AivKernelInfo = struct AivKernelInfoDef {
    const char* kernelName;
    HcclDataType dataType;
    KernelArgsType argsType;

    AivKernelInfoDef(const char* kernelName, HcclDataType dataType,
        KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER)
        : kernelName(kernelName), dataType(dataType), argsType(argsType)
    {
    }
};

// 非均匀算子AlltoAllV/AlltoAllVC/AllGatherV/ReduceScatterV需要的额外参数信息，A3场景
struct ExtraArgs {
    u64 sendCounts[MAX_RANK_SIZE] = {};
    u64 sendDispls[MAX_RANK_SIZE] = {};
    u64 recvCounts[MAX_RANK_SIZE] = {};
    u64 recvDispls[MAX_RANK_SIZE] = {};
};

// 算子计数信息
struct OpCounterInfo {
    u64 headCountMem = 0;
    u64 tailCountMem = 0;
    u64 addOneMem = 0;
    u32 memSize = 0;
    bool isEnableCounter = false;
};
 
// 表示算子属性的参数，相对固定
struct AivOpArgs {
    HcclCMDType cmdType = HcclCMDType::HCCL_CMD_MAX;
    std::string comm = {};
    u32 numBlocks = MAX_NUM_BLOCKS;
    rtStream_t stream = nullptr;
    uint64_t beginTime = 0;
    OpCounterInfo counter = {}; 
    void* buffersIn = nullptr;
    u64 input = 0;
    u64 output = 0;
    u32 rank = 0;
    u32 rankSize = 0;
    u64 xRankSize = 0;
    u64 yRankSize = 0;
    u64 zRankSize = 0;
    u64 count = 0;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32; 
    HcclReduceOp op = HcclReduceOp::HCCL_REDUCE_SUM;
    u32 root = 0;
    u32 aivCountTag = 0;
    u64 inputSliceStride = 0;
    u64 outputSliceStride = 0;
    u64 repeatNum = 0;
    u64 inputRepeatStride = 0;
    u64 outputRepeatStride = 0;
    bool isOpBase = false;
    ExtraArgs extraArgs = {}; 
    uint64_t topo_[TOPO_LEN] = {0}; 
    AivOpArgs() {};
    KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER;
};

using AivSuperKernelArgs = struct AivSuperKernelArgsDef {
    const void* buffersIn = nullptr; // 注册的CCLIN地址，所有卡可访问
    u64 rank{};
    u64 rankSize{};
    u64 len{};
    u64 dataType{};
    u64 unitSize{};
    u64 reduceOp{};
    u64 numBlocks{};
    s64 tag{}; // 第几次调用，定时重置成1
    s64 clearEnable{};
    uint64_t inputSliceStride{};
    uint64_t outputSliceStride{};
    uint64_t repeatNum{};
    uint64_t inputRepeatStride{};
    uint64_t outputRepeatStride{};
    u64 input{};
    u64 output{};
    u64 cclBufferSize{};
    AivSuperKernelArgsDef(u64 input, u64 output, u32 rank,
        u32 rankSize, u64 len, u32 dataType, u64 unitSize, u32 reduceOp,u32 numBlocks = 0, s32 tag = 0, bool clearEnable = true,
        uint64_t inputSliceStride = 0, uint64_t outputSliceStride = 0, uint64_t repeatNum = 0,
        uint64_t inputRepeatStride = 0, uint64_t outputRepeatStride = 0, u64 cclBufferSize = 0)
        : rank(rank), rankSize(rankSize), len(len), dataType(dataType), unitSize(unitSize), 
          reduceOp(reduceOp), numBlocks(numBlocks),tag(tag),
          clearEnable(clearEnable), inputSliceStride(inputSliceStride), outputSliceStride(outputSliceStride),
          repeatNum(repeatNum), inputRepeatStride(inputRepeatStride), outputRepeatStride(outputRepeatStride),
          input(input), output(output), cclBufferSize(cclBufferSize)
    {
    }
    AivSuperKernelArgsDef() {}
};

HcclResult RegisterKernel(HcclCMDType cmdType, const std::string &aivBinaryName, const std::vector<AivKernelInfo> &aivKernelInfoList);

HcclResult UnRegisterAivKernel();

HcclResult GetAivCountTag(const std::string &commTag, u32 rank, s32 &aivCountTag);

HcclResult ClearAivSyncBuf(const OpParam &param, AlgResourceCtxSerializable& resCtx);

HcclResult ExecuteKernelLaunchInner(const AivOpArgs &opArgs, void* args, u32 argsSize);
 
HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs);
}
 
#endif // HCCL_AIV_UTILS_H