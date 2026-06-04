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
#include <memory>
#include "hccl_types.h"
#include "acl/acl_rt.h"
#include "alg_param.h"

namespace ops_hccl {
constexpr u32 MAX_RANK_SIZE = 128; // 注意要和device侧的一致
 
constexpr s32 TOPO_LEN = 128;

constexpr u32 AIV_TAG_ADDR_OFFSET = 16 * 1024;
constexpr u32 AIV_TOPO_ADDR_OFFSET = 32 * 1024;
constexpr u32 AIV_TOPO_BUFF_LEN = 8 * 1024;
constexpr u32 AIV_FLAG_ADDR_OFFSET = 40 * 1024;
constexpr u32 AIV_FLAG_AREA_SIZE = 1000 * 1024;
constexpr u32 AIV_TAG_BUFF_LEN = 33 * 1024 * 1024;

constexpr u32 AIV_MAX_CCL_LOOP_NUM = 16;

constexpr u32 AIV_ATTRNUM_THREE = 3;

constexpr u32 AIV_CACHE_CTX_TAG_MAX_LENGTH = 64;
constexpr u32 AIV_CACHE_INDEX_MAX_ENTRY = 8192;
constexpr u32 AIV_CACHE_INDEX_CLEAR_PERCENT = 20;
constexpr u64 FNV_OFFSET_BASIS = 14695981039346656037ULL;
constexpr u64 FNV_PRIME = 1099511628211ULL;
constexpr char AIV_CACHE_INDEX_CTX_TAG[] = "AivCacheIndex";

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
    u32 sendRecvRemoteRank = 0;
    u32 rankSize = 0;
    u64 xRankSize = 0;
    u64 yRankSize = 0;
    u64 zRankSize = 0;
    u64 count = 0;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32; 
    HcclReduceOp op = HcclReduceOp::HCCL_REDUCE_SUM;
    u32 root = 0;
    u32 sliceId = 0;
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

// AIV Cache Definitions
struct AivOpCacheArgs {
    std::string commName;
    std::string algName;
    u64 count;
    HcclDataType dataType;
    HcclCMDType opType;
    HcclReduceOp reduceOp;
    u32 root;
    // For AlltoAll
    HcclDataType sendType;
    HcclDataType recvType;
    u64 sendCount;
    u64 recvCount;

    bool operator<(const AivOpCacheArgs& other) const {
        if (commName != other.commName) return commName < other.commName;
        if (algName != other.algName) return algName < other.algName;
        if (count != other.count) return count < other.count;
        if (dataType != other.dataType) return dataType < other.dataType;
        if (opType != other.opType) return opType < other.opType;
        if (reduceOp != other.reduceOp) return reduceOp < other.reduceOp;
        if (root != other.root) return root < other.root;
        if (sendType != other.sendType) return sendType < other.sendType;
        if (recvType != other.recvType) return recvType < other.recvType;
        if (sendCount != other.sendCount) return sendCount < other.sendCount;
        return recvCount < other.recvCount;
    }
};

struct AivCacheCtxHeader {
    u64 keyHash;
    u32 insCount;
    // insCount个instruction
};

struct AivCacheIndexCtx {
    u32 head = 0; // 起始位置
    u32 tail = 0; // 尾巴位置
    u32 size = 0; // 当前元素数
    char ctxTags[AIV_CACHE_INDEX_MAX_ENTRY][AIV_CACHE_CTX_TAG_MAX_LENGTH];
};

struct AivInstruction {
    AivOpArgs opArgs;
    u64 inputOffset;
    u64 outputOffset;
};

using InsQueue = std::vector<AivInstruction>;

extern thread_local std::shared_ptr<InsQueue> g_recordingQueue;
extern thread_local bool g_recordOnlyMode;
extern thread_local u64 g_baseInputAddr;
extern thread_local u64 g_baseOutputAddr;

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

HcclResult RegisterKernel();

HcclResult UnRegisterAivKernel();

HcclResult ExecuteKernelLaunchInner(const AivOpArgs &opArgs, void* args, u32 argsSize);
 
HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs);

u64 CalcAivCacheKeyHash(const AivOpCacheArgs &cacheKey);

HcclResult BuildAivCacheCtxTag(u64 keyHash, std::string &ctxTag);

HcclResult GetOrCreateAivCacheIndexCtx(HcclComm comm, AivCacheIndexCtx **indexCtx);

HcclResult EvictAivCacheIfNeeded(HcclComm comm, AivCacheIndexCtx *indexCtx);

HcclResult ReplayAivCacheCtx(HcclComm comm, const std::string &ctxTag, u64 keyHash, OpParam &param, bool &cacheHit);

HcclResult StoreAivCacheCtx(HcclComm comm, const std::string &ctxTag, u64 keyHash, AivCacheIndexCtx *indexCtx);
}
 
#endif // HCCL_AIV_UTILS_H