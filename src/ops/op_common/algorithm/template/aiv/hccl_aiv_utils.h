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
#include <cstddef>
#include <memory>
#include <mutex>
#include "hccl_types.h"
#include "acl/acl_rt.h"
#include "alg_param.h"

namespace ops_hccl {
constexpr u32 MAX_RANK_SIZE_V = 256; // 注意要和device侧的一致
constexpr u32 MAX_RANK_SIZE = 1024;  // 对端对称 VA 数组容量，注意要和device侧的一致

constexpr u32 AIV_TAG_ADDR_OFFSET = 16 * 1024;
constexpr u64 AIV_SYM_PEER_OUT_ADDR_OFFSET = 8 * 1024; // 注意要和device侧的一致
constexpr u64 AIV_SYM_PEER_IN_ADDR_OFFSET = 24 * 1024; // 注意要和device侧的一致
constexpr u32 AIV_TOPO_ADDR_OFFSET = 32 * 1024;
constexpr u32 AIV_TOPO_BUFF_LEN = 8 * 1024;
constexpr u32 AIV_FLAG_ADDR_OFFSET = 1 * 1024 * 1024;
constexpr u32 AIV_FLAG_CLEAR_OFFSET = 512 * 1024;
constexpr u32 AIV_FLAG_AREA_SIZE = 1000 * 1024;
constexpr u32 AIV_TAG_BUFF_LEN = 65 * 1024 * 1024;
constexpr u32 AIV_MAX_CCL_LOOP_NUM = 16;

// AIV_URMA 内存布局常量（host 侧）
constexpr u32 AIV_URMA_HANDLE_ADDR_OFFSET = 32 * 1024;             // 32KB
constexpr u32 AIV_URMA_SUBCOMM_RANKS_START_OFFSET = 256 * 1024;    // 256KB
constexpr u32 AIV_URMA_SUBCOMM_RANKS_PER_LEVEL_BYTES = 16 * 1024;  // 每级 16KB
constexpr u32 AIV_URMA_SUBCOMM_RANKS_AREA_END = 512 * 1024;        // 512KB（TAG 区起始）
constexpr u32 AIV_URMA_OUTPUT_ADDR_SLOT_OFFSET = 50 * 1024 * 1024; // 50MB
constexpr u32 MAX_CHANNELS_PER_RANK = 8;                           // 每对端最大 channel 数

// URMA channel 信息结构体（POD，Host 和 device 侧各定义一份，布局需一致）
struct AivUrmaChannelInfo {
    u32 rankSize = 0;
    u32 channelsPerRank[MAX_RANK_SIZE] = {};
    u64 channelHandle[MAX_RANK_SIZE * MAX_CHANNELS_PER_RANK] = {};
    u32 portNum[MAX_RANK_SIZE * MAX_CHANNELS_PER_RANK] = {};
};

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
    ARGS_TYPE_URMA = 2,
    ARGS_TYPE_DEFAULT
};

using AivKernelInfo = struct AivKernelInfoDef {
    const char* kernelName;
    HcclDataType dataType;
    KernelArgsType argsType;

    AivKernelInfoDef(
        const char* kernelName, HcclDataType dataType, KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER)
        : kernelName(kernelName),
          dataType(dataType),
          argsType(argsType)
    {}
};

// 非均匀算子AlltoAllV/AlltoAllVC/AllGatherV/ReduceScatterV需要的额外参数信息，A3场景
struct ExtraArgs {
    u64 sendCounts[MAX_RANK_SIZE_V] = {};
    u64 sendDispls[MAX_RANK_SIZE_V] = {};
    u64 recvCounts[MAX_RANK_SIZE_V] = {};
    u64 recvDispls[MAX_RANK_SIZE_V] = {};
};

// BatchSendRecv 单次 kernel 下发的最大任务条数，必须与 device 侧 BATCH_SEND_RECV_ITEM_MAX 一致：
// 2*rankSize*itemCapPerPair 在 rankSize<=256 时恰好取到上界 512，没有余量。
constexpr u32 AIV_BSR_ITEM_MAX = 512;

// 跨 rank 任务的私有视图，与 device 侧 AivBatchSendRecv::AivBsrPeerView 逐字段对应。
// 单独定义而不是匿名 struct：便于对 sizeof 做断言，也避开 -Wpedantic 对匿名 struct 的告警。
struct AivBsrPeerViewArgs {
    u32 cclOffset;  // @0 相对 CCL buffer base 的偏移（< rankSize*bucketCap <= 256MB）
    u16 remoteRank; // @4 对端 rank（< rankSize <= 256，u16 足够）
    u16 peerRsv;    // @6 显式占位，使视图恰好 8B
};

// 单条任务 24B：byteCount/cclOffset 用 u32 的前提是 host 把中转容量 clamp 到 256MB 再按 rankSize 分桶，
// 放开该上限须同步改回 u64；512 条占 12KB，加公共入参约 12.2KB，远低于 28000B 预算。
struct AivBsrItemArgs {
    u64 bufAddr;     // @0  本地数据地址（send/self_send: 源地址；recv/self_recv: 目的地址）
    u32 byteCount;   // @8  本片需要搬运的字节数（<= bucketCap）
    u16 sliceIdx;    // @12 本轮内该 pair 的片序号 [0, itemCapPerPair)，收发两侧一致
    u8 sendRecvType; // @14 0: send, 1: recv, 2: self_send, 3: self_recv
    u8 rsv;          // @15
    // 两种互斥视图，由 sendRecvType 决定取哪一个：SELF_SEND 取 dstAddr，SEND/RECV 取 peer，
    // SELF_RECV 两个都不读。device 侧读之前必须先判类型，因此这里填错视图不会被静默掩盖成零值。
    union {
        u64 dstAddr;             // @16 仅 SELF_SEND：本卡目的地址
        AivBsrPeerViewArgs peer; // @16 仅 SEND/RECV：桶位 + 对端 rank
    };
};

struct ExtraArgsBsr {
    u64 itemNum = 0;
    // 每对端在 flag 信箱预留的格数（= itemCapPerPair）：保证 rankSize*flagBlock <= 256，远低于 32768 格。
    u64 flagBlock = 0;
    AivBsrItemArgs items[AIV_BSR_ITEM_MAX] = {};
};

static_assert(sizeof(AivBsrPeerViewArgs) == 8, "AivBsrPeerViewArgs must be 8B to union with dstAddr");
static_assert(sizeof(AivBsrItemArgs) == 24, "AivBsrItemArgs layout must match device side AivBsrItem");
// 偏移钉死：只比 sizeof 挡不住"两字段互换、总大小不变"。union 内成员属 conditionally-supported，
// 不对其断言，起点与宽度由下面两条（sendRecvType 在 14、总长 24）间接锁死。
static_assert(
    offsetof(AivBsrItemArgs, bufAddr) == 0 && offsetof(AivBsrItemArgs, byteCount) == 8,
    "AivBsrItemArgs bufAddr/byteCount offsets must match device side AivBsrItem");
static_assert(
    offsetof(AivBsrItemArgs, sliceIdx) == 12 && offsetof(AivBsrItemArgs, sendRecvType) == 14,
    "AivBsrItemArgs sliceIdx/sendRecvType offsets must match device side AivBsrItem");
static_assert(
    sizeof(ExtraArgsBsr) == 2 * sizeof(u64) + sizeof(AivBsrItemArgs) * AIV_BSR_ITEM_MAX,
    "ExtraArgsBsr layout must match device side AivBatchSendRecv::ExtraArgsBsr");

static_assert(
    sizeof(ExtraArgsBsr) == 12304, "ExtraArgsBsr ABI size must match device side AivBatchSendRecv::ExtraArgsBsr");

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
    HcclComm hcclComm = nullptr;
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
    u64 count = 0;
    u64 totalDataCount = 0; // 算子上报dfx的总数据量，与AICPU对齐，不参与kernel launch
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
    bool isWrite = false;            // 写语义(write)标记，核函数内部据此走不同分支
    u32 subCommRankLevel = 0;        // 0=框内, 1=跨框
    u32 level0ChannelNumPerRank = 0; // Z轴绕路: 框内level0链路数(0或1)
    AivOpArgs() {};
    KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER;
};

// AIV Cache Definitions
struct AivOpCacheArgs {
    std::string commName;
    u64 count;
    HcclDataType dataType;
    HcclCMDType opType;
    HcclReduceOp reduceOp;
    u32 root;
    u32 numBlocksLimit;
    OpExecuteConfig opExecuteConfig{OpExecuteConfig::DEFAULT};

    bool operator<(const AivOpCacheArgs& other) const
    {
        if (commName != other.commName)
            return commName < other.commName;
        if (count != other.count)
            return count < other.count;
        if (dataType != other.dataType)
            return dataType < other.dataType;
        if (opType != other.opType)
            return opType < other.opType;
        if (reduceOp != other.reduceOp)
            return reduceOp < other.reduceOp;
        if (root != other.root)
            return root < other.root;
        if (numBlocksLimit != other.numBlocksLimit)
            return numBlocksLimit < other.numBlocksLimit;
        return opExecuteConfig < other.opExecuteConfig;
    }
};

// 对称内存签名：窗口句柄与算子内 buffer 偏移，决定对端 VA 数组的内容
struct AivSymSign {
    u64 inWin = 0;
    u64 outWin = 0;
    u64 inOffset = 0;
    u64 outOffset = 0;

    bool operator==(const AivSymSign& other) const
    {
        return inWin == other.inWin && outWin == other.outWin && inOffset == other.inOffset
               && outOffset == other.outOffset;
    }
};

struct AivCacheCtxHeader {
    u64 keyHash;
    u32 insCount;
    u32 algNameLen;
    u32 symMode; // 1=该指令集在对称模式下录制，replay 必须保持对称模式
    AivSymSign symSign;
    // 后续内存布局: char algName[algNameLen] + AivInstruction[insCount]
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
extern thread_local HcclComm g_aivCurrentComm;
extern thread_local std::string g_aivCurrentCommName;

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
    AivSuperKernelArgsDef(
        u64 input, u64 output, u32 rank, u32 rankSize, u64 len, u32 dataType, u64 unitSize, u32 reduceOp,
        u32 numBlocks = 0, s32 tag = 0, bool clearEnable = true, uint64_t inputSliceStride = 0,
        uint64_t outputSliceStride = 0, uint64_t repeatNum = 0, uint64_t inputRepeatStride = 0,
        uint64_t outputRepeatStride = 0, u64 cclBufferSize = 0)
        : rank(rank),
          rankSize(rankSize),
          len(len),
          dataType(dataType),
          unitSize(unitSize),
          reduceOp(reduceOp),
          numBlocks(numBlocks),
          tag(tag),
          clearEnable(clearEnable),
          inputSliceStride(inputSliceStride),
          outputSliceStride(outputSliceStride),
          repeatNum(repeatNum),
          inputRepeatStride(inputRepeatStride),
          outputRepeatStride(outputRepeatStride),
          input(input),
          output(output),
          cclBufferSize(cclBufferSize)
    {}
    AivSuperKernelArgsDef() {}
};

HcclResult RegisterKernel();

HcclResult UnRegisterAivKernel();

HcclResult ExecuteKernelLaunchInner(const AivOpArgs& opArgs, void* args, u32 argsSize);

HcclResult ExecuteKernelLaunch(const AivOpArgs& opArgs);

// BatchSendRecv 专用 launch 入口：入参多一份 12KB 任务表（AivBsrKernelArgs），只有 BSR 需要，
// 故不塞进通用入口。extraArgsBsr 由调用方持有，必须在本次 launch 期间存活。
HcclResult ExecuteKernelLaunchBsr(const AivOpArgs& opArgs, const ExtraArgsBsr& extraArgsBsr);

void ProcessAivExceptionCallBack(aclrtExceptionInfo* exceptionInfo);

u64 CalcAivCacheKeyHash(const AivOpCacheArgs& cacheKey);

HcclResult BuildAivCacheCtxTag(u64 keyHash, std::string& ctxTag);

HcclResult GetOrCreateAivCacheIndexCtx(HcclComm comm, AivCacheIndexCtx** indexCtx);

HcclResult EvictAivCacheIfNeeded(HcclComm comm, AivCacheIndexCtx* indexCtx);

// Hold this mutex from cache lookup until all replay reads of the returned context finish.
std::mutex& GetAivCacheMutex();

HcclResult LookupAivCacheCtx(
    HcclComm comm, const std::string& ctxTag, u64 keyHash, bool& cacheHit, std::string& algName,
    AivInstruction*& instructions, u32& insCount, u32& symMode, AivSymSign& symSign);

HcclResult ReplayAivInstructions(const AivInstruction* instructions, u32 insCount, OpParam& param);

HcclResult ReplayAivInstructionsV(const AivInstruction* instructions, u32 insCount, OpParam& param);

HcclResult StoreAivCacheCtx(
    HcclComm comm, const std::string& ctxTag, u64 keyHash, const std::string& algName, AivCacheIndexCtx* indexCtx,
    u32 symMode, const AivSymSign& symSign);

// 对端对称 VA 数组的设备侧内容：与最近一次写表签名一致时可跳过写表，否则需按该签名重填。
bool IsAivSymPeerTableFilled(const void* commInfo, const AivSymSign& sign);

void RecordAivSymPeerTable(const void* commInfo, const AivSymSign& sign);

HcclResult ClearAivTagCb(HcclComm comm, HcclCommStatePhase state, void* userPtr);

} // namespace ops_hccl

#endif // HCCL_AIV_UTILS_H
