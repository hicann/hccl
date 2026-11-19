/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_DEFINES_H
#define AIV_DEFINES_H

#include "kernel_operator.h"

using namespace AscendC;

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------
constexpr uint32_t MAX_RANK_SIZE = 1024; // server内最大卡数
constexpr uint32_t MAX_RANK_SIZE_V = 256;
constexpr uint32_t AIV_URMA_ALLTOALLV_GROUP_SIZE = 16U; // 每轮并发对端上限(host侧同名常量须同步)
constexpr uint32_t BR_CTRL_CORE_LIMIT_RANK_SIZE = 64;
constexpr uint64_t BUFFER_OUT_ADDR_OFFSET = 16 * 1024;
constexpr uint64_t LOCAL_FLAG_BUF_LEN = 2560;
constexpr uint64_t AIV_TAG_MOVE_RIGHT_BITS = 16;
constexpr uint64_t LOW_16_BITS = 0xFFFF;
constexpr uint64_t DATA_LIMIT = 512 * 1024;
constexpr uint32_t PING_PONG = 2;

constexpr uint64_t AIV_PING_PONG_FACTOR_TWO = 2;
constexpr uint32_t NUM_BLOCKS_FOUR_PER_RANK_A3 = 4;
constexpr uint32_t MAX_NUM_BLOCKS = 48;

constexpr uint64_t FLAG_SIZE = 128;
constexpr uint64_t UB_ALIGN_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE_4 = UB_FLAG_SIZE * 4;
constexpr uint64_t UB_FLAG_SIZE_8 = UB_FLAG_SIZE * 8;
constexpr uint64_t UB_MAX_DATA_SIZE = 190 * 1024;
constexpr uint64_t UB_DB_DATA_BATCH_SIZE = UB_MAX_DATA_SIZE / 2;
constexpr uint32_t MaxBufferSize = 200 * 1024 * 1024;

constexpr uint64_t ATOMIC_FLAG_SIZE = 512;
constexpr uint64_t FLAG_ONE_OFFSET = 0;
constexpr uint64_t FLAG_TWO_OFFSET = FLAG_SIZE;
constexpr uint64_t FLAG_THREE_OFFSET = FLAG_SIZE * 2;
constexpr uint64_t FLAG_FOUR_OFFSET = FLAG_SIZE * 3;
constexpr uint64_t FLAG_FIVE_OFFSET = FLAG_SIZE * 4;

constexpr uint64_t DOUBLE = 2;
constexpr uint64_t FLAG_BUF_NUM = 3;
// UB_MAX_DATA_SIZE(inOutQue) 切成几份给 inQueueX/inQueueY/outQueueZ：194560/4 = 48640B 一份
constexpr uint64_t TILING_NUM = 4;
constexpr uint64_t CHUNK_SIZE = 2048;

constexpr int32_t TAG_INIT_VALUE = 1;
constexpr int32_t TAG_RESET_COUNT = 1000;
constexpr uint32_t AIV_FLAG_CLEAR_OFFSET = 512 * 1024;

// AIV_URMA 内存布局常量（device 侧，与 host 侧 hccl_aiv_utils.h 一致）
constexpr uint32_t AIV_URMA_HANDLE_ADDR_OFFSET = 32 * 1024;
constexpr uint32_t AIV_URMA_SUBCOMM_RANKS_START_OFFSET = 256 * 1024;
constexpr uint32_t AIV_URMA_SUBCOMM_RANKS_PER_LEVEL_BYTES = 16 * 1024;
constexpr uint32_t AIV_URMA_SUBCOMM_RANKS_AREA_END = 512 * 1024;
constexpr uint32_t AIV_URMA_OUTPUT_ADDR_SLOT_OFFSET = 50 * 1024 * 1024;
constexpr uint32_t MAX_CHANNELS_PER_RANK = 8;
// 相对于GM_OUT，前同步、尾同步使用的同步标记区的偏移，也是普通标记区的大小
constexpr uint32_t FLAG1_OFFSET = 1 * 1024 * 1024;
constexpr uint32_t FLAG2_OFFSET = 5 * 1024 * 1024;
constexpr uint32_t BASE_FLAG_OFFSET = 9 * 1024 * 1024;
constexpr uint32_t AIV_FLAG_EMPTY_OFFSET = 10 * 1024 * 1024;
constexpr uint32_t GM_OUT_PING_OFFSET = 18 * 1024 * 1024;
constexpr uint32_t GM_OUT_PONG_OFFSET = 34 * 1024 * 1024;
constexpr uint64_t PINGPONG_TOTAL_DATA_LIMIT = (GM_OUT_PONG_OFFSET - GM_OUT_PING_OFFSET) / PING_PONG;

// A5/950 对称内存：buffIn 空闲区里的「对端 output/input 对称 VA 数组」，按 rank 索引，各 MAX_RANK_SIZE 项。
// host 按窗口布局算好每个 rank 的地址后下发；本 rank 条目存用户 VA（kernel 用 output_/input_ 减去它求
// 分循环偏移），某项为 0 表示该侧对称不可用。
constexpr uint64_t AIV_SYM_PEER_OUT_ADDR_OFFSET = 8 * 1024; // 8K~16K 空洞：对端 output 对称 VA 数组
constexpr uint64_t AIV_SYM_PEER_IN_ADDR_OFFSET = 24 * 1024; // 24K~32K 空洞：对端 input 对称 VA 数组

#if defined(HCCL_AIV_ACLGRAPH_SK)
// SK_BIND cap 掩码：blockdim_scale_up，表示算子已自行支持核数向上扩展
constexpr uint64_t SK_CAP_BLOCKDIM_SCALE_UP = 16;
#endif

/**
 * ccl buffers        GM_OUT               Tag(大小4)             flag1         flag2             BarrierBase Clear
 * data1              data2 0       |         16K          |         512K          |      1M      |       5M      | 9M
 * |               10M           |   18M             |    34M BUFFER_OUT_ADDR_OFFSET | AIV_FLAG_CLEAR_OFFSET |
 * FLAG1_OFFSET|  FLAG2_OFFSET |  BASE_FLAG_OFFSET |  AIV_FLAG_EMPTY_OFFSET |GM_OUT_PING_OFFSET | GM_OUT_PONG_OFFSET
 */

// ---------------------------------------------------------------------------
// 宏、结构体、枚举
// ---------------------------------------------------------------------------
#define EXPORT_AIV_META_INFO(kernel_name)                               \
    static const struct FunLevelKType kernel_name##_kernel_type_section \
        __attribute__((used, section(".ascend.meta." #kernel_name)))    \
        = {{F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}}

struct ExtraArgs {
    uint64_t sendCounts[MAX_RANK_SIZE_V] = {};
    uint64_t sendDispls[MAX_RANK_SIZE_V] = {};
    uint64_t recvCounts[MAX_RANK_SIZE_V] = {};
    uint64_t recvDispls[MAX_RANK_SIZE_V] = {};
};

using AivSuperKernelArgs = struct AivSuperKernelArgsDef {
    GM_ADDR buffersIn = nullptr; // 注册的CCLIN地址，所有卡可访问
    uint64_t rank;
    uint64_t rankSize;
    uint64_t len;
    uint64_t dataType;
    uint64_t unitSize;
    uint64_t reduceOp;
    uint64_t numBlocks;
    uint64_t tag; // 第几次调用，定时重置成1
    uint64_t clearEnable;
    uint64_t inputSliceStride;
    uint64_t outputSliceStride;
    uint64_t repeatNum;
    uint64_t inputRepeatStride;
    uint64_t outputRepeatStride;
    uint64_t input;
    uint64_t output;
    uint64_t cclBufferSize;
};

enum class AivNotifyType { ACK, DataSignal, Done };

enum class CommPattern {
    // server间
    interRank,
    // server内
    intraRank
};

#define KERNEL_ARGS_DEF                                                                                             \
    GM_ADDR buffIn, uint64_t input, uint64_t output, uint32_t rank, uint32_t sendRecvRemoteRank, uint32_t rankSize, \
        uint64_t len, uint32_t dataType, uint32_t reduceOp, uint32_t root, uint32_t sliceId,                        \
        uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t repeatNum, uint64_t inputRepeatStride,      \
        uint64_t outputRepeatStride, uint32_t numBlocks, bool isOpBase, GM_ADDR headCountMem, GM_ADDR tailCountMem, \
        GM_ADDR addOneMem, uint32_t counterMemSize, bool isEnableCounter, bool isWrite, uint32_t subCommRankLevel,  \
        uint32_t level0ChannelNumPerRank

#define EXTERN_KERNEL_ARGS_DEF_V2 KERNEL_ARGS_DEF, ExtraArgs extraArgs

#define KERNEL_ARGS_CALL                                                                                            \
    buffIn, input, output, rank, sendRecvRemoteRank, rankSize, len, dataType, reduceOp, root, sliceId,              \
        inputSliceStride, outputSliceStride, repeatNum, inputRepeatStride, outputRepeatStride, numBlocks, isOpBase, \
        headCountMem, tailCountMem, addOneMem, counterMemSize, isEnableCounter, isWrite, subCommRankLevel,          \
        level0ChannelNumPerRank

#define EXTERN_KERNEL_ARGS_CALL KERNEL_ARGS_CALL, extraArgs

#define KERNEL_CLASS_INIT                                                                                           \
    buffIn, input, output, rank, sendRecvRemoteRank, rankSize, len, dataType, reduceOp, root, inputSliceStride,     \
        outputSliceStride, repeatNum, inputRepeatStride, outputRepeatStride, headCountMem, tailCountMem, addOneMem, \
        counterMemSize, isEnableCounter, numBlocks

#define SUPERKERNEL_LITE_ARGS_DEF uint64_t args_offset

#define SUPERKERNEL_LITE_ARGS_EXTRACT                \
    GM_ADDR* param_base = (GM_ADDR*)get_para_base(); \
    GM_ADDR hiddenInput = param_base[args_offset++]; \
    GM_ADDR input = param_base[args_offset++];       \
    GM_ADDR output = param_base[args_offset++]

#define SUPERKERNEL_ARGS_DEF GM_ADDR hiddenInput, GM_ADDR input, GM_ADDR output

#define SUPERKERNEL_ARGS_CALL hiddenInput, input, output

#define SUPERKERNEL_CLASS_INIT hiddenInput, input, output

#define AIV_INFO(format, ...)                   \
    do {                                        \
        AscendC::PRINTF(format, ##__VA_ARGS__); \
    } while (0)

#define AIV_INFO_HINT AIV_INFO("Aiv log dump is enabled in %s\n", __func__)

template <typename T>
__aicore__ inline void
DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal, const uint32_t calCount)
{
    if ((calCount * sizeof(T)) % UB_ALIGN_SIZE == 0) {
        DataCopy(dstLocal, srcGlobal, calCount);
    } else {
        DataCopyExtParams copyParams{1, calCount * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = 1;
        DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
    }
}

template <typename T>
__aicore__ inline void
DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal, const uint32_t calCount)
{
    if ((calCount * sizeof(T)) % UB_ALIGN_SIZE == 0) {
        DataCopy(dstGlobal, srcLocal, calCount);
    } else {
        DataCopyExtParams copyParams{1, calCount * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPad(dstGlobal, srcLocal, copyParams);
    }
}

#endif // AIV_DEFINES_H
