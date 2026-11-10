/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALG_V2_TEMPLATE_UTILS
#define ALG_V2_TEMPLATE_UTILS

#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <list>
#include <cmath>
#include <algorithm>
#include "alg_param.h"
#include "binary_stream.h"

HcclResult __attribute__((weak)) HcommThreadJoin(ThreadHandle thread, uint32_t timeout);
namespace ops_hccl {

#define UINT32_MAX (4294967295U)
constexpr u32 INVALID_U32 = UINT32_MAX;
constexpr u32 MAX_JETTY_NUM = 4;
constexpr u32 SMALL_SIZE_512KB = 512 * 1024;

constexpr s32 INVALID_RANKID = INT32_MAX;

struct SliceInfo {
    u64 offset{0};
    u64 size{0};
};

struct SplitSliceInfo {
    u64 offset{0};
    u64 size{0};
    u64 count{0};

    SplitSliceInfo(const u64 offset, const u64 size, const u64 count) : offset(offset), size(size), count(count) {}
};

using RankSliceInfo = std::vector<std::vector<SliceInfo>>;

enum class BufferType { INPUT = 0, OUTPUT = 1, HCCL_BUFFER = 2, DEFAULT };

enum class BatchSendRecvOpType { RECORD = 0, SEND = 1, RECV = 2, FENCE = 3, DEFAULT };

struct DataSlice {
    void* addr_ = nullptr;
    u64 offset_{0}; // Slice相对于input/output的偏移字节数，gather类操作取output，scatter类操作取input
    u64 size_{0};   // Slice的数据大小，单位：字节
    u64 count_{0};  // 数据元素个数

    // 构造时校验：count > 0 时 size 不应为 0（避免下游除零）。
    // DataSlice 为热路径数据载体，此处仅做 debug 断言，不影响 release 性能。
    DataSlice(void* addr, u64 offset, u64 size, u64 count) : addr_(addr), offset_(offset), size_(size), count_(count)
    {
        if (count > 0 && size == 0) {
            HCCL_ERROR("[DataSlice] invalid params: addr=%p, offset=%lu, size=0, count=%lu.", addr, offset, count);
        }
    }

    DataSlice(void* addr, u64 offset, u64 size) : addr_(addr), offset_(offset), size_(size) { count_ = 0; }

    std::string Describe() const
    {
        std::ostringstream oss;
        oss << "DataSlice: addr=" << addr_ // 指针地址会自动格式化为十六进制
            << ", offset=" << offset_ << ", size=" << size_ << ", count=" << count_;
        return oss.str();
    }
};

struct SlicesList {
    std::vector<DataSlice> srcSlices_;
    std::vector<DataSlice> dstSlices_;

    SlicesList(const std::vector<DataSlice>& srcSlices, const std::vector<DataSlice>& dstSlices)
        : srcSlices_(srcSlices),
          dstSlices_(dstSlices)
    {}

    SlicesList(std::vector<DataSlice>&& srcSlices, std::vector<DataSlice>&& dstSlices)
        : srcSlices_(std::move(srcSlices)),
          dstSlices_(std::move(dstSlices))
    {}
};

struct A2ASendRecvInfo {
    // 存放数据长度和偏移长度
    std::vector<u64> sendLength;
    std::vector<u64> sendOffset;
    std::vector<u64> recvLength;
    std::vector<u64> recvOffset;
    // 存放数据个数和偏移个数
    std::vector<u64> sendCounts;
    std::vector<u64> sendDispls;
    std::vector<u64> recvCounts;
    std::vector<u64> recvDispls;
};

struct DataInfo {
    ChannelInfo channel_;
    SlicesList slices_;
    HcclDataType dataType_;
    DataInfo(const ChannelInfo& channel, const SlicesList& slices) : channel_(channel), slices_(slices) {}
    DataInfo(const ChannelInfo& channel, const SlicesList& slices, HcclDataType dataType)
        : channel_(channel),
          slices_(slices),
          dataType_(dataType)
    {}
};

struct DataReduceInfo {
    ChannelInfo channel_;
    SlicesList slices_;
    HcclDataType dataType_;
    HcclReduceOp reduceType_;
    DataReduceInfo(const ChannelInfo& channel, const SlicesList& slices, HcclDataType dataType, HcclReduceOp reduceType)
        : channel_(channel),
          slices_(slices),
          dataType_(dataType),
          reduceType_(reduceType)
    {}
};

struct TxRxChannels {
    ChannelInfo txChannel_;
    ChannelInfo rxChannel_;

    TxRxChannels(const ChannelInfo& txLink, const ChannelInfo& rxLink) : txChannel_(txLink), rxChannel_(rxLink) {}
};

struct TxRxSlicesList {
    SlicesList txSlicesList_;
    SlicesList rxSlicesList_;

    TxRxSlicesList(const SlicesList& txSlicesList, const SlicesList& rxSlicesList)
        : txSlicesList_(txSlicesList),
          rxSlicesList_(rxSlicesList)
    {}
};

struct SendRecvInfo {
    TxRxChannels sendRecvChannels_;
    TxRxSlicesList sendRecvSlices_;
    HcclDataType dataType_;

    SendRecvInfo(const TxRxChannels& sendRecvLinks, const TxRxSlicesList& sendRecvSlices)
        : sendRecvChannels_(sendRecvLinks),
          sendRecvSlices_(sendRecvSlices)
    {}

    SendRecvInfo(const TxRxChannels& sendRecvLinks, const TxRxSlicesList& sendRecvSlices, HcclDataType dataType)
        : sendRecvChannels_(sendRecvLinks),
          sendRecvSlices_(sendRecvSlices),
          dataType_(dataType)
    {}
};

struct SendRecvReduceInfo {
    TxRxChannels sendRecvChannels_;
    TxRxSlicesList sendRecvSlices_;
    HcclDataType dataType_;
    HcclReduceOp reduceType_;

    SendRecvReduceInfo(
        const TxRxChannels& sendRecvLinks, const TxRxSlicesList& sendRecvSlices, const HcclDataType dataType,
        const HcclReduceOp reduceOp)
        : sendRecvChannels_(sendRecvLinks),
          sendRecvSlices_(sendRecvSlices),
          dataType_(dataType),
          reduceType_(reduceOp)
    {}
};

struct BuffInfo {
    void* inputPtr = nullptr;  // userIn
    void* outputPtr = nullptr; // userOut
    HcclMem hcclBuff;          // 跨Rank缓存Buffer
    BufferType inBuffType{BufferType::DEFAULT};
    BufferType outBuffType{BufferType::DEFAULT};
    BufferType hcclBuffType{BufferType::DEFAULT};
    u64 inputSize = 0;
    u64 outputSize = 0;
    u64 hcclBuffSize = 0;
    u64 inBuffBaseOff = 0;
    u64 outBuffBaseOff = 0;
    u64 hcclBuffBaseOff = 0;
};

struct StepSliceInfo {
    BuffInfo buffInfo;
    std::vector<std::vector<u64>> stepCount;     // 每step上所有的rank参与的数据量
    std::vector<std::vector<u64>> stepSliceSize; // 每step上所有的rank参与的数据量
    std::vector<u64>
        stepInputSliceStride; // 数据连着放 buffertype.addr + inputSliceStride[rankid] + inputOmniPipeSliceStride[j]
    std::vector<u64> stepOutputSliceStride; // 数据连着放
    std::vector<std::vector<u64>> inputOmniPipeSliceStride;
    std::vector<std::vector<u64>> outputOmniPipeSliceStride;

    std::vector<char> Serialize() const
    {
        BinaryStream binaryStream;
        binaryStream << stepCount;
        binaryStream << stepSliceSize;
        binaryStream << stepInputSliceStride;
        binaryStream << stepOutputSliceStride;
        binaryStream << inputOmniPipeSliceStride;
        binaryStream << outputOmniPipeSliceStride;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char>& data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> stepCount;
        binaryStream >> stepSliceSize;
        binaryStream >> stepInputSliceStride;
        binaryStream >> stepOutputSliceStride;
        binaryStream >> inputOmniPipeSliceStride;
        binaryStream >> outputOmniPipeSliceStride;
    }
};

struct TemplateFastLaunchCtx {
    BuffInfo buffInfo;
    std::vector<ThreadHandle> threads;
    std::vector<CcuKernelSubmitInfo> ccuKernelSubmitInfos;
};

struct TemplateDataParams {
    BuffInfo buffInfo;
    u64 count{0};
    u64 sliceSize{0};
    u64 inputSliceStride{0};
    u64 outputSliceStride{0};
    u64 repeatNum{0};
    u64 inputRepeatStride{0};
    u64 outputRepeatStride{0};
    u64 tailSize{0};
    bool enableRemoteMemAccess{false};
    u64 processedDataCount{0};
    u64 root{0};
    HcclDataType dataType{HCCL_DATA_TYPE_INT8};
    std::vector<u64> allRankSliceSize;
    std::vector<u64> allRankDispls;
    std::vector<u64> allRankProcessedDataCount;
    // alltoallV loop内变长数据
    std::vector<u64> sendCounts;
    std::vector<u64> recvCounts;
    std::vector<u64> sdispls;
    std::vector<u64> rdispls;
    std::vector<u64> peerRdispls;
    StepSliceInfo stepSliceInfo;
    BatchSendRecvOpType opType{BatchSendRecvOpType::DEFAULT};
    StepSliceInfo omniReadDstStepSliceInfo;
    bool omniLastStepRead_ = false;
    u64 localCopyFlag{0};
    // 读写语义由执行期参数下传，模板本身无状态。
    bool isWrite{false};
    HcclReduceOp reduceOp{HcclReduceOp::HCCL_REDUCE_SUM};

    std::vector<char> Serialize() const
    {
        BinaryStream binaryStream;
        binaryStream << buffInfo;
        binaryStream << count;
        binaryStream << sliceSize;
        binaryStream << inputSliceStride;
        binaryStream << outputSliceStride;
        binaryStream << repeatNum;
        binaryStream << inputRepeatStride;
        binaryStream << outputRepeatStride;
        binaryStream << tailSize;
        binaryStream << enableRemoteMemAccess;
        binaryStream << allRankSliceSize;
        binaryStream << allRankDispls;
        binaryStream << sendCounts;
        binaryStream << recvCounts;
        binaryStream << sdispls;
        binaryStream << rdispls;
        binaryStream << peerRdispls;
        binaryStream << allRankProcessedDataCount;
        binaryStream << root;
        binaryStream << dataType;
        binaryStream << stepSliceInfo.Serialize();
        binaryStream << opType;
        binaryStream << omniReadDstStepSliceInfo.Serialize();
        binaryStream << omniLastStepRead_;
        binaryStream << localCopyFlag;
        binaryStream << isWrite;
        binaryStream << reduceOp;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char>& data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> buffInfo;
        binaryStream >> count;
        binaryStream >> sliceSize;
        binaryStream >> inputSliceStride;
        binaryStream >> outputSliceStride;
        binaryStream >> repeatNum;
        binaryStream >> inputRepeatStride;
        binaryStream >> outputRepeatStride;
        binaryStream >> tailSize;
        binaryStream >> enableRemoteMemAccess;
        binaryStream >> allRankSliceSize;
        binaryStream >> allRankDispls;
        binaryStream >> sendCounts;
        binaryStream >> recvCounts;
        binaryStream >> sdispls;
        binaryStream >> rdispls;
        binaryStream >> peerRdispls;
        binaryStream >> allRankProcessedDataCount;
        binaryStream >> root;
        binaryStream >> dataType;
        std::vector<char> stepSliceInfoData;
        binaryStream >> stepSliceInfoData;
        stepSliceInfo.DeSerialize(stepSliceInfoData);
        binaryStream >> opType;

        std::vector<char> omniReadDstStepSliceInfoData;
        binaryStream >> omniReadDstStepSliceInfoData;
        omniReadDstStepSliceInfo.DeSerialize(omniReadDstStepSliceInfoData);
        binaryStream >> omniLastStepRead_;
        binaryStream >> localCopyFlag;
        binaryStream >> isWrite;
        binaryStream >> reduceOp;
    }
};

struct TemplateResource {
    std::map<u32, std::vector<ChannelInfo>> channels;
    std::vector<ThreadHandle> threads;
    std::vector<CcuKernelHandle> ccuKernels;
    std::vector<CcuKernelSubmitInfo> submitInfos;
    double dieSplitRatio = 0.0;
    void* npu2DpuShmemPtr;
    void* dpu2NpuShmemPtr;
    void* aivCommInfoPtr = nullptr;
    bool isDpuThreadEnabled = false;
    ThreadHandle dpuThread = 0;
    u32 dpuThreadNotifyIdxOnMain = 0;
    bool isHcommChannelDrainSupported = false;
};

struct DPURunInfo {           // AICPU构造信息，写入共享内存
    std::string templateName; // DPU算法展开的template名
    TemplateDataParams tempAlgParams;
    std::map<uint32_t, std::vector<ChannelInfo>> channels;
    u32 myRank;
    std::vector<std::vector<uint32_t>> subCommRanks;
    u32 execTimeout{CUSTOM_TIMEOUT}; // DPU notify wait超时时间，单位为秒，0表示永不超时

    std::vector<char> Serialize() const
    {
        BinaryStream binaryStream;
        binaryStream << templateName;
        binaryStream << tempAlgParams.Serialize();
        binaryStream << channels;
        binaryStream << myRank;
        binaryStream << subCommRanks;
        binaryStream << execTimeout;

        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char>& data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> templateName;
        std::vector<char> tempAlgParamsData;
        binaryStream >> tempAlgParamsData;
        tempAlgParams.DeSerialize(tempAlgParamsData);
        binaryStream >> channels;
        binaryStream >> myRank;
        binaryStream >> subCommRanks;
        binaryStream >> execTimeout;
    }
};

struct AlltoAllSendRecvInfo {
    // 存放数据个数和偏移个数
    std::vector<u64> sendCounts;
    std::vector<u64> sendDispls;
    std::vector<u64> recvCounts;
    std::vector<u64> recvDispls;
};

struct AicpuNHRStepInfo {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    AicpuNHRStepInfo() : nSlices(0) {}
};

// ==================== RHD 步骤生成公共函数族 ====================
// 区段字段（txSeg*/rxSeg*）一律为物理单元空间（σ 布局）；字节长度经 RhdSegToBytes* 推导。
// 主体步 nSlices 为虚拟步距（非区段 unit 数），PRE/POST 为区段 unit 数；勿照搬 HD/NHR 的 nSlices 消费范式。

enum class RhdPhase : u8 { MAIN_RS, MAIN_AG, PRE, POST };

struct RhdStepInfo {
    u32 step = 0;                      // 步序（相位内 0 起）
    RhdPhase phase{RhdPhase::MAIN_RS}; // 相位
    u32 peer = 0;                      // 对端：MAIN 相位为主体有效序，PRE/POST 为配对 algRank
    u32 nSlices = 0;                   // 主体步为步距 dist，PRE/POST 为份额 slice 数
    u32 txSegStart = 0;                // 发送区段起（unit 空间，含）
    u32 txSegEnd = 0;                  // 发送区段止（unit 空间，不含）
    u32 rxSegStart = 0;                // 接收区段起（unit 空间，含）
    u32 rxSegEnd = 0;                  // 接收区段止（unit 空间，不含）
};

struct RhdPart1Info {
    u32 r = 0;                                // r = p − 2^⌊log₂p⌋（2^N 时 0）
    bool isPowerOfTwo = true;                 // 本层为 2^N（无前后处理）
    bool inPart1 = false;                     // 本 rank 属于 part1（algRank < 2r）
    bool isRepresentative = false;            // 本 rank 是所在配对的代表
    u32 pairedAlgRank = INVALID_VALUE_RANKID; // 配对对端（inPart1 时有效）
    u32 virtualRank = 0;                      // 主体有效序 v ∈ [0, P)
    u32 effectiveRankSize = 0;                // P = p − r
    u32 unitStart = 0;                        // 持有份额的 unit 起点（代表占 2 unit）
    u32 unitCount = 1;                        // 持有份额的 unit 数（代表 2，其余 1）
};

// 主体步数：⌊log₂rankSize⌋
u32 GetRhdStepNum(u32 rankSize);

// part1 划分：前 2r 个 rank 按 (t ↔ t+r) 配对，代表默认低位 t，root 落 part1 时跟随 root 所在侧
// （root = INVALID_VALUE_RANKID 表示恒低位）；root 类外的算子传 INVALID 即可
HcclResult CalcRhdPart1Info(u32 rankSize, u32 myAlgRank, u32 root, RhdPart1Info& info);

// RS（折半）相位步骤表：步 k 的 dist = P>>(k+1)、对端 = v⊕dist
HcclResult GenRhdReduceScatterSteps(const RhdPart1Info& info, std::vector<RhdStepInfo>& steps);

// AG（倍增）相位步骤表：步 k 的 dist = 1<<k（与 RS 相位互为镜像）
HcclResult GenRhdAllGatherSteps(const RhdPart1Info& info, std::vector<RhdStepInfo>& steps);

// 前/后处理（2^N 输出空步骤）：
//   pre：被合并方发送（SHARE = 自身份额段 / FULL = [0,p) 全量段），代表为接收镜像
//   post：代表发送，被合并方接收（SHARE / FULL 同上）
//   RS 族取 (FULL, SHARE)，AG 族取 (SHARE, FULL)
enum class RhdSegMode : u8 { SHARE, FULL };
HcclResult GenRhdPrePostSteps(
    const RhdPart1Info& info, RhdStepInfo& pre, RhdStepInfo& post, RhdSegMode preSegMode, RhdSegMode postSegMode);

// ---- 支撑辅助 ----
// σ：algRank → 自身份额槽位；越界返回 INVALID_VALUE_RANKID
u32 RhdUnitStartOfRank(u32 rankSize, u32 algRank);
// 虚拟序 → 承载 algRank；越界返回 INVALID_VALUE_RANKID
// ⚠ physRankSize 为物理规模 p（非有效规模 P），root 须与 CalcRhdPart1Info 一致
u32 RhdAlgRankOfVirtual(u32 physRankSize, u32 root, u32 virtualRank);

struct RhdSegBytes {
    u64 offset{0};
    u64 len{0};
};
// 均匀布局区段 → 字节（仅紧凑布局；tailSize == 0 视为无尾块，对齐 TemplateDataParams 约定）
RhdSegBytes RhdSegToBytesUniform(u32 unitStart, u32 unitEnd, u64 sliceSize, u64 tailSize, u32 tailUnit);
// 变长布局区段 → 字节（仅紧凑布局）；越界返回 HCCL_E_PARA，空区段返回 HCCL_SUCCESS + {0,0}
HcclResult RhdSegToBytesAllRank(u32 unitStart, u32 unitEnd, const std::vector<u64>& unitSizes, RhdSegBytes& bytes);
// algRank 索引的尺寸表 → unit 空间尺寸表（σ 反查）
HcclResult RhdBuildUnitSizes(u32 rankSize, const std::vector<u64>& allRankSliceSize, std::vector<u64>& unitSizes);

HcclResult GetAlgRank(const u32 virtRank, const std::vector<u32>& rankIds, u32& algRank);

u32 GetNHRStepNum(u32 rankSize);

inline u32 CalcChannelsPerRank(const std::vector<HcclChannelDesc>& channels)
{
    u32 channelsPerRank = 1;
    u32 currentRank = INVALID_VALUE_RANKID;
    u32 currentCount = 0;
    // channels的排列遵循相同远端的channel放在相邻位置
    for (const auto& channel : channels) {
        if (channel.remoteRank == currentRank) {
            // 如果remoteRank不变，则计数一直累加
            currentCount++;
        } else {
            // 如果remoteRank变化了，则更新channelsPerRank并重新开始给下一个remoteRank计数
            if (currentCount != channelsPerRank && currentRank != channels[0].remoteRank
                && currentRank != INVALID_VALUE_RANKID) {
                HCCL_WARNING(
                    "[CalcChannelsPerRank] channel num[%u] of remote rank[%u] is not equal to "
                    "channel num[%u] of previous ranks.",
                    currentCount, currentRank, channelsPerRank);
            }
            if (currentCount > channelsPerRank) {
                channelsPerRank = currentCount;
            }
            currentRank = channel.remoteRank;
            currentCount = 1;
        }
    }
    // 处理最后一个rank
    if (currentCount > channelsPerRank) {
        channelsPerRank = currentCount;
    }
    return channelsPerRank;
}

inline u32 CalcChannelsPerRankMin(const std::vector<HcclChannelDesc>& channels)
{
    u32 channelsPerRank = UINT32_MAX;
    u32 currentRank = INVALID_VALUE_RANKID;
    u32 currentCount = 0;

    auto updateChannelsPerRank = [&](u32 rank, u32 count) {
        if (rank != INVALID_VALUE_RANKID) {
            if (channelsPerRank != UINT32_MAX && count != channelsPerRank) {
                HCCL_WARNING(
                    "[CalcChannelsPerRank] channel num[%u] of remote rank[%u] is not equal to "
                    "channel num[%u] of previous ranks.",
                    count, rank, channelsPerRank);
            }
            if (count < channelsPerRank) {
                channelsPerRank = count;
            }
        }
    };

    // channels的排列遵循相同远端的channel放在相邻位置
    for (const auto& channel : channels) {
        if (channel.remoteRank == currentRank) {
            // 如果remoteRank不变，则计数一直累加
            currentCount++;
        } else {
            // 如果remoteRank变化了，则更新channelsPerRank并重新开始给下一个remoteRank计数
            updateChannelsPerRank(currentRank, currentCount);
            currentRank = channel.remoteRank;
            currentCount = 1;
        }
    }
    // 处理最后一个rank
    updateChannelsPerRank(currentRank, currentCount);
    if (channelsPerRank == UINT32_MAX) {
        channelsPerRank = 1;
    }
    return channelsPerRank;
}

inline u32 CalcChannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    u32 channelsPerRank = 1;
    for (const auto& channelsByRank : channels) {
        if (channelsByRank.second.size() > channelsPerRank) {
            channelsPerRank = static_cast<u32>(channelsByRank.second.size());
        }
    }
    return channelsPerRank;
}

inline u32 CalcChannelsPerRankMin(const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    u32 channelsPerRank = UINT32_MAX;
    for (const auto& channelsByRank : channels) {
        u32 count = static_cast<u32>(channelsByRank.second.size());
        if (count < channelsPerRank) {
            channelsPerRank = count;
        }
    }
    if (channelsPerRank == UINT32_MAX) {
        channelsPerRank = 1;
    }
    return channelsPerRank;
}

// roundup func for uint
inline u64 RoundUp(const u64 dividend, const u64 divisor)
{
    if (divisor == 0) {
        HCCL_WARNING("[RoundUp] divisor is 0.");
        return dividend;
    }
    return dividend / divisor + ((dividend % divisor != 0) ? 1 : 0);
}

// ccu快速下发arg填充
template <typename... Args>
HcclResult FillCachedArgs(CcuKernelSubmitInfo& info, Args... args)
{
    size_t argNum = sizeof...(Args);
    if (UNLIKELY(argNum > CCU_MAX_TASK_ARG_NUM)) {
        HCCL_ERROR("[FillCachedArgs] argNum is bigger than CCU_MAX_TASK_ARG_NUM[%d]", CCU_MAX_TASK_ARG_NUM);
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint64_t temp[] = {static_cast<uint64_t>(args)...};

    for (size_t i = 0; i < argNum; i++) {
        info.cachedArgs[i] = temp[i];
    }

    return HcclResult::HCCL_SUCCESS;
}
HcclResult CalcDataSplitByPortGroupCommon(
    const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
    std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset, const u32 channelsPerRank);

HcclResult CalcDataSplitByPortGroupZAxisDetour(
    const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
    std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset,
    const u32 level0ChannelNumPerRank, const u32 level1ChannelNumPerRank, const float level0DataRatio = 0.5f);

// Split by (server rank size - 1) and level1 port weights, preferring 4KB alignment without padding.
HcclResult CalcDataSplitByBandwidthZAxisDetour(
    const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
    std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset,
    const u32 level0ChannelNumPerRank, const u32 level1ChannelNumPerRank, const u32 serverRankSize);

bool IsAllConnetedWithTopo(const TopoInfoWithNetLayerDetails* topoInfo, const u32 netLayer, const CommTopo topoType);

enum class ParallelDataSplitType { REDUCE_SCATTER_WITH_LOCAL_REDUCE = 0, ALL_GATHER = 1, SCATTER = 2, BROADCAST = 3 };

bool GetPortGroupSize(const std::map<u32, std::vector<ChannelInfo>>& channels, uint64_t& portGroupSize);

// 返回值统一表示"先Mesh后Clos"数据片的目标比例，"先Clos后Mesh"数据片的比例为1减去该值。
// fallbackRatio需按同一语义传入：公式无法计算时直接返回该回退值(已裁剪到[0, 1])。
// 仅用于BUILTIN_FORMULA；物理层映射不是{{0}, {1}}或物理层信息不足时固定返回0.5。
// MESH_1D_CLOS拓扑不使用自适应切分，直接返回fallbackRatio，不依赖端口信息。
// resPortInfo有效时直接使用其中的端口信息（CCU模式），否则从ChannelInfo Map中提取（AICPU/DPU模式）。
double CalcParallelDataSplitRatio(
    uint64_t intraRankSize, uint64_t interRankSize, const std::map<u32, std::vector<ChannelInfo>>& intraChannels,
    const std::map<u32, std::vector<ChannelInfo>>& interChannels, const ParallelChannelPortInfo& resPortInfo,
    const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    ParallelDataSplitType splitType, double fallbackRatio);
double CalcParallelDataSplitRatio(
    uint64_t intraRankSize, uint64_t interRankSize, const std::vector<u32>& portNum,
    const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    ParallelDataSplitType splitType, double fallbackRatio);

#ifndef AICPU_COMPILE
// CCU模式的资源上下文中没有ChannelInfo，需在CalcRes阶段从机内/机间template的建链请求中采集端口信息。
// 采集失败时portInfo.isValid保持false，仅影响切分比例寻优，不返回错误。
HcclResult CollectParallelPortInfoFromCcuKernels(
    HcclComm comm, u32 userRank, const std::vector<CcuKernelInfo>& intraKernelInfos,
    const std::vector<CcuKernelInfo>& interKernelInfos, ParallelChannelPortInfo& portInfo);
#endif

const char* ParallelDataSplitTypeToStr(ParallelDataSplitType splitType);

HcclResult FillChannelSymWinPeerAddrs(
    void* inputSymWindow, u64 inputOffset, void* outputSymWindow, u64 outputOffset, ChannelInfo& channel);

// 输出vector由调用方预先resize；counts/displs仍使用元素数单位。
inline void CopyAllToAllVCountsAndDispls(const OpParam& param, u32 rankSize, A2ASendRecvInfo& info)
{
    for (u32 j = 0; j < rankSize; j++) {
        // Send info
        u64 curSendCounts = *(static_cast<const u64*>(param.all2AllVDataDes.sendCounts) + j);
        u64 curSendDispls = *(static_cast<const u64*>(param.all2AllVDataDes.sdispls) + j);
        info.sendCounts[j] = curSendCounts;
        info.sendDispls[j] = curSendDispls;

        // Recv info
        u64 curRecvCounts = *(static_cast<const u64*>(param.all2AllVDataDes.recvCounts) + j);
        u64 curRecvDispls = *(static_cast<const u64*>(param.all2AllVDataDes.rdispls) + j);
        info.recvCounts[j] = curRecvCounts;
        info.recvDispls[j] = curRecvDispls;
    }
}

// 本rank单jetty，远端按开关选择；保留resize和逐rank填写的顺序。
inline void FillMeshJettyNums(std::vector<uint32_t>& jettyNums, u32 rankSize, u32 myRank, bool multijetty)
{
    constexpr u32 SINGLE_JETTY_NUM = 1;
    jettyNums.resize(rankSize, 0);
    for (int i = 0; i < rankSize; i++) {
        if (i == myRank) {
            jettyNums[i] = SINGLE_JETTY_NUM;
        } else if (multijetty) {
            jettyNums[i] = MAX_JETTY_NUM;
        } else {
            jettyNums[i] = SINGLE_JETTY_NUM;
        }
    }
}

} // namespace ops_hccl
#endif
