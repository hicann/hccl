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

#include <cstring>
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <list>
#include "alg_param.h"
#include "binary_stream.h"

namespace ops_hccl {

# define UINT32_MAX     (4294967295U)
constexpr u32 INVALID_U32 = UINT32_MAX;

constexpr s32 INVALID_RANKID = INT32_MAX;

struct SliceInfo {
    u64 offset{0};
    u64 size{0};
};
 
using RankSliceInfo = std::vector<std::vector<SliceInfo>>;

enum class BufferType {
    INPUT = 0,
    OUTPUT = 1,
    HCCL_BUFFER = 2,
    DEFAULT
};

struct DataSlice {
    void* addr_ = nullptr;
    u64 offset_{0}; // Slice相对于input/output的偏移字节数，gather类操作取output，scatter类操作取input
    u64 size_{0};    // Slice的数据大小，单位：字节
    u64 count_{0};   // 数据元素个数

    DataSlice(void* addr, u64 offset, u64 size, u64 count)
    : addr_(addr), offset_(offset), size_(size), count_(count)
    {

    }

    DataSlice(void* addr, u64 offset, u64 size)
    : addr_(addr), offset_(offset), size_(size)
    {
        count_ = 0;
    }

    std::string Describe() const {
        std::ostringstream oss;
        oss << "DataSlice: addr=" << addr_ // 指针地址会自动格式化为十六进制
            << ", offset=" << offset_
            << ", size=" << size_
            << ", count=" << count_;
        return oss.str();
    }
};

struct SlicesList {
    std::vector<DataSlice> srcSlices_;
    std::vector<DataSlice> dstSlices_;

    SlicesList(const std::vector<DataSlice> &srcSlices, const std::vector<DataSlice> &dstSlices)
        : srcSlices_(srcSlices), dstSlices_(dstSlices)
    {
    }
};

struct DataInfo {
    ChannelInfo channel_;
    SlicesList slices_;
    DataInfo(const ChannelInfo &channel, const SlicesList &slices)
    : channel_(channel), slices_(slices)
    {
    }
};

struct DataReduceInfo {
    ChannelInfo channel_;
    SlicesList slices_;
    HcclDataType dataType_;
    HcclReduceOp reduceType_;
    DataReduceInfo(const ChannelInfo &channel, const SlicesList &slices,
             HcclDataType dataType, HcclReduceOp reduceType)
    : channel_(channel), slices_(slices), dataType_(dataType), reduceType_(reduceType)
    {
    }
};

struct TxRxChannels {
    ChannelInfo txChannel_;
    ChannelInfo rxChannel_;

    TxRxChannels(const ChannelInfo &txLink, const ChannelInfo &rxLink) : txChannel_(txLink), rxChannel_(rxLink)
    {
    }
};

struct TxRxSlicesList {
    SlicesList txSlicesList_;
    SlicesList rxSlicesList_;

    TxRxSlicesList(const SlicesList &txSlicesList, const SlicesList &rxSlicesList)
        : txSlicesList_(txSlicesList), rxSlicesList_(rxSlicesList)
    {
    }
};

struct SendRecvInfo {
    TxRxChannels      sendRecvChannels_;
    TxRxSlicesList    sendRecvSlices_;

    SendRecvInfo(const TxRxChannels &sendRecvLinks, const TxRxSlicesList &sendRecvSlices)
        : sendRecvChannels_(sendRecvLinks), sendRecvSlices_(sendRecvSlices)
    {
    }
};

struct SendRecvReduceInfo {
    TxRxChannels      sendRecvChannels_;
    TxRxSlicesList    sendRecvSlices_;
    HcclDataType dataType_;
    HcclReduceOp reduceType_;

    SendRecvReduceInfo(const TxRxChannels &sendRecvLinks, const TxRxSlicesList &sendRecvSlices,
                       const HcclDataType dataType, const HcclReduceOp reduceOp)
        : sendRecvChannels_(sendRecvLinks), sendRecvSlices_(sendRecvSlices), dataType_(dataType), reduceType_(reduceOp)
    {
    }
};

struct BuffInfo {
    void* inputPtr = nullptr; // userIn
    void* outputPtr = nullptr; // userOut
    HcclMem hcclBuff; // 跨Rank缓存Buffer
    BufferType inBuffType;
    BufferType outBuffType;
    BufferType hcclBuffType;
    u64        inputSize          = 0;
    u64        outputSize         = 0;
    u64        hcclBuffSize       = 0;
    u64        inBuffBaseOff      = 0;
    u64        outBuffBaseOff     = 0;
    u64        hcclBuffBaseOff    = 0;
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
    std::vector<u64> allRankSliceSize;
    std::vector<u64> allRankDispls;
    std::vector<u64> allRankProcessedDataCount;
    // alltoallV loop内变长数据
    std::vector<u64> sendCounts;
    std::vector<u64> recvCounts;
    std::vector<u64> sdispls;
    std::vector<u64> rdispls;

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
        binaryStream << allRankSliceSize;
        binaryStream << allRankDispls;

        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
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
        binaryStream >> allRankSliceSize;
        binaryStream >> allRankDispls;
    }
};

struct TemplateResource {
    std::map<u32, std::vector<ChannelInfo>> channels;
    std::vector<ThreadHandle> threads;
    std::vector<CcuKernelHandle> ccuKernels;
    void *npu2DpuShmemPtr;
    void *dpu2NpuShmemPtr;
    void* aivCommInfoPtr = nullptr;
};

struct DPURunInfo { // AICPU构造信息，写入共享内存
    std::string templateName; // DPU算法展开的template名
    TemplateDataParams tempAlgParams;
    std::map<uint32_t, std::vector<ChannelInfo>> channels;
    u32 myRank;
    std::vector<std::vector<uint32_t>> subCommRanks;

    std::vector<char> Serialize() const
    {
        BinaryStream binaryStream;
        binaryStream << templateName;
        binaryStream << tempAlgParams.Serialize();
        binaryStream << channels;
        binaryStream << myRank;
        binaryStream << subCommRanks;

        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> templateName;
        std::vector<char> tempAlgParamsData;
        binaryStream >> tempAlgParamsData;
        tempAlgParams.DeSerialize(tempAlgParamsData);
        binaryStream >> channels;
        binaryStream >> myRank;
        binaryStream >> subCommRanks;
    }
};

struct AicpuNHRStepInfo {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    AicpuNHRStepInfo() : nSlices(0)
    {
    }
};

HcclResult GetAlgRank(const u32 virtRank, const std::vector<u32> &rankIds, u32 &algRank);

u32 GetNHRStepNum(u32 rankSize);

// roundup func for uint
inline u64 RoundUp(const u64 dividend, const u64 divisor)
{
    if (divisor == 0) {
        HCCL_WARNING("[RoundUp] divisor is 0.");
        return dividend;
    }
    return dividend / divisor + ((dividend % divisor != 0) ? 1 : 0);
}

}
#endif