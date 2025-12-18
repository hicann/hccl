/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALG_TEMPLATE_BASE_PUB_H
#define ALG_TEMPLATE_BASE_PUB_H

#include <vector>
#include <memory>
#include <list>
#include "hccl/base.h"
#include "alg_param.h"
#include "utils.h"

namespace ops_hccl {

constexpr s32 HCCL_EXEC_STAGE_NOT_SET = -1;
constexpr s32 HCCL_EXEC_STEP_NOT_SET = -1;
constexpr s32 HCCL_EXEC_PLANE_NOT_SET = -1;
constexpr u64 ZERO_SLICE = 0;
constexpr u32 TWO_RANK_SIZE = 2;
constexpr u32 DMA_REDUCE_TWO_OFFSET = 2;
constexpr u32 DMA_REDUCE_THREE_OFFSET = 3;
constexpr u64 HCCL_CHUNK_SIZE = 1024 * 1024 * 1024; // 1024*1024*1024的size
constexpr u64 HCCL_MIN_PIPLINE_SLICE_ALIGN = 512;
constexpr u64 HCCL_MIN_SLICE_ALIGN_910B = 16384;
constexpr u64 HCCL_MIN_SLICE_ALIGN_910_93 = 16384;
constexpr u64 HCCL_MIN_SLICE_ALIGN_ONCHIP = 512;
constexpr u64 HCCL_MIN_SLICE_ALIGN = 128;
constexpr u64 HCCL_NIC_MAX_NUM = 8;
constexpr u64 DOUBLE_RING_NUM = 2;
constexpr u64 DOUBLE_RING_STREAM_NUM = 3;
constexpr u32 ALIGNED_SUB_RING_INDEX = 0;
constexpr u32 ALIGNED_MAIN_RING_INDEX = 1;

// AnyPath相关，SDMA数据量切分比例
constexpr u32 MAX_SPLIT_VALUE = 100;
constexpr u32 BEST_SPLIT_VALUE_SR = 87;
constexpr u32 BEST_SPLIT_VALUE_DR = 90;
constexpr u64 HCCL_SPLIT_SIZE_INTER_SERVER = 8388608; // 每卡通信量的切分边界

enum TemplateType {
    TEMPLATE_SCATTER_MESH = 0,               // ScatterMesh
    TEMPLATE_SCATTER_RING = 1,               // ScatterRing
    TEMPLATE_SCATTER_NB = 2,                 // ScatterNB
    TEMPLATE_SCATTER_NHR = 3,                // ScatterNHR
    TEMPLATE_SCATTER_RING_DIRECT = 4,        // ScatterRingDirect

    TEMPLATE_NATIVE_MAX_NUM,                        // 内置template最大值

    TEMPLATE_CUSTOM_BEGIN = 1000,                   // 用户自定义template起始值
    TEMPLATE_CUSTOM_MAX_NUM = 2000                  // 用户自定义template最大值
};

enum class SliceType {
    SLICE_TYPE_TX,
    SLICE_TYPE_RX
};

enum class HalvingDoublingType {
    BINARY_BLOCK_HALVING_DOUBLING,
    RECURSIVE_HALVING_DOUBLING,
    RESERVED_ALGORITHM_TYPE
};

using SliceType = enum SliceType;

enum class RunStage {
    RUN_PREPARE,
    RUN_REDUCE_SCATTER,
    RUN_ALLGATHER,
    RUN_ALLREDUCE,
    RUN_DEFAULT
};

struct PrepareData {
    u32 root = INVALID_VALUE_RANKID;
    u32 userRank = INVALID_VALUE_RANKID;
    u32 userRankSize = 0;
    u32 interRank = INVALID_VALUE_RANKID;
    u32 interRankSize = 0;

    u64 count = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reductionOp = HCCL_REDUCE_RESERVED;
    u64 baseOffset = 0;

    HcclMem inputMem;
    HcclMem outputMem;
    HcclMem scratchMem;
    HcclMem cclInMem;
    HcclMem cclOutMem;

    ThreadHandle thread;
    const std::vector<ThreadHandle>* subStreamsPtr = nullptr;

    const std::vector<Slice>* slicesPtr = nullptr;
    const std::vector<std::vector<Slice>>* multRingsSlicesPtr = nullptr;
    const std::vector<u32>* nicRankListPtr = nullptr;

    HcomCollOpInfo *opInfo = nullptr;
    bool disableDMAReduce = false;
    bool isSuPodAsym = false;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;

    u32 devNumInlocalPod = 0;
    u32 rankIdxInPod = 0;
    u64 reduceAttr = 0;
};

class AlgTemplateBase {
public:
    explicit AlgTemplateBase();
    virtual ~AlgTemplateBase();

    virtual HcclResult RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels);
     /* 12个参数 */
    virtual HcclResult Prepare(HcclMem &inputMem, HcclMem &outputMem, HcclMem &scratchMem, const u64 count,
                         const HcclDataType dataType, ThreadHandle thread,
                         const HcclReduceOp reductionOp = HCCL_REDUCE_RESERVED,
                         const u32 root = INVALID_VALUE_RANKID,
                         const std::vector<Slice> &slices = std::vector<Slice>(ZERO_SLICE),
                         const u64 baseOffset = 0, const bool disableDMAReduce = false);

    virtual HcclResult Prepare(PrepareData &param);

    // ScatterMesh
    virtual HcclResult Prepare(u32 interRank, u32 interRankSize);

    virtual HcclResult Prepare(HcomCollOpInfo *opInfo, const u32 userRank, const std::vector<u32> &ringsOrders,
        const std::vector<Slice> &userMemInputSlices);

    HcclResult Sum(const std::vector<Slice> &inputSlices, u32 start, u32 num, u64 &sizeOut);
    static HcclResult ExecEmptyTask(HcclMem &inputMem, HcclMem &outputMem, ThreadHandle thread);
    HcclResult CheckConcurrentDirectParameters(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels);
    u32 DataUnitSize(HcclDataType dataType) const
    {
        if (dataType >= HCCL_DATA_TYPE_RESERVED) {
            HCCL_ERROR("[AlgTemplateBase][DataUnitSize]data type[%s] out of range[%d, %d]",
                GetDataTypeEnumStr(dataType).c_str(), HCCL_DATA_TYPE_INT8, static_cast<int>(HCCL_DATA_TYPE_RESERVED) - 1);
            return 0;
        }

        return SIZE_TABLE[dataType];
    }

    static std::vector<bool> CalcLinksRelation(const u32 rank, const u32 rankSize, const u32 rootRank = 0,
        HalvingDoublingType algorithmType = HalvingDoublingType::RECURSIVE_HALVING_DOUBLING);

    static HcclResult PrepareSliceData(u64 dataCount, u32 unitSize, u32 sliceNum, u64 piplineOffset,
        std::vector<Slice> &dataSlice);
    static HcclResult PrepareSliceMeshStreams(const std::vector<Slice> &rankSegsSlice, u32 streamCount,
        std::vector<std::vector<Slice>> &mutliStreamsSlices);

    inline u64 ByteOffset(u64 countOffset) const
    {
        return countOffset * DataUnitSize(dataType_);
    }
    inline u64 SliceOffset(u32 sliceIndex, u64 countPerSlice) const
    {
        return sliceIndex * countPerSlice * DataUnitSize(dataType_);
    }
    inline void CloseBarrier()
    {
        barrierSwitchOn_ = false;
    }

protected:
    HcclResult ExecuteBarrier(ChannelInfo &channel, ThreadHandle thread);
    HcclResult ExecuteBarrier(ChannelInfo &preChannel, ChannelInfo &aftChannel);
    HcclResult ExecuteBarrier(ChannelInfo &preChannel, ChannelInfo &aftChannel, ThreadHandle thread);

    // 下面这组是否需要？
    HcclResult ExecuteBarrier(ChannelInfo &preChannel, ChannelInfo &aftChannel, u32 notifyIdx);
    HcclResult ExecuteBarrier(ChannelInfo &preChannel, ChannelInfo &aftChannel, u32 notifyIdx, ThreadHandle thread);

    std::vector<Slice> slicesDummy_;
    std::vector<Slice> &slices_;
    HcclMem inputMem_;   /* * 输入memory */
    HcclMem outputMem_;  /* * 输出memory */
    HcclMem scratchMem_; /* * 草稿memory */

    u64 count_; //  需处理的每块memory数据总个数
    u64 dataBytes_; //  数据所占的字节数
    HcclDataType dataType_;
    HcclReduceOp reductionOp_;
    u32 root_;
    u32 unitSize_;
    bool disableDMAReduce_;

    u64 baseOffset_;

    ThreadHandle thread_;

    bool barrierSwitchOn_;
    // 用于91093 aligend double ring算法
    std::vector<std::vector<Slice>> multRingsSlices_;
private:
    static void CalcBinaryBlockParams(u32 rank, u32 rankSize, u32 &stepsInBlock, u32 &lowerBlockSize,
        u32 &myBlockSize, u32 &rankInMyBlock, u32 &myBlockOffset, u32 &higherBlockSize);

    static void CalcLinkInBlock(u32 blockSize, u32 rankInBlock, std::list<u32> &linkRankIndexInBlock);
    static void CalcLinkBetweenParts(u32 part1Size, std::list<u32> &linkRankIndexInBlock,
                                             std::list<u32> &linkRankIndex, bool oddRank);
    static void CalcRecursiveHalvingDobuleLinkReleation(u32 rank, u32 rankSize, u32 rootRank,
                                                                   std::vector<bool> &linkRelation);
    static void CalcRecursiveHdLinkRelationForFirstScene(u32 rank,
        u32 part1Size, u32 blockSize, std::vector<bool> &linkRelation);
    static void CalcRecursiveHdLinkRelationForSecondScene(u32 rank,
        u32 part1Size, u32 blockSize, std::vector<bool> &linkRelation);
};
}

#endif /* EXECUTOR_BASE_PUB_H */
