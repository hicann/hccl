/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV1_CHECK_UTILS_H
#define HCCLV1_CHECK_UTILS_H

#include <map>
#include <set>
#include <vector>
#include <memory>
#include "log.h"
#include "checker_def.h"
#include "enum_factory.h"
#include "string_util.h"

namespace HcclSim {

const std::string FOUR_INDENT_SPACE = "    ";

struct SrcBufDes {
    RankId      rankId;  // 数据源的rankId
    BufferType  bufType; // 数据源的内存类型
    mutable u64 srcAddr; // 数据源的地址
    SrcBufDes(RankId id, BufferType type, u64 addr) : rankId(id), bufType(type), srcAddr(addr)
    {
    }
    inline bool operator<(const SrcBufDes &another) const
    {
        return rankId < another.rankId;
    }

    std::string Describe() const
    {
        std::stringstream ret;
        ret << "rankId is " << rankId << ", ";
        ret << "bufType  is " << bufType << ", ";
        ret << "srcAddr is " << srcAddr << std::endl;

        return ret.str();
    }
};

struct BufferSemantic {
    u64                         startAddr;  // 起始地址
    mutable u64                 size;       // 大小
    mutable bool                isReduce;   // 是否做了reduce操作
    mutable HcclReduceOp        reduceType; // reduce操作的类型
    mutable std::set<SrcBufDes> srcBufs;    // 这块数据来自哪个或哪些rank
    std::vector<u32>            affectedGlobalSteps;  // 表示这个语义块被哪个、哪些节点影响了，用于图形化界面展示

    BufferSemantic(u64 startAddr, u64 size, bool isReduce = false,
        HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED)
        : startAddr(startAddr), size(size), isReduce(isReduce), reduceType(reduceType)
    {
    }

    BufferSemantic(u64 startAddr, u64 size, bool isReduce, HcclReduceOp reduceType, std::set<SrcBufDes> srcBufs)
        : startAddr(startAddr), size(size), isReduce(isReduce), reduceType(reduceType), srcBufs(srcBufs)
    {
    }

    inline bool operator<(const BufferSemantic &another) const
    {
        return startAddr < another.startAddr;
    }

    std::string Describe() const
    {
        std::stringstream ret;
        ret << "startAddr is " << startAddr << ", ";
        ret << "size is " << size << ", ";
        if (isReduce) {
            ret << "HcclReduceOp is " << reduceType << "." << std::endl;
        } else {
            ret << "no reduce" << std::endl;
        }
        for (auto &ele : srcBufs) {
            ret << FOUR_INDENT_SPACE << FOUR_INDENT_SPACE << FOUR_INDENT_SPACE << ele.Describe();
        }
        return ret.str();
    }
};

using RankMemorySemantics = std::map<BufferType, std::set<BufferSemantic>>;

void CalcInputOutputSize(HcclCMDType opType, uint32_t rankSize, uint64_t count, HcclDataType dataType,
    u64 &inputSize, u64 &outputSize, RankId myRank, RankId srcRank = 0, RankId dstRank = 0);
void CalcDataSize(HcclCMDType opType, uint64_t count, HcclDataType dataType, u64 &dataSize);
bool IsAllToAllSeries(HcclCMDType opType);
void GenTopoMeta(TopoMeta &topoMate, int superPodNum, int serverNum, int rankNum);
u32 CalRankSize(const TopoMeta &topoMeta);
} // namespace HcclSim

#endif