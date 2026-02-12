/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "check_utils.h"
#include "sim_task.h"

namespace HcclSim {

bool IsAllToAllSeries(HcclCMDType opType)
{
    return (opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
            opType == HcclCMDType::HCCL_CMD_ALLTOALLVC);
}

bool IsSendRecvType(HcclCMDType opType)
{
    return opType == HcclCMDType::HCCL_CMD_SEND || opType == HcclCMDType::HCCL_CMD_RECEIVE;
}

void CalcInputOutputSize(HcclCMDType opType, uint32_t rankSize, uint64_t count, HcclDataType dataType, 
    u64 &inputSize, u64 &outputSize, RankId myRank, RankId srcRank, RankId dstRank,
    VDataDesTag vDataDes, All2AllDataDesTag all2AllDataDes)
{
    u32 unitSize = 0;
    if (!IsAllToAllSeries(opType) &&
        opType != HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V && opType != HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        unitSize = SIZE_TABLE[dataType];
    }

    if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        inputSize = count * unitSize;
        outputSize = count * unitSize;
    } else if (opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        inputSize = count * unitSize;
        outputSize = count * unitSize;
    } else if (IsSendRecvType(opType) && myRank == srcRank) {
        inputSize = count * unitSize;
        outputSize = 0;
    } else if (IsSendRecvType(opType) && myRank == dstRank) {
        inputSize = 0;
        outputSize = count * unitSize;
    } else if (opType == HcclCMDType::HCCL_CMD_REDUCE) {
        // 当前代码中非root节点还是会用到OUTPUT内存块
        outputSize = count * unitSize;
        inputSize = count * unitSize;
    } else if (opType == HcclCMDType::HCCL_CMD_ALLGATHER) {
        inputSize = count * unitSize;
        outputSize = count * unitSize * rankSize;
    } else if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        inputSize = count * unitSize * rankSize;
        outputSize = count * unitSize;
    } else if (opType == HcclCMDType::HCCL_CMD_SCATTER) {
        inputSize = count * unitSize * rankSize;
        outputSize = count * unitSize;
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        u64 curSendOffset = 0;
        u64 curRecvOffset = 0;
        for (u32 j = 0; j < rankSize; j++) {
            u64 curSendCounts = all2AllDataDes.sendCountMatrix[myRank * rankSize + j];
            u64 curSendLength = curSendCounts * SIZE_TABLE[all2AllDataDes.sendType];
            curSendOffset += curSendLength;
            u64 curRecvCounts = all2AllDataDes.sendCountMatrix[myRank + rankSize * j];
            u64 curRecvLength = curRecvCounts * SIZE_TABLE[all2AllDataDes.recvType];
            curRecvOffset += curRecvLength;
        }
        inputSize = curSendOffset;
        outputSize = curRecvOffset;
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        u64 curSendOffset = 0;
        u64 curRecvOffset = 0;
        for (u32 i = 0; i < rankSize; i++) {
            u64 curSendCounts = all2AllDataDes.sendCounts[i];
            u64 curSendLength = curSendCounts * SIZE_TABLE[all2AllDataDes.sendType];
            curSendOffset += curSendLength;

            u64 curRecvCounts = all2AllDataDes.recvCounts[i];
            u64 curRecvLength = curRecvCounts * SIZE_TABLE[all2AllDataDes.recvType];
            curRecvOffset += curRecvLength;
        }
        inputSize = curSendOffset;
        outputSize = curRecvOffset;
    } else if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        inputSize = count * unitSize * rankSize;
        outputSize = count * unitSize * rankSize;
    } else if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        inputSize = 0;
        for (u32 i = 0; i < rankSize; i++) {
            u64 curCounts = vDataDes.counts[i];
            u64 curLength = curCounts * SIZE_TABLE[vDataDes.dataType];
            inputSize += curLength;
        }
        outputSize = vDataDes.counts[myRank] * SIZE_TABLE[vDataDes.dataType];
    } else if (opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        outputSize = 0;
        for (u32 i = 0; i < rankSize; i++) {
            u64 curCounts = vDataDes.counts[i];
            u64 curLength = curCounts * SIZE_TABLE[vDataDes.dataType];
            outputSize += curLength;
        }
        inputSize = vDataDes.counts[myRank] * SIZE_TABLE[vDataDes.dataType];
    } else {
        printf("CalcInputOutputSize not support HcclCMDType [%d]\n", opType);
    }
}

// 如果输入、输出的count的大小不一样的话，那么opParam中的count是指较小的那个值
// 比如对于AllGather算子，count指输入；对于ReduceScatter算子，count指输入
// 如果输入、输出的count大小一样的话，那么opParam中的count既可以指代输入，也可以指代输出
void CalcDataSize(HcclCMDType opType, uint64_t count, HcclDataType dataType, u64 &dataSize)
{
    // 当前AllToAll系列以及不等长算子不使用dataSize，如果后续使用的话，需要适配这个地方
    if (!IsAllToAllSeries(opType) && opType != HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V &&
        opType != HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        u32 unitSize = SIZE_TABLE[dataType];
        dataSize = count * unitSize;
    }
}

std::vector<std::string> SplitString(const std::string &str, const char c)
{
    std::string::size_type startPos = 0;
    std::string::size_type foundPos = str.find(c);
 
    std::vector<std::string> strVector;
    while (foundPos != std::string::npos) {
        strVector.push_back(str.substr(startPos, foundPos - startPos));
        startPos = foundPos + 1;
        foundPos = str.find(c, startPos);
    }
    if (startPos != str.length()) {
        strVector.push_back(str.substr(startPos));
    }
    return strVector;
}

bool DataSliceSizeIsEqual(std::unique_ptr<DataSlice> &a, std::unique_ptr<DataSlice> &b)
{
    return a->GetSize() == b->GetSize();
}
 
bool DataSliceSizeIsEqual(std::unique_ptr<DataSlice> &a, std::unique_ptr<DataSlice> &b, std::unique_ptr<DataSlice> &c)
{
    return (a->GetSize() == b->GetSize()) && (b->GetSize() == c->GetSize());
}

void GenTopoMeta(TopoMeta &topoMate, int superPodNum, int serverNum, int rankNum)
{
    for (u32 i = 0; i < superPodNum; i++) {  // box
        SuperPodMeta superPodMeta;
        for (u32 j = 0; j < serverNum; j++) {  // serverNumPerBox
            ServerMeta serverMate;
            for (u32 k = 0; k < rankNum; k++) {
                serverMate.push_back(k);
            }
            superPodMeta.push_back(serverMate);
        }
        topoMate.push_back(superPodMeta);
    }
}

u32 CalRankSize(const TopoMeta &topoMeta)
{
    u32 rankNum = 0;
    for (const auto& superPod : topoMeta) {
        for (const auto& server : superPod) {
            for (const auto &phyId : server) {
                rankNum++;
            }
        }
    }

    return rankNum;
}

} // namespace hccl