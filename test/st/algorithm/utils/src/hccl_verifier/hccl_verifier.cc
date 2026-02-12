/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_verifier.h"
#include "checker.h"
#include "task_check_op_semantics.h"

using namespace HcclSim;
HcclResult CheckAllReduce(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount, HcclReduceOp reduceType)
{
    Checker checker;
    TaskCheckOpSemantics opSemanticsChecker(rankSize, HcclCMDType::HCCL_CMD_ALLREDUCE, dataType, dataCount);
    opSemanticsChecker.SetReduceType(reduceType);
    CHK_RET(checker.GenAndCheckGraph(taskQueues, opSemanticsChecker));
    return HCCL_SUCCESS;
}

HcclResult CheckAllGather(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount)
{
    Checker checker;
    TaskCheckOpSemantics opSemanticsChecker(rankSize, HcclCMDType::HCCL_CMD_ALLGATHER, dataType, dataCount);
    CHK_RET(checker.GenAndCheckGraph(taskQueues, opSemanticsChecker));
    return HCCL_SUCCESS;
}

HcclResult CheckReduceScatter(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount, HcclReduceOp reduceType)
{
    Checker checker;
    TaskCheckOpSemantics opSemanticsChecker(rankSize, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, dataType, dataCount);
    opSemanticsChecker.SetReduceType(reduceType);
    CHK_RET(checker.GenAndCheckGraph(taskQueues, opSemanticsChecker));
    return HCCL_SUCCESS;
}

HcclResult CheckSend(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount, u32 srcRank, u32 dstRank)
{
    Checker checker;
    TaskCheckOpSemantics opSemanticsChecker(rankSize, HcclCMDType::HCCL_CMD_SEND, dataType, dataCount);
    opSemanticsChecker.SetSrcRank(srcRank);
    opSemanticsChecker.SetDstRank(dstRank);
    CHK_RET(checker.GenAndCheckGraph(taskQueues, opSemanticsChecker));
    return HCCL_SUCCESS;
}

HcclResult CheckRecv(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount, u32 srcRank, u32 dstRank)
{
    Checker checker;
    TaskCheckOpSemantics opSemanticsChecker(rankSize, HcclCMDType::HCCL_CMD_RECEIVE, dataType, dataCount);
    opSemanticsChecker.SetSrcRank(srcRank);
    opSemanticsChecker.SetDstRank(dstRank);
    CHK_RET(checker.GenAndCheckGraph(taskQueues, opSemanticsChecker));
    return HCCL_SUCCESS;
}

HcclResult CheckBroadcast(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount, u32 root)
{
    Checker checker;
    TaskCheckOpSemantics opSemanticsChecker(rankSize, HcclCMDType::HCCL_CMD_BROADCAST, dataType, dataCount);
    opSemanticsChecker.SetRoot(root);
    CHK_RET(checker.GenAndCheckGraph(taskQueues, opSemanticsChecker));
    return HCCL_SUCCESS;
}

HcclResult CheckReduce(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount, HcclReduceOp reduceType, u32 root)
{
    Checker checker;
    TaskCheckOpSemantics opSemanticsChecker(rankSize, HcclCMDType::HCCL_CMD_REDUCE, dataType, dataCount);
    opSemanticsChecker.SetReduceType(reduceType);
    opSemanticsChecker.SetRoot(root);
    CHK_RET(checker.GenAndCheckGraph(taskQueues, opSemanticsChecker));
    return HCCL_SUCCESS;
}

HcclResult CheckScatter(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount, u32 root)
{
    Checker checker;
    TaskCheckOpSemantics opSemanticsChecker(rankSize, HcclCMDType::HCCL_CMD_SCATTER, dataType, dataCount);
    opSemanticsChecker.SetRoot(root);
    CHK_RET(checker.GenAndCheckGraph(taskQueues, opSemanticsChecker));
    return HCCL_SUCCESS;
}

HcclResult CheckBatchSendRecv(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount)
{
    Checker checker;
    TaskCheckOpSemantics opSemanticsChecker(rankSize, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, dataType, dataCount);
    CHK_RET(checker.GenAndCheckGraph(taskQueues, opSemanticsChecker));
    return HCCL_SUCCESS;
}

HcclResult CheckAll2All(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType sendType, u64 sendCount)
{
    All2AllDataDesTag all2AllDataDes;
    all2AllDataDes.sendType = sendType;
    all2AllDataDes.recvType = sendType;
    all2AllDataDes.sendCount = sendCount;
    all2AllDataDes.recvCount = sendCount;
    std::vector<u64> sendCountMatrix(rankSize * rankSize, sendCount);
    all2AllDataDes.sendCountMatrix = sendCountMatrix;
    Checker checker;
    TaskCheckOpSemantics opSemanticsChecker(rankSize, HcclCMDType::HCCL_CMD_ALLTOALL, sendType, sendCount);
    opSemanticsChecker.SetAll2AllDataDes(all2AllDataDes);
    CHK_RET(checker.GenAndCheckGraph(taskQueues, opSemanticsChecker));
    return HCCL_SUCCESS;
}

HcclResult CheckAll2AllV(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType sendType, std::vector<u64> sendCountMatrix)
{
    All2AllDataDesTag all2AllDataDes;
    all2AllDataDes.sendType = sendType;
    all2AllDataDes.recvType = sendType;
    all2AllDataDes.sendCountMatrix = sendCountMatrix;
    Checker checker;
    // 复用all2allVC的校验逻辑
    TaskCheckOpSemantics opSemanticsChecker(rankSize, HcclCMDType::HCCL_CMD_ALLTOALLVC, sendType, 0);
    opSemanticsChecker.SetAll2AllDataDes(all2AllDataDes);
    CHK_RET(checker.GenAndCheckGraph(taskQueues, opSemanticsChecker));
    return HCCL_SUCCESS;
}

HcclResult CheckAll2AllVC(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType sendType, std::vector<u64> sendCountMatrix)
{
    All2AllDataDesTag all2AllDataDes;
    all2AllDataDes.sendType = sendType;
    all2AllDataDes.recvType = sendType;
    all2AllDataDes.sendCountMatrix = sendCountMatrix;
    Checker checker;
    TaskCheckOpSemantics opSemanticsChecker(rankSize, HcclCMDType::HCCL_CMD_ALLTOALLVC, sendType, 0);
    opSemanticsChecker.SetAll2AllDataDes(all2AllDataDes);
    CHK_RET(checker.GenAndCheckGraph(taskQueues, opSemanticsChecker));
    return HCCL_SUCCESS;
}

HcclResult CheckAllGatherV(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, VDataDesTag vDataDes)
{
    Checker checker;
    TaskCheckOpSemantics opSemanticsChecker(rankSize, HcclCMDType::HCCL_CMD_ALLGATHER_V, vDataDes.dataType, 0);
    opSemanticsChecker.SetVDataDes(vDataDes);
    CHK_RET(checker.GenAndCheckGraph(taskQueues, opSemanticsChecker));
    return HCCL_SUCCESS;
}

HcclResult CheckReduceScatterV(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclReduceOp reduceType, VDataDesTag vDataDes)
{
    Checker checker;
    TaskCheckOpSemantics opSemanticsChecker(rankSize, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, vDataDes.dataType, 0);
    opSemanticsChecker.SetReduceType(reduceType);
    opSemanticsChecker.SetVDataDes(vDataDes);
    CHK_RET(checker.GenAndCheckGraph(taskQueues, opSemanticsChecker));
    return HCCL_SUCCESS;
}