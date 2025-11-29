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