/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VERIFIER_H
#define HCCL_VERIFIER_H
#include "hccl/hccl_types.h"
#include "hccl/base.h"
#include "sim_task_queue.h"
#include "checker.h"

/**
 * @brief 检查AllReduce操作的语义是否正确
 *
 * @param taskQueues 所有rank的任务队列
 * @param rankSize 总的rank数量
 * @param dataType 数据类型
 * @param dataCount 数据元素个数
 * @param reduceType 归约操作类型
 * @return HcclResult操作结果
 */
HcclResult CheckAllReduce(HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount,
    HcclReduceOp reduceType);

/**
 * @brief 检查AllGather操作的语义是否正确
 *
 * @param taskQueues 所有rank的任务队列
 * @param rankSize 总的rank数量
 * @param dataType 数据类型
 * @param dataCount 数据元素个数
 * @return HcclResult操作结果
 */
HcclResult CheckAllGather(HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount);

/**
 * @brief 检查ReduceScatter操作的语义是否正确
 *
 * @param taskQueues 所有rank的任务队列
 * @param rankSize 总的rank数量
 * @param dataType 数据类型
 * @param dataCount 数据元素个数
 * @param reduceType 归约操作类型
 * @return HcclResult操作结果
 */
HcclResult CheckReduceScatter(HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType,
    u64 dataCount, HcclReduceOp reduceType);

/**
 * @brief 检查Send操作的语义是否正确
 *
 * @param taskQueues 所有rank的任务队列
 * @param rankSize 总的rank数量
 * @param dataType 数据类型
 * @param dataCount 数据元素个数
 * @param srcRank 源rank编号
 * @param dstRank 目的rank编号
 * @return HcclResult操作结果
 */
HcclResult CheckSend(HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount,
    u32 srcRank, u32 dstRank);

/**
 * @brief 检查Recv操作的语义是否正确
 *
 * @param taskQueues 所有rank的任务队列
 * @param rankSize 总的rank数量
 * @param dataType 数据类型
 * @param dataCount 数据元素个数
 * @param srcRank 源rank编号
 * @param dstRank 目的rank编号
 * @return HcclResult操作结果
 */
HcclResult CheckRecv(HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount,
    u32 srcRank, u32 dstRank);

/**
 * @brief 检查Broadcast操作的语义是否正确
 *
 * @param taskQueues 所有rank的任务队列
 * @param rankSize 总的rank数量
 * @param dataType 数据类型
 * @param dataCount 数据元素个数
 * @param root 根节点rank编号
 * @return HcclResult操作结果
 */
HcclResult CheckBroadcast(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount, u32 root);

/**
 * @brief 检查Broadcast操作的语义是否正确
 *
 * @param taskQueues 所有rank的任务队列
 * @param rankSize 总的rank数量
 * @param dataType 数据类型
 * @param dataCount 数据元素个数
 * @param reduceType 归约操作类型
 * @param root 根节点rank编号
 * @return HcclResult操作结果
 */
HcclResult CheckReduce(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount, HcclReduceOp reduceType, u32 root);

/**
 * @brief 检查Scatter操作的语义是否正确
 *
 * @param taskQueues 所有rank的任务队列
 * @param rankSize 总的rank数量
 * @param dataType 数据类型
 * @param dataCount 数据元素个数
 * @param root 根节点rank编号
 * @return HcclResult操作结果
 */
HcclResult CheckScatter(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount, u32 root);

/**
 * @brief 检查BatchSendRecv操作的语义是否正确
 *
 * @param taskQueues 所有rank的任务队列
 * @param rankSize 总的rank数量
 * @param dataType 数据类型
 * @param dataCount 数据元素个数
 * @return HcclResult操作结果
 */
HcclResult CheckBatchSendRecv(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount);

HcclResult CheckAll2All(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType sendType, u64 sendCount);

HcclResult CheckAll2AllV(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType sendType, std::vector<u64> sendCountMatrix);

HcclResult CheckAll2AllVC(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclDataType sendType, std::vector<u64> sendCountMatrix);

HcclResult CheckAllGatherV(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, VDataDesTag vDataDes);

HcclResult CheckReduceScatterV(
    HcclSim::AllRankTaskQueues &taskQueues, u32 rankSize, HcclReduceOp reduceType, VDataDesTag vDataDes);

#endif