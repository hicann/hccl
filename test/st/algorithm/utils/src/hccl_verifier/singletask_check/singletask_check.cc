/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <map>
#include "singletask_check.h"
#include "log.h"
#include "checker_toolkit.h"

#ifdef HCCL_ALG_ANALYZER_DAVID
#include "alg_adapt_v2_interface.h"
#endif

namespace HcclSim {

// 判断从流开始是不是localwaitfrom开始，localpostto结尾
HcclResult SingleTaskCheck::CheckSlaveTaskQueue(AllRankTaskQueues& allRankTaskQueues)
{
    for (const auto& podIter : allRankTaskQueues) {
        const PodId& podId = podIter.first;
        const auto& serMap = podIter.second;
        for (const auto& serIter : serMap) {
            const SerId& serId = serIter.first;
            const auto& phyMap = serIter.second;
            for (const auto& phyIter : phyMap) {
                const PhyId& phyId = phyIter.first;
                const SingleTaskQueue& taskQueue = phyIter.second;
                u32 queueNum = taskQueue.size();
                for (u32 queueId = 1; queueId < queueNum; queueId++) {
                    u32 taskSize = taskQueue[queueId].size();
                    if (taskSize < 2) {
                        return HCCL_SUCCESS;
                    }
                    std::shared_ptr<TaskStub> firstTask = taskQueue[queueId][0];
                    std::shared_ptr<TaskStub> lastTask = taskQueue[queueId][taskSize - 1];

                    // 为了适配FFTS成图问题，有些算法在从流后面添加了empty task
                    u32 backStep = 1;
                    while (lastTask->GetType() == TaskTypeStub::LOCAL_COPY) {
                        auto task = dynamic_cast<TaskStubLocalCopy *>(lastTask.get());
                        const DataSlice& srcSlice = task->GetSrcSlice();
                        const DataSlice& dstSlice = task->GetDstSlice();
                        if (srcSlice.GetSize() == 0 && dstSlice.GetSize() == 0) {
                            lastTask = taskQueue[queueId][taskSize - 1 - backStep];
                            backStep++;
                            continue;
                        } else {
                            break;
                        }
                    }

                    if (firstTask->GetType() != TaskTypeStub::LOCAL_WAIT_FROM) {
                        HCCL_ERROR("[SlaveStreamCheck]podId:%u, serId:%u, phyId:%u, queueId:%u first task type should be LOCAL_WAIT_FROM, while is %s",
                            podId, serId, phyId, queueId, firstTask->GetType().Describe().c_str());
                        return HCCL_E_INTERNAL;
                    }

                    if (lastTask->GetType() != TaskTypeStub::LOCAL_POST_TO) {
                        HCCL_ERROR("[SlaveStreamCheck]podId:%u, serId:%u, phyId:%u, queueId:%u last task type should be LOCAL_POST_TO, while is %s",
                            podId, serId, phyId, queueId, lastTask->GetType().Describe().c_str());
                        return HCCL_E_INTERNAL;
                    }
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult SingleTaskCheck::CheckSingleSlice(RankId taskRank, u32 queueId, u32 taskId, const DataSlice& slice, RankId sliceRank)
{
    BufferType type = slice.GetType();
    u64 offset = slice.GetOffset();
    u64 size = slice.GetSize();
    u64 blockSize = GetBlockSize(sliceRank, type);
    if (offset + size > blockSize) {
        HCCL_ERROR("Failed to check slice in [rankId:%u, queueId:%u, index:%u], slice is %s, block size is %llu",
            taskRank, queueId, taskId, slice.Describe().c_str(), blockSize);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult SingleTaskCheck::CheckTwoSliceOverlap(RankId rank, u32 queueId, u32 taskId, const DataSlice& sliceA, const DataSlice& sliceB)
{
    if (sliceA.GetType() != sliceB.GetType()) {
        return HCCL_SUCCESS;
    }

    if (sliceA.GetSize() == 0 || sliceB.GetSize() == 0) {
        return HCCL_SUCCESS;
    }

    bool conflictCase1 = sliceA.GetOffset() >= sliceB.GetOffset() && sliceA.GetOffset() < (sliceB.GetOffset() + sliceB.GetSize());
    bool conflictCase2 = sliceB.GetOffset() >= sliceA.GetOffset() && sliceB.GetOffset() < (sliceA.GetOffset() + sliceA.GetSize());
    if (conflictCase1 || conflictCase2) {
        HCCL_ERROR("Slice is conflict in [rankId:%u, queueId:%u, index:%u], one slice is %s, another slice is %s",
            rank, queueId, taskId, sliceA.Describe().c_str(), sliceB.Describe().c_str());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

void SingleTaskCheck::AddChildrenToQueue(TaskNode *node, std::set<TaskNode *> &visitedNodes,
    std::queue<TaskNode *> &walkQue, std::set<TaskNode *> &simulatedNodes)
{
#ifdef HCCL_ALG_ANALYZER_DAVID
    node = Hccl::UpdateNodeForCcuGraph(node, simulatedNodes);
#endif
    for (auto &child : node->children) {
        if (visitedNodes.count(child) != 0) {
            continue;
        }
        walkQue.push(child);
        visitedNodes.insert(child);
    }
    return;
}

HcclResult SingleTaskCheck:: CheckSingleTaskMem(TaskNodePtr curTask)
{
    u32 rankId = curTask->rankIdx;
    u32 queueId = curTask->queIdx;
    u32 taskId = curTask->pos;

    if (!curTask->task){
        return HCCL_SUCCESS;
    }

    TaskTypeStub taskType = curTask->task->GetType();
    if (taskType == TaskTypeStub::LOCAL_COPY) {
        auto task = dynamic_cast<TaskStubLocalCopy *>(curTask->task);
        const DataSlice& srcSlice = task->GetSrcSlice();
        const DataSlice& dstSlice = task->GetDstSlice();
        CHK_RET(CheckSingleSlice(rankId, queueId, taskId, srcSlice, rankId));
        CHK_RET(CheckSingleSlice(rankId, queueId, taskId, dstSlice, rankId));
        CHK_RET(CheckTwoSliceOverlap(rankId, queueId, taskId, srcSlice, dstSlice));
    } else if (taskType == TaskTypeStub::LOCAL_REDUCE) {
        auto task = dynamic_cast<TaskStubLocalReduce *>(curTask->task);
        const DataSlice& srcSlice = task->GetSrcSlice();
        const DataSlice& dstSlice = task->GetDstSlice();
        CHK_RET(CheckSingleSlice(rankId, queueId, taskId, srcSlice, rankId));
        CHK_RET(CheckSingleSlice(rankId, queueId, taskId, dstSlice, rankId));
        CHK_RET(CheckTwoSliceOverlap(rankId, queueId, taskId, srcSlice, dstSlice));
    } else if (taskType == TaskTypeStub::READ) {
        auto task = dynamic_cast<TaskStubRead *>(curTask->task);
        const DataSlice& localSlice = task->GetLocalSlice();
        RankId remoteRank = task->GetRemoteRank();
        const DataSlice& remoteSlice = task->GetRemoteSlice();
        CHK_RET(CheckSingleSlice(rankId, queueId, taskId, localSlice, rankId));
        CHK_RET(CheckSingleSlice(rankId, queueId, taskId, remoteSlice, remoteRank));
    } 
    else if (taskType == TaskTypeStub::READ_REDUCE) {
        auto task = dynamic_cast<TaskStubReadReduce *>(curTask->task);
        const DataSlice& localSlice = task->GetLocalSlice();
        RankId remoteRank = task->GetRemoteRank();
        const DataSlice& remoteSlice = task->GetRemoteSlice();
        CHK_RET(CheckSingleSlice(rankId, queueId, taskId, localSlice, rankId));
        CHK_RET(CheckSingleSlice(rankId, queueId, taskId, remoteSlice, remoteRank));
    } else if (taskType == TaskTypeStub::WRITE) {
        auto task = dynamic_cast<TaskStubWrite *>(curTask->task);
        const DataSlice& localSlice = task->GetLocalSlice();
        RankId remoteRank = task->GetRemoteRank();
        const DataSlice& remoteSlice = task->GetRemoteSlice();
        CHK_RET(CheckSingleSlice(rankId, queueId, taskId, localSlice, rankId));
        CHK_RET(CheckSingleSlice(rankId, queueId, taskId, remoteSlice, remoteRank));
    } else if (taskType == TaskTypeStub::WRITE_REDUCE) {
        auto task = dynamic_cast<TaskStubWriteReduce *>(curTask->task);
        const DataSlice& localSlice = task->GetLocalSlice();
        RankId remoteRank = task->GetRemoteRank();
        const DataSlice& remoteSlice = task->GetRemoteSlice();
        CHK_RET(CheckSingleSlice(rankId, queueId, taskId, localSlice, rankId));
        CHK_RET(CheckSingleSlice(rankId, queueId, taskId, remoteSlice, remoteRank));
    }
    return HCCL_SUCCESS;
}

HcclResult SingleTaskCheck::CheckTaskMem(TaskNodePtr dummyStart)
{
    std::queue<TaskNode*> candNode;
    std::set<TaskNode*> isVisitedNode;
    std::set<TaskNode*> simulatedNodes;

    for (auto child : dummyStart->children) {
        isVisitedNode.insert(child);
        candNode.push(child);
    }

    while(!candNode.empty()) {
        TaskNodePtr curNode = candNode.front();
        candNode.pop();
        AddChildrenToQueue(curNode, isVisitedNode, candNode, simulatedNodes);
        simulatedNodes.insert(curNode);
    }

    for (auto node : simulatedNodes) {
        CHK_RET(CheckSingleTaskMem(node));
    }
    return HCCL_SUCCESS;
}

}

