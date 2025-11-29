/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_GRAPH_GENERATOR_H
#define TASK_GRAPH_GENERATOR_H

#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <tuple>

#include "sim_task.h"
#include "log.h"
#include "task_def.h"
#include "sim_task_queue.h"

namespace HcclSim {
using namespace std;

using SeenLocalPost = std::vector<TaskNodePtr>;
using SeenInterRankPosts = std::map<RankId, std::map<RankId, std::vector<TaskNodePtr>>>; // seen Post nodes
using SeenInterRankSendSync = std::map<int32_t*, std::tuple<int32_t, std::vector<TaskNodePtr>, bool>>; // seen SendSync or SendSyncReduce nodes

class TaskGraphGenerator {
public:
    HcclResult GenGraph(const AllRankTaskQueues &taskQueues, TaskNodePtr dummyStart);

private:
    HcclResult GenGraph4Rank(const SingleTaskQueue *rankTaskQueues, const RankId rankIdx,
                             TaskNodePtr dummyStart);
    HcclResult InitRankNodeQue(const SingleTaskQueue *rankTaskQueues, const RankId rankIdx, TaskNodePtr dummyStart,
                               std::vector<TaskNodePtr> &rankNodeQue);
    void       LocateUnmatchedNode(const std::vector<TaskNodePtr> &rankNodeQue);
    HcclResult ExecFlitPrim(const SingleTaskQueue *rankTaskQueues, TaskNodePtr currNode,
                            std::vector<TaskNodePtr> &rankNodeQue, u64 &unmatchedCnt);
    HcclResult ConnectNextAndPushInQue(const SingleTaskQueue *rankTaskQueues, TaskNodePtr currNode,
                                       std::vector<TaskNodePtr> &rankNodeQue);
    HcclResult ExecLocalPostPrim(const SingleTaskQueue *rankTaskQueues, TaskNodePtr currNode,
                                 std::vector<TaskNodePtr> &rankNodeQue, SeenLocalPost &seenLocalPosts,
                                 u64 &unmatchedCnt);
    HcclResult ExecLocalWaitPrim(const SingleTaskQueue *rankTaskQueues, TaskNodePtr currNode,
                                 std::vector<TaskNodePtr> &rankNodeQue, SeenLocalPost &seenLocalPosts,
                                 u64 &unmatchedCnt);
    bool       IsSemPeer(const TaskNodePtr postNode, const TaskNodePtr waitNode);

    HcclResult GenGraphInterRanks(TaskNodePtr dummyStart);
    HcclResult ExecNode4Graph(TaskNodePtr node, std::vector<TaskNodePtr> &graphNodeQue);
    HcclResult ProcNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                              SeenInterRankPosts &seenInterRankPosts, u64 &unmatchedCnt);
    bool       IsExecutable(TaskNodePtr currNode);
    HcclResult ProcInterRankPostNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                           SeenInterRankPosts &seenInterRankPosts, u64 &unmatchedCnt);
    HcclResult ProcInterRankWaitNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                           SeenInterRankPosts &seenInterRankPosts, u64 &unmatchedCnt);
    bool       IsPostWaitPeer(const TaskNodePtr postNode, const TaskNodePtr waitNode);

    //用于处理aiv task相关
    
    // 用于收集分配的node节点并进行析构
    // 如果直接在TaskNode的parent或children节点中放入共享指针，在图规模很大的时候，递归析构可能导致堆栈溢出
    std::vector<std::shared_ptr<TaskNode>> nodes_;
};

} // namespace hccl

#endif