/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_graph_generator.h"

namespace HcclSim {

HcclResult TaskGraphGenerator::GenGraph(const AllRankTaskQueues &allRankTaskQueues, TaskNodePtr dummyStart)
{
    u32 rankId = 0;
    for (auto& podIter : allRankTaskQueues) {
        auto& serMap = podIter.second;
        for (auto& serIter : serMap) {
            auto& phyMap = serIter.second;
            for (auto& phyIter : phyMap) {
                const SingleTaskQueue *taskQueue = &phyIter.second;
                CHK_RET(GenGraph4Rank(taskQueue, rankId, dummyStart));
                HCCL_DEBUG("[TaskGraphGenerator] Rank [%d], local dependency graph generation done.", rankId);
                rankId++;
            }
        }
    }
    HCCL_DEBUG("[TaskGraphGenerator] rankSize [%u] and numChildren of dummyStart [%u].", rankId,
               dummyStart->children.size());

    /*
    Mismatch may occur when: 1) fail to generate local dependency graph correctly --> ERROR
                             2) a group prim is placed at the begining of the primitive queue
    */

    CHK_RET(GenGraphInterRanks(dummyStart));

    //Handling the synchronization relationship between AIV, if exist.
    // 将CCU微码序列转换为Task子图

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::GenGraph4Rank(const SingleTaskQueue* rankTaskQueues, const RankId rankIdx,
                                             TaskNodePtr dummyStart)
{
    std::vector<TaskNodePtr> rankNodeQue;      // executable task nodes
    SeenLocalPost            seenLocalPosts;   // seen Local Posts
    u64                      unmatchedCnt = 0; // for deadlock checking

    CHK_PRT_RET(InitRankNodeQue(rankTaskQueues, rankIdx, dummyStart, rankNodeQue) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to  init rankNodeQue.", rankIdx),
                HcclResult::HCCL_E_INTERNAL);

    while (!rankNodeQue.empty()) {
        if (unmatchedCnt >= rankNodeQue.size()) {
            // DeadLocking
            for (auto &rankNodeUnmatch : rankNodeQue) {
                rankNodeUnmatch->unmatch = true;
            }
            HCCL_ERROR("[TaskGraphGenerator] deadLocking occurs due to mismatch of LOCAL_POST_TO and LOCAL_WAIT_FROM.");
            LocateUnmatchedNode(rankNodeQue);
            return HcclResult::HCCL_E_INTERNAL;
        }

        TaskNodePtr currNode = rankNodeQue[0];
        rankNodeQue.erase(rankNodeQue.begin());

        switch (currNode->task->GetType()) {
            case TaskTypeStub::LOCAL_COPY:
            case TaskTypeStub::LOCAL_REDUCE:
            case TaskTypeStub::POST:
            case TaskTypeStub::WAIT:
            case TaskTypeStub::READ:
            case TaskTypeStub::READ_REDUCE:
            case TaskTypeStub::WRITE:
            case TaskTypeStub::WRITE_REDUCE:
                CHK_RET(ExecFlitPrim(rankTaskQueues, currNode, rankNodeQue, unmatchedCnt));
                break;

            case TaskTypeStub::LOCAL_POST_TO:
                CHK_RET(ExecLocalPostPrim(rankTaskQueues, currNode, rankNodeQue, seenLocalPosts, unmatchedCnt));
                break;

            case TaskTypeStub::LOCAL_WAIT_FROM:
                CHK_RET(ExecLocalWaitPrim(rankTaskQueues, currNode, rankNodeQue, seenLocalPosts, unmatchedCnt));
                break;

            default:
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], taskType not supported.", rankIdx);
                return HcclResult::HCCL_E_INTERNAL;
        }
    }
    if (!seenLocalPosts.empty()){
        for (auto &localPost : seenLocalPosts) {
            localPost->unmatch = true;
            HCCL_ERROR("[TaskGraphGenerator] unmatched local_post: %s.",
                       localPost->GenPosInfo().c_str());
            return HcclResult::HCCL_E_INTERNAL;
	    }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::InitRankNodeQue(const SingleTaskQueue *rankTaskQueues, const RankId rankIdx,
                                               TaskNodePtr dummyStart, std::vector<TaskNodePtr> &rankNodeQue)
{
    // first task of master queue
    auto currNode = std::make_shared<TaskNode>(rankTaskQueues[0][0][0].get(), rankIdx, 0, 0);
    CHK_PTR_NULL(currNode);
    nodes_.push_back(currNode);

    dummyStart->children.push_back(currNode.get());
    currNode->parents.push_back(dummyStart);
    rankNodeQue.push_back(currNode.get());
    HCCL_DEBUG("[TaskGraphGenerator] Rank [%d], connect dummyStart -> first taskNode of master queue, put taskNode in "
               "rankNodeQue",
               rankIdx);

    // first task of slave queues: first task should be local wait
    for (u32 qIdx = 1; qIdx < rankTaskQueues[0].size(); qIdx++) {
        // 表明这条流未下发任何Task
        if (rankTaskQueues[0][qIdx].size() == 0) {
            continue;
        }
        auto currNode = std::make_shared<TaskNode>(rankTaskQueues[0][qIdx][0].get(), rankIdx, 0, 0);

        CHK_PTR_NULL(currNode);
        nodes_.push_back(currNode);
        CHK_PRT_RET(
            currNode->task->GetType() != TaskTypeStub::LOCAL_WAIT_FROM,
            HCCL_ERROR("[TaskGraphGenerator] Rank[%d], Que [%u], first task of slave queue should be localWaitFrom.",
                       currNode->rankIdx, currNode->queIdx),
            HcclResult::HCCL_E_INTERNAL);
        rankNodeQue.push_back(currNode.get());
        HCCL_DEBUG("[TaskGraphGenerator] Rank [%d], put first taskNode of slave queues in rankNodeQue", rankIdx);
    }

    return HcclResult::HCCL_SUCCESS;
}

void TaskGraphGenerator::LocateUnmatchedNode(const std::vector<TaskNodePtr> &rankNodeQue)
{
    auto rankNodeIter = rankNodeQue.begin();
    for (; rankNodeIter != rankNodeQue.end(); rankNodeIter++) {
        HCCL_ERROR("[TaskGraphGenerator] unmatched task locates in: %s", (*rankNodeIter)->GenPosInfo().c_str());
        return;
    }
    HCCL_ERROR("[TaskGraphGenerator] Checker internal error, deadlock is not due to mismatch of local sync.");
    return;
}

HcclResult TaskGraphGenerator::ExecFlitPrim(const SingleTaskQueue *rankTaskQueues, TaskNodePtr currNode,
                                            std::vector<TaskNodePtr> &rankNodeQue, u64 &unmatchedCnt)
{
    // curr -> its nxt, push nxt to nodeQue
    CHK_PRT_RET(ConnectNextAndPushInQue(rankTaskQueues, currNode, rankNodeQue) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to generate dependency graph: TaskType [%s].",
                           currNode->rankIdx, currNode->task->GetType().Describe().c_str()),
                HcclResult::HCCL_E_INTERNAL);
    unmatchedCnt = 0;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ConnectNextAndPushInQue(const SingleTaskQueue *rankTaskQueues, TaskNodePtr currNode,
                                                       std::vector<TaskNodePtr> &rankNodeQue)
{
    if (currNode->pos < rankTaskQueues[0][currNode->queIdx].size() - 1) {
        auto nxtNode = std::make_shared<TaskNode>(rankTaskQueues[0][currNode->queIdx][currNode->pos + 1].get(),
                                                  currNode->rankIdx, currNode->queIdx, currNode->pos + 1);
        CHK_PTR_NULL(nxtNode);
        nodes_.push_back(nxtNode);
        nxtNode->parents.push_back(currNode);
        currNode->children.push_back(nxtNode.get());
        rankNodeQue.push_back(nxtNode.get());
    } else {
        HCCL_DEBUG("[TaskGraphGenerator] Rank [%d], end of current Que [%u]: TrimType [%s].", currNode->rankIdx,
                   currNode->queIdx, currNode->task->GetType().Describe().c_str());
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecLocalPostPrim(const SingleTaskQueue *rankTaskQueues, TaskNodePtr currNode,
                                                 std::vector<TaskNodePtr> &rankNodeQue, SeenLocalPost &seenLocalPosts,
                                                 u64 &unmatchedCnt)
{
    // curr -> its nxt, nxt in nodeQue
    CHK_PRT_RET(ConnectNextAndPushInQue(rankTaskQueues, currNode, rankNodeQue) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to generate dependency graph: TaskType [%s].",
                           currNode->rankIdx, "LocalPostTo"),
                HcclResult::HCCL_E_INTERNAL);

    seenLocalPosts.push_back(currNode);
    unmatchedCnt = 0;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecLocalWaitPrim(const SingleTaskQueue *rankTaskQueues, TaskNodePtr currNode,
                                                 std::vector<TaskNodePtr> &rankNodeQue, SeenLocalPost &seenLocalPosts,
                                                 u64 &unmatchedCnt)
{
    std::vector<TaskNodePtr>::iterator postIter;
    for (postIter = seenLocalPosts.begin(); postIter != seenLocalPosts.end(); postIter++) {
        if (IsSemPeer((*postIter), currNode)) {
            (*postIter)->children.push_back(currNode);
            currNode->parents.push_back((*postIter));    // local_post_to -> curr local_wait_from
            (seenLocalPosts).erase(postIter); // remove local_post_to from seenLocalPosts
            // curr local_wait_from -> its nxt
            CHK_PRT_RET(
                ConnectNextAndPushInQue(rankTaskQueues, currNode, rankNodeQue) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to generate dependency graph: TaskType [%s].",
                            currNode->rankIdx, "LocalWaitFrom"),
                HcclResult::HCCL_E_INTERNAL);
            unmatchedCnt = 0;
            return HcclResult::HCCL_SUCCESS;
        }
    }

    // corresponding post not seen yet
    rankNodeQue.push_back(currNode); // local_wait_from cannot be executed, push back to node que
    unmatchedCnt++;
    return HcclResult::HCCL_SUCCESS;
}

bool TaskGraphGenerator::IsSemPeer(const TaskNodePtr postNode, const TaskNodePtr waitNode)
{
    return false;
}

HcclResult TaskGraphGenerator::GenGraphInterRanks(TaskNodePtr dummyStart)
{
    std::vector<TaskNodePtr> graphNodeQue;       // executable primnodes
    SeenInterRankPosts       seenInterRankPosts; // seen inter-rank Posts
    u64                      unmatchedCnt = 0;   // for deadlock checking

    CHK_PRT_RET(ExecNode4Graph(dummyStart, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Fail to init graphNodeQue."), HcclResult::HCCL_E_INTERNAL);

    while (!graphNodeQue.empty()) {
        if (unmatchedCnt >= graphNodeQue.size()) {
            for (auto &graphNodeUnmatch : graphNodeQue) {
                graphNodeUnmatch->unmatch = true;
            }
            HCCL_ERROR("[TaskGraphGenerator] deadLocking occurs due to mismatch of inter-rank Post/Wait.");
            LocateUnmatchedNode(graphNodeQue);
            return HcclResult::HCCL_E_INTERNAL;
        }

        TaskNodePtr currNode = graphNodeQue[0];
        graphNodeQue.erase(graphNodeQue.begin());

        CHK_PRT_RET(ProcNode4Graph(currNode, graphNodeQue, seenInterRankPosts, unmatchedCnt)
                        != HcclResult::HCCL_SUCCESS,
                    HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to proceed taskNode.", currNode->rankIdx),
                    HcclResult::HCCL_E_INTERNAL);
    }

    bool hasChanged = false;
    if (!seenInterRankPosts.empty()) {
        for (auto &curRankPosts : seenInterRankPosts) {
            for (auto &peerRankPosts : curRankPosts.second) {
                for (auto &post : peerRankPosts.second) {
                    post->unmatch = true;
                    HCCL_ERROR("[TaskGraphGenerator] unmatched inter-rank post: %s, PeerRank [%d],  ",
                               post->GenPosInfo().c_str(), peerRankPosts.first);
                    hasChanged = true;
                }
            }
        }
    }
    if (hasChanged) {
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecNode4Graph(TaskNodePtr node, std::vector<TaskNodePtr> &graphNodeQue)
{
    node->execFlag = true;
    std::vector<TaskNodePtr>::iterator childIter = node->children.begin();
    for (; childIter != node->children.end(); childIter++) {
        if (!(*childIter)->travFlag) {
            (*childIter)->travFlag = true;
            graphNodeQue.push_back((*childIter));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ProcNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                              SeenInterRankPosts &seenInterRankPosts, u64 &unmatchedCnt)
{
    // taskNode not executable
    if (!IsExecutable(currNode)) {
        graphNodeQue.push_back(currNode);
        unmatchedCnt++;
        HCCL_DEBUG("[TaskGraphGenerator] taskNode not executable, push back to the queue.");
        return HcclResult::HCCL_SUCCESS;
    }

    switch (currNode->task->GetType()) {
        case TaskTypeStub::LOCAL_COPY:
        case TaskTypeStub::LOCAL_REDUCE:
        case TaskTypeStub::LOCAL_POST_TO:
        case TaskTypeStub::LOCAL_WAIT_FROM:
        case TaskTypeStub::READ:
        case TaskTypeStub::READ_REDUCE:
        case TaskTypeStub::WRITE:
        case TaskTypeStub::WRITE_REDUCE:
            CHK_PRT_RET(ExecNode4Graph(currNode, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                        HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to execute taskNode.", currNode->rankIdx),
                        HcclResult::HCCL_E_INTERNAL);
            unmatchedCnt = 0;
            return HcclResult::HCCL_SUCCESS;

        case TaskTypeStub::POST:
            // put in seenInterRankPosts
            CHK_RET(ProcInterRankPostNode4Graph(currNode, graphNodeQue, seenInterRankPosts, unmatchedCnt));
            return HcclResult::HCCL_SUCCESS;

        case TaskTypeStub::WAIT:
            // check if peer task in seenInterRankPosts
            CHK_RET(ProcInterRankWaitNode4Graph(currNode, graphNodeQue, seenInterRankPosts, unmatchedCnt));
            break;

        default:
            HCCL_ERROR("[TaskGraphGenerator] taskType %s not supported.", currNode->task->GetType().Describe().c_str());
            return HcclResult::HCCL_E_INTERNAL;
    }

    return HcclResult::HCCL_SUCCESS;
}

bool TaskGraphGenerator::IsExecutable(TaskNodePtr currNode)
{
    std::vector<TaskNodePtr>::iterator parentIter = currNode->parents.begin();
    for (; parentIter != currNode->parents.end(); parentIter++) {
        TaskNodePtr tmpParent = *parentIter;
        if (!tmpParent->execFlag) {
            return false;
        }
    }
    return true;
}

HcclResult TaskGraphGenerator::ProcInterRankPostNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                                           SeenInterRankPosts &seenInterRankPosts,
                                                           u64                &unmatchedCnt)
{
    RankId        currRank = currNode->rankIdx;
    TaskStubPost *post     = dynamic_cast<TaskStubPost *>(currNode->task);
    RankId        peerRank = post->GetRemoteRank();

    if (seenInterRankPosts.find(currRank) == seenInterRankPosts.end()) {
        std::vector<TaskNodePtr> tmpPosts;
        tmpPosts.push_back(currNode);
        std::map<RankId, std::vector<TaskNodePtr>> tmpRankPosts;
        tmpRankPosts.insert(std::make_pair(peerRank, tmpPosts));
        seenInterRankPosts.insert(std::make_pair(currRank, tmpRankPosts));
    } else {
        if (seenInterRankPosts[currRank].find(peerRank) == seenInterRankPosts[currRank].end()) {
            std::vector<TaskNodePtr> tmpPosts;
            tmpPosts.push_back(currNode);
            seenInterRankPosts[currRank].insert(std::make_pair(peerRank, tmpPosts));
        } else {
            seenInterRankPosts[currRank][peerRank].push_back(currNode);
        }
    }
    CHK_PRT_RET(
        ExecNode4Graph(currNode, graphNodeQue) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[TaskGraphGenerator] Fail to execute node %s: TaskType [%s].",
                   currNode->GenPosInfo().c_str(), currNode->task->GetType().Describe().c_str()),
        HcclResult::HCCL_E_INTERNAL);
    unmatchedCnt = 0;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ProcInterRankWaitNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                                           SeenInterRankPosts &seenInterRankPosts,
                                                           u64                &unmatchedCnt)
{
    RankId        currRank = currNode->rankIdx;
    TaskStubWait *wait     = dynamic_cast<TaskStubWait *>(currNode->task);
    RankId        peerRank = wait->GetRemoteRank();
    if ((seenInterRankPosts.find(peerRank) != seenInterRankPosts.end())
        && (seenInterRankPosts[peerRank].find(currRank) != seenInterRankPosts[peerRank].end())) {
        std::vector<TaskNodePtr>::iterator postIter = seenInterRankPosts[peerRank][currRank].begin();
        for (; postIter != seenInterRankPosts[peerRank][currRank].end(); postIter++) {
            if (IsPostWaitPeer((*postIter), currNode)) {
                HCCL_DEBUG("[TaskGraphGenerator] peer PostNode of current WaitNode has already been seen.");
                (*postIter)->children.push_back(currNode);
                currNode->parents.push_back((*postIter));
                seenInterRankPosts[peerRank][currRank].erase(postIter); // remove post from seenInterRankPosts
                CHK_PRT_RET(ExecNode4Graph(currNode, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                            HCCL_ERROR("[TaskGraphGenerator] Fail to execute node %s: TaskType [%s].",
                                       currNode->GenPosInfo().c_str(),
                                       currNode->task->GetType().Describe().c_str()),
                            HcclResult::HCCL_E_INTERNAL);
                unmatchedCnt = 0;
                return HcclResult::HCCL_SUCCESS;
            }
        }
    }

    HCCL_DEBUG("[TaskGraphGenerator] peer PostNode of current WaitNode has not been seen yet.");
    graphNodeQue.push_back(currNode);
    unmatchedCnt++;
    return HcclResult::HCCL_SUCCESS;
}

bool TaskGraphGenerator::IsPostWaitPeer(const TaskNodePtr postNode, const TaskNodePtr waitNode)
{
    TaskStubPost *post = dynamic_cast<TaskStubPost *>(postNode->task);
    TaskStubWait *wait = dynamic_cast<TaskStubWait *>(waitNode->task);

    // check rankId
    if ((postNode->rankIdx != wait->GetRemoteRank()) || (waitNode->rankIdx != post->GetRemoteRank())) {
        return false;
    }

    // check LinkType
    if (post->GetLinkType() != wait->GetLinkType()) {
        return false;
    }

    // check topicId
    if (post->GetTopicId() != wait->GetTopicId()) {
        return false;
    }

    return post->GetNotifyType() == wait->GetNotifyType();
}

} // namespace hccl
