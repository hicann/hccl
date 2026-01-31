/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_graph_revamp.h"
#include <queue>
#include <set>
#include "log.h"
#include "task_def.h"

namespace HcclSim {
GraphRevampBilateralSemantics::~GraphRevampBilateralSemantics()
{
    for (auto& ele : toDeleteTaskResource_) {
        if (ele == nullptr) {
            continue;
        }
        delete ele;
    }
    for (auto& ele : toDeleteTaskNodeResource_) {
        if (ele == nullptr) {
            continue;
        }
        delete ele;
    }
}

HcclResult GraphRevampBilateralSemantics::Revamp(TaskNodePtr dummyStart, AllRankTaskQueues& taskQueue)
{
    std::map<RankId, TaskNodePtr> rank2Head;
    CHK_PRT_RET(InitRankHead(dummyStart, rank2Head, taskQueue) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[GraphRevampBilateralSemantics] Unable to initialize head nodes for each rank."),
                HcclResult::HCCL_E_INTERNAL);

    // revamp the graph for two-side semantics and rdma doorbell specs
    CHK_PRT_RET(RevampGraph(dummyStart, rank2Head) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[GraphRevampBilateralSemantics] Unable to revamp graph."), HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::InitRankHead(TaskNodePtr dummyStart, std::map<RankId, TaskNodePtr> &rank2Head, AllRankTaskQueues& taskQueue)
{
    auto childIter = dummyStart->children.begin();
    for (; childIter != dummyStart->children.end(); childIter++) {
        RankId myRank = (*childIter)->rankIdx;
        rank2Head.insert(std::make_pair(myRank, (*childIter)));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::RevampGraph(TaskNodePtr dummyStart, std::map<RankId, TaskNodePtr> &rank2Head)
{
    VirtQueMgr virtQueManager;
    std::queue<TaskNodePtr> graphNodeQue;
    // queue结构体无法直接查找是否存在某个元素，此处利用set结构体查找是否存在某个元素
    std::set<TaskNodePtr> isVisited;
    // init graphNodeQue
    dummyStart->procFlag = true;
    CHK_RET(ProceedNode(dummyStart, graphNodeQue, isVisited));
    int count = 0;
    while (!graphNodeQue.empty()) {
        TaskNodePtr currNode = graphNodeQue.front();
        graphNodeQue.pop();
        // 这边塞回队列的逻辑放在IsProceedParentNode里面，会影响可读性
        if (IsProceedParentNode(currNode, graphNodeQue, isVisited) && !currNode->procFlag) {
            currNode->procFlag = true;
            CHK_RET(ProceedNode(currNode, graphNodeQue, isVisited));
            switch (currNode->task->GetType()) {
                case TaskTypeStub::READ:
                case TaskTypeStub::READ_REDUCE:
                    CHK_RET(ProcReadNode(dummyStart, currNode, rank2Head, virtQueManager));
                    break;
                case TaskTypeStub::WRITE:
                case TaskTypeStub::WRITE_REDUCE:
                    CHK_RET(ProcWriteNode(dummyStart, currNode, rank2Head, virtQueManager));
                    break;
                default:
                    break;
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::ProceedNode(TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue,
                                                      std::set<TaskNodePtr> &isVisited)
{
    for (auto childIter = currNode->children.begin(); childIter != currNode->children.end(); childIter++) {
        if (!(*childIter)->procFlag) {
            graphNodeQue.push((*childIter));
            if (!isVisited.count(*childIter)) {
                isVisited.insert((*childIter));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

bool GraphRevampBilateralSemantics::IsProceedParentNode(TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue,
                                                        std::set<TaskNodePtr> &isVisited)
{
    for (auto parentIter = currNode->parents.begin(); parentIter != currNode->parents.end(); parentIter++) {
        if (!(*parentIter)->procFlag) {
            graphNodeQue.push(currNode);
            if (!isVisited.count(*parentIter)) {
                graphNodeQue.push((*parentIter));
                isVisited.insert(*parentIter);
            }
            return false;
        }
    }
    return true;
}

HcclResult GraphRevampBilateralSemantics::ProcReadNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
    std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    LinkProtoStub link;
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));
    // aiv产生read write 给的链路类型
    CHK_RET(GetLinkProtoStubByTaskNode(currNode, link));

    if (link == LinkProtoStub::SDMA) {
        CHK_PRT_RET(
            ProcSdmaRWNode(dummyStart, currNode, rank2Head, virtQueManager) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[GraphRevampBilateralSemantics] fail to proceed READ taskNode locates in Rank [%d] - Que [%u] - Pos [%u], "
                "reading from Rank [%u].", currNode->rankIdx, currNode->queIdx, currNode->pos, peerRank),
            HcclResult::HCCL_E_INTERNAL);
    } else {
        HCCL_ERROR("[GraphRevampBilateralSemantics] Rank [%d], linkProto %s not supported yet.", currNode->rankIdx, link.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::ProcSdmaRWNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
    std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    // Search backward and add beingRead in peerRank
    CHK_RET(SearchBackwardSdmaRW(dummyStart, currNode, rank2Head, virtQueManager));

    // Search forward and see if beingRead can be terminated
    CHK_RET(SearchForwardSdmaRW(dummyStart, currNode, rank2Head, virtQueManager));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::SearchBackwardSdmaRW(TaskNodePtr dummyStart, TaskNodePtr currNode,
    std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));

    std::queue<TaskNodePtr> candParents;
    std::set<TaskNodePtr> isVisited;
    if (currNode->parents.size() != 1) {
        HCCL_ERROR("[GraphRevampBilateralSemantics] read taskNode parent num is not 1, is [%d].", currNode->parents.size());
        return HcclResult::HCCL_E_INTERNAL;
    }
    candParents.push(currNode->parents[0]);
    isVisited.insert(currNode->parents[0]);
    // Search backward till wait(peerRank) or Read/Write(peerRank) being Found
    while (!candParents.empty()) {
        TaskNodePtr candNode = candParents.front();
        candParents.pop();
        if (candNode->task->GetType() == TaskTypeStub::WAIT) {
            TaskStubWait *candWait = dynamic_cast<TaskStubWait *>(candNode->task);
            if (candWait->GetRemoteRank() == peerRank) {
                // get Wait(peerRank)
                return HcclResult::HCCL_SUCCESS;
            }
        } else if (IsReadWriteWithSameRank(peerRank, candNode)) {
            // get read(currank)
            return HcclResult::HCCL_SUCCESS;
        }

        // update candParents
        auto candParentIter = candNode->parents.begin();
        for (; candParentIter != candNode->parents.end(); candParentIter++) {
            TaskNodePtr tmpCandParent = (*candParentIter);
            if ((tmpCandParent->rankIdx == currNode->rankIdx) && (!isVisited.count(tmpCandParent))) {
                candParents.push(tmpCandParent);
                isVisited.insert(tmpCandParent);
            }
        }
    }

    // no Wait(peerRank)
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::SearchForwardSdmaRW(TaskNodePtr dummyStart, TaskNodePtr currNode,
    std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));

    // Search forward till post(peerRank) or Read(peerRank) being Found
    std::queue<TaskNodePtr> candChildren;
    std::set<TaskNodePtr> isVisited;
    if (currNode->children.size() != 1) {
        HCCL_ERROR("[GraphRevampBilateralSemantics] read taskNode child num is not 1, is [%d].", currNode->children.size());
        return HcclResult::HCCL_E_INTERNAL;
    }
    candChildren.push(currNode->children[0]);
    isVisited.insert(currNode->parents[0]);
    while (!candChildren.empty()) {
        TaskNodePtr candNode = candChildren.front();
        candChildren.pop();
        if (candNode->task->GetType() == TaskTypeStub::POST) {
            TaskStubPost *candPost = dynamic_cast<TaskStubPost *>(candNode->task);
            if (candPost->GetRemoteRank() == peerRank) {
                // get Post(peerRank)
                return HcclResult::HCCL_SUCCESS;
            }
        } else if (IsReadWriteWithSameRank(peerRank, candNode)) {
            // no Post(peerRank)
            return HcclResult::HCCL_SUCCESS;
        }

        // update candChildren
        auto candChildrenIter = candNode->children.begin();
        for (; candChildrenIter != candNode->children.end(); candChildrenIter++) {
            TaskNodePtr tmpCandChildren = (*candChildrenIter);
            if ((tmpCandChildren->rankIdx == currNode->rankIdx) && (!isVisited.count(tmpCandChildren))) {
                candChildren.push(tmpCandChildren);
                isVisited.insert(tmpCandChildren);
            }
        }
    }

    // no Post(peerRank)
    return HcclResult::HCCL_SUCCESS;
}

// 非虚拟队列下，在某个节点后插入一个节点
HcclResult GraphRevampBilateralSemantics::InsertNode(TaskNodePtr headNode, TaskNodePtr insertNode)
{
    // headNode --> insertNode --> originalNxtNode
    auto childIter = headNode->children.begin();
    for (; childIter != headNode->children.end(); childIter++) {
        TaskNodePtr originalNxtNode = (*childIter);
        if ((originalNxtNode->rankIdx == headNode->rankIdx) && (originalNxtNode->queIdx == headNode->queIdx)
        ) {
            InsertNode(originalNxtNode, insertNode);
            return HcclResult::HCCL_SUCCESS;
        } else if ((originalNxtNode->rankIdx == headNode->rankIdx) && (originalNxtNode->queIdx == headNode->queIdx)) {
            headNode->children.erase(childIter);

            // remove headNode from parents of originalNxtNode
            auto removeIter = std::remove(originalNxtNode->parents.begin(), originalNxtNode->parents.end(), headNode);
            originalNxtNode->parents.erase(removeIter, originalNxtNode->parents.end());

            headNode->children.push_back(insertNode);
            insertNode->parents.push_back(headNode);
            insertNode->children.push_back(originalNxtNode);
            originalNxtNode->parents.push_back(insertNode);
            return HcclResult::HCCL_SUCCESS;
        }
    }

    // headNode is last node of currQue
    headNode->children.push_back(insertNode);
    insertNode->parents.push_back(headNode);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::ProcWriteNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
                                     std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    LinkProtoStub link;
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));
    CHK_RET(GetLinkProtoStubByTaskNode(currNode, link));

    if (link == LinkProtoStub::RDMA) {
        CHK_PRT_RET(
            ProcRdmaWriteNode(dummyStart, currNode, rank2Head, virtQueManager) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[GraphRevampBilateralSemantics] fail to proceed WRITE taskNode locates in Rank [%d] - Que [%u] - Pos [%u], "
                       "reading from Rank [%u].",
                       currNode->rankIdx, currNode->queIdx, currNode->pos, peerRank),
            HcclResult::HCCL_E_INTERNAL);
    } else if (link == LinkProtoStub::SDMA) {
        CHK_PRT_RET(
            ProcSdmaRWNode(dummyStart, currNode, rank2Head, virtQueManager) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[GraphRevampBilateralSemantics] fail to proceed WRITE taskNode locates in Rank [%d] - Que [%u] - Pos [%u], "
                       "reading from Rank [%u].",
                       currNode->rankIdx, currNode->queIdx, currNode->pos, peerRank),
            HcclResult::HCCL_E_INTERNAL);
    } else {
        HCCL_ERROR("[GraphRevampBilateralSemantics] Rank [%d], linkProto %s not supported yet.", currNode->rankIdx, link.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::ProcRdmaWriteNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
                                          std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    // Search backward and add beingRead in peerRank
    CHK_RET(SearchBackwardRdmaWrite(dummyStart, currNode, rank2Head, virtQueManager));

    // Search forward and see if beingRead can be terminated
    // In this Function transfer write to virtual queue
    CHK_RET(SearchForwardRdmaWrite(dummyStart, currNode, rank2Head, virtQueManager));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::SearchBackwardRdmaWrite(TaskNodePtr dummyStart, TaskNodePtr currNode,
                                   std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));

    // Search backward till Read/Write(peerRank) being Found
    std::queue<TaskNodePtr> candParents;
    std::set<TaskNodePtr> isVisited;
    if (currNode->parents.size() != 1) {
        HCCL_ERROR("[GraphRevampBilateralSemantics] write taskNode parent num is not 1, is [%d].", currNode->parents.size());
        return HcclResult::HCCL_E_INTERNAL;
    }
    candParents.push(currNode->parents[0]);
    isVisited.insert(currNode->parents[0]);
    while (!candParents.empty()) {
        TaskNodePtr candNode = candParents.front();
        candParents.pop();
        if (candNode->task->GetType() == TaskTypeStub::WAIT) {
            TaskStubWait *candWait = dynamic_cast<TaskStubWait *>(candNode->task);
            if (candWait->GetRemoteRank() == peerRank) {
                return HcclResult::HCCL_SUCCESS;
            }
        } 

        // update candChildren
        auto candParentsIter = candNode->parents.begin();
        for (; candParentsIter != candNode->parents.end(); candParentsIter++) {
            TaskNodePtr tmpCandParents = (*candParentsIter);
            if ((tmpCandParents->rankIdx == currNode->rankIdx) && (!isVisited.count(tmpCandParents))) {
                candParents.push(tmpCandParents);
                isVisited.insert(tmpCandParents);
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::SearchForwardRdmaWrite(TaskNodePtr dummyStart, TaskNodePtr currNode,
                                  std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));

    std::queue<TaskNodePtr> candChildren;
    std::set<TaskNodePtr> isVisited;
    if (currNode->children.size() != 1) {
        HCCL_ERROR("[GraphRevampBilateralSemantics] write taskNode children num is not 1, is [%d].", currNode->children.size());
        return HcclResult::HCCL_E_INTERNAL;
    }
    candChildren.push(currNode->children[0]);
    isVisited.insert(currNode->parents[0]);

    while (!candChildren.empty()) {
        TaskNodePtr candNode = candChildren.front();
        candChildren.pop();
        if (candNode->task->GetType() == TaskTypeStub::POST) {
            TaskStubPost *candPost = dynamic_cast<TaskStubPost *>(candNode->task);
            if (candPost->GetRemoteRank() == peerRank) {
            }
        } else if (candNode->task->GetType() == TaskTypeStub::WAIT) {
            TaskStubWait *candWait = dynamic_cast<TaskStubWait *>(candNode->task);
            if (candWait->GetRemoteRank() == peerRank) {
                return HcclResult::HCCL_SUCCESS;
            }
        } else if (IsReadWriteWithSameRank(peerRank, candNode)) {
            return HcclResult::HCCL_SUCCESS;
        }

        // update candChildren
        auto candChildrenIter = candNode->children.begin();
        for (; candChildrenIter != candNode->children.end(); candChildrenIter++) {
            TaskNodePtr tmpCandChildren = (*candChildrenIter);
            if ((tmpCandChildren->rankIdx == currNode->rankIdx) && (!isVisited.count(tmpCandChildren))) {
                candChildren.push(tmpCandChildren);
                isVisited.insert(tmpCandChildren);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::GetPeerRankByTaskNode(TaskNodePtr currNode, RankId &peerRank)
{
    if (currNode->task->GetType() == TaskTypeStub::READ) {
        TaskStubRead *read = dynamic_cast<TaskStubRead *>(currNode->task);
        peerRank = read->GetRemoteRank();
    } else if (currNode->task->GetType() == TaskTypeStub::READ_REDUCE) {
        TaskStubReadReduce *read = dynamic_cast<TaskStubReadReduce *>(currNode->task);
        peerRank = read->GetRemoteRank();
    } else if (currNode->task->GetType() == TaskTypeStub::WRITE) {
        TaskStubWrite *write = dynamic_cast<TaskStubWrite *>(currNode->task);
        peerRank = write->GetRemoteRank();
    } else if (currNode->task->GetType() == TaskTypeStub::WRITE_REDUCE) {
        TaskStubWriteReduce *write = dynamic_cast<TaskStubWriteReduce *>(currNode->task);
        peerRank = write->GetRemoteRank();
    }
    return HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::GetLinkProtoStubByTaskNode(TaskNodePtr currNode, LinkProtoStub &link)
{
    if (currNode->task->GetType() == TaskTypeStub::READ) {
        TaskStubRead *read = dynamic_cast<TaskStubRead *>(currNode->task);
        link = read->GetLinkType();
    } else if (currNode->task->GetType() == TaskTypeStub::READ_REDUCE) {
        TaskStubReadReduce *read = dynamic_cast<TaskStubReadReduce *>(currNode->task);
        link = read->GetLinkType();
    } else if (currNode->task->GetType() == TaskTypeStub::WRITE) {
        TaskStubWrite *write = dynamic_cast<TaskStubWrite *>(currNode->task);
        link = write->GetLinkType();
    } else if (currNode->task->GetType() == TaskTypeStub::WRITE_REDUCE) {
        TaskStubWriteReduce *write = dynamic_cast<TaskStubWriteReduce *>(currNode->task);
        link = write->GetLinkType();
    }
    return HCCL_SUCCESS;
}


bool GraphRevampBilateralSemantics::IsReadWriteWithSameRank(RankId peerRank, TaskNodePtr candNode)
{
    if (candNode->task->GetType() == TaskTypeStub::READ) {
        TaskStubRead *candRead = dynamic_cast<TaskStubRead *>(candNode->task);
        if (candRead->GetRemoteRank() == peerRank) {
            return true;
        }
    } else if (candNode->task->GetType() == TaskTypeStub::WRITE) {
        TaskStubWrite *candWrite = dynamic_cast<TaskStubWrite *>(candNode->task);
        if (candWrite->GetRemoteRank() == peerRank) {
            return true;
        }
    } else if (candNode->task->GetType() == TaskTypeStub::READ_REDUCE) {
        HCCL_ERROR("[GraphRevampBilateralSemantics::IsReadWriteWithSameRank] TaskType READ_REDUCE not support");
    } else if (candNode->task->GetType() == TaskTypeStub::WRITE_REDUCE) {
        HCCL_ERROR("[GraphRevampBilateralSemantics::IsReadWriteWithSameRank] TaskType WRITE_REDUCE not support");
    }
    return false;
}

} // namespace Hccl