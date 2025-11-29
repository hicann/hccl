/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "check_rank_mem.h"

#include <queue>
#include <set>

#include "check_utils.h"

namespace HcclSim {

bool IsGenFromSync(TaskStub* task)
{
    bool isGenFromSync = false;
    if (task->GetType() == TaskTypeStub::LOCAL_COPY) {
        TaskStubLocalCopy *candLocalCopy = dynamic_cast<TaskStubLocalCopy *>(task);
        isGenFromSync = candLocalCopy->IsGenFromSync();
    } else if (task->GetType() == TaskTypeStub::READ) {
        TaskStubRead *candRead = dynamic_cast<TaskStubRead *>(task);
        isGenFromSync = candRead->IsGenFromSync();
    }
    return isGenFromSync;
}

MemoryStatus operator|(MemoryStatus a, MemoryStatus b)
{
    return static_cast<MemoryStatus>(static_cast<u32>(a) | static_cast<u32>(b));
}

MemoryStatus operator&(MemoryStatus a, MemoryStatus b)
{
    return static_cast<MemoryStatus>(static_cast<u32>(a) & static_cast<u32>(b));
}

MemoryStatus &operator|=(MemoryStatus &a, MemoryStatus b)
{
    return a = a | b;
}

// 边界节点，用于将一个原语队列切分为多个碎片
bool IsBoardType(TaskTypeStub type)
{
    const std::set<TaskTypeStub> boardTypes = {
                                                TaskTypeStub::LOCAL_POST_TO,
                                                TaskTypeStub::LOCAL_WAIT_FROM,
                                               };
    return boardTypes.count(type) != 0;
}

std::string GenFragQueueMemDes(FragQueueMemStatus &fragQueMemStatus)
{
    std::stringstream ret;
    for (auto iter = fragQueMemStatus.begin(); iter != fragQueMemStatus.end(); iter++) {
        BufferType type = iter->first;
        ret << FOUR_INDENT_SPACE << FOUR_INDENT_SPACE << "BufferType is " << type << std::endl;
        for (auto &ele : iter->second) {
            ret << FOUR_INDENT_SPACE << FOUR_INDENT_SPACE << FOUR_INDENT_SPACE << ele.Describe();
        }
    }
    return ret.str();
}

void CheckRankMem::GenFragQueueInOneQueue(TaskNode *head, std::set<u32> &seenQueues)
{
    TaskNode *fragStart = nullptr;
    TaskNode *fragEnd = nullptr;

    std::set<TaskNode *> visitedNodes;
    visitedNodes.insert(head);
    std::queue<TaskNode *> walkQue;
    walkQue.push(head);

    // 出于灵活性考虑，一个queue的头节点不一定是Post/Wait类型
    if (IsBoardType(head->task->GetType())) {
        fragStart = head;
    }

    while (!walkQue.empty()) {
        TaskNode *curNode = walkQue.front();
        walkQue.pop();
        for (auto &child : curNode->children) {
            // 不是同一个rank上的不考虑
            if (child->rankIdx != head->rankIdx) {
                continue;
            }

            if ((child->queIdx != head->queIdx) && (seenQueues.count(child->queIdx) == 0)) {
                bool isHeadNode = true;
                for (int i = 0; i < child->parents.size(); i++) {
                    if (child->parents[i]->queIdx == child->queIdx) {
                        isHeadNode = false;
                        break;
                    }
                }
                if (isHeadNode) {
                    seenQueues.insert(child->queIdx);
                    GenFragQueueInOneQueue(child, seenQueues);
                }
            }

            if (child->queIdx != head->queIdx) {
                continue;
            }

            if (visitedNodes.count(child) == 0) {
                walkQue.push(child);
                visitedNodes.insert(child);
            }
        }

        if (curNode == fragStart) {
            continue;
        }

        // 1）遇到边界节点 2）待循环队列已经为空
        if (IsBoardType(curNode->task->GetType()) || walkQue.empty()) {
            if (IsBoardType(curNode->task->GetType())) {
                fragEnd = curNode;
            }
            FragmentQueue ele{head->queIdx, 0, 0, false, fragStart, fragEnd};
            rank2FragQueue_[head->rankIdx].insert(ele);
            fragStart = curNode;
            fragEnd   = nullptr;
        }
    }
    return;
}

void CheckRankMem::GenFragQueueInOneRank(TaskNode *node)
{
    u32 queueId = node->queIdx;
    // 头结点链接的都应该是主流，queIdx=0
    if (queueId != 0) {
        HCCL_ERROR("The node connecting the head node should be the mainstream.");
    }
    std::set<u32> seenQueues;
    seenQueues.insert(queueId);
    GenFragQueueInOneQueue(node, seenQueues);
    return;
}

void CheckRankMem::GenFragQueue()
{
    // 头节点的每个child应该代表了一个rank
    for (auto &child : graphHead_->children) {
        GenFragQueueInOneRank(child);
    }
    return;
}

void CheckRankMem::FindPostWaitNode(TaskNode *node, std::set<TaskNode *> &postNodes, std::set<TaskNode *> &waitNodes) const
{
    if (node == nullptr) {
        return;
    }
    switch (node->GetNodeType()) {
        case TaskTypeStub::LOCAL_POST_TO:
            postNodes.insert(node);
            break;
        case TaskTypeStub::LOCAL_WAIT_FROM:
            waitNodes.insert(node);
            break;
        default:
            return;
    }
    return;
}

HcclResult CheckRankMem::FindPostWaitPair(RankId rankId)
{
    std::set<TaskNode *> postNodes;
    std::set<TaskNode *> waitNodes;
    for (auto &ele : rank2FragQueue_[rankId]) {
        FindPostWaitNode(ele.head, postNodes, waitNodes);
        FindPostWaitNode(ele.tail, postNodes, waitNodes);
    }

    for (auto &post : postNodes) {
        TaskNode *wait = nullptr;
        for (auto &child : post->children) {
            if (child->queIdx == post->queIdx) {
                continue;
            }
            if (waitNodes.count(child) == 1) {
                wait = child;
            }
        }
        if (wait == nullptr) {
            HCCL_ERROR("Can not find corresponding wait node for post node");
            return HcclResult::HCCL_E_PARA;
        }

        rank2PostWaitPairs_[rankId][post] = wait;
        rank2PostWaitPairs_[rankId][wait] = post;
    }
    return HcclResult::HCCL_SUCCESS;
}

void CheckRankMem::ProcessEqualToTargetStartAddr(u64 &sliceStartAddr, u64 sliceEndAddr,
                                                 std::vector<SliceMemoryStatus> &addedEles, MemoryStatus sliceStatus,
                                                 std::set<SliceMemoryStatus>::iterator target) const
{
    u64 eleEndAddr = target->startAddr + target->size;
    // 已经打过相同的标记位，不需要重复打
    if ((static_cast<u32>(target->status) & static_cast<u32>(sliceStatus)) != 0) {
        sliceStartAddr = eleEndAddr;
        return;
    }

    if (sliceEndAddr < eleEndAddr) {
        SliceMemoryStatus sliceMemStatus{sliceEndAddr, eleEndAddr - sliceEndAddr, target->status};
        addedEles.push_back(sliceMemStatus);
        target->size = sliceEndAddr - target->startAddr;
        target->status |= sliceStatus;
        sliceStartAddr = sliceEndAddr;
    } else if (sliceEndAddr == eleEndAddr) {
        target->status |= sliceStatus;
        sliceStartAddr = sliceEndAddr;
    } else { // sliceEndAddr > eleEndAddr
        target->status |= sliceStatus;
        sliceStartAddr = eleEndAddr;
    }
}

void CheckRankMem::ProcessGreatThanTargetStartAddr(u64 &sliceStartAddr, u64 sliceEndAddr,
                                                   std::vector<SliceMemoryStatus> &addedEles, MemoryStatus sliceStatus,
                                                   std::set<SliceMemoryStatus>::iterator target) const
{
    u64 eleEndAddr = target->startAddr + target->size;
    // 已经打过相同的标记位，不需要重复打
    if ((static_cast<u32>(target->status) & static_cast<u32>(sliceStatus)) != 0) {
        sliceStartAddr = eleEndAddr;
        return;
    }

    if (sliceEndAddr < eleEndAddr) {
        SliceMemoryStatus sliceMemStatus{sliceStartAddr, sliceEndAddr - sliceStartAddr, target->status | sliceStatus};
        addedEles.push_back(sliceMemStatus);

        SliceMemoryStatus tmp{sliceEndAddr, eleEndAddr - sliceEndAddr, target->status};
        addedEles.push_back(tmp);

        target->size   = sliceStartAddr - target->startAddr;
        sliceStartAddr = sliceEndAddr;
    } else if (sliceEndAddr == eleEndAddr) {
        SliceMemoryStatus sliceMemStatus{sliceStartAddr, sliceEndAddr - sliceStartAddr, target->status | sliceStatus};
        addedEles.push_back(sliceMemStatus);

        target->size   = sliceStartAddr - target->startAddr;
        sliceStartAddr = sliceEndAddr;
    } else { // sliceEndAddr > eleEndAddr
        SliceMemoryStatus sliceMemStatus{sliceStartAddr, eleEndAddr - sliceStartAddr, target->status | sliceStatus};
        addedEles.push_back(sliceMemStatus);

        target->size   = sliceStartAddr - target->startAddr;
        sliceStartAddr = eleEndAddr;
    }
}

void CheckRankMem::GenSliceMemoryInfo(DataSlice &slice, MemoryStatus sliceStatus, FragQueueMemStatus &result)
{
    BufferType sliceBufferType = slice.GetType();
    u64        sliceStartAddr  = slice.GetOffset(); // offset
    u64        sliceEndAddr    = sliceStartAddr + slice.GetSize();

    std::vector<SliceMemoryStatus> addedEles;
    for (auto ele = result[sliceBufferType].begin(); ele != result[sliceBufferType].end(); ele++) {
        u64 eleEndAddr = ele->startAddr + ele->size;
        // 下面两个判断保证了slice和ele有交集部分
        if ((sliceStartAddr >= eleEndAddr) || (sliceEndAddr <= ele->startAddr)) {
            continue;
        }

        if (sliceStartAddr < ele->startAddr) {
            SliceMemoryStatus sliceMemStatus{sliceStartAddr, ele->startAddr - sliceStartAddr, sliceStatus};
            addedEles.push_back(sliceMemStatus);
            sliceStartAddr = ele->startAddr;
        } else if (sliceStartAddr == ele->startAddr) {
            ProcessEqualToTargetStartAddr(sliceStartAddr, sliceEndAddr, addedEles, sliceStatus, ele);
        } else { // sliceStartAddr > ele->startAddr
            ProcessGreatThanTargetStartAddr(sliceStartAddr, sliceEndAddr, addedEles, sliceStatus, ele);
        }
    }

    if (sliceEndAddr > sliceStartAddr) {
        SliceMemoryStatus sliceMemStatus{sliceStartAddr, sliceEndAddr - sliceStartAddr, sliceStatus};
        addedEles.push_back(sliceMemStatus);
    }

    // 将addedElem给刷新上去
    for (auto &ele : addedEles) {
        result[sliceBufferType].insert(ele);
    }
    return;
}

HcclResult CheckRankMem::GenPrimNodeMemoryInfo(TaskNode *node, FragQueueMemStatus &result)
{
    std::vector<DataSlice> readSlices;
    std::vector<DataSlice> writeSlices;
    GetReadSlice(node, readSlices);
    GetWriteSlice(node, writeSlices);

    for (auto &ele : readSlices) {
        GenSliceMemoryInfo(ele, MemoryStatus::READ, result);
    }

    for (auto &ele : writeSlices) {
        GenSliceMemoryInfo(ele, MemoryStatus::WRITE, result);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CheckRankMem::GenFragQueueMemoryInfo(FragmentQueue &fragQueue, FragQueueMemStatus &result)
{
    std::queue<TaskNode *> walkQueue;
    walkQueue.push(fragQueue.head);

    std::set<TaskNode *> visitedNodes;
    visitedNodes.insert(fragQueue.head);

    while (!walkQueue.empty()) {
        TaskNode *curNode = walkQueue.front();
        walkQueue.pop();

        // 只有主流最前面的原语碎片才会出现头节点为空的情况，主流最前面的原语碎片不会和其他的原语碎片冲突，不生成内存信息也没关系
        if (curNode == nullptr) {
            continue;
        }

        for (auto &child : curNode->children) {
            if (curNode->isAivNode) {
                if (child->rankIdx != curNode->rankIdx or child->blockIdx != curNode->blockIdx or child->pipeIdx != curNode->pipeIdx) {
                    continue;
                }
            }
            else {
                if (child->rankIdx != curNode->rankIdx or child->queIdx != curNode->queIdx) {
                    continue;
                }
            }

            if (visitedNodes.count(child) == 0) {
                walkQueue.push(child);
                visitedNodes.insert(child);
            }
        }

        GenPrimNodeMemoryInfo(curNode, result);

        if (curNode == fragQueue.tail) {
            break;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CheckRankMem::CompareBufferTypeMemoryInfo(std::set<SliceMemoryStatus> &left,
                                                     std::set<SliceMemoryStatus> &right)
{
    std::set<SliceMemoryStatus>::iterator leftIter  = left.begin();
    std::set<SliceMemoryStatus>::iterator rightIter = right.begin();

    while (leftIter != left.end() && rightIter != right.end()) {
        if (leftIter->size == 0) {
            leftIter++;
            continue;
        }

        if (rightIter->size == 0) {
            rightIter++;
            continue;
        }

        if (leftIter->startAddr + leftIter->size <= rightIter->startAddr) {
            leftIter++;
            continue;
        }

        if (rightIter->startAddr + rightIter->size <= leftIter->startAddr) {
            rightIter++;
            continue;
        }

        if ((leftIter->status & MemoryStatus::WRITE) == MemoryStatus::WRITE
            or (rightIter->status & MemoryStatus::WRITE) == MemoryStatus::WRITE) {
            HCCL_ERROR("there is memory use confilict in two SliceMemoryStatus");
            HCCL_ERROR("one is %s", leftIter->Describe().c_str());
            HCCL_ERROR("another is %s", rightIter->Describe().c_str());
            return HcclResult::HCCL_E_MEMORY;
        }

        if (leftIter->startAddr + leftIter->size <= rightIter->startAddr + rightIter->size) {
            leftIter++;
        } else {
            rightIter++;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

// 仅内部dump使用，不需要对外提供
HcclResult CheckRankMem::CompareBufferTypeMemoryInfo(std::set<SliceMemoryStatus> &left,
                                                     std::set<SliceMemoryStatus> &right,
                                                     SliceMemoryStatus &conflictEleA,
                                                     SliceMemoryStatus &conflictEleB)
{
    std::set<SliceMemoryStatus>::iterator leftIter  = left.begin();
    std::set<SliceMemoryStatus>::iterator rightIter = right.begin();

    while (leftIter != left.end() && rightIter != right.end()) {
        if (leftIter->size == 0) {
            leftIter++;
            continue;
        }

        if (rightIter->size == 0) {
            rightIter++;
            continue;
        }

        if (leftIter->startAddr + leftIter->size <= rightIter->startAddr) {
            leftIter++;
            continue;
        }

        if (rightIter->startAddr + rightIter->size <= leftIter->startAddr) {
            rightIter++;
            continue;
        }

        if ((leftIter->status & MemoryStatus::WRITE) == MemoryStatus::WRITE
            or (rightIter->status & MemoryStatus::WRITE) == MemoryStatus::WRITE) {
            conflictEleA = *leftIter;
            conflictEleB = *rightIter;
            return HcclResult::HCCL_E_MEMORY;
        }

        if (leftIter->startAddr + leftIter->size <= rightIter->startAddr + rightIter->size) {
            leftIter++;
        } else {
            rightIter++;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CheckRankMem::CompareSliceMemoryInfo(FragQueueMemStatus &left, FragQueueMemStatus &right)
{
    for (auto iter = left.begin(); iter != left.end(); iter++) {
        BufferType type = iter->first;
        if (right.count(type) != 0) {
            auto ret = CompareBufferTypeMemoryInfo(iter->second, right[type]);
            if (ret != HcclResult::HCCL_SUCCESS) {
                HCCL_ERROR("failed to check memory %s", type.Describe().c_str());
                return ret;
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

std::string GenConflictDetailInfo(TaskNode *node)
{
    if (node->realPeerNode) {
        return node->realPeerNode->GenPosInfo();
    }
    return node->GenPosInfo();
}

HcclResult CheckRankMem::CompareFragQueStatus(u32 fragQueueSize, std::map<u32, FragmentQueue> &index2FragQueue,
                                              std::vector<std::vector<bool>> &fragQueueMatrix)
{
    std::map<FragmentQueue, FragQueueMemStatus> fragQueue2MemStatus;
    for (u32 i = 0; i < fragQueueSize; i++) {
        for (u32 j = i + 1; j < fragQueueSize; j++) {
            if (fragQueueMatrix[i][j] == false) {
                continue;
            }

            // 产生两条queue的内存状态
            if (fragQueue2MemStatus.count(index2FragQueue[i]) == 0) {
                GenFragQueueMemoryInfo(index2FragQueue[i], fragQueue2MemStatus[index2FragQueue[i]]);
            }

            if (fragQueue2MemStatus.count(index2FragQueue[j]) == 0) {
                GenFragQueueMemoryInfo(index2FragQueue[j], fragQueue2MemStatus[index2FragQueue[j]]);
            }

            HcclResult ret = CompareSliceMemoryInfo(fragQueue2MemStatus[index2FragQueue[i]],
                                                    fragQueue2MemStatus[index2FragQueue[j]]);
            if (ret != HcclResult::HCCL_SUCCESS) {
                for (TaskNode *nodeA = index2FragQueue[i].head; nodeA != index2FragQueue[i].tail;) {
                    for (TaskNode *nodeB = index2FragQueue[j].head; nodeB != index2FragQueue[j].tail;) {
                        // 判断是否有冲突，如果有，就dump数据
                        SliceMemoryStatus conflictEleA;
                        SliceMemoryStatus conflictEleB;
                        if (IsConfilictBetweenTwoNodes(nodeA, nodeB, conflictEleA, conflictEleB)) {
                            HCCL_ERROR("memory conflict between node %s and node %s",
                                nodeA->GenPosInfo().c_str(), nodeB->GenPosInfo().c_str());
                            break;
                        }

                        for (auto &child : nodeB->children) {
                            if (child->rankIdx != nodeB->rankIdx || child->queIdx != nodeB->queIdx) {
                                continue;
                            }
                            nodeB = child;
                            break;
                        }

                        if (nodeB->children.size() == 0) {
                            break;
                        }
                    }
                    for (auto &child : nodeA->children) {
                        if (child->rankIdx != nodeA->rankIdx || child->queIdx != nodeA->queIdx) {
                            continue;
                        }
                        nodeA = child;
                        break;
                    }

                    if (nodeA->children.size() == 0) {
                        break;
                    }
                }
                return ret;
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

bool CheckRankMem::IsConfilictBetweenTwoNodes(TaskNode* nodeA, TaskNode* nodeB,
                                              SliceMemoryStatus &conflictEleA, SliceMemoryStatus &conflictEleB)
{
    FragQueueMemStatus resultA;
    FragQueueMemStatus resultB;
    
    GenPrimNodeMemoryInfo(nodeA, resultA);

    GenPrimNodeMemoryInfo(nodeB, resultB);

    HcclResult ret;
    for (auto iter = resultA.begin(); iter != resultA.end(); iter++) {
        BufferType type = iter->first;
        if (resultB.count(type) != 0) {
            ret = CompareBufferTypeMemoryInfo(iter->second, resultB[type], conflictEleA, conflictEleB);
            if (ret != HcclResult::HCCL_SUCCESS) {
                break;
            }
        }
    }

    if (ret != HcclResult::HCCL_SUCCESS) {
        return true;
    }
    return false;
}

// 判断是否是连接AivEnd的最后一个TaskNode
bool CheckRankMem::IsLastTaskNode(TaskNode* node)
{
    return false;
}

HcclResult CheckRankMem::GenFragQueConcurrencyMatrixAndCompare(RankId rankId)
{
    u32 fragQueueSize = rank2FragQueue_[rankId].size();
    std::vector<std::vector<bool>> fragQueueMatrix(fragQueueSize, std::vector<bool>(fragQueueSize, true));
    // 自己不能与自己比较
    for (u32 i = 0; i < fragQueueSize; i++) {
        fragQueueMatrix[i][i] = false;
    }

    std::map<u32, FragmentQueue> index2FragQueue;
    std::map<TaskNode*, FragmentQueue> headNode2FragQueue;
    std::map<TaskNode*, u32> headNode2Index;
    u32 index = 0;
    for (auto &fragQueue : rank2FragQueue_[rankId]) {
        index2FragQueue[index] = fragQueue;
        headNode2FragQueue[fragQueue.head] = fragQueue;
        headNode2Index[fragQueue.head] = index;
        index++;
    }

    for (int i = 0; i < fragQueueSize; i++) {
        TaskNode* curTailNode = index2FragQueue[i].tail;
        if (curTailNode == nullptr) {
            continue;
        }
        std::queue<TaskNode*> walkQue;
        std::set<TaskNode*> visitedNodes;
        walkQue.push(curTailNode);
        while(!walkQue.empty()) {
            TaskNode* curNode = walkQue.front();
            walkQue.pop();
            if (curNode == nullptr) {
                continue;
            }

            if (curNode->GetNodeType() == TaskTypeStub::LOCAL_POST_TO) {
                // 找到以该post节点为起点的碎片队列，并打上不可能并行的标签
                // 将以该post节点为起点的碎片队列的结束点加进walkQue队列中
                // 将该post节点对应的wait节点加进walkQue队列中
                if (!curNode->isAivNode) {
                    fragQueueMatrix[i][headNode2Index[curNode]] = false;
                    fragQueueMatrix[headNode2Index[curNode]][i] = false;
                } else {
                    // 对于AIV，如果是连接AivEnd的最后一个TaskNode，headNode2Index找不到对应的片段，不应该置false
                    if (IsLastTaskNode(curNode) == false) {
                        fragQueueMatrix[i][headNode2Index[curNode]] = false;
                        fragQueueMatrix[headNode2Index[curNode]][i] = false;
                    }
                }

                if (headNode2FragQueue.find(curNode) != headNode2FragQueue.end() && !visitedNodes.count(headNode2FragQueue[curNode].tail)) {
                    walkQue.push(headNode2FragQueue[curNode].tail);
                    visitedNodes.insert(headNode2FragQueue[curNode].tail);
                }
                if (!visitedNodes.count(rank2PostWaitPairs_[rankId][curNode])) {
                    walkQue.push(rank2PostWaitPairs_[rankId][curNode]);
                    visitedNodes.insert(rank2PostWaitPairs_[rankId][curNode]);
                }
            } else if (curNode->GetNodeType() == TaskTypeStub::LOCAL_WAIT_FROM) {
                // 找到以该wait节点为起点的碎片队列，并打上不可能并行的标签
                // 将该wait节点加进walkQue队列中
                fragQueueMatrix[i][headNode2Index[curNode]] = false;
                fragQueueMatrix[headNode2Index[curNode]][i] = false;

                if (headNode2FragQueue.find(curNode) != headNode2FragQueue.end() && !visitedNodes.count(headNode2FragQueue[curNode].tail)) {
                    walkQue.push(headNode2FragQueue[curNode].tail);
                    visitedNodes.insert(headNode2FragQueue[curNode].tail);
                }
            } 
        }
    }

    return CompareFragQueStatus(fragQueueSize, index2FragQueue, fragQueueMatrix);
}

// 被读的内存块
void CheckRankMem::GetReadSlice(TaskNode *node, std::vector<DataSlice> &slices)
{
    TaskTypeStub type = node->task->GetType();
    bool isGenFromSync = IsGenFromSync(node->task);
    if (type == TaskTypeStub::LOCAL_COPY) {
        auto task = dynamic_cast<TaskStubLocalCopy *>(node->task);
        slices.push_back(task->GetSrcSlice());
    } else if (type == TaskTypeStub::LOCAL_REDUCE) {
        HCCL_ERROR("[CheckRankMem::GetReadSlice] TaskType LOCAL_REDUCE not support");
    } else if (type == TaskTypeStub::WRITE) {
        auto task = dynamic_cast<TaskStubWrite *>(node->task);
        slices.push_back(task->GetLocalSlice());
    } else if (type == TaskTypeStub::WRITE_REDUCE) {
        HCCL_ERROR("[CheckRankMem::GetReadSlice] TaskType WRITE_REDUCE not support");
    }
    return;
}

void CheckRankMem::GetWriteSlice(TaskNode *node, std::vector<DataSlice> &slices)
{
    TaskTypeStub type = node->task->GetType();
    bool isGenFromSync = IsGenFromSync(node->task);
    if (type == TaskTypeStub::LOCAL_COPY) {
        auto task = dynamic_cast<TaskStubLocalCopy *>(node->task);
        slices.push_back(task->GetDstSlice());
    } else if (type == TaskTypeStub::LOCAL_REDUCE && !isGenFromSync) {
        HCCL_ERROR("[CheckRankMem::GetWriteSlice] TaskType LOCAL_REDUCE not support");
    } else if (type == TaskTypeStub::READ) {
        auto task = dynamic_cast<TaskStubRead *>(node->task);
        slices.push_back(task->GetLocalSlice());
    } else if (type == TaskTypeStub::READ_REDUCE) {
        HCCL_ERROR("[CheckRankMem::GetWriteSlice] TaskType READ_REDUCE not support");
    }
    return;
}

HcclResult CheckRankMem::Execute()
{
    // 如果有AivTask，不管是纯AIV算法还是混编算法，均先处理AIV子图内层的冲突校验

    // 先从整图中提取信息
    GenFragQueue();

    // 从每个rank中提取post/wait队列
    for (auto &child : graphHead_->children) {
        RankId rankId = child->rankIdx;
        CHK_RET(FindPostWaitPair(rankId));
        auto ret = GenFragQueConcurrencyMatrixAndCompare(rankId);
        if (ret != HcclResult::HCCL_SUCCESS) {
            // DataDumper::Global()->SetResultStatus(gui::ResultStatus::MEMORY_CONFLICT);
            HCCL_ERROR("check rank memory conflict failed for rank %d", rankId);
            return ret;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl