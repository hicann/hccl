/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "checker.h"
#include <thread>
#include <string>
#include "check_utils.h"
#include "singletask_check.h"
#include "sim_task.h"
#include "task_def.h"
#include "checker_def.h"
#include "task_graph_generator.h"
#include "check_rank_mem.h"
#include "task_graph_revamp.h"
#include "task_check_op_semantics.h"

using namespace std;

namespace HcclSim {

Checker::~Checker()
{
    for (auto& ele : toDeleteCopyTaskNodeResource_) {
        if (ele == nullptr) {
            continue;
        }
        delete ele;
    }

    for (auto& ele : toDeleteCopyTaskResource_) {
        if (ele == nullptr) {
            continue;
        }
        delete ele;
    }
}

void Checker::CloseRankMemCheck()
{
    closeRankMemCheck_ = true;
}

HcclResult Checker::GenAndCheckGraph(AllRankTaskQueues& allRankTaskQueues, TaskCheckOpSemantics& opSemanticsChcker)
{
    if (allRankTaskQueues.empty()) {
        HCCL_ERROR("[Checker][GenAndCheckGraph] taskQueues is empty!");
        return HCCL_E_PARA;
    }

    // 打印Task队列
    PrintTask(allRankTaskQueues);

    // 1. 检查从流
    HCCL_INFO("1. 检查从流");
    SingleTaskCheck taskChecker;
    CHK_RET(taskChecker.CheckSlaveTaskQueue(allRankTaskQueues));

    // 2. 成图
    HCCL_INFO("2. 成图");
    TaskNode dummyStart = TaskNode(nullptr, -1, 0, 0);
    TaskNode dummyStartCopy = TaskNode(nullptr, -1, 0, 0);
    TaskGraphGenerator graphGenerator;
    CHK_RET(graphGenerator.GenGraph(allRankTaskQueues, &dummyStart));

    // 3. Task内存校验
    // 是否可以复用 taskChecker
    HCCL_INFO("3. Task内存校验");
    CHK_RET(taskChecker.CheckTaskMem(&dummyStart));

    // 4. 图复制
    HCCL_INFO("4. 图复制");
    CopyTaskGraph(&dummyStart, &dummyStartCopy);

    if (!closeRankMemCheck_) {
        // 5. 图改造
        HCCL_INFO("5. 图改造");
        GraphRevampBilateralSemantics graphRevamp;
        CHK_RET(graphRevamp.Revamp(&dummyStartCopy, allRankTaskQueues));

        // 6. Rank内存校验
        HCCL_INFO("6. Rank内存校验");
        CheckRankMem checkRankmem(&dummyStartCopy);
        CHK_RET(checkRankmem.Execute());
    }

    // 7. 语义校验
    HCCL_INFO("7. 语义校验");
    opSemanticsChcker.SetGraphHead(&dummyStart);
    CHK_RET(opSemanticsChcker.Execute());

    // 成图及校验成功
    return HCCL_SUCCESS;
}

void Checker::CopyTaskGraph(TaskNodePtr originNode, TaskNodePtr copyNode)
{
    // 该函数无修改
    // 遍历两遍，先将所有节点拷贝出来，再建立父子关系
    std::map<TaskNodePtr, TaskNodePtr> originNode2copyNode; // 用来收录原节点到新节点的映射
    std::vector<TaskNodePtr> candTaskNodePtr;
    std::set<TaskNodePtr> isVisited;

    originNode2copyNode[originNode] = copyNode;
    for (int i = 0; i < originNode->children.size(); i++) {
        candTaskNodePtr.push_back(originNode->children[i]);
        isVisited.insert(originNode->children[i]);
    }

    while (!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr[0];
        candTaskNodePtr.erase(candTaskNodePtr.begin());

        TaskNodePtr newNodePtr = new TaskNode(curNode->task, curNode->rankIdx, curNode->queIdx, curNode->pos);
        toDeleteCopyTaskNodeResource_.push_back(newNodePtr);
        originNode2copyNode[curNode] = newNodePtr;

        for (auto &child : curNode->children) {
            if (isVisited.find(child) == isVisited.end()) {
                isVisited.insert(child);
                candTaskNodePtr.push_back(child);
            }
        }
    }

    isVisited.clear();
    for (int i = 0; i < originNode->children.size(); i++) {
        candTaskNodePtr.push_back(originNode->children[i]);
        isVisited.insert(originNode->children[i]);
        copyNode->children.push_back(originNode2copyNode[originNode->children[i]]);
    }
    while (!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr[0];
        candTaskNodePtr.erase(candTaskNodePtr.begin());
        for (auto &parent : curNode->parents) {
            originNode2copyNode[curNode]->parents.push_back(originNode2copyNode[parent]);
        }
        for (auto &child : curNode->children) {
            originNode2copyNode[curNode]->children.push_back(originNode2copyNode[child]);
            if (isVisited.count(child) == 0) {
                isVisited.insert(child);
                candTaskNodePtr.push_back(child);
            }
        }
    }
}

void Checker::PrintTask(const AllRankTaskQueues &allRankTaskQueues)
{
    u32 rankIdx = 0;
    for (auto& podIter : allRankTaskQueues) {
        auto& serMap = podIter.second;
        for (auto& serIter : serMap) {
            auto& phyMap = serIter.second;
            for (auto& phyIter : phyMap) {
                printf("=======================================================\n");
                printf("rankId is : %d \n", rankIdx);
                u32 threadIdx = 0;
                const SingleTaskQueue& taskQueue = phyIter.second;
                for (auto& thread : taskQueue) {
                    printf("threadIdx : %d, taskNum : %ld \n", threadIdx, thread.size());
                    printf("-------------------------------------------------------\n");
                    for (auto& task : thread) {
                        std::string tempStr = task->Describe();
                        printf("rankIdx:%d, threadIdx:%d, %s\n", rankIdx, threadIdx, tempStr.c_str());
                    }
                    threadIdx++;
                }
                rankIdx++;
            }
        }
    }
}
}