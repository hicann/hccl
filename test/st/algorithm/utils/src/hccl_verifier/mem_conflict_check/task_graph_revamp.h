/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV1_GRAPH_REVAMP_H
#define HCCLV1_GRAPH_REVAMP_H

#include <map>
#include <set>
#include <vector>
#include <memory>
#include <queue>

#include "log.h"
#include "sim_task.h"
#include "task_graph_generator.h"
#include "sim_common.h"
#include "task_def.h"

namespace HcclSim {
// vector struct is preserved for multi-QP
// 只维护最后一个TaskNodePtr的信息
using VirtQueMgr = std::map<RankId, std::map<RankId, std::vector<TaskNodePtr>>>;
using VirtAivBlockMgr = std::map<std::pair<RankId, u32>, std::map<std::pair<RankId, u32>, std::vector<TaskNodePtr>>>;

class GraphRevampBilateralSemantics {
public:
    HcclResult Revamp(TaskNodePtr dummyStart, AllRankTaskQueues& taskQueue);
    virtual ~GraphRevampBilateralSemantics();
private:
    HcclResult InitRankHead(TaskNodePtr dummyStart, std::map<RankId, TaskNodePtr> &rank2Head, AllRankTaskQueues& taskQueue);
    HcclResult RevampGraph(TaskNodePtr dummyStart, std::map<RankId, TaskNodePtr> &rank2Head);
    HcclResult ProceedNode(TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue, std::set<TaskNodePtr> &visited);

    HcclResult ProcReadNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult ProcSdmaRWNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult SearchBackwardSdmaRW(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult SearchForwardSdmaRW(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    bool IsVirtualTask(TaskNodePtr node);
    bool IsReadWriteWithSameRank(RankId peerRank, TaskNodePtr candNode);
    bool IsBeingReadOrWrittenTask(TaskNodePtr candNode);
    bool IsProceedParentNode(TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue, std::set<TaskNodePtr> &visited);

    HcclResult AddBeingRWNodeToVirtualQue(TaskNodePtr currNode, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult AddBeingRWNodeToVirtualQueWithWait(TaskNodePtr waitNode, TaskNodePtr currNode, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult InsertNode(TaskNodePtr headNode, TaskNodePtr insertNode);
    HcclResult AddTerminalNodePeerRankVirtualQue(TaskNodePtr candNode, TaskNodePtr currNode, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult PrepAvailVirtQueTail(const RankId myRank, const RankId remRank, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult AddVirtQueTail(const RankId myRank, const RankId remRank, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, TaskNodePtr waitFromHeadQueNode);

    HcclResult ProcWriteNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult ProcRdmaWriteNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult SearchBackwardRdmaWrite(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult SearchForwardRdmaWrite(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult AddBeingWrittenRdmaWithWait(TaskNodePtr candNode, TaskNodePtr currNode, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult AddTerminalNodePeerRankVirtualQue(TaskNodePtr candNode, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult AddWaitToCurRankVitualQue(TaskNodePtr currNode, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult TransferCurNodeToVitualQue(TaskNodePtr currNode, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult AddTerminalNodeCurRankVirtualQue(TaskNodePtr candNode, RankId peerRank, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    TaskStub* GenTaskStubBeingReadOrWrittern(TaskNodePtr currNode);
    HcclResult GetPeerRankByTaskNode(TaskNodePtr currNode, RankId &peerRank);
    HcclResult GetLinkProtoStubByTaskNode(TaskNodePtr currNode, LinkProtoStub &link);
    map<RankId, u32> rank2QueSize_;
    std::vector<TaskStub*> toDeleteTaskResource_;
    std::vector<TaskNodePtr> toDeleteTaskNodeResource_;

    //for Aiv revamp
    std::map<RankId, std::map<u32, TaskNodePtr>> rank2AivStart_;
    std::map<RankId, std::map<u32, u32>> rank2AivStartSize_;
    HcclResult ProcAivNode(TaskNodePtr aivTaskNode);
    HcclResult ProcAivRWNode(TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult ProcSdmaAivRWNode(TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult SearchBackwardSdmaAivRW(TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult SearchForwardSdmaAivRW(TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult AddBeingAivRWNodeToVirtualQueWithRecvSync(TaskNodePtr recvNode, TaskNodePtr sendNode, 
        TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult AddBeingAivRWNodeToVirtualQue(TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult AddAivTerminalNodePeerRankVirtualQue(TaskNodePtr sendNode, TaskNodePtr recvNode, 
        TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult PrepAvailAivVirtQueTail(std::pair<RankId, u32> peerRankAndPosPair, 
        std::pair<RankId, u32> currRankAndPosPair, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult InsertAivNode(TaskNodePtr headNode, TaskNodePtr insertNode);
    HcclResult ProceedAivNode(TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue,
                                                      std::set<TaskNodePtr> &isVisited);
    bool IsProceedAivParentNode(TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue,
                                                        std::set<TaskNodePtr> &isVisited);
};
} // namespace Hccl

#endif