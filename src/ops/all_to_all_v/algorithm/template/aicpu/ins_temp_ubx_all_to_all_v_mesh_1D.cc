/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/ins_temp_ubx_all_to_all_v_mesh_1D.h"

#define NET_NUM 2

namespace ops_hccl {
constexpr u32 UBX_BOARD_PAIR_SIZE = 2;
// 跨框CLOS端口数：与 InsTempAlltoAllVMesh1D 约定一致，CLOS 场景 portNum 传 8；
// CalcMeshParam 在 isPod=true 时内部 /2，得 4jetty 并发（对应 maxPathNum_ 的“跨框最多4jetty”）。
constexpr u32 UBX_CLOS_PORT_NUM = 4;
InsTempUBXAllToAllVMesh1D::InsTempUBXAllToAllVMesh1D(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempUBXAllToAllVMesh1D::~InsTempUBXAllToAllVMesh1D() {}

std::vector<CostModelParam> InsTempUBXAllToAllVMesh1D::CalcCostCoeff(CalcCostCoeffParam param)
{
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;
    int taskNum
        = CostModelManager::CalcTransTaskNum(param.rankSize) + CostModelManager::CalcSyncTaskNum(param.rankSize) * 2;
    // UBX拓扑: 板内fullmesh(rankNumPerBoard卡, 单链) + 跨框CLOS(boardNum板, 4jetty), R = B*P
    // 板内卡数P取layer0的1DMESH实例rank数(与运行时GetRankNumPerBoard同源于rank graph), 板数B = rankSize/P
    u32 rankNumPerBoard = 0;
    if (param.topoInfo != nullptr && !param.topoInfo->topoInstDetailsOfLayer.empty()) {
        const auto& rankNumMap = param.topoInfo->topoInstDetailsOfLayer[0].rankNumForTopoType;
        auto meshItr = rankNumMap.find(CommTopo::COMM_TOPO_1DMESH);
        if (meshItr != rankNumMap.end() && !meshItr->second.empty()) {
            rankNumPerBoard = meshItr->second[0];
        }
    }
    if (rankNumPerBoard == 0 || rankNumPerBoard > param.rankSize || param.rankSize % rankNumPerBoard != 0) {
        // 兜底: UBX机型fullmesh内最多4卡
        rankNumPerBoard = std::min(param.rankSize, 4U);
    }
    u32 boardNum = param.rankSize / rankNumPerBoard;

    // A: 传输时间, 以总数据量S为单位
    // 跨框: (B-1)轮板配对 * 每轮P步 * 每步S/R数据, 4jetty并发 => (B-1)*S/(4B*bw)
    float aClos = 0.0f;
    if (boardNum > 1) {
        // 跨框CLOS 4jetty 并发，portNum 取 UBX_CLOS_PORT_NUM(8)，不能直接用 executor 传入的 param.portNum[0]（其值为
        // 1）： isPod=true 时 CalcMeshParam 内部会 /2，1/2 整数除法得 0，导致 A = n*(rankSize-1)/(0*bw) = inf。
        CostModelManager::Global()->CalcMeshParam(
            1.0f * rankNumPerBoard, CommTopo::COMM_TOPO_CLOS, static_cast<int>(UBX_CLOS_PORT_NUM), boardNum, aClos,
            param.isPod);
    }
    // 板内: fullmesh一轮, 每链路S/R数据(链路间并行) => S/(R*bw)
    float aMesh = 0.0f;
    CostModelManager::Global()->CalcMeshParam(1.0f, CommTopo::COMM_TOPO_1DMESH, 1, rankNumPerBoard, aMesh, param.isPod);
    // 板内fullmesh与首轮跨框并发, 取较大值
    A = std::max(aClos, aMesh);

    // B: 落地拷贝(scratch->output)与下一轮跨框传输流水重叠被掩盖, 仅最后一轮暴露S/B
    // (单板B=1时无传输可掩盖, 恰好退化为全量落地)
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(1.0f, EngineType::AICPU, B);
    }

    // C: 任务数 = (B-1)轮*P步跨框收发 + 板内fullmesh(P-1次) + 自环, 约等于rankSize
    CostModelManager::Global()->CalcLatencyParams(static_cast<int>(rankNumPerBoard), EngineType::AICPU, C);

    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AICPU, D);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    HCCL_INFO(
        "[%s] rankSize=%u boardNum=%u rankNumPerBoard=%u, CalcCostCoeff A=%f B=%f C=%f D=%f.", __func__, param.rankSize,
        boardNum, rankNumPerBoard, A, B, C, D);
    return params;
}

HcclResult InsTempUBXAllToAllVMesh1D::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> level0Channels;
    std::vector<HcclChannelDesc> myChannelDescs;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        bool isIsolation = !(IsAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH));
        CHK_RET(
            CalcChannelRequestMeshClosMultiJetty(comm, param, topoInfo, subCommRanks_, myChannelDescs, isIsolation));
        for (auto channel : myChannelDescs) {
            if (channel.channelProtocol == COMM_PROTOCOL_UB_CTP) {
                level0Channels.push_back(channel);
            }
        }
        HCCL_DEBUG("[InsTempUBXAllToAllVMesh1D::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    channelsPerRank_ = CalcChannelsPerRank(level0Channels);
    HCCL_INFO("[InsTempUBXAllToAllVMesh1D][CalcRes] channelsPerRank_ is [%u]", channelsPerRank_);
    // 按照当前来看，channelsPerRank_ 就是maxPathNum_
    if (channelsPerRank_ > maxPathNum_) {
        HCCL_ERROR(
            "[InsTempUBXAllToAllVMesh1D][CalcRes] channelsPerRank_[%u] is more than [%u]", channelsPerRank_,
            maxPathNum_);
        return HCCL_E_NOT_SUPPORT;
    }

    // UBX机型，fullmesh内最多四卡
    // 跨框的时候，最多4 jetty
    // fullmesh 和 clos 并发
    resourceRequest.slaveThreadNum = maxPathNum_ + maxRankNumPerBoard_;
    HCCL_INFO("[InsTempUBXAllToAllVMesh1D::CalcRes] slaveThreadNum is [%u]", resourceRequest.slaveThreadNum);
    for (u32 index = 0; index < resourceRequest.slaveThreadNum; index++) {
        // 从流的notify数量以rank间channel数的最大值为准，用于和主流同步以及同一个rank多条链路间的同步
        resourceRequest.notifyNumPerThread.push_back(channelsPerRank_);
    }
    resourceRequest.notifyNumOnMainThread = resourceRequest.slaveThreadNum;
    return HCCL_SUCCESS;
}

u64 InsTempUBXAllToAllVMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // 直接切成rankSize份
    return templateRankSize_;
}

HcclResult InsTempUBXAllToAllVMesh1D::GetBoardSendRecvMatrix(u32 n, std::vector<std::vector<u32>>& sendRecvMatrix)
{
    // n必须是
    if (n < UBX_BOARD_PAIR_SIZE || n % UBX_BOARD_PAIR_SIZE != 0) {
        HCCL_ERROR("n is [%u]", n);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    // 创建环形排列：0固定，其余1到n-1
    std::vector<int> ring(n);
    ring[0] = 0;
    for (int i = 1; i < n; i++) {
        ring[i] = i;
    }

    // 生成n-1轮调度
    for (int round = 0; round < n - 1; round++) {
        // 生成本轮配对：首尾对称配对
        for (int i = 0; i < n / UBX_BOARD_PAIR_SIZE; i++) {
            int a = ring[i];
            int b = ring[n - 1 - i];

            // 写入调度矩阵
            sendRecvMatrix[a][round] = b;
            sendRecvMatrix[b][round] = a;
        }

        // 旋转环形排列：固定第一个元素0，其余旋转一位
        // 保存最后一个元素
        int last = ring[n - 1];
        // 从倒数第二个元素开始向前移动
        for (int i = n - 1; i > 1; i--) {
            ring[i] = ring[i - 1];
        }
        // 将原最后一个元素放到第二个位置
        ring[1] = last;
    }
    // 打印矩阵
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - 1; j++) {
            HCCL_DEBUG("sendRecvMatrix[%d][%d] = %d", i, j, sendRecvMatrix[i][j]);
        }
    }
    // 举个例子：n = 8 的时候，矩阵如下
    // board	step1	step2	step3	step4	step5	step6	step7
    //     0	    7	    6	    5	    4	    3	    2	    1
    //     1	    6	    4	    2	    7	    5	    3	    0
    //     2	    5	    3	    1	    6	    4	    0	    7
    //     3	    4	    2	    7	    5	    0	    1	    6
    //     4	    3	    1	    6	    0	    2	    7	    5
    //     5	    2	    7	    0	    3	    1	    6	    4
    //     6	    1	    0	    4	    2	    7	    5	    3
    //     7	    0	    5	    3	    1	    6	    4	    2
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempUBXAllToAllVMesh1D::GetRankSendRecvMatrix(
    u32 board1, u32 board2, std::vector<std::vector<u32>>& rankSendRecvMatrix)
{
    u32 boardSmall = std::min(board1, board2);
    u32 boardBig = std::max(board1, board2);

    for (u32 rankIndex = boardSmall * rankNumPerBoard_; rankIndex < (boardSmall + 1) * rankNumPerBoard_; rankIndex++) {
        for (u32 step = 0; step < rankNumPerBoard_; step++) {
            u32 targetRank = boardBig * rankNumPerBoard_ + (rankIndex + step) % rankNumPerBoard_;
            rankSendRecvMatrix[rankIndex][step] = targetRank;
            rankSendRecvMatrix[targetRank][step] = rankIndex;
        }
    }
    // 打印矩阵
    for (int i = 0; i < rankSendRecvMatrix.size(); i++) {
        for (int j = 0; j < rankNumPerBoard_; j++) {
            HCCL_DEBUG(
                "myRank is [%u], board1 is[%u], board2 is [%u], rankSendRecvMatrix[%d][%d] = [%d]", myAlgRank_, board1,
                board2, i, j, rankSendRecvMatrix[i][j]);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempUBXAllToAllVMesh1D::GetRankNumPerBoard(TemplateResource& templateResource)
{
    // 根据channel，和远端rank只有一条链路的，说明是fullmesh内的
    // 遍历所有卡
    u32 fullMeshRankNum = 1;
    std::map<u32, std::vector<ChannelInfo>>& channels = templateResource.channels;
    for (u32 rank = 0; rank < templateRankSize_; rank++) {
        if (rank == myAlgRank_) {
            continue;
        }
        u32 linkNumSendRecv = channels.at(rank).size();
        if (linkNumSendRecv == 1) { // 说明是fullmesh内
            HCCL_INFO("myRank[%u] and curRank[%u] is fullMesh link", myAlgRank_, rank);
            fullMeshRankNum++;
        }
    }

    rankNumPerBoard_ = fullMeshRankNum;
    if (rankNumPerBoard_ > maxRankNumPerBoard_) {
        // fullmesh 内最多4P
        HCCL_ERROR(
            "[InsTempUBXAllToAllVMesh1D][GetRankNumPerBoard] rankNumPerBoard_[%u] is more than [%u]", rankNumPerBoard_,
            maxRankNumPerBoard_);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    // 纯clos不支持
    if (rankNumPerBoard_ == 1) {
        HCCL_ERROR(
            "[InsTempUBXAllToAllVMesh1D][GetRankNumPerBoard] rankNumPerBoard_ is [%u], "
            "templateRankSize_ is [%u], only clos is not support",
            rankNumPerBoard_, templateRankSize_);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    // 纯fullmesh暂时先支持
    if (rankNumPerBoard_ == templateRankSize_) {
        HCCL_WARNING(
            "[InsTempUBXAllToAllVMesh1D][GetRankNumPerBoard] rankNumPerBoard_ is [%u], "
            "templateRankSize_ is [%u], here is only fullmesh",
            rankNumPerBoard_, templateRankSize_);
    }
    // rankSize要是rankNumPerBoard_的整数倍
    if (templateRankSize_ % rankNumPerBoard_ != 0) {
        HCCL_ERROR(
            "[InsTempUBXAllToAllVMesh1D][GetRankNumPerBoard] rankNumPerBoard_ is [%u], "
            "templateRankSize_ is [%u]",
            rankNumPerBoard_, templateRankSize_);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempUBXAllToAllVMesh1D::CheckPathNum(TemplateResource& templateResource)
{
    // 校验是不是最多4jetty
    std::map<u32, std::vector<ChannelInfo>>& channels = templateResource.channels;
    for (u32 rank = 0; rank < templateRankSize_; rank++) {
        if (rank == myAlgRank_) {
            continue;
        }
        u32 linkNumSendRecv = channels.at(rank).size();
        HCCL_INFO("myRank is [%u], targetRank is [%u], linkNum is [%u] ", myAlgRank_, rank, linkNumSendRecv);
        if (linkNumSendRecv > maxPathNum_) {
            HCCL_ERROR(
                "myRank is [%u], targetRank is [%u], linkNum is [%u], linkNum should not bigger than [%u]", myAlgRank_,
                rank, linkNumSendRecv, maxPathNum_);
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempUBXAllToAllVMesh1D::RunFullMesh(const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    std::map<u32, std::vector<ChannelInfo>>& channels = templateResource.channels;
    std::vector<ThreadHandle>& threads = templateResource.threads;
    // 用后面的 maxRankNumPerBoard_ 条流做fullmesh内的收发
    localCopyInfoFullMesh_.clear();
    for (u32 i = 0; i < rankNumPerBoard_; i++) {
        u32 targetRank = currBoard_ * rankNumPerBoard_ + i;
        HCCL_INFO("[InsTempUBXAllToAllVMesh1D] targetRank is [%u]", targetRank);
        u32 fullMeshThreadId = maxPathNum_ + 1 + i;
        if (targetRank == myAlgRank_) { // 发给自己
            DataSlice usrInSlices = DataSlice(
                tempAlgParams.buffInfo.inputPtr, tempAlgParams.sdispls[myAlgRank_] * dataTypeSize_,
                tempAlgParams.sendCounts[myAlgRank_] * dataTypeSize_, tempAlgParams.sendCounts[myAlgRank_]);
            DataSlice usrOutSlices = DataSlice(
                tempAlgParams.buffInfo.outputPtr, tempAlgParams.rdispls[myAlgRank_] * dataTypeSize_,
                tempAlgParams.recvCounts[myAlgRank_] * dataTypeSize_, tempAlgParams.recvCounts[myAlgRank_]);
            CHK_RET(LocalCopy(threads[fullMeshThreadId], usrInSlices, usrOutSlices));
        } else {                                                             // 和板内其他卡去收发
            const ChannelInfo& channelSendRecv = channels.at(targetRank)[0]; // 和对端收发
            // 对称路径直读对端input中发往本卡的数据并写入本地output，不经过scratch。
            if (enableRemoteMemAccess_) {
                const u32 peerRank = subCommRanks_[0][targetRank];
                void* peerInput = nullptr;
                CHK_RET(GetSymmetricPeerInput(peerRank, &peerInput));
                DataSlice rxSrcSlice(
                    peerInput, tempAlgParams.sdispls[myAlgRank_] * dataTypeSize_,
                    tempAlgParams.recvCounts[targetRank] * dataTypeSize_, tempAlgParams.recvCounts[targetRank]);
                DataSlice rxDstSlice(
                    tempAlgParams.buffInfo.outputPtr, tempAlgParams.rdispls[targetRank] * dataTypeSize_,
                    tempAlgParams.recvCounts[targetRank] * dataTypeSize_, tempAlgParams.recvCounts[targetRank]);
                std::vector<DataSlice> emptySlices;
                TxRxSlicesList sendRecvSlicesList({emptySlices, emptySlices}, {{rxSrcSlice}, {rxDstSlice}});
                TxRxChannels sendRecvChannels(channelSendRecv, channelSendRecv);
                SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
                CHK_RET(SendRecvBatchRead(sendRecvInfo, threads[fullMeshThreadId]));
                continue;
            }
            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            DataSlice txSrcSlice = DataSlice(
                tempAlgParams.buffInfo.inputPtr, tempAlgParams.sdispls[targetRank] * dataTypeSize_,
                tempAlgParams.sendCounts[targetRank] * dataTypeSize_, tempAlgParams.sendCounts[targetRank]);
            DataSlice txDstSlice = DataSlice(
                channelSendRecv.remoteCclMem.addr, myAlgRank_ * scratchBufferSizePerRank_,
                tempAlgParams.sendCounts[targetRank] * dataTypeSize_, tempAlgParams.sendCounts[targetRank]);
            txSrcSlices.push_back(txSrcSlice);
            txDstSlices.push_back(txDstSlice);
            std::vector<DataSlice> rxSrcSlices;
            std::vector<DataSlice> rxDstSlices;
            DataSlice rxSrcSlice = DataSlice(
                channelSendRecv.remoteCclMem.addr, targetRank * scratchBufferSizePerRank_,
                tempAlgParams.recvCounts[targetRank] * dataTypeSize_, tempAlgParams.recvCounts[targetRank]);
            DataSlice rxDstSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr, targetRank * scratchBufferSizePerRank_,
                tempAlgParams.recvCounts[targetRank] * dataTypeSize_, tempAlgParams.recvCounts[targetRank]);
            rxSrcSlices.push_back(rxSrcSlice);
            rxDstSlices.push_back(rxDstSlice);
            SendRecvInfo sendRecvInfo{
                {channelSendRecv, channelSendRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
            if (tempAlgParams.sendCounts[targetRank] > 0 && tempAlgParams.recvCounts[targetRank] > 0) {
                CHK_RET(SendRecvWrite(sendRecvInfo, threads[fullMeshThreadId]));
            } else if (tempAlgParams.sendCounts[targetRank] > 0) {
                DataInfo sendInfo{channelSendRecv, {txSrcSlices, txDstSlices}, dataType_};
                CHK_RET(SendWrite(sendInfo, threads[fullMeshThreadId]));
            } else if (tempAlgParams.recvCounts[targetRank] > 0) {
                DataInfo recvInfo{channelSendRecv, {rxSrcSlices, rxDstSlices}, dataType_};
                CHK_RET(RecvWrite(recvInfo, threads[fullMeshThreadId]));
            }
            // 构造下一step localCopy的数据
            DataSlice usrOutSlices = DataSlice(
                tempAlgParams.buffInfo.outputPtr, tempAlgParams.rdispls[targetRank] * dataTypeSize_,
                tempAlgParams.recvCounts[targetRank] * dataTypeSize_, tempAlgParams.recvCounts[targetRank]);
            if (tempAlgParams.recvCounts[targetRank] > 0) {
                localCopyInfoFullMesh_.push_back(std::vector<DataSlice>{rxDstSlice, usrOutSlices});
            }
        }
    }
    needDealWithFullMeshInfo_ = true;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempUBXAllToAllVMesh1D::RunPairwise(
    const TemplateDataParams& tempAlgParams, TemplateResource& templateResource, u32 targetBoard)
{
    std::map<u32, std::vector<ChannelInfo>>& channels = templateResource.channels;
    std::vector<ThreadHandle>& threads = templateResource.threads;

    // curBoard 和 targetBoard 两两收发，要按照编排好的次序
    std::vector<std::vector<u32>> rankSendRecvMatrix(
        algBoardNum_ * rankNumPerBoard_, std::vector<u32>(rankNumPerBoard_, 0));
    CHK_RET(GetRankSendRecvMatrix(currBoard_, targetBoard, rankSendRecvMatrix));
    // 按照一定的顺序，遍历board内的卡
    for (u32 step = 0; step < rankNumPerBoard_; step++) {
        u32 targetRank = rankSendRecvMatrix[myAlgRank_][step];

        // 跳过虚拟board 的收发
        if (targetRank >= templateRankSize_) {
            HCCL_INFO(
                "[InsTempUBXAllToAllVMesh1D] myRank is [%u], targetRank is [%u], "
                "targetRank is out of rankSize, skip",
                myAlgRank_, targetRank);
            continue;
        }

        // 每个targetRank都是一个单独的step
        if (threadNum_ > 1) {
            GetNotifyIdxMainToClos(notifyIdxMainToSub_);
            CHK_RET(PreSyncInterThreads(threads[0], subThreadsBoard_, notifyIdxMainToSub_));
        }
        // 一条流做上一个step的后拷贝
        for (u32 i = 0; i < localCopyInfo_.size(); i++) {
            CHK_RET(LocalCopy(threads[0], localCopyInfo_[i][0], localCopyInfo_[i][1]));
        }

        // 这里要处理一次 localCopyInfoFullMesh_，第一次进来的时候需要处理一下
        if (needDealWithFullMeshInfo_) {
            for (u32 i = 0; i < localCopyInfoFullMesh_.size(); i++) {
                CHK_RET(LocalCopy(threads[0], localCopyInfoFullMesh_[i][0], localCopyInfoFullMesh_[i][1]));
            }
            needDealWithFullMeshInfo_ = false;
            localCopyInfoFullMesh_.clear(); // 用完要清空
        }

        // 其他流开始和对端卡 4 jetty 收发
        u32 linkNumSendRecv = channels.at(targetRank).size();
        HCCL_INFO(
            "[InsTempUBXAllToAllVMesh1D] myRank is [%u], targetRank is [%u], currBoard is [%u], "
            "targetBoard is [%u], linkNumSendRecv is[%u]",
            myAlgRank_, targetRank, currBoard_, targetBoard, linkNumSendRecv);
        const std::vector<ChannelInfo>& channelSendRecv = channels.at(targetRank);
        void* peerInput = nullptr;
        if (enableRemoteMemAccess_) {
            const u32 peerRank = subCommRanks_[0][targetRank];
            CHK_RET(GetSymmetricPeerInput(peerRank, &peerInput));
        }
        std::vector<float> dataSplitRate(linkNumSendRecv, (float)1.0 / (float)linkNumSendRecv);
        u64 innerSendOffset = 0;
        u64 innerRecvOffset = 0;
        u64 curSendDataCount = tempAlgParams.sendCounts[targetRank];
        u64 curRecvDataCount = tempAlgParams.recvCounts[targetRank];
        for (u32 j = 0; j < linkNumSendRecv; j++) {
            u64 innerCurrSendDataCount = curSendDataCount * dataSplitRate[j];
            u64 innerCurrRecvDataCount = curRecvDataCount * dataSplitRate[j];
            if (j == linkNumSendRecv - 1) {
                innerCurrSendDataCount = curSendDataCount - innerSendOffset;
                innerCurrRecvDataCount = curRecvDataCount - innerRecvOffset;
            }
            u64 innerCurrSendDataSize = innerCurrSendDataCount * dataTypeSize_;
            u64 innerCurrRecvDataSize = innerCurrRecvDataCount * dataTypeSize_;
            u32 queId = j + 1; // 主流用来后同步了
            // 对称路径只提交read任务，数据从对端input直接落入本地output。
            if (enableRemoteMemAccess_) {
                DataSlice rxSrcSlice(
                    peerInput, (tempAlgParams.sdispls[myAlgRank_] + innerRecvOffset) * dataTypeSize_,
                    innerCurrRecvDataSize, innerCurrRecvDataCount);
                DataSlice rxDstSlice(
                    tempAlgParams.buffInfo.outputPtr,
                    (tempAlgParams.rdispls[targetRank] + innerRecvOffset) * dataTypeSize_, innerCurrRecvDataSize,
                    innerCurrRecvDataCount);
                std::vector<DataSlice> emptySlices;
                TxRxSlicesList sendRecvSlicesList({emptySlices, emptySlices}, {{rxSrcSlice}, {rxDstSlice}});
                TxRxChannels sendRecvChannels(channelSendRecv[j], channelSendRecv[j]);
                SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
                CHK_RET(SendRecvBatchRead(sendRecvInfo, threads[queId]));
                innerSendOffset += innerCurrSendDataCount;
                innerRecvOffset += innerCurrRecvDataCount;
                continue;
            }
            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;

            DataSlice txSrcSlice = DataSlice(
                tempAlgParams.buffInfo.inputPtr, (tempAlgParams.sdispls[targetRank] + innerSendOffset) * dataTypeSize_,
                innerCurrSendDataSize, innerCurrSendDataCount);
            DataSlice txDstSlice = DataSlice(
                channelSendRecv[j].remoteCclMem.addr,
                myAlgRank_ * scratchBufferSizePerRank_ + innerSendOffset * dataTypeSize_, innerCurrSendDataSize,
                innerCurrSendDataCount);
            txSrcSlices.push_back(txSrcSlice);
            txDstSlices.push_back(txDstSlice);

            std::vector<DataSlice> rxSrcSlices;
            std::vector<DataSlice> rxDstSlices;
            DataSlice rxSrcSlice = DataSlice(
                channelSendRecv[j].remoteCclMem.addr,
                targetRank * scratchBufferSizePerRank_ + innerRecvOffset * dataTypeSize_, innerCurrRecvDataSize,
                innerCurrRecvDataCount);
            DataSlice rxDstSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                targetRank * scratchBufferSizePerRank_ + innerRecvOffset * dataTypeSize_, innerCurrRecvDataSize,
                innerCurrRecvDataCount);
            rxSrcSlices.push_back(rxSrcSlice);
            rxDstSlices.push_back(rxDstSlice);

            if (innerCurrSendDataSize > 0 && innerCurrRecvDataSize > 0) {
                SendRecvInfo sendRecvInfo{
                    {channelSendRecv[j], channelSendRecv[j]}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
                CHK_RET(SendRecvWrite(sendRecvInfo, threads[queId]));
            } else if (innerCurrSendDataSize > 0) {
                DataInfo sendInfo{channelSendRecv[j], {txSrcSlices, txDstSlices}, dataType_};
                CHK_RET(SendWrite(sendInfo, threads[queId]));
            } else if (innerCurrRecvDataSize > 0) {
                DataInfo recvInfo{channelSendRecv[j], {rxSrcSlices, rxDstSlices}, dataType_};
                CHK_RET(RecvWrite(recvInfo, threads[queId]));
            }
            innerSendOffset += innerCurrSendDataCount;
            innerRecvOffset += innerCurrRecvDataCount;
        }

        // 普通路径保留scratch到output的后拷贝；对称路径已经直接写入output。
        if (!enableRemoteMemAccess_) {
            DataSlice scratchSlices = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr, targetRank * scratchBufferSizePerRank_,
                curRecvDataCount * dataTypeSize_, curRecvDataCount);
            DataSlice usrOutSlices = DataSlice(
                tempAlgParams.buffInfo.outputPtr, tempAlgParams.rdispls[targetRank] * dataTypeSize_,
                curRecvDataCount * dataTypeSize_, curRecvDataCount);
            localCopyInfo_.clear();
            if (curRecvDataCount > 0) {
                localCopyInfo_.push_back(std::vector<DataSlice>{scratchSlices, usrOutSlices});
            }
        }

        if (threadNum_ > 1) {
            GetNotifyIdxClosToMain(notifyIdxSubToMain_);
            CHK_RET(PostSyncInterThreads(threads[0], subThreadsBoard_, notifyIdxSubToMain_));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempUBXAllToAllVMesh1D::InitParam(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempUBXAllToAllVMesh1D][InitParam] Run Start");

    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = HCCL_SIZE_TABLE[dataType_];

    // myAlgRank_应该是逻辑rank，myRank_ 是物理rank，如果全打平来看，应该是一样的
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        myAlgRank_ = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempUBXAllToAllVMesh1D][InitParam] subCommRanks_ or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }

    CHK_RET(GetRankNumPerBoard(templateResource));
    CHK_RET(CheckPathNum(templateResource));

    currBoard_ = myAlgRank_ / rankNumPerBoard_;
    currRankIndex_ = myAlgRank_ % rankNumPerBoard_;
    boardNum_ = templateRankSize_ / rankNumPerBoard_;
    // 考虑boardNum 不是偶数的场景,先配置成偶数board，实际收发再跳过
    algBoardNum_ = boardNum_;
    if (boardNum_ % UBX_BOARD_PAIR_SIZE == 1) {
        algBoardNum_ = algBoardNum_ + 1;
    }

    scratchBufferSizePerRank_ = tempAlgParams.inputSliceStride;
    dataStridePerRank_ = tempAlgParams.outputSliceStride;
    curProcessedDataCount_ = tempAlgParams.processedDataCount;
    curDataCount_ = tempAlgParams.count;
    curDataSize_ = curDataCount_ * dataTypeSize_;
    threadNum_ = templateResource.threads.size();
    if (threadNum_ != maxPathNum_ + maxRankNumPerBoard_ + 1) {
        HCCL_ERROR(
            "[InsTempUBXAllToAllVMesh1D] tempInsQues.size() is [%u], but it should be [%u]", threadNum_,
            maxPathNum_ + maxRankNumPerBoard_ + 1);
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_INFO(
        "[InsTempUBXAllToAllVMesh1D] myRank_ is [%u], myAlgRank_ is [%u], rankNumPerBoard_ is [%u], "
        "templateRankSize_ is [%u], currBoard_ is [%u], currRankIndex_ is [%u], boardNum_ is [%u], "
        "algBoardNum_ is [%u], threadNum_ is [%u], scratchBufferSizePerRank_ is [%llu], dataStridePerRank_ is [%llu], "
        "curProcessedDataCount_ is [%llu], curDataCount_ is [%llu], curDataSize_ is [%llu]",
        myRank_, myAlgRank_, rankNumPerBoard_, templateRankSize_, currBoard_, currRankIndex_, boardNum_, algBoardNum_,
        threadNum_, scratchBufferSizePerRank_, dataStridePerRank_, curProcessedDataCount_, curDataCount_, curDataSize_);

    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_DEBUG(
            "[InsTempUBXAllToAllVMesh1D] rank is [%u], sendCounts[%u] is [%llu], recvCounts[%u] is [%llu], "
            "sdispls[%u] is [%llu], rdispls[%u] is [%llu]",
            myAlgRank_, i, tempAlgParams.sendCounts[i], i, tempAlgParams.recvCounts[i], i, tempAlgParams.sdispls[i], i,
            tempAlgParams.rdispls[i]);
    }

    std::vector<ThreadHandle>& threads = templateResource.threads;
    if (threadNum_ > 1) {
        subThreadsBoard_.assign(threads.begin() + 1, threads.begin() + 1 + maxPathNum_); // clos用
        subThreadsFullMesh_.assign(
            threads.begin() + 1 + maxPathNum_,
            threads.begin() + 1 + maxPathNum_ + maxRankNumPerBoard_); // fullmesh用
    }

    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    if (enableRemoteMemAccess_) {
        inputSymWindow_ = param.inputSymWindow;
        outputSymWindow_ = param.outputSymWindow;
        inputOffset_ = param.inputOffset;
        outputOffset_ = param.outputOffset;
        HCCL_INFO(
            "[InsTempUBXAllToAllVMesh1D][InitParam] symmetric memory enabled, "
            "inputSymWin[%p] outputSymWin[%p] inOff[%llu] outOff[%llu].",
            inputSymWindow_, outputSymWindow_, inputOffset_, outputOffset_);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempUBXAllToAllVMesh1D::GetSymmetricPeerInput(u32 peerRank, void** peerInput)
{
    CHK_PTR_NULL(peerInput);
    HcclResult ret = GetSymWinRemoteMem(inputSymWindow_, inputOffset_, peerRank, peerInput);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS || *peerInput == nullptr,
        HCCL_ERROR(
            "[InsTempUBXAllToAllVMesh1D][GetSymmetricPeerInput] failed, peerRank[%u] ret[%d] ptr[%p].", peerRank, ret,
            *peerInput),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempUBXAllToAllVMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempUBXAllToAllVMesh1D][KernelRun] Run Start");

    CHK_RET(InitParam(param, tempAlgParams, templateResource));

    localCopyInfo_.clear();
    localCopyInfoFullMesh_.clear();

    std::vector<ThreadHandle>& threads = templateResource.threads;
    std::vector<std::vector<u32>> sendRecvMatrix(algBoardNum_, std::vector<u32>(algBoardNum_ - 1, 0));
    CHK_RET(GetBoardSendRecvMatrix(algBoardNum_, sendRecvMatrix));

    // 这里是fullmesh的前同步
    if (threadNum_ > 1) {
        GetNotifyIdxMainToFullMesh(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(threads[0], subThreadsFullMesh_, notifyIdxMainToSub_));
    }

    // 前 maxPathNum_条从流用来做clos通信
    // 第一轮的pairwise和fullmesh并行
    u32 targetBoard = sendRecvMatrix[currBoard_][0];
    CHK_RET(RunPairwise(tempAlgParams, templateResource, targetBoard));

    // fullmesh的前后流同步要取出来，不然就变串行了
    CHK_RET(RunFullMesh(tempAlgParams, templateResource));

    // 这里是fullmesh的后同步
    if (threadNum_ > 1) {
        GetNotifyIdxFullMeshToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(threads[0], subThreadsFullMesh_, notifyIdxSubToMain_));
    }

    for (u32 boardIndex = 1; boardIndex < algBoardNum_ - 1; boardIndex++) {
        targetBoard = sendRecvMatrix[currBoard_][boardIndex];
        CHK_RET(RunPairwise(tempAlgParams, templateResource, targetBoard));
    }

    // 最后做一次后同步
    for (u32 i = 0; i < localCopyInfo_.size(); i++) {
        CHK_RET(LocalCopy(threads[0], localCopyInfo_[i][0], localCopyInfo_[i][1]));
    }
    for (u32 i = 0; i < localCopyInfoFullMesh_.size(); i++) {
        CHK_RET(LocalCopy(threads[0], localCopyInfoFullMesh_[i][0], localCopyInfoFullMesh_[i][1]));
    }

    HCCL_INFO("[InsTempUBXAllToAllVMesh1D] AllToAll full mesh rank[%d] finish.", myAlgRank_);
    return HcclResult::HCCL_SUCCESS;
}

void InsTempUBXAllToAllVMesh1D::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub)
{
    (void)notifyIdxMianToSub;
    return;
}

void InsTempUBXAllToAllVMesh1D::GetNotifyIdxMainToClos(std::vector<u32>& notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    if (threadNum_ <= 1) {
        return;
    }
    u32 slaveThreadNum = threadNum_ - 1 - maxRankNumPerBoard_;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempUBXAllToAllVMesh1D::GetNotifyIdxMainToFullMesh(std::vector<u32>& notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    if (threadNum_ <= 1) {
        return;
    }
    u32 slaveThreadNum = threadNum_ - 1 - maxPathNum_;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempUBXAllToAllVMesh1D::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    (void)notifyIdxSubToMain;
    return;
}

void InsTempUBXAllToAllVMesh1D::GetNotifyIdxClosToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 notifyNum = threadNum_ - 1 - maxRankNumPerBoard_;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

void InsTempUBXAllToAllVMesh1D::GetNotifyIdxFullMeshToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 notifyNum = threadNum_ - 1;
    for (u32 notifyIdx = maxPathNum_; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

} // namespace ops_hccl
