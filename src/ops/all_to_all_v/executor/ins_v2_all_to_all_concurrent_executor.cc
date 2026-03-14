/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "alg_data_trans_wrapper.h"
#include "channel.h"
#include "hccl_ccu_res.h"
#include "ins_v2_all_to_all_concurrent_executor.h"
#include "aicpu/ins_temp_all_to_all_v_mesh_1D.h"

#ifndef AICPU_COMPILE
#include "ccu/ccu_temp_all_to_all_mesh1d_multi_jetty.h"
#endif

namespace ops_hccl {
using namespace std;
constexpr uint32_t CONCURRENT_NUM = 2;
constexpr uint32_t CONST_0 = 0;
constexpr uint32_t CONST_1 = 1;
constexpr uint32_t CONST_2 = 2;
constexpr uint32_t CONST_3 = 3;
constexpr uint32_t CONST_4 = 4;

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2AllToAllConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2AllToAllConcurrentExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitCommInfo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // sole_exector只考虑通信域只有一级算法的情况，多级情况需要使用algHierarchyInfo
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ =  SIZE_TABLE[dataType_];
    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO("[InsV2AllToAllConcurrentExecutor][InitCommInfo] myRank = %u, rankSize = %u, devType = %u, "
        "dataType = %u, dataTypeSize = %u", myRank_, rankSize_, devType_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm,
    TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    // 初始化基本成员变量
    u32 topoNum = 2;
    if (algHierarchyInfo.infos[0].size() != topoNum) {
        HCCL_ERROR("[InsV2AllToAllConcurrentExecutor[%s] toposize = %u", __FUNCTION__, algHierarchyInfo.infos[0].size());
        return HCCL_E_PARA;
    }

    // 获取子通信域
    std::vector<std::vector<u32>> subCommRanks0 = {algHierarchyInfo.infos[0][0]};
    std::vector<std::vector<u32>> subCommRanks1 = {algHierarchyInfo.infos[0][1]};

    // 构建template
    std::shared_ptr<InsAlgTemplate0> interTempAlg0 = std::make_shared<InsAlgTemplate0>(param, topoInfo->userRank,
                                                                                       subCommRanks0);
    std::shared_ptr<InsAlgTemplate1> intraTempAlg1 = std::make_shared<InsAlgTemplate1>(param, topoInfo->userRank,
                                                                                       subCommRanks1);

    // 调用计算资源的函数
    AlgResourceRequest resReq0, resReq1;
    interTempAlg0->CalcRes(comm, param, topoInfo, resReq0);
    intraTempAlg1->CalcRes(comm, param, topoInfo, resReq1);
    // temp0的主流负责和temp1主流同步
    resourceRequest.slaveThreadNum = resReq0.slaveThreadNum + resReq1.slaveThreadNum + 1;   // +1用于temp0和temp1主流之间的同步流
    resourceRequest.notifyNumOnMainThread = resReq0.notifyNumOnMainThread + 1;              // +1用于2个template间同步
    resourceRequest.notifyNumPerThread.reserve(resReq0.notifyNumPerThread.size() +
                                               resReq1.notifyNumPerThread.size() + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                            resReq0.notifyNumPerThread.begin(),
                                            resReq0.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(resReq1.notifyNumOnMainThread + 1); // +1用于两个template间同步
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                            resReq1.notifyNumPerThread.begin(),
                                            resReq1.notifyNumPerThread.end());

    std::vector<HcclChannelDesc> channelDescs0, channelDescs1;
    CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks0,
                                                    channelDescs0, CommTopo::COMM_TOPO_1DMESH));
    CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks1,
                                                    channelDescs1, CommTopo::COMM_TOPO_CLOS));

    if ((param.engine == CommEngine::COMM_ENGINE_CCU)) {
        resReq0.ccuKernelInfos[0].channels = channelDescs0;
        resReq1.ccuKernelInfos[0].channels = channelDescs1;
        resourceRequest.ccuKernelNum.push_back(resReq0.ccuKernelNum[0]);
        resourceRequest.ccuKernelNum.push_back(resReq1.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                              resReq0.ccuKernelInfos.begin(),
                                              resReq0.ccuKernelInfos.end());
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                              resReq1.ccuKernelInfos.begin(),
                                              resReq1.ccuKernelInfos.end());
    } else if (param.engine == CommEngine::COMM_ENGINE_AICPU) {
        resourceRequest.channels[0].insert(resourceRequest.channels[0].end(),
                                              channelDescs0.begin(),
                                              channelDescs0.end());
        resourceRequest.channels[0].insert(resourceRequest.channels[0].end(),
                                              channelDescs1.begin(),
                                              channelDescs1.end());
    } else {
        HCCL_ERROR("[InsV2AllToAllConcurrentExecutor][CalcRes] the communication engine is not supported currently"
                    ", please check");
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllToAllConcurrentExecutor][Orchestrate] Orchestrate Start");
    // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    maxTmpMemSize_ = resCtx.cclMem.size;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine == CommEngine::COMM_ENGINE_AICPU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = SIZE_TABLE[dataType_];
    rankSize_ = resCtx.topoInfo.userRankSize;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[InsV2AllToAllConcurrentExecutor][Orchestrate]errNo[0x%016llx] Reduce scatter excutor "
                "kernel run failed", HCCL_ERROR_CODE(ret)),
                ret);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::FillTemplateResource(
    const OpParam &param, const AlgResourceCtxSerializable& resCtx, TemplateResource& templateAlgRes, uint32_t index)
{
    templateAlgRes.threads = {resCtx.threads[index]};
    templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    if (param.engine == COMM_ENGINE_CCU) {
        templateAlgRes.ccuKernels = {resCtx.ccuKernels[index]};
    }
    if (param.engine == COMM_ENGINE_AICPU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
        templateAlgRes.channels = remoteRankToChannelInfo_[index];
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitTemplateDataParams(
    const OpParam &param, const AlgResourceCtxSerializable& resCtx, TemplateDataParams& tempAlgParams) const
{
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.sendCounts.resize(rankSize_, 0);
    tempAlgParams.recvCounts.resize(rankSize_, 0);
    tempAlgParams.sdispls.resize(rankSize_, 0);
    tempAlgParams.rdispls.resize(rankSize_, 0);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::RestoreSendRecvData(
    const OpParam &param)
{
    // 从varData把值取出来
    const u64* data = reinterpret_cast<const u64*>(param.varData);
    for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * rankSize_; i++) {
        HCCL_INFO("OrchestrateLoop, param.varData[%u] is [%u]", i, data[i]);
    }
    for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * rankSize_; i++) {
        u64 val = i / rankSize_;
        switch(val) {
            case CONST_0:
                sendCounts_.push_back(data[i]);
                break;
            case CONST_1:
                recvCounts_.push_back(data[i]);
                break;
            case CONST_2:
                sdispls_.push_back(data[i]);
                break;
            case CONST_3:
                rdispls_.push_back(data[i]);
                break;
            default:
                break;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::SplitSendRecvData(
    std::vector<SendRecvData>& splitData)
{
    splitData.resize(CONCURRENT_NUM);
    for (u32 i = 0; i < CONCURRENT_NUM; i++) {
        splitData[i].sendCounts.resize(rankSize_);
        splitData[i].recvCounts.resize(rankSize_);
        splitData[i].sdispls.resize(rankSize_);
        splitData[i].rdispls.resize(rankSize_);
    }

    // 按topo切分数据：0为topo 0，1为topo 1
    uint32_t factorMesh = rankSize_ - 1;
    uint32_t factorClos = CONST_4;                // 端口数获取
    uint32_t factor = factorMesh + factorClos;
    for (u64 i = 0; i < rankSize_; i++) {
        uint64_t sendQuotient = sendCounts_[i] / factor;
        uint64_t recvQuotient = recvCounts_[i] / factor;
        uint64_t sendCountsMesh = sendQuotient * factorMesh;
        uint64_t recvCountsMesh = recvQuotient * factorMesh;
        splitData[0].sendCounts[i] = sendCountsMesh;                        // mesh拓扑处理的数据量
        splitData[1].sendCounts[i] = sendCounts_[i] - sendCountsMesh;       // clos拓扑处理的数据量
        splitData[0].recvCounts[i] = recvCountsMesh;
        splitData[1].recvCounts[i] = recvCounts_[i] - recvCountsMesh;
        splitData[0].sdispls[i] = sdispls_[i];
        splitData[1].sdispls[i] = sdispls_[i] + sendCountsMesh;
        splitData[0].rdispls[i] = rdispls_[i];
        splitData[1].rdispls[i] = rdispls_[i] + recvCountsMesh;
    }

    for (u32 i = 0; i < CONCURRENT_NUM; i++) {
        for (u32 rankId = 0; rankId < rankSize_; rankId++) {
            HCCL_INFO("OrchestrateLoop, rank=%u, sendCounts[%u] is [%u]", rankId, i, splitData[i].sendCounts[rankId]);
            HCCL_INFO("OrchestrateLoop, rank=%u, recvCounts[%u] is [%u]", rankId, i, splitData[i].recvCounts[rankId]);
            HCCL_INFO("OrchestrateLoop, rank=%u, sdispls[%u] is [%u]", rankId, i, splitData[i].sdispls[rankId]);
            HCCL_INFO("OrchestrateLoop, rank=%u, rdispls[%u] is [%u]", rankId, i, splitData[i].rdispls[rankId]);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetMaxSendRecvDataCount(
    u64& maxSendRecvDataCount, const SendRecvData& splitData) const
{
    u64 max = 0;
    for (u64 i = 0; i < rankSize_; i++) {
        max = std::max(max, splitData.sendCounts[i]);
        max = std::max(max, splitData.recvCounts[i]);
    }

    HCCL_INFO("[InsV2AllToAllConcurrentExecutor] maxSendRecvDataCount = %u", max);
    maxSendRecvDataCount = max;
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllToAllConcurrentExecutor][OrchestrateLoop] Start");
    HCCL_INFO("[OrchestrateLoop] ALL_TO_ALL_V_VECTOR_NUM is [%u]", ALL_TO_ALL_V_VECTOR_NUM);
    HCCL_INFO("[OrchestrateLoop] rankSize_ is [%u]", rankSize_);
    HCCL_INFO("[OrchestrateLoop] param.varMemSize is [%u]", param.varMemSize);
    CHK_PRT_RET(param.varMemSize != ALL_TO_ALL_V_VECTOR_NUM * rankSize_ * sizeof(u64),
                HCCL_ERROR("[InsV2AllToAllConcurrentExecutor][OrchestrateLoop] param.varMemSize [%llu] is invalid",
                            param.varMemSize),
                HCCL_E_PARA);

    // 获取子通信域
    std::vector<std::vector<u32>> subCommRanks0 = {algHierarchyInfo_.infos[0][0]};
    std::vector<std::vector<u32>> subCommRanks1 = {algHierarchyInfo_.infos[0][1]};
    TemplateResource templateAlgRes0, templateAlgRes1;
    FillTemplateResource(param, resCtx, templateAlgRes0, 0);
    FillTemplateResource(param, resCtx, templateAlgRes1, 1);

    // 获取SendRecv数据并切分到各template上
    std::vector<SendRecvData> splitData;
    RestoreSendRecvData(param);
    SplitSendRecvData(splitData);

    u64 maxSendOrRecvDataCount0, maxSendOrRecvDataCount1;
    GetMaxSendRecvDataCount(maxSendOrRecvDataCount0, splitData[0]);
    GetMaxSendRecvDataCount(maxSendOrRecvDataCount1, splitData[1]);

    // 构建template
    std::shared_ptr<InsAlgTemplate0> algTemplate0 =
        std::make_shared<InsAlgTemplate0>(param, resCtx.topoInfo.userRank, subCommRanks0);
    std::shared_ptr<InsAlgTemplate1> algTemplate1 =
        std::make_shared<InsAlgTemplate1>(param, resCtx.topoInfo.userRank, subCommRanks1);

    TemplateDataParams tempAlgParams0, tempAlgParams1;
    InitTemplateDataParams(param, resCtx, tempAlgParams0);
    InitTemplateDataParams(param, resCtx, tempAlgParams1);

    std::vector<u64> scratchMulti;
    scratchMulti.push_back(algTemplate0->CalcScratchMultiple(tempAlgParams0.buffInfo.inBuffType,
                                                            tempAlgParams0.buffInfo.outBuffType));
    scratchMulti.push_back(algTemplate1->CalcScratchMultiple(tempAlgParams1.buffInfo.inBuffType,
                                                            tempAlgParams1.buffInfo.outBuffType));
    std::vector<u64> maxDataCountPerLoop(CONCURRENT_NUM, 1);
    CalcMaxDataCountPerLoop(param, scratchMulti, maxDataCountPerLoop);

    // alltoallv的时候，loopTimes可能是0
    u64 loopTimes0 = (maxSendOrRecvDataCount0 + maxDataCountPerLoop[0] - 1) / maxDataCountPerLoop[0];
    u64 loopTimes1 = (maxSendOrRecvDataCount1 + maxDataCountPerLoop[1] - 1) / maxDataCountPerLoop[1];
    HCCL_INFO("[OrchestrateLoop] loopTimes0 = %llu, loopTimes1 = %llu.", loopTimes0, loopTimes1);

    std::vector<u64> processedDataCount = {0, 0};
    std::vector<u32> notify = {0};
    u64 loop = 0;
    CHK_RET(PreSyncInterThreads(resCtx.threads[0], {resCtx.threads[1]}, notify));
    while (loop < loopTimes0 || loop < loopTimes1) {
        if (loop < loopTimes0) {
            u64 currDataCount = (loop == loopTimes0 - 1) ?
                                (maxSendOrRecvDataCount0 - processedDataCount[0]) : (maxDataCountPerLoop[0]);
            SetTemplateDataParams(tempAlgParams0, splitData[0], loop, currDataCount, processedDataCount[0],
                                  maxDataCountPerLoop[0]);
            CHK_RET(algTemplate0->KernelRun(param, tempAlgParams0, templateAlgRes0));
            processedDataCount[0] += currDataCount;
        }

        if (loop < loopTimes1) {
            u64 currDataCount = (loop == loopTimes1 - 1) ?
                                (maxSendOrRecvDataCount1 - processedDataCount[1]) : (maxDataCountPerLoop[1]);
            SetTemplateDataParams(tempAlgParams1, splitData[1], loop, currDataCount, processedDataCount[1],
                                  maxDataCountPerLoop[1]);
            CHK_RET(algTemplate1->KernelRun(param, tempAlgParams1, templateAlgRes1));
            processedDataCount[1] += currDataCount;
        }
        loop++;
    }
    CHK_RET(PostSyncInterThreads(resCtx.threads[0], {resCtx.threads[1]}, notify));

    HCCL_INFO("[InsV2AllToAllConcurrentExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcMaxDataCountPerLoop(
    const OpParam &param, const std::vector<u64> scratchMulti, std::vector<u64>& maxDataCountPerLoop) const
{
    // 计算最小传输大小
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 maxDataSizePerLoop;
    u64 scratchMultiSum = 0;
    for (const auto& x : scratchMulti) {
        scratchMultiSum += x;
    }

    for (u32 i = 0; i < scratchMulti.size(); i++) {
        if ((scratchMulti[i] != 0) && (scratchMultiSum != 0)) {
            HCCL_INFO("[InsV2AllToAllConcurrentExecutor]maxTmpMemSize_ = %lu", maxTmpMemSize_);
            u64 scratchBoundDataSize = maxTmpMemSize_ / scratchMultiSum / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
            maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
        } else {
            maxDataSizePerLoop = transportBoundDataSize;
        }

        // 单次循环处理的数据量大小
        // 使用ccu的时候不需要除以rank_size
        if (param.engine == CommEngine::COMM_ENGINE_AICPU) {
            maxDataCountPerLoop[i] = maxDataSizePerLoop / dataTypeSize_ / rankSize_; // 发往单卡的数据量  使用scratch buffer
        } else {
            maxDataCountPerLoop[i] = maxDataSizePerLoop / dataTypeSize_;
        }

        HCCL_INFO(
            "[InsV2AllToAllConcurrentExecutor][CalcMaxDataCountPerLoop] maxDataCountPerLoop = %llu, "
            "maxDataSizePerLoop = %llu, transportBoundDataSize = %llu, scratchMulti = %llu",
            maxDataCountPerLoop[i], maxDataSizePerLoop, transportBoundDataSize, scratchMulti[i]);
        CHK_PRT_RET(maxDataCountPerLoop[i] == 0,
                    HCCL_ERROR("[InsV2AllToAllConcurrentExecutor][CalcMaxDataCountPerLoop] maxDataCountPerLoop[%u] is 0"
                               ".", i),
                    HCCL_E_INTERNAL);
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllToAllConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::SetTemplateDataParams(
    TemplateDataParams &tempAlgParams, const SendRecvData &splitData,
    u32 loop, u64 currDataCount, u64 processedDataCount, u64 maxDataCountPerLoop) const
{
    tempAlgParams.count = currDataCount;
    tempAlgParams.buffInfo.inBuffBaseOff = (splitData.sdispls[0] + processedDataCount) * dataTypeSize_;
    tempAlgParams.buffInfo.outBuffBaseOff = (splitData.rdispls[0] + processedDataCount) * dataTypeSize_;
    tempAlgParams.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParams.sliceSize = currDataCount * dataTypeSize_; // 这是每次循环处理的数据大小
    tempAlgParams.tailSize = tempAlgParams.sliceSize;
    // 这里的stride当成传统意义上的sreide 间隔
    tempAlgParams.inputSliceStride = 0; // 变长算子不涉及,这里是每一块数据的大小，这个值被sendCounts代替了
    tempAlgParams.outputSliceStride = maxDataCountPerLoop * dataTypeSize_; // 这里用来放每张卡可以用的cclBuffer的大小，数据从ureIn到cclBuffer的时候，以这个量来分隔

    for (u64 i = 0; i < rankSize_; i++) {
        if (splitData.sendCounts[i] > processedDataCount) {
            tempAlgParams.sendCounts[i] = std::min(currDataCount, splitData.sendCounts[i] - processedDataCount);
            tempAlgParams.sdispls[i] = splitData.sdispls[i] + processedDataCount;
        } else {
            tempAlgParams.sendCounts[i] = 0;
            tempAlgParams.sdispls[i] = splitData.sdispls[i] + splitData.sendCounts[i];
        }

        if (splitData.recvCounts[i] > processedDataCount) {
            tempAlgParams.recvCounts[i] = std::min(currDataCount, splitData.recvCounts[i] - processedDataCount);
            tempAlgParams.rdispls[i] = splitData.rdispls[i] + processedDataCount;
        } else {
            tempAlgParams.recvCounts[i] = 0;
            tempAlgParams.rdispls[i] = splitData.rdispls[i] + splitData.recvCounts[i];
        }
        HCCL_DEBUG("[%s]loop = %u, rank = %u, tempAlgParams.sendCounts = %u, tempAlgParams.sdispls = %u, "
                   "tempAlgParams.recvCounts = %u, tempAlgParams.rdispls = %u",
                   __FUNCTION__, loop, i, tempAlgParams.sendCounts[i], tempAlgParams.sdispls[i],
                   tempAlgParams.recvCounts[i], tempAlgParams.rdispls[i]);
    }

    HCCL_INFO("[InsV2AllToAllConcurrentExecutor] loop = %u, tempAlgParams.buffInfo.inBuffBaseOff = %u,"
        "tempAlgParams.buffInfo.outBuffBaseOff = %u, tempAlgParams.inputSliceStride = %u,"
        "tempAlgParams.outputSliceStride = %u, tempAlgParams.sliceSize = %u",
        loop, tempAlgParams.inputSliceStride, tempAlgParams.outputSliceStride, tempAlgParams.sliceSize,
        tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.outBuffBaseOff);

    // 不需要重复
    tempAlgParams.repeatNum = 1;
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;
    return HCCL_SUCCESS;
}

// 第1个模板走mesh拓扑
// 第2个模板走clos拓扑
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLTOALL,
                                InsAllToAllMesh1DConcurrent,
                                InsV2AllToAllConcurrentExecutor,
                                TopoMatchUBX,
                                InsTempAlltoAllVMesh1D,
                                InsTempAlltoAllVMesh1D);
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLTOALLV,
                                InsAllToAllVMesh1DConcurrent,
                                InsV2AllToAllConcurrentExecutor,
                                TopoMatchUBX,
                                InsTempAlltoAllVMesh1D,
                                InsTempAlltoAllVMesh1D);
#ifndef AICPU_COMPILE
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLTOALL,
                                CcuAllToAllMesh1DConcurrent,
                                InsV2AllToAllConcurrentExecutor,
                                TopoMatchUBX,
                                CcuTempAllToAllMesh1dMultiJetty,
                                CcuTempAllToAllMesh1dMultiJetty);
#endif

}