/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_reduce_concurrent_executor.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_temp_all_reduce_mesh_1D.h"
#include "ccu_temp_all_reduce_mesh_1D_mem2mem.h"
#include "ccu_temp_all_reduce_nhr_mem2mem_1D_multi_jetty.h"
#include "ins_temp_all_reduce_nhr.h"
#include "ins_temp_all_reduce_mesh_1D_two_shot.h"

constexpr u32 CLOS_PORT_NUM = 4;

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2AllReduceConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2AllReduceConcurrentExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitCommInfo(
    const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    algHierarchyInfo_ = algHierarchyInfo;
    
    HCCL_INFO("[%s][InitCommInfo] myRank [%u], rankSize [%u], devType [%u], redOp [%u], "
              "dataType [%u] dataTypeSize [%u]",
              __func__, myRank_, rankSize_, devType_, reduceOp_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(HcclComm comm,
    TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo)); // match mesh+clos 拓扑
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcChannelRequest(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, const std::vector<std::vector<u32>> &subCommRanks,
    std::vector<HcclChannelDesc> &channelDescs, CommTopo topo) const
{
    std::vector<HcclChannelDesc> channelDescsTemp;

    if (topo == CommTopo::COMM_TOPO_1DMESH) {
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks, channelDescsTemp, topo));
    } else if (topo == CommTopo::COMM_TOPO_CLOS) {
        CHK_RET(CalcChannelRequestNHRWithPriorityTopo(comm, param, topoInfo, subCommRanks, channelDescsTemp, topo));
    }

    channelDescs.clear();
    std::copy_if(channelDescsTemp.begin(), channelDescsTemp.end(), std::back_inserter(channelDescs),
                 [](const HcclChannelDesc &c) { return c.channelProtocol == COMM_PROTOCOL_UBC_CTP; });

    CHK_PRT_RET(channelDescs.empty(),
                HCCL_ERROR("[%s] channelDescs.size()[%zu] is zero.", __func__, channelDescs.size()),
                HcclResult::HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
     // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, topoInfo, algHierarchyInfo));

    // SubCommRanks拆分
    std::vector<std::vector<u32>> subCommRanks0{ algHierarchyInfo.infos[0][0] };
    std::vector<std::vector<u32>> subCommRanks1{ algHierarchyInfo.infos[0][1] };
    
    // 构造template
    std::shared_ptr<InsAlgTemplate0> temp0 = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0);
    std::shared_ptr<InsAlgTemplate1> temp1 = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1);

    AlgResourceRequest resReq0;
    AlgResourceRequest resReq1;
    CHK_RET(temp0->CalcRes(comm, param, topoInfo, resReq0));
    CHK_RET(temp1->CalcRes(comm, param, topoInfo, resReq1));

    // 两个template并行执行，需要的资源是两者相加，temp0的主thread负责和temp1主thread同步
    resourceRequest.slaveThreadNum = resReq0.slaveThreadNum + resReq1.slaveThreadNum + 1; // 1: temp1的主thread
    resourceRequest.notifyNumOnMainThread = resReq0.notifyNumOnMainThread + 1;  // 1: 用于两个template间同步
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReq0.notifyNumPerThread.begin(), resReq0.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(resReq1.notifyNumOnMainThread + 1); // 1: 用于两个template间同步
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReq1.notifyNumPerThread.begin(), resReq1.notifyNumPerThread.end());

     // 分别获取两种拓扑的链路，这里约束temp0为mesh拓扑，走mesh算法；temp1为clos拓扑，走nhr算法
    std::vector<HcclChannelDesc> channelDescs0, channelDescs1, channelDescsTemp;
    CHK_RET(CalcChannelRequest(comm, param, topoInfo, subCommRanks0, channelDescs0, CommTopo::COMM_TOPO_1DMESH));
    CHK_RET(CalcChannelRequest(comm, param, topoInfo, subCommRanks1, channelDescs1, CommTopo::COMM_TOPO_CLOS));

    HCCL_INFO("[%s] CalcRes channelDescs0.size()[%zu], channelDescs1.size())[%zu]", __func__, channelDescs0.size(),
              channelDescs1.size());

    // 两者数量应相等
    CHK_PRT_RET(channelDescs0.size() != channelDescs1.size(),
                HCCL_ERROR("[%s] channelDescs0.size()[%zu] is not equal to channelDescs1.size()[%zu]", __func__,
                        channelDescs0.size(), channelDescs1.size()),
                HcclResult::HCCL_E_INTERNAL);

    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        for (auto &kernelInfo : resReq0.ccuKernelInfos) {
            kernelInfo.channels = channelDescs0;
        }
        resourceRequest.ccuKernelNum.insert(resourceRequest.ccuKernelNum.end(), resReq0.ccuKernelNum.begin(),
                                            resReq0.ccuKernelNum.end());
        resourceRequest.ccuKernelNum.insert(resourceRequest.ccuKernelNum.end(), resReq1.ccuKernelNum.begin(),
                                            resReq1.ccuKernelNum.end());
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(), resReq0.ccuKernelInfos.begin(),
                                              resReq0.ccuKernelInfos.end());
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(), resReq1.ccuKernelInfos.begin(),
                                              resReq1.ccuKernelInfos.end());
    } else if (param.engine == CommEngine::COMM_ENGINE_AICPU) {
        // 都放在level0，前面放temp0的channels，后面放temp1的channels，两者数量应相等
        resourceRequest.channels.resize(1);
        resourceRequest.channels[0].insert(resourceRequest.channels[0].end(), channelDescs0.begin(),
                                            channelDescs0.end());
        resourceRequest.channels[0].insert(resourceRequest.channels[0].end(), channelDescs1.begin(),
                                            channelDescs1.end());
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[%s] Start", __func__);
    // 参数填充
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    // 算法展开
    CHK_RET(OrchestrateLoop(param, resCtx));

    HCCL_INFO("[%s] End.", __func__);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[%s] Start", __func__);

    // SubCommRanks拆分
    std::vector<std::vector<u32>> subCommRanks0{ algHierarchyInfo_.infos[0][0] };
    std::vector<std::vector<u32>> subCommRanks1{ algHierarchyInfo_.infos[0][1] };

    // 构造template
    std::shared_ptr<InsAlgTemplate0> temp0 = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0);
    std::shared_ptr<InsAlgTemplate1> temp1 = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1);

    // 获取每个template会用多少的buffer
    const u32 temp0ScratchMultiple = temp0->CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    const u32 temp1ScratchMultiple = temp1->CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    const u32 totalScratchMultiple = temp0ScratchMultiple + temp1ScratchMultiple;

    // 计算数据切分比例，获取端口数量
    const u32 portNum0 = rankSize_ - 1; // mesh端口数为rank size - 1
    const u32 portNum1 = CLOS_PORT_NUM;
    const u64 totalCounts = param.DataDes.count;
    const u64 sliceAlignCount = HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    const u64 totalCount0 = (totalCounts * portNum0) / (portNum0 + portNum1) / sliceAlignCount * sliceAlignCount;
    const u64 totalCount1 = totalCounts - totalCount0;
    const u64 dataOffset = totalCount0 * dataTypeSize_;

    HCCL_INFO("[%s]portNum0[%u], portNum1[%u], totalCounts[%llu], totalCount0[%llu], totalCount1[%llu]", __func__,
              portNum0, portNum1, totalCounts, totalCount0, totalCount1);

    void *cclMemAddr = resCtx.cclMem.addr;
    const u64 cclMemSize = resCtx.cclMem.size;
    const auto cclMemType = resCtx.cclMem.type;
    HcclMem cclMem0 = {cclMemType, cclMemAddr, cclMemSize};
    HcclMem cclMem1 = {cclMemType, cclMemAddr, cclMemSize};

    // ccl buffer 按数据比例和ScratchMultiple比例划分给两个template用
    if (temp0ScratchMultiple > 0 || temp1ScratchMultiple > 0) {
        const u64 bufferRatioTerm0 = portNum0 * temp0ScratchMultiple;
        const u64 bufferRatioTerm1 = portNum1 * temp1ScratchMultiple;
        const double bufferRatio0 = bufferRatioTerm0 / (bufferRatioTerm0 + bufferRatioTerm1);
        cclMem0.size = cclMemSize * bufferRatio0;
        cclMem1.addr = static_cast<void *>(static_cast<s8 *>(cclMemAddr) + cclMem0.size);
        cclMem1.size = cclMemSize - cclMem0.size;
    }

    // 初始的data参数
    TemplateDataParams tempAlgParams0;
    tempAlgParams0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams0.buffInfo.hcclBuff = cclMem0;
    tempAlgParams0.buffInfo.hcclBuffBaseOff = 0;
    tempAlgParams0.buffInfo.inBuffBaseOff = 0;
    tempAlgParams0.buffInfo.outBuffBaseOff = 0;

    TemplateDataParams tempAlgParams1;
    tempAlgParams1.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams1.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams1.buffInfo.hcclBuff = cclMem1;
    tempAlgParams1.buffInfo.hcclBuffBaseOff = 0;
    tempAlgParams1.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParams1.buffInfo.outBuffBaseOff = dataOffset;

    TemplateResource tempAlgResource0;
    TemplateResource tempAlgResource1;

    u64 temp0SlaveThreadNum = 0;
    u64 temp1SlaveThreadNum = 0;

    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        constexpr u32 EXPECTED_CCU_KERNEL_NUM_2 = 2;
        CHK_PRT_RET(resCtx.ccuKernels.size() != EXPECTED_CCU_KERNEL_NUM_2,
                    HCCL_ERROR("[%s] resCtx.ccuKernels.size[%zu] is not %u.", __func__, resCtx.ccuKernels.size(), EXPECTED_CCU_KERNEL_NUM_2),
                    HcclResult::HCCL_E_INTERNAL);
        // CCU模式
        tempAlgResource0.ccuKernels.push_back(resCtx.ccuKernels[0]);
        tempAlgResource1.ccuKernels.push_back(resCtx.ccuKernels[1]);
    } else if (param.engine == CommEngine::COMM_ENGINE_AIV) {
        // AIV模式
        tempAlgResource0.aivCommInfoPtr = resCtx.aivCommInfoPtr;
        tempAlgResource1.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    } else if (param.engine == CommEngine::COMM_ENGINE_AICPU) {
        // AICPU模式
        const auto &channels = resCtx.channels[0];
        const size_t channelCount = channels.size();

        for (u32 i = 0; i < channelCount; ++i) {
            const auto &channel = channels[i];
            auto &targetChannels = (i < rankSize_) ? tempAlgResource0.channels : tempAlgResource1.channels;
            targetChannels[channel.remoteRank].push_back(channel);
        }

        temp0SlaveThreadNum = temp0->GetThreadNum();
        temp1SlaveThreadNum = temp1->GetThreadNum();
    }

    const u64 temp0ThreadsNum = temp0SlaveThreadNum + 1;
    const u64 temp1ThreadsNum = temp1SlaveThreadNum + 1;
    CHK_PRT_RET(threads_.size() < temp0ThreadsNum + temp1ThreadsNum,
        HCCL_ERROR(
            "[%s] threads resource is not enough. threads.size=[%zu], temp0ThreadsNum=[%llu], temp1ThreadsNum=[%llu].",
            __func__, threads_.size(), temp0ThreadsNum, temp1ThreadsNum),
        HcclResult::HCCL_E_INTERNAL);

    // 划分thread
    u64 threadIdx = 0;
    for (auto i = 0; i < temp0ThreadsNum; ++i) {
        tempAlgResource0.threads.push_back(threads_[threadIdx++]);
    }
    for (auto i = 0; i < temp1ThreadsNum; ++i) {
        tempAlgResource1.threads.push_back(threads_[threadIdx++]);
    }

    // 分别计算两个template的maxCountPerLoop
    const u64 maxCountUBLimit = UB_MAX_DATA_SIZE / dataTypeSize_;
    u64 maxCountPerLoop0 = maxCountUBLimit;
    u64 maxCountPerLoop1 = maxCountUBLimit;
    if (temp0ScratchMultiple > 0) {
        maxCountPerLoop0 = std::min(maxCountUBLimit, cclMem0.size / temp0ScratchMultiple / HCCL_MIN_SLICE_ALIGN *
                                                         HCCL_MIN_SLICE_ALIGN / dataTypeSize_);
    }
    if (temp1ScratchMultiple > 0) {
        maxCountPerLoop1 = std::min(maxCountUBLimit, cclMem1.size / temp1ScratchMultiple / HCCL_MIN_SLICE_ALIGN *
                                                         HCCL_MIN_SLICE_ALIGN / dataTypeSize_);
    }

    // template间同步所需信息计算
    ThreadHandle mainThread = tempAlgResource0.threads[0];
    std::vector<ThreadHandle> syncThreads{tempAlgResource1.threads[0]};
    std::vector<u32> notifyIdxesMainToSub{static_cast<u32>(temp1SlaveThreadNum)}; // 统一使用第[slaveThreadNum + 1]个notify做template间同步
    std::vector<u32> notifyIdxesSubToMain{static_cast<u32>(temp0SlaveThreadNum)};

    // Template间前同步
    CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));

    // 交替下发两个template的任务
    u64 countLeft0 = totalCount0;
    u64 countLeft1 = totalCount1;
    
    while (countLeft0 > 0 || countLeft1 > 0) {
        if (countLeft0 > 0) {
            const u64 curCount0 = std::min(countLeft0, maxCountPerLoop0);
            tempAlgParams0.count = curCount0;
            tempAlgParams0.sliceSize = curCount0 * dataTypeSize_;
            
            CHK_RET(temp0->KernelRun(param, tempAlgParams0, tempAlgResource0));
            
            tempAlgParams0.buffInfo.inBuffBaseOff += tempAlgParams0.sliceSize;
            tempAlgParams0.buffInfo.outBuffBaseOff += tempAlgParams0.sliceSize;
            countLeft0 -= curCount0;
            HCCL_DEBUG("curCount0[%llu], countLeft0[%llu]", curCount0, countLeft0);
        }

        if (countLeft1 > 0) {
            const u64 curCount1 = std::min(countLeft1, maxCountPerLoop1);
            tempAlgParams1.count = curCount1;
            tempAlgParams1.sliceSize = curCount1 * dataTypeSize_;
            
            CHK_RET(temp1->KernelRun(param, tempAlgParams1, tempAlgResource1));
            
            tempAlgParams1.buffInfo.inBuffBaseOff += tempAlgParams1.sliceSize;
            tempAlgParams1.buffInfo.outBuffBaseOff += tempAlgParams1.sliceSize;
            countLeft1 -= curCount1;
            HCCL_DEBUG("curCount1[%llu], countLeft1[%llu]", curCount1, countLeft1);
        }
    }

    // Template间尾同步
    CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));

    HCCL_INFO("[%s] End.", __func__);
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE, CcuAllReduceConcurrentSche, InsV2AllReduceConcurrentExecutor, TopoMatchUBX,
    CcuTempAllReduceMeshMem2Mem1D, CcuTempAllReduceNhrMem2Mem1DMultiJetty);
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE, CcuAllReduceConcurrentMs, InsV2AllReduceConcurrentExecutor, TopoMatchUBX,
    CcuTempAllReduceMesh1D, CcuTempAllReduceNhrMem2Mem1DMultiJetty);
#endif
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE, InsAllReduceConcurrent, InsV2AllReduceConcurrentExecutor, TopoMatchUBX,
    InsTempAllReduceMesh1DTwoShot, InsTempAllReduceNHR);
}