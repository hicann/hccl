/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_reduce_sequence_executor.h"
#include "ins_temp_reduce_scatter_mesh_1D_intra.h"
#include "ins_temp_reduce_scatter_mesh_1D_dpu_inter.h"
#include "ins_temp_all_gather_nhr_dpu_inter.h"
#include "ins_temp_all_gather_mesh_1D_intra.h"
#include "alg_attrs_registry.h"
#include "auto_selector_base.h"

namespace ops_hccl {
// ! 已经编码完成
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
std::vector<CostModelParam>
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    CalcCostCoeff(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)algName;
    (void)comm;
    AlgHierarchyInfoForAllLevel algHierarchyInfo; // TODO: unused for now, costmodel fallback
    (void)algHierarchyInfo;
    // TODO: CalcAlgHierarchyInfo(comm, topoInfo, algHierarchyInfo);
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = true;
    auto rs = CostModelManager::Global()->CalcRankSizeByTopo(topoInfo);
    u32 rankSizeLevel0 = rs.level0;
    u32 rankSizeLevel1 = rs.level1;
    // TODO: CommTopo netTypeLevel0 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[0]);
    CommTopo netTypeLevel0 = CommTopo::COMM_TOPO_1DMESH;
    // TODO: CommTopo netTypeLevel1 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[1]);
    CommTopo netTypeLevel1 = CommTopo::COMM_TOPO_CLOS;
    // TODO: std::vector<u32> portNumLevel0 = GetPortNumLevel(topoInfo, algHierarchyInfo.index[0]);
    std::vector<u32> portNumLevel0 = {1};
    // TODO: std::vector<u32> portNumLevel1 = GetPortNumLevel(topoInfo, algHierarchyInfo.index[1]);
    std::vector<u32> portNumLevel1 = {8};
    HCCL_INFO(
        "[CalcCostCoeff] rankSize=%d, rankSizeLevel0=%d, rankSizeLevel1=%d, portNumLevel0=%d, portNumLevel1=%d, "
        "netTypeLevel0=%d, netTypeLevel1=%d",
        rankSize, rankSizeLevel0, rankSizeLevel1, portNumLevel0, portNumLevel1, static_cast<int>(netTypeLevel0),
        static_cast<int>(netTypeLevel1));
    std::vector<CostModelParam> params = [rankSize, rankSizeLevel0, rankSizeLevel1, portNumLevel0, portNumLevel1,
                                          netTypeLevel0, netTypeLevel1, isPod] {
        std::vector<CostModelParam> v;
        auto p0 = InsAlgTemplate0::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel0, 1.0f / rankSize, netTypeLevel0, BufferType::INPUT, BufferType::HCCL_BUFFER,
            BufferType::HCCL_BUFFER, portNumLevel0, isPod});
        v.insert(v.end(), p0.begin(), p0.end());
        auto p1 = InsAlgTemplate1::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel1, 1.0f / rankSize, netTypeLevel1, BufferType::INPUT, BufferType::HCCL_BUFFER,
            BufferType::HCCL_BUFFER, portNumLevel1, isPod});
        v.insert(v.end(), p1.begin(), p1.end());
        auto p2 = InsAlgTemplate2::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel1, 1.0f / rankSize, netTypeLevel1, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
            BufferType::HCCL_BUFFER, portNumLevel1, isPod});
        v.insert(v.end(), p2.begin(), p2.end());
        auto p3 = InsAlgTemplate3::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel0, 1.0f / rankSize, netTypeLevel0, BufferType::HCCL_BUFFER, BufferType::OUTPUT,
            BufferType::HCCL_BUFFER, portNumLevel0, isPod});
        v.insert(v.end(), p3.begin(), p3.end());
        return v;
    }();
    return params;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
AlgNetMeta
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const
{
    (void)param;
    auto rs = CostModelManager::Global()->CalcRankSizeByTopo(topoInfo);
    u32 rankSizeLevel0 = rs.level0;
    u32 rankSizeLevel1 = rs.level1;
    u32 rankSize = topoInfo->userRankSize;
    // TODO: CommTopo netTypeLevel0 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[0]);
    CommTopo netTypeLevel0 = CommTopo::COMM_TOPO_1DMESH;
    // TODO: CommTopo netTypeLevel1 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[1]);
    CommTopo netTypeLevel1 = CommTopo::COMM_TOPO_CLOS;
    AlgNetMeta meta;
    meta.netTypes.push_back(netTypeLevel0);
    meta.netTypes.push_back(netTypeLevel1);
    meta.netTypes.push_back(netTypeLevel1);
    meta.netTypes.push_back(netTypeLevel0);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1, 1, 1, 1};
    meta.dataRatios = {1.0f / rankSizeLevel0, 1.0f / rankSize, 1.0f / rankSize, 1.0f / rankSizeLevel0};
    meta.rankSizes = {rankSizeLevel0, rankSizeLevel1, rankSizeLevel1, rankSizeLevel0};
    return meta;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
InsV2AllReduceSequenceExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::InsV2AllReduceSequenceExecutor()
{}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    InitCommInfo(
        const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutor][InitCommInfo] myRank [%u], rankSize [%u], devType [%u], redOp [%u], "
        "dataType [%u] dataTypeSize [%u]",
        myRank_, rankSize_, devType_, reduceOp_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

// 实例化实际执行以来AutoMatchMeshNhr这个类的实现
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    InitCommInfo(param, topoInfo, algHierarchyInfo);

    if (algHierarchyInfo.infos.empty() || algHierarchyInfo.infos.size() < TOPO_LEVEL_NUM_2) {
        HCCL_ERROR("[%s] algHierarchyInfo.infos is invalid (empty or size < 2).", __func__);
        return HCCL_E_PARA;
    }
    rankSizeLevel0_ = algHierarchyInfo.infos[0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[1].size();

    std::shared_ptr<InsAlgTemplate0> reduceScatterMesh1DTempAlg
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> reduceScatterMesh1dDpuTempAlg
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate2> allGatherNhrDPUTempAlg
        = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate3> allGatherMesh1DTempAlg
        = std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo.infos[0]);

    AlgResourceRequest resReqReduceScatterMesh1D;
    AlgResourceRequest resReqReduceScatterMesh1dDpu;
    AlgResourceRequest resReqAllGatherNhrDPU;
    AlgResourceRequest resReqAllGatherMesh1D;
    CHK_RET(reduceScatterMesh1DTempAlg->CalcRes(comm, param, topoInfo, resReqReduceScatterMesh1D));
    CHK_RET(reduceScatterMesh1dDpuTempAlg->CalcRes(comm, param, topoInfo, resReqReduceScatterMesh1dDpu));
    CHK_RET(allGatherNhrDPUTempAlg->CalcRes(comm, param, topoInfo, resReqAllGatherNhrDPU));
    CHK_RET(allGatherMesh1DTempAlg->CalcRes(comm, param, topoInfo, resReqAllGatherMesh1D));

    // step1、2、3、4为串行，因此slaveThread和对应notify可以复用
    resourceRequest.slaveThreadNum
        = std::max(resReqReduceScatterMesh1D.slaveThreadNum, resReqReduceScatterMesh1dDpu.slaveThreadNum);
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.resize(resourceRequest.slaveThreadNum);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        if (i < resReqReduceScatterMesh1D.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReqReduceScatterMesh1D.notifyNumPerThread[i]);
        }
        if (i < resReqReduceScatterMesh1dDpu.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReqReduceScatterMesh1dDpu.notifyNumPerThread[i]);
        }
    }
    resourceRequest.notifyNumOnMainThread
        = std::max(resReqReduceScatterMesh1D.notifyNumOnMainThread, resReqReduceScatterMesh1dDpu.notifyNumOnMainThread);

    resourceRequest.channels = {resReqReduceScatterMesh1D.channels[0], resReqReduceScatterMesh1dDpu.channels[0]};
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceSequenceExecutor][Orchestrate] Orchestrate Start");
    // 参数填充
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    if (algHierarchyInfo_.infos.empty() || algHierarchyInfo_.infos.size() < TOPO_LEVEL_NUM_2
        || algHierarchyInfo_.infos[0][0].empty() || algHierarchyInfo_.infos[1][0].empty()) {
        HCCL_ERROR("[%s] invalid algHierarchyInfo infos.", __func__);
        return HCCL_E_PARA;
    }
    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = myRank_ / algHierarchyInfo_.infos[0][0].size();

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2AllReduceSequenceExecutor][Orchestrate]errNo[0x%016llx] AllReduce executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceSequenceExecutor][OrchestrateLoop] Start");
    // 将ccl-buffer分成ccl-in和ccl-out 2部分区分使用
    void* cclInAddr = resCtx.cclMem.addr;
    HcclMem cclInMem = {resCtx.cclMem.type, cclInAddr, resCtx.cclMem.size / 2};
    void* cclOutAddr = static_cast<void*>(static_cast<s8*>(resCtx.cclMem.addr) + resCtx.cclMem.size / 2);
    HcclMem cclOutMem = {resCtx.cclMem.type, cclOutAddr, resCtx.cclMem.size / 2};

    // 声明框内ReduceScatterMesh1D的templateargs
    TemplateDataParams tempAlgParamsStepOne;
    tempAlgParamsStepOne.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsStepOne.buffInfo.outputPtr = cclOutMem.addr;
    tempAlgParamsStepOne.buffInfo.hcclBuff = cclInMem;
    tempAlgParamsStepOne.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsStepOne.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepOne.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建框内ReduceScatterMesh1D的template
    std::shared_ptr<InsAlgTemplate0> algTemplateStepOne
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);

    // 声明框间ReduceScatterMesh1dDpu的templateargs
    TemplateDataParams tempAlgParamsStepTwo;
    tempAlgParamsStepTwo.buffInfo.inputPtr = cclOutMem.addr;
    tempAlgParamsStepTwo.buffInfo.outputPtr = cclOutMem.addr;
    tempAlgParamsStepTwo.buffInfo.hcclBuff = cclInMem;
    tempAlgParamsStepTwo.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepTwo.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepTwo.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建框间ReduceScatterMesh1dDpu的template
    std::shared_ptr<InsAlgTemplate1> algTemplateStepTwo
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);

    // 声明框间AllGatherNhr的templateargs，ccl-out搬运到ccl-out
    TemplateDataParams tempAlgParamsStepThree;
    tempAlgParamsStepThree.buffInfo.inputPtr = cclOutMem.addr;
    tempAlgParamsStepThree.buffInfo.outputPtr = cclOutMem.addr;
    tempAlgParamsStepThree.buffInfo.hcclBuff = cclInMem;
    tempAlgParamsStepThree.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepThree.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepThree.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建框间AllGatherNhrDPU的template
    std::shared_ptr<InsAlgTemplate2> algTemplateStepThree
        = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[1]);

    // 声明框内AllGatherMesh1D的templateargs，ccl-out搬运到user-out
    TemplateDataParams tempAlgParamsStepFour;
    tempAlgParamsStepFour.buffInfo.inputPtr = cclOutMem.addr;
    tempAlgParamsStepFour.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsStepFour.buffInfo.hcclBuff = cclInMem;
    tempAlgParamsStepFour.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsStepFour.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsStepFour.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建框内AllGatherMesh1D的template
    std::shared_ptr<InsAlgTemplate3> algTemplateStepFour
        = std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo_.infos[0]);

    // 构造框内ReduceScatterMesh1D的template资源
    TemplateResource templateResourceReduceScatterMesh1D;
    templateResourceReduceScatterMesh1D.channels = remoteRankToChannelInfo_[0];
    templateResourceReduceScatterMesh1D.threads = resCtx.threads;
    templateResourceReduceScatterMesh1D.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceReduceScatterMesh1D.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    // 构造框间ReduceScatterMesh1dDpu的template资源
    TemplateResource templateResourceReduceScatterMesh1dDpu;
    templateResourceReduceScatterMesh1dDpu.channels = remoteRankToChannelInfo_[1];
    templateResourceReduceScatterMesh1dDpu.threads = resCtx.threads;
    templateResourceReduceScatterMesh1dDpu.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceReduceScatterMesh1dDpu.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    // 构造框间AllGatherNhrDPU的template资源
    TemplateResource templateResourceAllGatherNhrDPU;
    templateResourceAllGatherNhrDPU.channels = remoteRankToChannelInfo_[1];
    templateResourceAllGatherNhrDPU.threads = resCtx.threads;
    templateResourceAllGatherNhrDPU.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceAllGatherNhrDPU.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    // 构造框内AllGatherMesh1D的template资源
    TemplateResource templateResourceAllGatherMesh1D;
    templateResourceAllGatherMesh1D.channels = remoteRankToChannelInfo_[0];
    templateResourceAllGatherMesh1D.threads = resCtx.threads;
    templateResourceAllGatherMesh1D.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceAllGatherMesh1D.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;

    // 中转内存单次最多能够接受的output count，注意是count不是size
    u64 maxCountPerLoop = tempAlgParamsStepOne.buffInfo.hcclBuff.size / 2 / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN
                          / dataTypeSize_; // 这边看前面有/10*10，不知道要不要加上
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount :
                                                      maxCountPerLoop; // 判断是最后一轮，就处理尾块长度

        // ----------- 框内ReduceScatter数据搬运 -----------
        // 框内的数据偏移和搬运计算
        tempAlgParamsStepOne.count = currDataCount;
        tempAlgParamsStepOne.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamsStepOne.buffInfo.outBuffBaseOff = 0;  // CCL-OUT
        tempAlgParamsStepOne.buffInfo.hcclBuffBaseOff = 0; // CCL-IN
        CHK_RET(
            SplitData(currDataCount, rankSizeLevel0_, tempAlgParamsStepOne)); // 计算每个卡对应位置的offset,count,size
        CHK_PRT_RET(
            tempAlgParamsStepOne.allRankSliceSize.size() != rankSizeLevel0_,
            HCCL_ERROR(
                "[InsV2AllReduceSequenceExecutor][tempAlgParamsStepOne] slice num[%u] is not equal to rank size[%u].",
                tempAlgParamsStepOne.allRankSliceSize.size(), rankSizeLevel0_),
            HcclResult::HCCL_E_INTERNAL);
        tempAlgParamsStepOne.sliceSize = 0; // 没用到，template里面用SplitData算了
        tempAlgParamsStepOne.tailSize = 0;  // 没用到
        // 这里的stride当成传统意义上的stride间隔
        tempAlgParamsStepOne.inputSliceStride = 0;  // 没用到
        tempAlgParamsStepOne.outputSliceStride = 0; // 没用到

        HCCL_INFO(
            "[InsV2AllReduceSequenceExecutor] loop [%u] tempAlgParamsStepOne.inputSliceStride [%u],"
            "tempAlgParamsStepOne.outputSliceStride [%u] tempAlgParamsStepOne.sliceSize [%u]",
            loop, tempAlgParamsStepOne.inputSliceStride, tempAlgParamsStepOne.outputSliceStride,
            tempAlgParamsStepOne.sliceSize);
        HCCL_INFO(
            "[InsV2AllReduceSequenceExecutor] loop [%u] tempAlgParamsStepOne.buffInfo.inBuffBaseOff [%u],"
            "tempAlgParamsStepOne.buffInfo.outBuffBaseOff [%u]",
            loop, tempAlgParamsStepOne.buffInfo.inBuffBaseOff, tempAlgParamsStepOne.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsStepOne.repeatNum = 1;
        tempAlgParamsStepOne.inputRepeatStride = 0;
        tempAlgParamsStepOne.outputRepeatStride = 0;
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplateStepOne->KernelRun(param, tempAlgParamsStepOne, templateResourceReduceScatterMesh1D));

        // ----------- 框间ReduceScatterMesh1dDpu数据搬运 -----------
        // 框间的数据偏移和搬运量计算
        tempAlgParamsStepTwo.count = tempAlgParamsStepOne.allRankProcessedDataCount.at(rankIdxLevel0_);
        if (tempAlgParamsStepTwo.count != 0) { // 如果卡里没有数据，不参与框间，让该框有数据的其他卡参与跨框
            tempAlgParamsStepTwo.buffInfo.inBuffBaseOff = 0;   // ccl-out偏移量，每次更新，所以是0
            tempAlgParamsStepTwo.buffInfo.outBuffBaseOff = 0;  // ccl-out偏移量，每次更新，所以是0
            tempAlgParamsStepTwo.buffInfo.hcclBuffBaseOff = 0; // ccl-in
            tempAlgParamsStepTwo.sliceSize = 0;
            tempAlgParamsStepTwo.tailSize = 0;
            CHK_RET(SplitData(tempAlgParamsStepTwo.count, rankSizeLevel1_, tempAlgParamsStepTwo));
            CHK_PRT_RET(
                tempAlgParamsStepTwo.allRankSliceSize.size() != rankSizeLevel1_,
                HCCL_ERROR(
                    "[InsV2AllReduceSequenceExecutor][tempAlgParamsStepTwo] slice num[%u] is not "
                    "equal to rank size[%u].",
                    tempAlgParamsStepTwo.allRankSliceSize.size(), rankSizeLevel1_),
                HcclResult::HCCL_E_INTERNAL);
            // 这里的stride当成传统意义上的stride 间隔
            tempAlgParamsStepTwo.inputSliceStride = 0;  // 没用到
            tempAlgParamsStepTwo.outputSliceStride = 0; // 没用到
            //
            HCCL_INFO(
                "[InsV2AllReduceSequenceExecutor] loop [%u] tempAlgParamsStepTwo.inputSliceStride [%u],"
                "tempAlgParamsStepTwo.outputSliceStride [%u] "
                "tempAlgParamsStepTwo.sliceSize [%u]",
                loop, tempAlgParamsStepTwo.inputSliceStride, tempAlgParamsStepTwo.outputSliceStride,
                tempAlgParamsStepTwo.sliceSize);
            HCCL_INFO(
                "[InsV2AllReduceSequenceExecutor] loop [%u] "
                "tempAlgParamsStepTwo.buffInfo.inBuffBaseOff [%u],"
                "tempAlgParamsStepTwo.buffInfo.outBuffBaseOff [%u]",
                loop, tempAlgParamsStepTwo.buffInfo.inBuffBaseOff, tempAlgParamsStepTwo.buffInfo.outBuffBaseOff);

            tempAlgParamsStepTwo.repeatNum = 1;
            tempAlgParamsStepTwo.inputRepeatStride = 0;
            tempAlgParamsStepTwo.outputRepeatStride = 0;

            CHK_RET(algTemplateStepTwo->KernelRun(param, tempAlgParamsStepTwo, templateResourceReduceScatterMesh1dDpu));
        }

        // ----------- 框间AllGatherNhr数据搬运 -----------
        // 框间的数据偏移和搬运量计算
        tempAlgParamsStepThree.count = tempAlgParamsStepOne.allRankProcessedDataCount.at(rankIdxLevel0_);
        if (tempAlgParamsStepTwo.count != 0) {
            tempAlgParamsStepThree.buffInfo.inBuffBaseOff = 0;   // ccl-out偏移量，每次更新，所以是0
            tempAlgParamsStepThree.buffInfo.outBuffBaseOff = 0;  // ccl-out偏移量，每次更新，所以是0
            tempAlgParamsStepThree.buffInfo.hcclBuffBaseOff = 0; // ccl-in
            tempAlgParamsStepThree.allRankDispls = tempAlgParamsStepTwo.allRankDispls;
            tempAlgParamsStepThree.allRankSliceSize = tempAlgParamsStepTwo.allRankSliceSize;
            tempAlgParamsStepThree.allRankProcessedDataCount = tempAlgParamsStepTwo.allRankProcessedDataCount;

            tempAlgParamsStepThree.sliceSize = 0;
            tempAlgParamsStepThree.tailSize = 0;
            // 这里的stride当成传统意义上的stride 间隔
            tempAlgParamsStepThree.inputSliceStride = 0;
            tempAlgParamsStepThree.outputSliceStride = 0;

            HCCL_INFO(
                "[InsV2AllReduceSequenceExecutor] loop [%u] tempAlgParamsStepThree.inputSliceStride [%u],"
                "tempAlgParamsStepThree.outputSliceStride [%u] tempAlgParamsStepThree.sliceSize [%u]",
                loop, tempAlgParamsStepThree.inputSliceStride, tempAlgParamsStepThree.outputSliceStride,
                tempAlgParamsStepThree.sliceSize);
            HCCL_INFO(
                "[InsV2AllReduceSequenceExecutor] loop [%u] tempAlgParamsStepThree.buffInfo.inBuffBaseOff [%u],"
                "tempAlgParamsStepThree.buffInfo.outBuffBaseOff [%u]",
                loop, tempAlgParamsStepThree.buffInfo.inBuffBaseOff, tempAlgParamsStepThree.buffInfo.outBuffBaseOff);

            tempAlgParamsStepThree.repeatNum = 1;
            tempAlgParamsStepThree.inputRepeatStride = 0;
            tempAlgParamsStepThree.outputRepeatStride = 0;

            CHK_RET(algTemplateStepThree->KernelRun(param, tempAlgParamsStepThree, templateResourceAllGatherNhrDPU));
        }

        // ----------- 框内AllGatherMesh数据搬运 -----------
        // 框内的数据偏移和搬运计算
        tempAlgParamsStepFour.count = currDataCount;
        tempAlgParamsStepFour.buffInfo.inBuffBaseOff = 0;                                   // CCL-OUT
        tempAlgParamsStepFour.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_; // USR-OUT
        tempAlgParamsStepFour.buffInfo.hcclBuffBaseOff = 0;                                 // CCL-IN
        tempAlgParamsStepFour.allRankDispls = tempAlgParamsStepOne.allRankDispls;
        tempAlgParamsStepFour.allRankSliceSize = tempAlgParamsStepOne.allRankSliceSize;
        tempAlgParamsStepFour.allRankProcessedDataCount = tempAlgParamsStepOne.allRankProcessedDataCount;

        tempAlgParamsStepFour.sliceSize = 0;
        tempAlgParamsStepFour.tailSize = 0;
        // 这里的stride当成传统意义上的stride间隔
        tempAlgParamsStepFour.inputSliceStride = 0;
        tempAlgParamsStepFour.outputSliceStride = 0;

        HCCL_INFO(
            "[InsV2AllReduceSequenceExecutor] loop [%u] tempAlgParamsStepFour.inputSliceStride [%u],"
            "tempAlgParamsStepFour.outputSliceStride [%u] tempAlgParamsStepFour.sliceSize [%u]",
            loop, tempAlgParamsStepFour.inputSliceStride, tempAlgParamsStepFour.outputSliceStride,
            tempAlgParamsStepFour.sliceSize);
        HCCL_INFO(
            "[InsV2AllReduceSequenceExecutor] loop [%u] tempAlgParamsStepFour.buffInfo.inBuffBaseOff [%u],"
            "tempAlgParamsStepFour.buffInfo.outBuffBaseOff [%u]",
            loop, tempAlgParamsStepFour.buffInfo.inBuffBaseOff, tempAlgParamsStepFour.buffInfo.outBuffBaseOff);

        tempAlgParamsStepFour.repeatNum = 1;
        tempAlgParamsStepFour.inputRepeatStride = 0;
        tempAlgParamsStepFour.outputRepeatStride = 0;

        CHK_RET(algTemplateStepFour->KernelRun(param, tempAlgParamsStepFour, templateResourceAllGatherMesh1D));

        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2AllReduceSequenceExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
u64 InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    RoundUp(const u64 dividend, const u64 divisor)
{
    if (divisor == 0) {
        HCCL_WARNING("[InsV2AllReduceSequenceExecutor][RoundUp] divisor is 0.");
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    SplitData(const u64& dataCount, const uint64_t& rankSize, TemplateDataParams& tempAlgParams)
{
    u32 sliceNum = rankSize;
    tempAlgParams.allRankSliceSize.clear();
    tempAlgParams.allRankProcessedDataCount.clear();
    tempAlgParams.allRankDispls.clear();
    tempAlgParams.allRankSliceSize.reserve(sliceNum);
    tempAlgParams.allRankDispls.reserve(sliceNum);
    tempAlgParams.allRankProcessedDataCount.reserve(sliceNum);

    u64 offsetCount = 0;
    u64 offsetSize = 0;
    u64 sliceCount = RoundUp(dataCount, sliceNum);
    u64 sliceSize = sliceCount * dataTypeSize_;

    for (u32 sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
        if (dataCount - offsetCount >= sliceCount) {
            tempAlgParams.allRankSliceSize.emplace_back(sliceSize);
            tempAlgParams.allRankProcessedDataCount.emplace_back(sliceCount);
            tempAlgParams.allRankDispls.emplace_back(offsetSize);
            offsetCount += sliceCount;
            offsetSize = offsetCount * dataTypeSize_;
        } else {
            u64 curSliceCount = dataCount - offsetCount;
            u64 curSliceSize = curSliceCount * dataTypeSize_;
            tempAlgParams.allRankSliceSize.emplace_back(curSliceSize);
            tempAlgParams.allRankProcessedDataCount.emplace_back(curSliceCount);
            tempAlgParams.allRankDispls.emplace_back(offsetSize);
            offsetCount = dataCount;
            offsetSize = offsetCount * dataTypeSize_;
        }
    }

    for (u32 i = 0; i < tempAlgParams.allRankSliceSize.size(); ++i) {
        HCCL_DEBUG(
            "[InsV2AllReduceSequenceExecutor] SliceInfo: offset[%u] size[%u] count[%u]",
            tempAlgParams.allRankDispls.at(i), tempAlgParams.allRankSliceSize.at(i),
            tempAlgParams.allRankProcessedDataCount.at(i));
    }

    return HcclResult::HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_FOUR_TEMPS(
    HcclCMDType::HCCL_CMD_ALLREDUCE, DpuAllReduceSequenceMeshNHR, InsV2AllReduceSequenceExecutor, TopoMatchMultilevel,
    InsTempReduceScatterMesh1DIntra, InsTempReduceScatterMesh1dDpuInter, InsTempAllGatherNhrDpuInter,
    InsTempAllGatherMesh1dIntra);
REGISTER_ALG_ATTRS(
    DpuAllReduceSequenceMeshNHR,
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.minTopoLevelNum = 2; topo.isSupportLevel0PcieMix = true; topo.isHostDpuOnly = true;
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->deviceNumPerModule == 1;
    });
} // namespace ops_hccl
