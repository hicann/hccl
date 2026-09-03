/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_sequence_executor.h"
#include "alg_attrs_registry.h"
#include "ins_temp_reduce_scatter_mesh_1D_intra.h"
#include "ins_temp_reduce_scatter_mesh_1D_dpu_inter.h"
#include "ins_temp_gather_dpu_inter.h"
#include "ins_temp_gather_mesh_1D_intra.h"

namespace ops_hccl {
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
InsV2ReduceSequenceExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::InsV2ReduceSequenceExecutor()
{}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
std::vector<CostModelParam>
InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    CalcCostCoeff(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)algName;
    (void)comm;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    (void)algHierarchyInfo;
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = true;
    auto rs = CostModelManager::Global()->CalcRankSizeByTopo(topoInfo);
    u32 rankSizeLevel0 = rs.level0;
    u32 rankSizeLevel1 = rs.level1;
    CommTopo netTypeLevel0 = CommTopo::COMM_TOPO_1DMESH;
    CommTopo netTypeLevel1 = CommTopo::COMM_TOPO_CLOS;
    std::vector<u32> portNumLevel0 = {1};
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
            rankSizeLevel0, 1.0f / rankSizeLevel0, netTypeLevel0, BufferType::INPUT, BufferType::HCCL_BUFFER,
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
            rankSizeLevel0, 1.0f / rankSizeLevel0, netTypeLevel0, BufferType::HCCL_BUFFER, BufferType::OUTPUT,
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
InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const
{
    (void)param;
    auto rs = CostModelManager::Global()->CalcRankSizeByTopo(topoInfo);
    u32 rankSizeLevel0 = rs.level0;
    u32 rankSizeLevel1 = rs.level1;
    u32 rankSize = topoInfo->userRankSize;
    CommTopo netTypeLevel0 = CommTopo::COMM_TOPO_1DMESH;
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
HcclResult
InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
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
        "[InsV2ReduceSequenceExecutor][InitCommInfo] myRank [%u], rankSize [%u], devType [%u], redOp [%u], "
        "dataType [%u], dataCount [%u], dataTypeSize [%u]",
        myRank_, rankSize_, devType_, reduceOp_, dataType_, dataCount_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
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

// ! 已编码完成
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    InitCommInfo(param, topoInfo, algHierarchyInfo);

    rankSizeLevel1_ = algHierarchyInfo.infos[1].size();
    rankSizeLevel0_ = algHierarchyInfo.infos[0].size();

    std::shared_ptr<InsAlgTemplate0> reduceScatterMesh1DTempAlg
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> reduceScatterMesh1dDpuTempAlg
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate2> GatherDpuTempAlg
        = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate3> GatherMesh1DTempAlg
        = std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo.infos[0]);

    AlgResourceRequest resReqReduceScatterMesh1D;
    AlgResourceRequest resReqReduceScatterMesh1dDpu;
    AlgResourceRequest resReqGatherDpu;
    AlgResourceRequest resReqGatherMesh1D;
    CHK_RET(reduceScatterMesh1DTempAlg->CalcRes(comm, param, topoInfo, resReqReduceScatterMesh1D));
    CHK_RET(reduceScatterMesh1dDpuTempAlg->CalcRes(comm, param, topoInfo, resReqReduceScatterMesh1dDpu));
    CHK_RET(GatherDpuTempAlg->CalcRes(comm, param, topoInfo, resReqGatherDpu));
    CHK_RET(GatherMesh1DTempAlg->CalcRes(comm, param, topoInfo, resReqGatherMesh1D));

    // step1、2、3、4为串行，因此slaveThread和对应notify可以复用
    resourceRequest.slaveThreadNum
        = std::max(resReqReduceScatterMesh1D.slaveThreadNum, resReqReduceScatterMesh1dDpu.slaveThreadNum);
    resourceRequest.notifyNumPerThread
        = std::max(resReqReduceScatterMesh1D.notifyNumPerThread, resReqReduceScatterMesh1dDpu.notifyNumPerThread);
    resourceRequest.notifyNumOnMainThread
        = std::max(resReqReduceScatterMesh1D.notifyNumOnMainThread, resReqReduceScatterMesh1dDpu.notifyNumOnMainThread);

    resourceRequest.channels = {resReqReduceScatterMesh1D.channels[0], resReqReduceScatterMesh1dDpu.channels[0]};
    return HCCL_SUCCESS;
}

// ! 已编码完成
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2ReduceSequenceExecutor][Orchestrate] Orchestrate Start");
    // 参数填充
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

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
            "[InsV2ReduceSequenceExecutor][Orchestrate]errNo[0x%016llx] Reduce executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

// ! 已编码完成
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2ReduceSequenceExecutor][OrchestrateLoop] Start");
    // 将ccl-buffer分成ccl-in和ccl-out 2部分区分使用
    void* cclInAddr = resCtx.cclMem.addr;
    HcclMem cclInMem = {resCtx.cclMem.type, cclInAddr, resCtx.cclMem.size / 2};
    void* cclOutAddr = static_cast<void*>(static_cast<s8*>(resCtx.cclMem.addr) + resCtx.cclMem.size / 2);
    HcclMem cclOutMem = {resCtx.cclMem.type, cclOutAddr, resCtx.cclMem.size / 2};

    // 声明框内ReduceScatterMesh1D的templateargs
    TemplateDataParams tempAlgParamsReduceScatterMesh1D;
    tempAlgParamsReduceScatterMesh1D.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsReduceScatterMesh1D.buffInfo.outputPtr = cclOutMem.addr;
    tempAlgParamsReduceScatterMesh1D.buffInfo.hcclBuff = cclInMem;
    tempAlgParamsReduceScatterMesh1D.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsReduceScatterMesh1D.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsReduceScatterMesh1D.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsReduceScatterMesh1D.root = param.root;

    // 构建框内ReduceScatterMesh1D的template
    std::shared_ptr<InsAlgTemplate0> algTemplateReduceScatterMesh1D
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);

    // 声明框间ReduceScatterMesh1dDpu的templateargs
    TemplateDataParams tempAlgParamsReduceScatterMesh1dDpu;
    tempAlgParamsReduceScatterMesh1dDpu.buffInfo.inputPtr = cclOutMem.addr;
    tempAlgParamsReduceScatterMesh1dDpu.buffInfo.outputPtr = cclOutMem.addr;
    tempAlgParamsReduceScatterMesh1dDpu.buffInfo.hcclBuff = cclInMem;
    tempAlgParamsReduceScatterMesh1dDpu.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsReduceScatterMesh1dDpu.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsReduceScatterMesh1dDpu.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsReduceScatterMesh1dDpu.root = param.root;

    // 构建框间ReduceScatterMesh1dDpu的template
    std::shared_ptr<InsAlgTemplate1> algTemplateReduceScatterMesh1dDpu
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);

    // 声明框间GatherDpu的templateargs，ccl-out搬运到ccl-out
    TemplateDataParams tempAlgParamsGatherDpu;
    tempAlgParamsGatherDpu.buffInfo.inputPtr = cclOutMem.addr;
    tempAlgParamsGatherDpu.buffInfo.outputPtr = cclOutMem.addr;
    tempAlgParamsGatherDpu.buffInfo.hcclBuff = cclInMem;
    tempAlgParamsGatherDpu.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsGatherDpu.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsGatherDpu.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsGatherDpu.root = param.root;

    // 构建框间GatherDpu的template
    std::shared_ptr<InsAlgTemplate2> algTemplateGatherDpu
        = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[1]);

    // 声明框内GatherMesh1D的templateargs，ccl-out搬运到user-out
    TemplateDataParams tempAlgParamsGatherMesh1D;
    tempAlgParamsGatherMesh1D.buffInfo.inputPtr = cclOutMem.addr;
    tempAlgParamsGatherMesh1D.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsGatherMesh1D.buffInfo.hcclBuff = cclInMem;
    tempAlgParamsGatherMesh1D.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsGatherMesh1D.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsGatherMesh1D.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsGatherMesh1D.root = param.root;

    // 构建框内GatherMesh1D的template
    std::shared_ptr<InsAlgTemplate3> algTemplateGatherMesh1D
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
    // 构造框间GatherDpu的template资源
    TemplateResource templateResourceGatherDpu;
    templateResourceGatherDpu.channels = remoteRankToChannelInfo_[1];
    templateResourceGatherDpu.threads = resCtx.threads;
    templateResourceGatherDpu.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceGatherDpu.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    // 构造框内GatherMesh1D的template资源
    TemplateResource templateResourceGatherMesh1D;
    templateResourceGatherMesh1D.channels = remoteRankToChannelInfo_[0];
    templateResourceGatherMesh1D.threads = resCtx.threads;
    templateResourceGatherMesh1D.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceGatherMesh1D.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;

    // 中转内存单次最多能够接受的output count，注意是count不是size
    u64 maxCountPerLoop = tempAlgParamsReduceScatterMesh1D.buffInfo.hcclBuff.size / 2 / HCCL_MIN_SLICE_ALIGN
                          * HCCL_MIN_SLICE_ALIGN / dataTypeSize_; // 这边看前面有/10*10，不知道要不要加上
    // 防止除零：当 hcclBuff.size < 2*HCCL_MIN_SLICE_ALIGN 或 dataTypeSize_==0 时 maxCountPerLoop 可能为 0
    if (maxCountPerLoop == 0) {
        HCCL_ERROR(
            "[InsV2ReduceSequenceExecutor][OrchestrateLoop] maxCountPerLoop is 0, hcclBuff.size[%llu], "
            "dataTypeSize[%llu], dataCount[%llu].",
            tempAlgParamsReduceScatterMesh1D.buffInfo.hcclBuff.size, dataTypeSize_, dataCount_);
        return HCCL_E_INTERNAL;
    }
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount :
                                                      maxCountPerLoop; // 判断是最后一轮，就处理尾块长度

        // ----------- 框内ReduceScatter数据搬运 -----------
        // 框内的数据偏移和搬运计算
        tempAlgParamsReduceScatterMesh1D.count = currDataCount;
        tempAlgParamsReduceScatterMesh1D.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamsReduceScatterMesh1D.buffInfo.outBuffBaseOff = 0;  // CCL-OUT
        tempAlgParamsReduceScatterMesh1D.buffInfo.hcclBuffBaseOff = 0; // CCL-IN
        CHK_RET(SplitData(
            currDataCount, rankSizeLevel0_, tempAlgParamsReduceScatterMesh1D)); // 计算每个卡对应位置的offset,count,size
        CHK_PRT_RET(
            tempAlgParamsReduceScatterMesh1D.allRankSliceSize.size() != rankSizeLevel0_,
            HCCL_ERROR(
                "[InsV2ReduceSequenceExecutor][tempAlgParamsReduceScatterMesh1D] slice num[%u] is not equal to rank "
                "size[%u].",
                tempAlgParamsReduceScatterMesh1D.allRankSliceSize.size(), rankSizeLevel0_),
            HcclResult::HCCL_E_INTERNAL);
        tempAlgParamsReduceScatterMesh1D.sliceSize = 0; // 没用到，template里面用SplitData算了
        tempAlgParamsReduceScatterMesh1D.tailSize = 0;  // 没用到
        // 这里的stride当成传统意义上的stride间隔
        tempAlgParamsReduceScatterMesh1D.inputSliceStride = 0;  // 没用到
        tempAlgParamsReduceScatterMesh1D.outputSliceStride = 0; // 没用到

        HCCL_INFO(
            "[InsV2ReduceSequenceExecutor] loop [%u] tempAlgParamsReduceScatterMesh.inputSliceStride [%u],"
            "tempAlgParamsReduceScatterMesh.outputSliceStride [%u] tempAlgParamsReduceScatterMesh.sliceSize [%u]",
            loop, tempAlgParamsReduceScatterMesh1D.inputSliceStride, tempAlgParamsReduceScatterMesh1D.outputSliceStride,
            tempAlgParamsReduceScatterMesh1D.sliceSize);
        HCCL_INFO(
            "[InsV2ReduceSequenceExecutor] loop [%u] tempAlgParamsReduceScatterMesh.buffInfo.inBuffBaseOff [%u],"
            "tempAlgParamsReduceScatterMesh.buffInfo.outBuffBaseOff [%u]",
            loop, tempAlgParamsReduceScatterMesh1D.buffInfo.inBuffBaseOff,
            tempAlgParamsReduceScatterMesh1D.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsReduceScatterMesh1D.repeatNum = 1;
        tempAlgParamsReduceScatterMesh1D.inputRepeatStride = 0;
        tempAlgParamsReduceScatterMesh1D.outputRepeatStride = 0;
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplateReduceScatterMesh1D->KernelRun(
            param, tempAlgParamsReduceScatterMesh1D, templateResourceReduceScatterMesh1D));

        // ----------- 框间ReduceScatterMesh1dDpu数据搬运 -----------
        // 框间的数据偏移和搬运量计算
        tempAlgParamsReduceScatterMesh1dDpu.count
            = tempAlgParamsReduceScatterMesh1D.allRankProcessedDataCount.at(rankIdxLevel0_);
        if (tempAlgParamsReduceScatterMesh1dDpu.count
            != 0) { // 如果卡里没有数据，不参与框间，让该框有数据的其他卡参与跨框
            tempAlgParamsReduceScatterMesh1dDpu.buffInfo.inBuffBaseOff = 0; // ccl-out偏移量，每次更新，所以是0
            tempAlgParamsReduceScatterMesh1dDpu.buffInfo.outBuffBaseOff = 0; // ccl-out偏移量，每次更新，所以是0
            tempAlgParamsReduceScatterMesh1dDpu.buffInfo.hcclBuffBaseOff = 0; // ccl-in
            tempAlgParamsReduceScatterMesh1dDpu.sliceSize = 0;
            tempAlgParamsReduceScatterMesh1dDpu.tailSize = 0;
            CHK_RET(SplitData(
                tempAlgParamsReduceScatterMesh1dDpu.count, rankSizeLevel1_, tempAlgParamsReduceScatterMesh1dDpu));
            CHK_PRT_RET(
                tempAlgParamsReduceScatterMesh1dDpu.allRankSliceSize.size() != rankSizeLevel1_,
                HCCL_ERROR(
                    "[InsV2ReduceSequenceExecutor][tempAlgParamsReduceScatterMesh1dDpu] slice num[%u] is not "
                    "equal to rank size[%u].",
                    tempAlgParamsReduceScatterMesh1dDpu.allRankSliceSize.size(), rankSizeLevel1_),
                HcclResult::HCCL_E_INTERNAL);
            // 这里的stride当成传统意义上的stride 间隔
            tempAlgParamsReduceScatterMesh1dDpu.inputSliceStride = 0;  // 没用到
            tempAlgParamsReduceScatterMesh1dDpu.outputSliceStride = 0; // 没用到
            //
            HCCL_INFO(
                "[InsV2ReduceSequenceExecutor] loop [%u] tempAlgParamsReduceScatterMesh1dDpu.inputSliceStride [%u],"
                "tempAlgParamsReduceScatterMesh1dDpu.outputSliceStride [%u] "
                "tempAlgParamsReduceScatterMesh1dDpu.sliceSize [%u]",
                loop, tempAlgParamsReduceScatterMesh1dDpu.inputSliceStride,
                tempAlgParamsReduceScatterMesh1dDpu.outputSliceStride, tempAlgParamsReduceScatterMesh1dDpu.sliceSize);
            HCCL_INFO(
                "[InsV2ReduceSequenceExecutor] loop [%u] "
                "tempAlgParamsReduceScatterMesh1dDpu.buffInfo.inBuffBaseOff [%u],"
                "tempAlgParamsReduceScatterMesh1dDpu.buffInfo.outBuffBaseOff [%u]",
                loop, tempAlgParamsReduceScatterMesh1dDpu.buffInfo.inBuffBaseOff,
                tempAlgParamsReduceScatterMesh1dDpu.buffInfo.outBuffBaseOff);
            // 不需要重复
            tempAlgParamsReduceScatterMesh1dDpu.repeatNum = 1;
            tempAlgParamsReduceScatterMesh1dDpu.inputRepeatStride = 0;
            tempAlgParamsReduceScatterMesh1dDpu.outputRepeatStride = 0;
            // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
            CHK_RET(algTemplateReduceScatterMesh1dDpu->KernelRun(
                param, tempAlgParamsReduceScatterMesh1dDpu, templateResourceReduceScatterMesh1dDpu));
        }

        // ----------- 框间GatherDpu数据搬运 -----------
        // 框间的数据偏移和搬运量计算
        tempAlgParamsGatherDpu.count = tempAlgParamsReduceScatterMesh1D.allRankProcessedDataCount.at(rankIdxLevel0_);
        if (tempAlgParamsReduceScatterMesh1dDpu.count != 0) {
            tempAlgParamsGatherDpu.buffInfo.inBuffBaseOff = 0;   // ccl-out偏移量，每次更新，所以是0
            tempAlgParamsGatherDpu.buffInfo.outBuffBaseOff = 0;  // ccl-out偏移量，每次更新，所以是0
            tempAlgParamsGatherDpu.buffInfo.hcclBuffBaseOff = 0; // ccl-in
            tempAlgParamsGatherDpu.allRankDispls = tempAlgParamsReduceScatterMesh1dDpu.allRankDispls;
            tempAlgParamsGatherDpu.allRankSliceSize = tempAlgParamsReduceScatterMesh1dDpu.allRankSliceSize;
            tempAlgParamsGatherDpu.allRankProcessedDataCount
                = tempAlgParamsReduceScatterMesh1dDpu.allRankProcessedDataCount;

            tempAlgParamsGatherDpu.sliceSize = 0;
            tempAlgParamsGatherDpu.tailSize = 0;
            // 这里的stride当成传统意义上的stride 间隔
            tempAlgParamsGatherDpu.inputSliceStride = 0;
            tempAlgParamsGatherDpu.outputSliceStride = 0;

            HCCL_INFO(
                "[InsV2ReduceSequenceExecutor] loop [%u] tempAlgParamsGatherDpu.inputSliceStride [%u],"
                "tempAlgParamsGatherDpu.outputSliceStride [%u] tempAlgParamsGatherDpu.sliceSize [%u]",
                loop, tempAlgParamsGatherDpu.inputSliceStride, tempAlgParamsGatherDpu.outputSliceStride,
                tempAlgParamsGatherDpu.sliceSize);
            HCCL_INFO(
                "[InsV2ReduceSequenceExecutor] loop [%u] tempAlgParamsGatherDpu.buffInfo.inBuffBaseOff [%u],"
                "tempAlgParamsGatherDpu.buffInfo.outBuffBaseOff [%u]",
                loop, tempAlgParamsGatherDpu.buffInfo.inBuffBaseOff, tempAlgParamsGatherDpu.buffInfo.outBuffBaseOff);
            // 不需要重复
            tempAlgParamsGatherDpu.repeatNum = 1;
            tempAlgParamsGatherDpu.inputRepeatStride = 0;
            tempAlgParamsGatherDpu.outputRepeatStride = 0;
            // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
            CHK_RET(algTemplateGatherDpu->KernelRun(param, tempAlgParamsGatherDpu, templateResourceGatherDpu));
        }

        // ----------- 框内GatherMesh数据搬运 -----------
        // 框内的数据偏移和搬运计算
        tempAlgParamsGatherMesh1D.count = currDataCount;
        tempAlgParamsGatherMesh1D.buffInfo.inBuffBaseOff = 0;                                   // CCL-OUT
        tempAlgParamsGatherMesh1D.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_; // USR-OUT
        tempAlgParamsGatherMesh1D.buffInfo.hcclBuffBaseOff = 0;                                 // CCL-IN
        tempAlgParamsGatherMesh1D.allRankDispls = tempAlgParamsReduceScatterMesh1D.allRankDispls;
        tempAlgParamsGatherMesh1D.allRankSliceSize = tempAlgParamsReduceScatterMesh1D.allRankSliceSize;
        tempAlgParamsGatherMesh1D.allRankProcessedDataCount
            = tempAlgParamsReduceScatterMesh1D.allRankProcessedDataCount;

        tempAlgParamsGatherMesh1D.sliceSize = 0;
        tempAlgParamsGatherMesh1D.tailSize = 0;
        // 这里的stride当成传统意义上的stride间隔
        tempAlgParamsGatherMesh1D.inputSliceStride = 0;
        tempAlgParamsGatherMesh1D.outputSliceStride = 0;

        HCCL_INFO(
            "[InsV2ReduceSequenceExecutor] loop [%u] tempAlgParamsGatherMesh.inputSliceStride [%u],"
            "tempAlgParamsGatherMesh.outputSliceStride [%u] tempAlgParamsGatherMesh.sliceSize [%u]",
            loop, tempAlgParamsGatherMesh1D.inputSliceStride, tempAlgParamsGatherMesh1D.outputSliceStride,
            tempAlgParamsGatherMesh1D.sliceSize);
        HCCL_INFO(
            "[InsV2ReduceSequenceExecutor] loop [%u] tempAlgParamsGatherMesh.buffInfo.inBuffBaseOff [%u],"
            "tempAlgParamsGatherMesh.buffInfo.outBuffBaseOff [%u]",
            loop, tempAlgParamsGatherMesh1D.buffInfo.inBuffBaseOff, tempAlgParamsGatherMesh1D.buffInfo.outBuffBaseOff);
        // 应该不需要重复
        tempAlgParamsGatherMesh1D.repeatNum = 1;
        tempAlgParamsGatherMesh1D.inputRepeatStride = 0;
        tempAlgParamsGatherMesh1D.outputRepeatStride = 0;
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplateGatherMesh1D->KernelRun(param, tempAlgParamsGatherMesh1D, templateResourceGatherMesh1D));

        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2ReduceSequenceExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
u64 InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    RoundUp(const u64 dividend, const u64 divisor)
{
    if (divisor == 0) {
        HCCL_WARNING("[InsV2ReduceSequenceExecutor][RoundUp] divisor is 0.");
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    SplitData(const u64& dataCount, const uint64_t& rankSize, TemplateDataParams& tempAlgParams)
{
    u32 sliceNum = rankSize;
    tempAlgParams.allRankSliceSize.clear();
    tempAlgParams.allRankDispls.clear();
    tempAlgParams.allRankProcessedDataCount.clear();
    tempAlgParams.allRankSliceSize.reserve(sliceNum);
    tempAlgParams.allRankDispls.reserve(sliceNum);
    tempAlgParams.allRankProcessedDataCount.reserve(sliceNum);

    u64 sliceCount = RoundUp(dataCount, sliceNum);
    u64 sliceSize = sliceCount * dataTypeSize_;
    u64 offsetSize = 0;
    u64 offsetCount = 0;
    for (u32 sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
        if (dataCount - offsetCount >= sliceCount) {
            tempAlgParams.allRankSliceSize.emplace_back(sliceSize);
            tempAlgParams.allRankDispls.emplace_back(offsetSize);
            tempAlgParams.allRankProcessedDataCount.emplace_back(sliceCount);
            offsetCount += sliceCount;
            offsetSize = offsetCount * dataTypeSize_;
        } else {
            u64 curSliceCount = dataCount - offsetCount;
            u64 curSliceSize = curSliceCount * dataTypeSize_;
            tempAlgParams.allRankSliceSize.emplace_back(curSliceSize);
            tempAlgParams.allRankDispls.emplace_back(offsetSize);
            tempAlgParams.allRankProcessedDataCount.emplace_back(curSliceCount);
            offsetCount = dataCount;
            offsetSize = offsetCount * dataTypeSize_;
        }
    }

    for (u32 i = 0; i < tempAlgParams.allRankSliceSize.size(); ++i) {
        HCCL_DEBUG(
            "[InsV2ReduceSequenceExecutor] SliceInfo: offset[%u] size[%u] count[%u]", tempAlgParams.allRankDispls.at(i),
            tempAlgParams.allRankSliceSize.at(i), tempAlgParams.allRankProcessedDataCount.at(i));
    }

    return HcclResult::HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_FOUR_TEMPS(
    HcclCMDType::HCCL_CMD_REDUCE, DpuReduceSequenceMeshNHR, InsV2ReduceSequenceExecutor, TopoMatchMultilevel,
    InsTempReduceScatterMesh1DIntra, InsTempReduceScatterMesh1dDpuInter, InsTempGatherDpuInter,
    InsTempGatherMesh1dIntra);
} // namespace ops_hccl
