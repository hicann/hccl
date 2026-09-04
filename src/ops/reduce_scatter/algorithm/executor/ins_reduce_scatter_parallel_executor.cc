/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_reduce_scatter_parallel_executor.h"
#include <cmath>
#include "ins_temp_reduce_scatter_mesh_1D.h"
#include "ins_temp_reduce_scatter_mesh_1d_dpu.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "alg_data_trans_wrapper.h"
#include "topo_match_squeeze_2d.h"
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_reduce_scatter_nhr_1D_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh_1D_mem2mem.h"
#include "ccu_temp_reduce_scatter_nhr_1D_multi_jetty_mem2mem.h"
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif

#include "alg_attrs_registry.h"
#include "auto_selector_base.h"
namespace ops_hccl {
constexpr u32 DEVICE_NUM_PER_MODULE_8 = 8;
constexpr u32 MAX_RANK_NUM_FOR_CONCURRENT_ALGO = 4;
constexpr u64 OMNI_PCIE_RS_DATA_SIZE = 4 * 1024 * 1024; // pcie/UBX机型并行与流水算法的数据量分界，与selector保持一致
constexpr u64 OMNI_UBX_RS_SCHED_DATA_SIZE = 4 * 1024 * 1024; // UBX机型ccu并行与流水算法的数据量分界，与selector保持一致

// 并行执行器中两个 template 算法所需的 notify 数量
constexpr u32 TEMPLATE_NOTIFY_NUM = 2;
// 并行执行器中两个 template 算法的主 thread 数量
constexpr u32 TEMPLATE_MAIN_THREAD_NUM = 2;
constexpr u64 PARALLEL_DATA_ALIGN_SIZE = 16 * 1024;

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsReduceScatterParallelExecutor()
    : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::~InsReduceScatterParallelExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
std::vector<CostModelParam>
InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)algName;
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
    // 数据按 ratio 分成 part0 + part1，默认 0.5
    float ratio = 0.5f;
    float ratio0 = ratio;
    float ratio1 = 1.0f - ratio;
    HCCL_INFO(
        "[CalcCostCoeff] rankSize=%d, rankSizeLevel0=%d, rankSizeLevel1=%d, portNumLevel0=%d, portNumLevel1=%d, "
        "netTypeLevel0=%d, netTypeLevel1=%d, ratio=%f",
        rankSize, rankSizeLevel0, rankSizeLevel1, portNumLevel0, portNumLevel1, static_cast<int>(netTypeLevel0),
        static_cast<int>(netTypeLevel1), ratio);
    std::vector<CostModelParam> params = [rankSizeLevel0, rankSizeLevel1, ratio0, ratio1, portNumLevel0, portNumLevel1,
                                          netTypeLevel0, netTypeLevel1, isPod] {
        std::vector<CostModelParam> v;
        // Step1: intra 处理 ratio0 比例数据, inter 处理 ratio1 比例数据（并行）
        auto p0 = InsAlgTemplate0::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel0, ratio0 * rankSizeLevel1, netTypeLevel0, BufferType::INPUT, BufferType::HCCL_BUFFER,
            BufferType::HCCL_BUFFER, portNumLevel0, isPod});
        auto p1 = InsAlgTemplate1::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel1, ratio1 * rankSizeLevel0, netTypeLevel1, BufferType::INPUT, BufferType::HCCL_BUFFER,
            BufferType::HCCL_BUFFER, portNumLevel1, isPod});
        // Step2: inter 处理 ratio0 比例数据, intra 处理 ratio1 比例数据（并行，从 scratch，无 localCopy）
        auto p2 = InsAlgTemplate1::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel1, ratio0, netTypeLevel1, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
            BufferType::HCCL_BUFFER, portNumLevel1, isPod});
        auto p3 = InsAlgTemplate0::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel0, ratio1, netTypeLevel0, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
            BufferType::HCCL_BUFFER, portNumLevel0, isPod});
        // 任一 template 未实现 CalcCostCoeff（返回空）则整个算法不参与 CostModel
        if (p0.empty() || p1.empty() || p2.empty() || p3.empty()) {
            HCCL_WARNING(
                "[InsReduceScatterParallelExecutor] CalcCostCoeff incomplete, skip (p0=%zu p1=%zu p2=%zu p3=%zu).",
                p0.size(), p1.size(), p2.size(), p3.size());
            return v;
        }
        v.insert(v.end(), p0.begin(), p0.end());
        v.insert(v.end(), p1.begin(), p1.end());
        v.insert(v.end(), p2.begin(), p2.end());
        v.insert(v.end(), p3.begin(), p3.end());
        return v;
    }();
    return params;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
AlgNetMeta InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const
{
    auto rs = CostModelManager::Global()->CalcRankSizeByTopo(topoInfo);
    u32 rankSizeLevel0 = rs.level0;
    u32 rankSizeLevel1 = rs.level1;
    float ratio = param.opConfig.multipleDimensionSplitRatio;
    if (param.opConfig.multipleDimensionSplitRatioSource == MultipleDimensionSplitRatioSource::BUILTIN_FORMULA) {
        ratio = 0.5f;
    }
    float ratio0 = ratio;
    float ratio1 = 1.0f - ratio;
    // TODO: CommTopo netTypeLevel0 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[0]);
    CommTopo netTypeLevel0 = CommTopo::COMM_TOPO_1DMESH;
    // TODO: CommTopo netTypeLevel1 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[1]);
    CommTopo netTypeLevel1 = CommTopo::COMM_TOPO_CLOS;
    AlgNetMeta meta;
    meta.netTypes.push_back(netTypeLevel0);
    meta.netTypes.push_back(netTypeLevel1);
    meta.netTypes.push_back(netTypeLevel1);
    meta.netTypes.push_back(netTypeLevel0);
    meta.intraGroupMode = CostAggMode::MAX;
    meta.groupSizes = {2, 2};
    meta.dataRatios = {ratio0 * rankSizeLevel1, ratio1 * rankSizeLevel0, ratio0, ratio1};
    meta.rankSizes = {rankSizeLevel0, rankSizeLevel1, rankSizeLevel1, rankSizeLevel0};
    return meta;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 构建template
    std::vector<std::vector<u32>> temp0HierarchyInfo;
    std::vector<std::vector<u32>> temp1HierarchyInfo;
    if (algHierarchyInfo.infos.empty() || algHierarchyInfo.infos.size() < TOPO_LEVEL_NUM_2) {
        HCCL_ERROR("[%s] algHierarchyInfo.infos is invalid (empty or size < 2).", __func__);
        return HCCL_E_PARA;
    }
    CHK_PRT_RET(
        algHierarchyInfo.infos.size() < TOPO_LEVEL_NUM_2 || algHierarchyInfo.infos[0].empty()
            || algHierarchyInfo.infos[1].empty(),
        HCCL_ERROR("[%s] algHierarchyInfo.infos is invalid.", __func__), HcclResult::HCCL_E_PARA);
    temp0HierarchyInfo = algHierarchyInfo.infos[0];
    temp1HierarchyInfo = algHierarchyInfo.infos[1];
    InsAlgTemplate0 intraTempAlg(param, topoInfo->userRank, temp0HierarchyInfo);
    InsAlgTemplate1 interTempAlg(param, topoInfo->userRank, temp1HierarchyInfo);

    // 调用计算资源的函数
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    CHK_RET(intraTempAlg.CalcRes(comm, param, topoInfo, intraTempRequest));
    CHK_RET(interTempAlg.CalcRes(comm, param, topoInfo, interTempRequest));
    // 申请一条控制thread作为主thread，该thread仅用于两个template之间同步
    resourceRequest.notifyNumOnMainThread = TEMPLATE_NOTIFY_NUM;
    // 由于主thread被单独作为控制thread，因此总的slaveThread需要额外加上两个template的主thread
    resourceRequest.slaveThreadNum
        = intraTempRequest.slaveThreadNum + interTempRequest.slaveThreadNum + TEMPLATE_MAIN_THREAD_NUM;
    // 第一个template的zhuthread需要的notify数量，+1是因为需要和控制thread做同步
    resourceRequest.notifyNumPerThread.emplace_back(intraTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(
        resourceRequest.notifyNumPerThread.end(), intraTempRequest.notifyNumPerThread.begin(),
        intraTempRequest.notifyNumPerThread.end());
    // 这一条是interTemplate的主thread，需要+1是为了和控制thread进行同步
    resourceRequest.notifyNumPerThread.emplace_back(interTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(
        resourceRequest.notifyNumPerThread.end(), interTempRequest.notifyNumPerThread.begin(),
        interTempRequest.notifyNumPerThread.end());
    if (param.engine != COMM_ENGINE_CCU) {
        resourceRequest.channels.emplace_back(intraTempRequest.channels[0]);
        resourceRequest.channels.emplace_back(interTempRequest.channels[0]);
    } else {
        // ccu
        HCCL_INFO(
            "[InsReduceScatterParallelExecutor][CalcRes] intraTemplate has [%d] kernels.",
            intraTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(
            resourceRequest.ccuKernelInfos.end(), intraTempRequest.ccuKernelInfos.begin(),
            intraTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(intraTempRequest.ccuKernelNum[0]);
        HCCL_INFO(
            "[InsReduceScatterParallelExecutor][CalcRes] interTemplate has [%d] kernels.",
            interTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(
            resourceRequest.ccuKernelInfos.end(), interTempRequest.ccuKernelInfos.begin(),
            interTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(interTempRequest.ccuKernelNum[0]);
        resourceRequest.dieSplitRatio = interTempRequest.dieSplitRatio;
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void)comm;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, AlgAttrs{}));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

// HOST 侧算法入口，将对应的 instruction 添加到指令队列中
// 传入的insQue为一条主流
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra0(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAxis0, std::vector<u64>& scratchOffVec, TemplateDataParams& tempAlgParamsIntra0) const
{
    tempAlgParamsIntra0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsIntra0.buffInfo.inputSize = param.inputSize;
    tempAlgParamsIntra0.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsIntra0.buffInfo.outputSize = resCtx.cclMem.size;
    tempAlgParamsIntra0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsIntra0.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsIntra0.buffInfo.outBuffType = BufferType::HCCL_BUFFER; // 第一步最后的数据存储在scratch buffer上
    tempAlgParamsIntra0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra0.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsIntra0.buffInfo.outBuffBaseOff
        = scratchOffVec[0] + rankIdxLevel0_ * dataCountPerLoopAxis0 * dataTypeSize_;
    tempAlgParamsIntra0.buffInfo.hcclBuffBaseOff = scratchOffVec[0];
    tempAlgParamsIntra0.sliceSize = dataCountPerLoopAxis0 * dataTypeSize_;
    tempAlgParamsIntra0.tailSize = tempAlgParamsIntra0.sliceSize;
    tempAlgParamsIntra0.count = dataCountPerLoopAxis0;

    tempAlgParamsIntra0.inputSliceStride = dataSize_;
    tempAlgParamsIntra0.outputSliceStride = 0;
    tempAlgParamsIntra0.repeatNum = rankSizeLevel1_;
    tempAlgParamsIntra0.inputRepeatStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsIntra0.outputRepeatStride = dataCountPerLoopAxis0 * dataTypeSize_ * rankSizeLevel0_;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter0(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAxis0, std::vector<u64>& scratchOffVec, TemplateDataParams& tempAlgParamsInter0) const
{
    tempAlgParamsInter0.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsInter0.buffInfo.inputSize = resCtx.cclMem.size;
    tempAlgParamsInter0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsInter0.buffInfo.outputSize = param.outputSize;
    tempAlgParamsInter0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsInter0.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsInter0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter0.buffInfo.inBuffBaseOff
        = scratchOffVec[0] + rankIdxLevel0_ * dataCountPerLoopAxis0 * dataTypeSize_;
    tempAlgParamsInter0.buffInfo.outBuffBaseOff = dataOffset;
    if (engine_ == CommEngine::COMM_ENGINE_CCU) {
        tempAlgParamsInter0.buffInfo.hcclBuffBaseOff = scratchOffVec[2];
    } else {
        tempAlgParamsInter0.buffInfo.hcclBuffBaseOff
            = scratchOffVec[0] + rankIdxLevel0_ * dataCountPerLoopAxis0 * dataTypeSize_;
    }
    tempAlgParamsInter0.sliceSize = dataCountPerLoopAxis0 * dataTypeSize_;
    tempAlgParamsInter0.tailSize = tempAlgParamsInter0.sliceSize;
    tempAlgParamsInter0.count = dataCountPerLoopAxis0;

    tempAlgParamsInter0.inputSliceStride = dataCountPerLoopAxis0 * dataTypeSize_ * rankSizeLevel0_;
    tempAlgParamsInter0.outputSliceStride = 0;
    tempAlgParamsInter0.repeatNum = 1;
    tempAlgParamsInter0.inputRepeatStride = 0;
    tempAlgParamsInter0.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter1(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAxis1, std::vector<u64>& scratchOffVec, TemplateDataParams& tempAlgParamsInter1) const
{
    tempAlgParamsInter1.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsInter1.buffInfo.inputSize = param.inputSize;
    tempAlgParamsInter1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsInter1.buffInfo.outputSize = resCtx.cclMem.size;
    tempAlgParamsInter1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsInter1.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsInter1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter1.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsInter1.buffInfo.outBuffBaseOff = scratchOffVec[3];
    tempAlgParamsInter1.buffInfo.hcclBuffBaseOff = scratchOffVec[3];
    tempAlgParamsInter1.sliceSize = dataCountPerLoopAxis1 * dataTypeSize_;
    tempAlgParamsInter1.tailSize = tempAlgParamsInter1.sliceSize;
    tempAlgParamsInter1.count = dataCountPerLoopAxis1;

    tempAlgParamsInter1.inputSliceStride = dataSize_ * rankSizeLevel0_;
    if (engine_ == CommEngine::COMM_ENGINE_CCU) {
        tempAlgParamsInter1.outputSliceStride = 0;
    } else {
        tempAlgParamsInter1.outputSliceStride = dataCountPerLoopAxis1 * dataTypeSize_;
    }
    tempAlgParamsInter1.repeatNum = rankSizeLevel0_;
    tempAlgParamsInter1.inputRepeatStride = dataSize_;
    tempAlgParamsInter1.outputRepeatStride = dataCountPerLoopAxis1 * dataTypeSize_ * rankSizeLevel1_;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra1(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAxis1, std::vector<u64>& scratchOffVec, TemplateDataParams& tempAlgParamsIntra1) const
{
    tempAlgParamsIntra1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsIntra1.buffInfo.inputSize = resCtx.cclMem.size;
    tempAlgParamsIntra1.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsIntra1.buffInfo.outputSize = param.outputSize;
    tempAlgParamsIntra1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsIntra1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    if (engine_ == CommEngine::COMM_ENGINE_CCU) {
        tempAlgParamsIntra1.buffInfo.inBuffBaseOff = scratchOffVec[3];
    } else {
        tempAlgParamsIntra1.buffInfo.inBuffBaseOff
            = scratchOffVec[3] + rankIdxLevel1_ * dataCountPerLoopAxis1 * dataTypeSize_;
    }
    tempAlgParamsIntra1.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsIntra1.buffInfo.hcclBuffBaseOff = scratchOffVec[1];
    tempAlgParamsIntra1.sliceSize = dataCountPerLoopAxis1 * dataTypeSize_;
    tempAlgParamsIntra1.tailSize = tempAlgParamsIntra1.sliceSize;
    tempAlgParamsIntra1.count = dataCountPerLoopAxis1;

    tempAlgParamsIntra1.inputSliceStride = dataCountPerLoopAxis1 * dataTypeSize_ * rankSizeLevel1_;
    tempAlgParamsIntra1.outputSliceStride = 0;
    tempAlgParamsIntra1.repeatNum = 1;
    tempAlgParamsIntra1.inputRepeatStride = 0;
    tempAlgParamsIntra1.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(
    std::vector<float>& splitDataSize) const
{
    float ratio = multipleDimensionSplitRatio_;
    if (multipleDimensionSplitRatioSource_ == MultipleDimensionSplitRatioSource::BUILTIN_FORMULA) {
        // TODO: CalcParallelDataSplitRatio 未实现，暂用默认 0.5
        // ratio = CalcParallelDataSplitRatio(
        //     rankSizeLevel0_, rankSizeLevel1_, intraChannelMap_, interChannelMap_,
        //     ParallelDataSplitType::REDUCE_SCATTER_WITH_LOCAL_REDUCE, multipleDimensionSplitRatio_);
        ratio = 0.5;
    }
    splitDataSize.push_back(ratio);
    splitDataSize.push_back(1.0 - ratio);
    HCCL_INFO(
        "[InsReduceScatterParallelExecutor] meshFirstRatio[%f], closFirstRatio[%f]", splitDataSize[0],
        splitDataSize[1]);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    const InsAlgTemplate0& tempAlgIntra, const InsAlgTemplate1& tempAlgInter)
{
    u64 intraThreadsNum = tempAlgIntra.GetThreadNum();
    u64 interThreadsNum = tempAlgInter.GetThreadNum();
    if (threads_.size() < intraThreadsNum + interThreadsNum + 1) {
        HCCL_ERROR(
            "[InsReduceScatterParallelExecutor][PrepareResForTemplate] threads size is %d, but intraThreadsNum is %d, "
            "interThreadsNum is %d",
            threads_.size(), intraThreadsNum, interThreadsNum);
        return HCCL_E_PARA;
    }
    intraThreads_.assign(threads_.begin() + 1, threads_.begin() + 1 + intraThreadsNum);
    interThreads_.assign(threads_.begin() + 1 + intraThreadsNum, threads_.end());
    // 用于两个算法同步
    controlThread_ = threads_.at(0);
    templateMainThreads_.push_back(intraThreads_.at(0));
    templateMainThreads_.push_back(interThreads_.at(0));
    // 获取两个template各自的主thread上有多少notify
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    CHK_RET(tempAlgIntra.GetRes(intraTempRequest));
    CHK_RET(tempAlgInter.GetRes(interTempRequest));
    notifyIdxControlToTemplates_.push_back(intraTempRequest.notifyNumOnMainThread);
    notifyIdxControlToTemplates_.push_back(interTempRequest.notifyNumOnMainThread);
    notifyIdxTemplatesToControl_.push_back(0);
    notifyIdxTemplatesToControl_.push_back(1);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsReduceScatterParallelExecutor][Orchestrate] Orchestrate Start");

    // cclBuffer的大小
    maxTmpMemSize_ = resCtx.cclMem.size;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
        intraChannelMap_ = remoteRankToChannelInfo_[0];
        interChannelMap_ = remoteRankToChannelInfo_[1];
    }
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    engine_ = param.engine;

    std::vector<std::vector<u32>> temp0HierarchyInfo;
    std::vector<std::vector<u32>> temp1HierarchyInfo;
    if (resCtx.algHierarchyInfo.infos.empty() || resCtx.algHierarchyInfo.infos.size() < TOPO_LEVEL_NUM_2) {
        HCCL_ERROR("[%s] algHierarchyInfo.infos is invalid (empty or size < 2).", __func__);
        return HCCL_E_PARA;
    }
    CHK_PRT_RET(
        resCtx.algHierarchyInfo.infos.size() < TOPO_LEVEL_NUM_2 || resCtx.algHierarchyInfo.infos[0].empty()
            || resCtx.algHierarchyInfo.infos[1].empty(),
        HCCL_ERROR("[%s] algHierarchyInfo.infos is invalid.", __func__), HcclResult::HCCL_E_PARA);
    temp0HierarchyInfo = resCtx.algHierarchyInfo.infos[0];
    temp1HierarchyInfo = resCtx.algHierarchyInfo.infos[1];

    rankSizeLevel0_ = GetRankSize(temp0HierarchyInfo);
    rankSizeLevel1_ = GetRankSize(temp1HierarchyInfo);
    myRank_ = resCtx.topoInfo.userRank;
    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel0_ is 0.", __func__);
        return HCCL_E_PARA;
    }
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    // 实例化算法模板类
    // 构建template
    InsAlgTemplate0 intraTempAlg(param, resCtx.topoInfo.userRank, temp0HierarchyInfo);
    InsAlgTemplate1 interTempAlg(param, resCtx.topoInfo.userRank, temp1HierarchyInfo);
    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS) {
        interTempAlg.SetchannelsPerRank(interChannelMap_);
    }
    // 将计算资源分配个每个算法
    PrepareResForTemplate(intraTempAlg, interTempAlg);
    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx, intraTempAlg, interTempAlg);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsReduceScatterParallelExecutor][Orchestrate]errNo[0x%016llx] Reduce scatter executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, InsAlgTemplate0& tempAlgIntra,
    InsAlgTemplate1& tempAlgInter)
{
    HCCL_INFO("[InsReduceScatterParallelExecutor][OrchestrateLoop] Start");
    HCCL_INFO("[InsReduceScatterParallelExecutor] AlgTemplate inter server is [%s]", tempAlgIntra.Describe().c_str());
    HCCL_INFO("[InsReduceScatterParallelExecutor] AlgTemplate intra server is [%s]", tempAlgInter.Describe().c_str());
    multipleDimensionSplitRatio_ = param.opConfig.multipleDimensionSplitRatio;
    multipleDimensionSplitRatioSource_ = param.opConfig.multipleDimensionSplitRatioSource;
    std::vector<float> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);
    u64 alignedSize = PARALLEL_DATA_ALIGN_SIZE;
    BufferType inBuffType = BufferType::INPUT;
    BufferType outBuffType = BufferType::OUTPUT;
    u32 intraScatchteMultipleStage0 = tempAlgIntra.CalcScratchMultiple(inBuffType, outBuffType);
    u32 interScatchteMultipleStage0 = tempAlgInter.CalcScratchMultiple(inBuffType, outBuffType);
    u32 intraScatchteMultipleStage1 = tempAlgIntra.CalcScratchMultiple(outBuffType, outBuffType);
    u32 interScatchteMultipleStage1 = tempAlgInter.CalcScratchMultiple(outBuffType, outBuffType);
    if (interScatchteMultipleStage0 == 0 || interScatchteMultipleStage1 == 0) {
        interScatchteMultipleStage0 = rankSizeLevel1_;
        interScatchteMultipleStage1 = rankSizeLevel1_;
    }
    u32 scratchMultipleIntra0
        = static_cast<u32>(std::ceil(dataSplitSize[0] * intraScatchteMultipleStage0 * rankSizeLevel1_));
    u32 scratchMultipleIntra1 = static_cast<u32>(std::ceil(dataSplitSize[1] * intraScatchteMultipleStage1));
    u32 scratchMultipleInter1
        = static_cast<u32>(std::ceil(dataSplitSize[1] * interScatchteMultipleStage0 * rankSizeLevel0_));
    u32 scratchMultipleInter0 = static_cast<u32>(std::ceil(dataSplitSize[0] * interScatchteMultipleStage1));
    u32 totalScratchMultiple
        = scratchMultipleIntra0 + scratchMultipleIntra1 + scratchMultipleInter0 + scratchMultipleInter1;
    u64 scratchMemBlockSize = maxTmpMemSize_;
    if (totalScratchMultiple > 0) {
        scratchMemBlockSize = (maxTmpMemSize_ / alignedSize / totalScratchMultiple) * alignedSize;
    }
    u64 intra0ScratchOffset = 0;
    u64 intra1ScratchOffset = intra0ScratchOffset + scratchMultipleIntra0 * scratchMemBlockSize;
    u64 inter0ScratchOffset = intra1ScratchOffset + scratchMultipleIntra1 * scratchMemBlockSize;
    u64 inter1ScratchOffset = inter0ScratchOffset + scratchMultipleInter0 * scratchMemBlockSize;
    std::vector<u64> scratchOffVec
        = {intra0ScratchOffset, intra1ScratchOffset, inter0ScratchOffset, inter1ScratchOffset};

    u64 maxCountPerLoop
        = std::min(static_cast<u64>(scratchMemBlockSize), static_cast<u64>(UB_MAX_DATA_SIZE)) / dataTypeSize_;
    CHK_PRT_RET(
        maxCountPerLoop == 0, HCCL_ERROR("[InsReduceScatterParallelExecutor][OrchestrateLoop] maxCountPerLoop is 0"),
        HcclResult::HCCL_E_INTERNAL);

    u64 alignSize = AICPU_ALIGN_SIZE;
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    TemplateDataParams tempAlgParamsIntra0;
    TemplateDataParams tempAlgParamsInter0;
    TemplateDataParams tempAlgParamsInter1;
    TemplateDataParams tempAlgParamsIntra1;

    TemplateResource templateAlgResIntra;
    TemplateResource templateAlgResInter;
    if (param.engine == COMM_ENGINE_CCU) {
        CHK_PRT_RET(
            resCtx.ccuKernelNum.size() <= 1,
            HCCL_ERROR("[%s] ccuKernelNum size[%zu] is less than 2", __func__, resCtx.ccuKernelNum.size()),
            HCCL_E_INTERNAL);
        templateAlgResIntra.ccuKernels.insert(
            templateAlgResIntra.ccuKernels.end(), resCtx.ccuKernels.begin(),
            resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
        templateAlgResInter.ccuKernels.insert(
            templateAlgResInter.ccuKernels.end(), resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
            resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);
        templateAlgResInter.dieSplitRatio = resCtx.dieSplitRatio;
    } else {
        templateAlgResIntra.channels = intraChannelMap_;
        templateAlgResInter.channels = interChannelMap_;
    }
    templateAlgResIntra.threads = intraThreads_;
    templateAlgResInter.threads = interThreads_;

    u64 processedCount = 0;
    u32 loopIndex = 0;
    while (processedCount < dataCount_) {
        u64 remainingCount = dataCount_ - processedCount;
        u32 remainingLoopTimes = (loopIndex < loopTimes) ? (loopTimes - loopIndex) : 1;
        u64 currCount = (remainingCount + remainingLoopTimes - 1) / remainingLoopTimes;
        currCount = std::min(currCount, maxCountPerLoop);
        u64 currCountPart0 = static_cast<u64>(dataSplitSize[0] * currCount);
        u64 currCountPart1 = currCount - currCountPart0;
        if (remainingLoopTimes > 1) {
            u64 alignedCountPart0 = currCountPart0;
            u64 alignedCountPart1 = currCountPart1;
            alignedCountPart0 = alignedCountPart0 * dataTypeSize_ / alignSize * alignSize / dataTypeSize_;
            alignedCountPart1 = alignedCountPart1 * dataTypeSize_ / alignSize * alignSize / dataTypeSize_;
            if (alignedCountPart0 + alignedCountPart1 > 0) {
                currCountPart0 = alignedCountPart0;
                currCountPart1 = alignedCountPart1;
            }
        }
        CHK_PRT_RET(
            currCountPart0 + currCountPart1 == 0,
            HCCL_ERROR("[InsReduceScatterParallelExecutor][OrchestrateLoop] currCount is 0"),
            HcclResult::HCCL_E_INTERNAL);

        u64 dataOffset0 = processedCount * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + currCountPart0 * dataTypeSize_;

        // 第一步开始前同步
        CHK_RET(PreSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxControlToTemplates_));
        // 数据0的server内的mesh算法
        GenTemplateAlgParamsIntra0(param, resCtx, dataOffset0, currCountPart0, scratchOffVec, tempAlgParamsIntra0);
        // 把每个template需要的queue传进去，比如stars的mesh要传多条queue
        CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra0, templateAlgResIntra));
        // 数据1的server间的nhr算法
        GenTemplateAlgParamsInter1(param, resCtx, dataOffset1, currCountPart1, scratchOffVec, tempAlgParamsInter1);
        CHK_RET(tempAlgInter.KernelRun(param, tempAlgParamsInter1, templateAlgResInter));
        // 第一步做完后回到主流做尾同步
        CHK_RET(PostSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxTemplatesToControl_));

#ifndef AICPU_COMPILE
        if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU) {
            ccuKernelLaunchNumIntra0_ = templateAlgResIntra.submitInfos.size();
            ccuKernelLaunchNumInter1_ = templateAlgResInter.submitInfos.size();
        }
#endif

        // 第二步开始前同步
        CHK_RET(PreSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxControlToTemplates_));
        // 数据0的server间的nhr算法
        GenTemplateAlgParamsInter0(param, resCtx, dataOffset0, currCountPart0, scratchOffVec, tempAlgParamsInter0);
        CHK_RET(tempAlgInter.KernelRun(param, tempAlgParamsInter0, templateAlgResInter));
        // 数据1的server内的mesh算法
        GenTemplateAlgParamsIntra1(param, resCtx, dataOffset1, currCountPart1, scratchOffVec, tempAlgParamsIntra1);
        CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra1, templateAlgResIntra));
        // 尾同步
        CHK_RET(PostSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxTemplatesToControl_));

        processedCount += currCountPart0 + currCountPart1;
        loopIndex++;
    }

#ifndef AICPU_COMPILE
    if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU && param.opMode != OpMode::OFFLOAD) {
        CHK_RET(FastLaunchSaveCtx(param, templateAlgResIntra, templateAlgResInter, resCtx.notifyNumOnMainThread));
    }
#endif

    HCCL_INFO("[InsReduceScatterParallelExecutor][OrchestrateLoop] End.");
    return HcclResult::HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::FastLaunchSaveCtx(
    const OpParam& param, const TemplateResource& templateAlgResIntra, const TemplateResource& templateAlgResInter,
    u32 notifyNumOnMainThread)
{
    HCCL_INFO("[InsReduceScatterParallelExecutor] loopTimes==1, save fast launch ctx.");
    ccuKernelLaunchNumIntra1_ = templateAlgResIntra.submitInfos.size() - ccuKernelLaunchNumIntra0_;
    ccuKernelLaunchNumInter0_ = templateAlgResInter.submitInfos.size() - ccuKernelLaunchNumInter1_;
    u32 threadNum = threads_.size();
    u32 ccuKernelNum
        = ccuKernelLaunchNumIntra1_ + ccuKernelLaunchNumInter0_ + ccuKernelLaunchNumIntra0_ + ccuKernelLaunchNumInter1_;
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsReduceScatterParallelExecutor] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "[InsReduceScatterParallelExecutor][HcclEngineCtxCreate] threadNum[%llu], ccuKernelNum[%llu]", threadNum,
        ccuKernelNum);

    std::vector<u32> ccuKernelNumList
        = {ccuKernelLaunchNumIntra0_, ccuKernelLaunchNumInter1_, ccuKernelLaunchNumInter0_, ccuKernelLaunchNumIntra1_};
    std::vector<std::vector<CcuKernelSubmitInfo>> submitInfosList
        = {templateAlgResIntra.submitInfos, templateAlgResInter.submitInfos};
    return FastLaunchSaveCtxTwoTemplate(
        param, threadNum, ccuKernelNum, threads_, ccuKernelNumList, submitInfosList, notifyNumOnMainThread);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::FastLaunch(
    const OpParam& param, const CcuFastLaunchCtx* ctx)
{
    InsAlgTemplate0 intraTempAlg{};
    InsAlgTemplate1 interTempAlg{};

    TemplateFastLaunchCtx tempFastLaunchCtxIntra0, tempFastLaunchCtxInter0;
    TemplateFastLaunchCtx tempFastLaunchCtxInter1, tempFastLaunchCtxIntra1;

    TemplateResource templateAlgResIntra, templateAlgResInter;
    ThreadHandle* threads = ctx->GetThreadHandlePtr();
    threads_.assign(threads, threads + ctx->threadNum);
    PrepareResForTemplate(intraTempAlg, interTempAlg);

    CcuKernelSubmitInfo* ccuKernelSubmitInfos = ctx->GetCcuKernelSubmitInfoPtr();

    // 第一步开始前同步
    HCCL_INFO("[InsReduceScatterParallelExecutor][FastLaunch] Intra0 ccuKernelNum[%llu]", ctx->ccuKernelNum[0]);
    CHK_RET(PreSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxControlToTemplates_));
    // 数据0的server内的mesh算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra0, param.inputPtr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxIntra0.threads = intraThreads_;
    tempFastLaunchCtxIntra0.ccuKernelSubmitInfos.assign(
        ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[0]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[0];
    // 把每个template需要的queue传进去，比如stars的mesh要传多条queue
    CHK_RET(intraTempAlg.FastLaunch(param, tempFastLaunchCtxIntra0));
    // 数据1的server间的nhr算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter1, param.inputPtr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxInter1.threads = interThreads_;
    tempFastLaunchCtxInter1.ccuKernelSubmitInfos.assign(
        ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[1]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[1];
    CHK_RET(interTempAlg.FastLaunch(param, tempFastLaunchCtxInter1));
    // 第一步做完后回到主流做尾同步
    CHK_RET(PostSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxTemplatesToControl_));

    // 第二步开始前同步
    CHK_RET(PreSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxControlToTemplates_));
    // 数据0的server间的nhr算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter0, param.hcclBuff.addr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxInter0.threads = interThreads_;
    tempFastLaunchCtxInter0.ccuKernelSubmitInfos.assign(
        ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[2]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[2];
    CHK_RET(interTempAlg.FastLaunch(param, tempFastLaunchCtxInter0));
    // 数据1的server内的mesh算法
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra1, param.hcclBuff.addr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxIntra1.threads = intraThreads_;
    tempFastLaunchCtxIntra1.ccuKernelSubmitInfos.assign(
        ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[3]);
    CHK_RET(intraTempAlg.FastLaunch(param, tempFastLaunchCtxIntra1));
    // 尾同步
    CHK_RET(PostSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxTemplatesToControl_));

    HCCL_INFO("[InsReduceScatterParallelExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
uint64_t InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetRankSize(
    const std::vector<std::vector<u32>>& subCommRanks) const
{
    uint64_t count = 1;
    for (const auto& i : subCommRanks) {
        count *= i.size();
    }
    return count;
}

// 算法注册
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
// 合并注册：标准 MESH_1D 两级 / PCIE-SW / Uboe 三级场景共用，分流由 selector 保证
REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AicpuReduceScatterParallelMeshNHR, InsReduceScatterParallelExecutor,
    TopoMatchTwoLevel, InsTempReduceScatterMesh1D, InsTempReduceScatterNHR);
REGISTER_ALG_ATTRS(
    AicpuReduceScatterParallelMeshNHR, topo.supportLevel0Topos = LEVEL0_TOPO_ANY; topo.isSupportLevel0PcieMix = true;
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return (topo->level0PcieMix
                && !AutoSelectorBase::IsLayerAllConnetedWithTopo(topo, 0, CommTopo::COMM_TOPO_1DMESH))
               || (topo->topLevelUboe
                   && !(
                       (topo->level0Symmetric && topo->level1Symmetric)
                       && topo->deviceNumPerModule == DEVICE_NUM_PER_MODULE_8)
                   && !(
                       !(topo->level0Symmetric && topo->level1Symmetric)
                       || topo->netLayerDetails.localNetInsSizeOfLayer[1] == 1));
    });
REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AicpuReduceScatterParallelMeshNHRMultiJetty, InsReduceScatterParallelExecutor,
    TopoMatchTwoLevel, InsTempReduceScatterMesh1D, InsTempReduceScatterNHR);
REGISTER_ALG_ATTRS(
    AicpuReduceScatterParallelMeshNHRMultiJetty, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS;
    topo.maxTopoLevelNum = 1; op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_64BIT;
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        bool isMultiple = false;
        AutoSelectorBase::CheckClosNumMultipleOfMeshNum(topo, isMultiple);
        return isMultiple && !AutoSelectorBase::IsLayerAllConnetedWithTopo(topo, 0, CommTopo::COMM_TOPO_1DMESH);
    };
    // 对称内存场景由PipeLineUBX参选，本算法不参与候选
    op.opCustomCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails*) -> bool {
        return !opParam.supportSymmetricMemory;
    });
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)

#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CcuSchedReduceScatterParallelMeshNHR, InsReduceScatterParallelExecutor,
    TopoMatchTwoLevel, CcuTempReduceScatterMesh1DMem2Mem, CcuTempReduceScatterNHR1DMem2Mem);
REGISTER_ALG_ATTRS(CcuSchedReduceScatterParallelMeshNHR, topo.minTopoLevelNum = 2; topo.maxTopoLevelNum = 2;
                   op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_INT8_AND_64BIT;
                   op.isSupportInplace = false);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CcuSchedReduceScatterParallelMeshNHRMultiLink,
    InsReduceScatterParallelExecutor, TopoMatchTwoLevel, CcuTempReduceScatterMesh1DMem2Mem,
    CcuTempReduceScatterNhrMultiJettyMem2Mem1D);
REGISTER_ALG_ATTRS(
    CcuSchedReduceScatterParallelMeshNHRMultiLink, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS;
    topo.maxTopoLevelNum = 1; op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_INT8_AND_64BIT;
    op.isSupportInplace = false; topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        bool isEqual = false;
        bool isMultiple = false;
        AutoSelectorBase::CheckMeshNumEqualToClosNum(topo, isEqual);
        AutoSelectorBase::CheckClosNumMultipleOfMeshNum(topo, isMultiple);
        return !(isEqual && topo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) && isMultiple
               && AutoSelectorBase::CalcFrameNum(topo) <= MAX_FRAME_NUM_FOR_CCU_ALGO;
    };);

#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif
} // namespace ops_hccl
