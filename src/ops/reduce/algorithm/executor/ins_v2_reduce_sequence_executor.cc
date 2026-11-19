/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_sequence_executor.h"
#include "reduce_sequence_rhd_tail.h"
#include "alg_attrs_registry.h"
#include "ins_temp_reduce_scatter_mesh_1D_intra.h"
#include "ins_temp_reduce_scatter_mesh_1D_dpu_inter.h"
#include "ins_temp_reduce_scatter_nhr_dpu.h"
#include "ins_temp_reduce_scatter_mesh_1D_nda_inter.h"
#include "ins_temp_gather_dpu_inter.h"
#include "ins_temp_all_gather_nhr_nda_inter.h"
#include "ins_temp_gather_mesh_1D_intra.h"
#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "ins_temp_reduce_scatter_rhd_dpu_inter.h"
#include "ins_temp_all_gather_rhd_dpu.h"
#include "alg_attrs_registry.h"
#include "dpu_inline_reduce_selector.h"
#include "template_utils.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

/* RHD 模板编译期判定（对齐 AllReduce ins_v2_all_reduce_sequence_3level_utils.h）：
 * 仅 InsAlgTemplate1=InsTempReduceScatterRhdDpuInter（props.algoType==RHD）命中，
 * NHR/Mesh 等默认注册落入 false_type 走原有路径，编译期隔离、零运行时开销。 */
template <typename T, typename = void>
struct ReduceSeqIsRhdTemplate : std::false_type {};

template <typename T>
struct ReduceSeqIsRhdTemplate<T, std::enable_if_t<(T::props.algoType == AlgoType::RHD)>> : std::true_type {};

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
    (void)comm;
    (void)topoInfo;
    (void)algName;
    (void)param;
    return {{0.0f, 0.0f, 1.0f, 0.0f}};
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
AlgNetMeta
InsV2ReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)param;
    (void)algName;
    AlgNetMeta meta;
    meta.netTypes.push_back(CommTopo::COMM_TOPO_1DMESH);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1};
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
    InitCommonCommInfo(param, topoInfo);

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
    CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
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

    if (algHierarchyInfo.infos.empty() || algHierarchyInfo.infos.size() < TOPO_LEVEL_NUM_2) {
        HCCL_ERROR("[InsV2ReduceSequenceExecutor][CalcRes] algHierarchyInfo.infos is invalid (empty or size < 2).");
        return HCCL_E_PARA;
    }

    // MESH_1D_CLOS without pcieMix: intra uses infos[0][1]
    std::vector<std::vector<u32>> intraHierarchyInfo;
    std::vector<std::vector<u32>> interHierarchyInfo;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix
        && algHierarchyInfo.infos[0].size() > 1) {
        intraHierarchyInfo = {algHierarchyInfo.infos[0][1]};
        interHierarchyInfo = algHierarchyInfo.infos[1];
    } else {
        intraHierarchyInfo = algHierarchyInfo.infos[0];
        interHierarchyInfo = algHierarchyInfo.infos[1];
    }

    std::shared_ptr<InsAlgTemplate0> reduceScatterMesh1DTempAlg
        = std::make_shared<InsAlgTemplate0>(param, myRank_, intraHierarchyInfo);
    std::shared_ptr<InsAlgTemplate1> reduceScatterDpuTempAlg
        = std::make_shared<InsAlgTemplate1>(param, myRank_, interHierarchyInfo);
    std::shared_ptr<InsAlgTemplate2> GatherDpuTempAlg
        = std::make_shared<InsAlgTemplate2>(param, myRank_, interHierarchyInfo);
    std::shared_ptr<InsAlgTemplate3> GatherMesh1DTempAlg
        = std::make_shared<InsAlgTemplate3>(param, myRank_, intraHierarchyInfo);

    AlgResourceRequest resReqReduceScatterMesh1D;
    AlgResourceRequest resReqReduceScatterDpu;
    AlgResourceRequest resReqGatherDpu;
    AlgResourceRequest resReqGatherMesh1D;
    CHK_RET(reduceScatterMesh1DTempAlg->CalcRes(comm, param, topoInfo, resReqReduceScatterMesh1D));
    CHK_RET(reduceScatterDpuTempAlg->CalcRes(comm, param, topoInfo, resReqReduceScatterDpu));
    CHK_RET(GatherDpuTempAlg->CalcRes(comm, param, topoInfo, resReqGatherDpu));
    CHK_RET(GatherMesh1DTempAlg->CalcRes(comm, param, topoInfo, resReqGatherMesh1D));

    // step1、2、3、4为串行，因此slaveThread和对应notify可以复用
    resourceRequest.slaveThreadNum
        = std::max(resReqReduceScatterMesh1D.slaveThreadNum, resReqReduceScatterDpu.slaveThreadNum);
    resourceRequest.notifyNumPerThread
        = std::max(resReqReduceScatterMesh1D.notifyNumPerThread, resReqReduceScatterDpu.notifyNumPerThread);
    resourceRequest.notifyNumOnMainThread
        = std::max(resReqReduceScatterMesh1D.notifyNumOnMainThread, resReqReduceScatterDpu.notifyNumOnMainThread);

    resourceRequest.channels = {resReqReduceScatterMesh1D.channels[0], resReqReduceScatterDpu.channels[0]};
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

    if (algHierarchyInfo_.infos.empty() || algHierarchyInfo_.infos.size() < TOPO_LEVEL_NUM_2
        || algHierarchyInfo_.infos[0][0].empty() || algHierarchyInfo_.infos[1][0].empty()) {
        HCCL_ERROR("[InsV2ReduceSequenceExecutor][Orchestrate] invalid algHierarchyInfo infos.");
        return HCCL_E_PARA;
    }
    if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix
        && resCtx.algHierarchyInfo.infos[0].size() > 1) {
        intraHierarchyInfo_ = {resCtx.algHierarchyInfo.infos[0][1]};
        interHierarchyInfo_ = resCtx.algHierarchyInfo.infos[1];
    } else {
        intraHierarchyInfo_ = resCtx.algHierarchyInfo.infos[0];
        interHierarchyInfo_ = resCtx.algHierarchyInfo.infos[1];
    }
    rankSizeLevel0_ = intraHierarchyInfo_[0].size();
    rankSizeLevel1_ = interHierarchyInfo_[0].size();
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;

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
    // 非整除余量尾 staging（仅 RHD 实例）：CCL 末端预留双 rankAlign tile，
    // 每轮只跑 totalRankAlign 整除前缀，余量单独一轮 staged
    AlgResourceCtxSerializable loopResCtx = resCtx;
    ReduceRhdTail rhdTail{};
    if constexpr (ReduceSeqIsRhdTemplate<InsAlgTemplate1>::value) {
        const u32 tailRankAlign = static_cast<u32>(rankSizeLevel0_ * rankSizeLevel1_);
        CHK_RET(ReduceRhdPrepareTailWorkspace(dataCount_, tailRankAlign, dataTypeSize_, loopResCtx.cclMem, rhdTail));
    }
    // 将ccl-buffer分成ccl-in和ccl-out 2部分区分使用（用收缩后的 loopResCtx）
    void* cclInAddr = loopResCtx.cclMem.addr;
    HcclMem cclInMem = {loopResCtx.cclMem.type, cclInAddr, loopResCtx.cclMem.size / 2};
    void* cclOutAddr = static_cast<void*>(static_cast<s8*>(loopResCtx.cclMem.addr) + loopResCtx.cclMem.size / 2);
    HcclMem cclOutMem = {loopResCtx.cclMem.type, cclOutAddr, loopResCtx.cclMem.size / 2};

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
        = std::make_shared<InsAlgTemplate0>(param, myRank_, intraHierarchyInfo_);

    // 声明框间ReduceScatterDpu的templateargs
    TemplateDataParams tempAlgParamsReduceScatterDpu;
    tempAlgParamsReduceScatterDpu.buffInfo.inputPtr = cclOutMem.addr;
    // RHD DpuInter：RS_L1 终态须留在 CCL 供 AG_L1 装载消费（Reduce 是 RS+AG 组合，
    // 非独立 RS 的直出 user）——绑 ccl-out
    tempAlgParamsReduceScatterDpu.buffInfo.outputPtr = cclOutMem.addr;
    tempAlgParamsReduceScatterDpu.buffInfo.hcclBuff = cclInMem;
    tempAlgParamsReduceScatterDpu.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsReduceScatterDpu.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsReduceScatterDpu.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsReduceScatterDpu.root = param.root;

    // 构建框间ReduceScatterDpu的template
    std::shared_ptr<InsAlgTemplate1> algTemplateReduceScatterDpu
        = std::make_shared<InsAlgTemplate1>(param, myRank_, interHierarchyInfo_);

    // 声明框间GatherDpu的templateargs，ccl-out搬运到ccl-out
    TemplateDataParams tempAlgParamsGatherDpu;
    tempAlgParamsGatherDpu.buffInfo.inputPtr = cclOutMem.addr;
    tempAlgParamsGatherDpu.buffInfo.outputPtr = cclOutMem.addr;
    tempAlgParamsGatherDpu.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    if constexpr (ReduceSeqIsRhdTemplate<InsAlgTemplate1>::value) {
        // RHD：hcclBuff 绑 cclMem 整块视图（网格基址为 CCL 绝对偏移 ccl/2）
        tempAlgParamsGatherDpu.buffInfo.hcclBuff = loopResCtx.cclMem;
    } else {
        tempAlgParamsGatherDpu.buffInfo.hcclBuff = cclInMem;
    }
    tempAlgParamsGatherDpu.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsGatherDpu.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsGatherDpu.root = param.root;

    // 构建框间GatherDpu的template
    std::shared_ptr<InsAlgTemplate2> algTemplateGatherDpu
        = std::make_shared<InsAlgTemplate2>(param, myRank_, interHierarchyInfo_);

    // 声明框内GatherMesh1D的templateargs，ccl-out搬运到user-out
    TemplateDataParams tempAlgParamsGatherMesh1D;
    tempAlgParamsGatherMesh1D.buffInfo.inputPtr = cclOutMem.addr;
    tempAlgParamsGatherMesh1D.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    // Reduce 的 recvBuf 仅 root 有效：RHD 的 L0 收尾为 AllGather detour（全 rank 有终写），
    // 非 root 的终写重定向到 CCL，交换相位仍全员参与
    if constexpr (ReduceSeqIsRhdTemplate<InsAlgTemplate1>::value) {
        if (myRank_ != param.root) {
            tempAlgParamsGatherMesh1D.buffInfo.outputPtr = resCtx.cclMem.addr;
            tempAlgParamsGatherMesh1D.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
        } else {
            tempAlgParamsGatherMesh1D.buffInfo.outputPtr = param.outputPtr;
            tempAlgParamsGatherMesh1D.buffInfo.outBuffType = BufferType::OUTPUT;
        }
    } else {
        tempAlgParamsGatherMesh1D.buffInfo.outputPtr = param.outputPtr;
        tempAlgParamsGatherMesh1D.buffInfo.outBuffType = BufferType::OUTPUT;
    }
    tempAlgParamsGatherMesh1D.buffInfo.hcclBuff = cclInMem;
    tempAlgParamsGatherMesh1D.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsGatherMesh1D.root = param.root;

    // 构建框内GatherMesh1D的template
    std::shared_ptr<InsAlgTemplate3> algTemplateGatherMesh1D
        = std::make_shared<InsAlgTemplate3>(param, myRank_, intraHierarchyInfo_);

    // 构造框内ReduceScatterMesh1D的template资源
    TemplateResource templateResourceReduceScatterMesh1D;
    SetTemplateCommResource(templateResourceReduceScatterMesh1D, remoteRankToChannelInfo_[0], resCtx);
    // 构造框间ReduceScatterDpu的template资源
    TemplateResource templateResourceReduceScatterDpu;
    SetTemplateCommResource(templateResourceReduceScatterDpu, remoteRankToChannelInfo_[1], resCtx);
    templateResourceReduceScatterDpu.isDpuThreadEnabled = resCtx.isDpuThreadEnabled;
    templateResourceReduceScatterDpu.dpuThread = resCtx.dpuThread;
    templateResourceReduceScatterDpu.dpuThreadNotifyIdxOnMain = resCtx.notifyNumOnMainThread + 1;
    templateResourceReduceScatterDpu.isHcommChannelDrainSupported = resCtx.isHcommChannelDrainSupported;
    // 构造框间GatherDpu的template资源
    TemplateResource templateResourceGatherDpu;
    SetTemplateCommResource(templateResourceGatherDpu, remoteRankToChannelInfo_[1], resCtx);
    templateResourceGatherDpu.isDpuThreadEnabled = resCtx.isDpuThreadEnabled;
    templateResourceGatherDpu.dpuThread = resCtx.dpuThread;
    templateResourceGatherDpu.dpuThreadNotifyIdxOnMain = resCtx.notifyNumOnMainThread + 1;
    templateResourceGatherDpu.isHcommChannelDrainSupported = resCtx.isHcommChannelDrainSupported;
    // 构造框内GatherMesh1D的template资源
    TemplateResource templateResourceGatherMesh1D;
    SetTemplateCommResource(templateResourceGatherMesh1D, remoteRankToChannelInfo_[0], resCtx);

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
    u64 processedDataCount = 0;
    // ---- RHD 二级 DPU 专属前置 ----
    if constexpr (ReduceSeqIsRhdTemplate<InsAlgTemplate1>::value) {
        // 对称内存路径不支持（AG RHDDPU remote 直写防御性拒绝），显式拒绝走重选
        CHK_PRT_RET(
            param.supportSymmetricMemory,
            HCCL_ERROR(
                "[%s] algName[%s] RHD sequence not support symmetric memory, fallback by reselecting HCCL_ALGO.",
                __func__, param.algName),
            HcclResult::HCCL_E_NOT_SUPPORT);
        // 执行期按通道映射刷新 channelsPerRank（GetThreadNum 随之缩放）；
        // L0 双平面按 locType==DEVICE 过滤，跨 server 数据流由 RS_L1/GA_L1 承担
        std::map<u32, std::vector<ChannelInfo>> deviceOnlyLevel0;
        for (const auto& rankChannelPair : remoteRankToChannelInfo_[0]) {
            std::vector<ChannelInfo> kept;
            for (const auto& channel : rankChannelPair.second) {
                if (channel.locationType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE) {
                    kept.push_back(channel);
                }
            }
            if (!kept.empty()) {
                deviceOnlyLevel0.emplace(rankChannelPair.first, std::move(kept));
            }
        }
        // 设备侧过滤后为空（全 HOST 链路的仿真形态）时回退原始通道表
        const std::map<u32, std::vector<ChannelInfo>>& level0ChannelsForDetour
            = deviceOnlyLevel0.empty() ? remoteRankToChannelInfo_[0] : deviceOnlyLevel0;
        if (rankSizeLevel0_ > 1) {
            CHK_RET(algTemplateReduceScatterMesh1D->SetchannelsPerRank(level0ChannelsForDetour));
            CHK_RET(algTemplateGatherMesh1D->SetchannelsPerRank(level0ChannelsForDetour));
        }
        if (rankSizeLevel1_ > 1) {
            CHK_RET(algTemplateReduceScatterDpu->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
            CHK_RET(algTemplateGatherDpu->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
        }
        // GA_L0 线程池裁剪到模板 GetThreadNum，保持 PreSyncInterThreads
        // 的 subThreads 与 notifyIdx 一致
        const u64 rhdGaL0ThreadNum = algTemplateGatherMesh1D->GetThreadNum();
        if (templateResourceGatherMesh1D.threads.size() > rhdGaL0ThreadNum) {
            templateResourceGatherMesh1D.threads.resize(static_cast<size_t>(rhdGaL0ThreadNum));
        }
        // maxCountPerLoop 对齐到 d0·d1 的整数倍（均匀槽契约）
        const u32 totalRankAlign = static_cast<u32>(rankSizeLevel0_ * rankSizeLevel1_);
        maxCountPerLoop = maxCountPerLoop / totalRankAlign * totalRankAlign;

        CHK_PRT_RET(
            maxCountPerLoop == 0,
            HCCL_ERROR("[%s] RHD sequence maxCountPerLoop aligns to 0 (totalRankAlign[%u])", __func__, totalRankAlign),
            HcclResult::HCCL_E_INTERNAL);
    }
    u64 loop = 0;
    while (processedDataCount < dataCount_) {
        u64 currDataCount;
        const u64 remaining = dataCount_ - processedDataCount;
        u64 validDataCount = 0;
        bool paddedTailLoop = false;
        if constexpr (ReduceSeqIsRhdTemplate<InsAlgTemplate1>::value) {
            // 每轮只跑 totalRankAlign 整除前缀；末尾余量 staged 进隔离 tile，仅回拷有效字节
            const u32 rhdTotalAlign = static_cast<u32>(rankSizeLevel0_ * rankSizeLevel1_);
            validDataCount = ReduceRhdSelectLoopCount(remaining, maxCountPerLoop, rhdTotalAlign);
            currDataCount = std::max(validDataCount, static_cast<u64>(rhdTotalAlign));
            paddedTailLoop = validDataCount != currDataCount;
            if (paddedTailLoop) {
                CHK_RET(ReduceRhdStageTail(param, rhdTail, processedDataCount, validDataCount, resCtx.threads[0]));
            }
        } else {
            currDataCount = (remaining <= maxCountPerLoop) ? remaining : maxCountPerLoop;
        }

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
                "[InsV2ReduceSequenceExecutor][tempAlgParamsReduceScatterMesh1D] slice num[%zu] is not equal to rank "
                "size[%u].",
                tempAlgParamsReduceScatterMesh1D.allRankSliceSize.size(), rankSizeLevel0_),
            HcclResult::HCCL_E_INTERNAL);
        tempAlgParamsReduceScatterMesh1D.sliceSize = 0; // 没用到，template里面用SplitData算了
        tempAlgParamsReduceScatterMesh1D.tailSize = 0;  // 没用到
        // 这里的stride当成传统意义上的stride间隔
        tempAlgParamsReduceScatterMesh1D.inputSliceStride = 0;  // 没用到
        tempAlgParamsReduceScatterMesh1D.outputSliceStride = 0; // 没用到

        HCCL_INFO(
            "[InsV2ReduceSequenceExecutor] loop [%llu] tempAlgParamsReduceScatterMesh.inputSliceStride [%llu],"
            "tempAlgParamsReduceScatterMesh.outputSliceStride [%llu] tempAlgParamsReduceScatterMesh.sliceSize [%llu]",
            loop, tempAlgParamsReduceScatterMesh1D.inputSliceStride, tempAlgParamsReduceScatterMesh1D.outputSliceStride,
            tempAlgParamsReduceScatterMesh1D.sliceSize);
        HCCL_INFO(
            "[InsV2ReduceSequenceExecutor] loop [%llu] tempAlgParamsReduceScatterMesh.buffInfo.inBuffBaseOff [%llu],"
            "tempAlgParamsReduceScatterMesh.buffInfo.outBuffBaseOff [%llu]",
            loop, tempAlgParamsReduceScatterMesh1D.buffInfo.inBuffBaseOff,
            tempAlgParamsReduceScatterMesh1D.buffInfo.outBuffBaseOff);
        if constexpr (ReduceSeqIsRhdTemplate<InsAlgTemplate1>::value) {
            // sliceSize=ISS=chunkSize（本 rank L0 份额），OSS=0（输出为本 rank chunk），
            // 切分表由通用段 SplitData 产出；尾轮输入重定向 staging tile
            if (paddedTailLoop) {
                tempAlgParamsReduceScatterMesh1D.buffInfo.inputPtr = rhdTail.storage.addr;
                tempAlgParamsReduceScatterMesh1D.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
                tempAlgParamsReduceScatterMesh1D.buffInfo.inBuffBaseOff = rhdTail.inputOffset;
            }
            const u64 rhdChunkSize = currDataCount * dataTypeSize_ / rankSizeLevel0_;
            tempAlgParamsReduceScatterMesh1D.sliceSize = rhdChunkSize;
            tempAlgParamsReduceScatterMesh1D.tailSize = rhdChunkSize;
            tempAlgParamsReduceScatterMesh1D.inputSliceStride = rhdChunkSize;
            tempAlgParamsReduceScatterMesh1D.outputSliceStride = 0;
            tempAlgParamsReduceScatterMesh1D.repeatNum = 1;
            tempAlgParamsReduceScatterMesh1D.inputRepeatStride = 0;
            tempAlgParamsReduceScatterMesh1D.outputRepeatStride = 0;
        } else {
            tempAlgParamsReduceScatterMesh1D.repeatNum = 1;
            tempAlgParamsReduceScatterMesh1D.inputRepeatStride = 0;
            tempAlgParamsReduceScatterMesh1D.outputRepeatStride = 0;
        }
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplateReduceScatterMesh1D->KernelRun(
            param, tempAlgParamsReduceScatterMesh1D, templateResourceReduceScatterMesh1D));

        // ----------- 框间ReduceScatterDpu数据搬运 -----------
        // 框间的数据偏移和搬运量计算
        tempAlgParamsReduceScatterDpu.count
            = tempAlgParamsReduceScatterMesh1D.allRankProcessedDataCount.at(rankIdxLevel0_);
        if constexpr (ReduceSeqIsRhdTemplate<InsAlgTemplate1>::value) {
            // subSlice = chunk/d1 均匀；输入=本 rank chunk（ccl-out[0]），
            // 输出=本 rank 子份额，scratch=ccl-in（acc+recv 双区）
            const u64 chunkCount = tempAlgParamsReduceScatterDpu.count;
            const u64 subSliceCount = chunkCount / rankSizeLevel1_;
            const u64 subSliceSize = subSliceCount * dataTypeSize_;
            tempAlgParamsReduceScatterDpu.sliceSize = subSliceSize;
            tempAlgParamsReduceScatterDpu.tailSize = subSliceSize;
            // ISS=OSS=subSliceSize：模板按 j·ISS 自推导均匀表
            tempAlgParamsReduceScatterDpu.inputSliceStride = subSliceSize;
            tempAlgParamsReduceScatterDpu.outputSliceStride = subSliceSize;
            tempAlgParamsReduceScatterDpu.buffInfo.inBuffBaseOff = 0;   // 输入 = ccl-out[0] 本 rank chunk
            tempAlgParamsReduceScatterDpu.buffInfo.outBuffBaseOff = 0;  // 输出 = ccl-out[0] 本 rank 子份额
            tempAlgParamsReduceScatterDpu.buffInfo.hcclBuffBaseOff = 0; // scratch = ccl-in 基址区
            tempAlgParamsReduceScatterDpu.repeatNum = 1;
            tempAlgParamsReduceScatterDpu.inputRepeatStride = 0;
            tempAlgParamsReduceScatterDpu.outputRepeatStride = 0;
            // 跳过 POST 与被合并方终态回搬：子份额留在 acc 槽，由桥接与 AG 相位承担
            tempAlgParamsReduceScatterDpu.rhdSkipPost = true;
            // allRank* 由模板入口按 ISS 自推导，不预填
            CHK_RET(algTemplateReduceScatterDpu->KernelRun(
                param, tempAlgParamsReduceScatterDpu, templateResourceReduceScatterDpu));
            // 代表侧桥接：被合并方终态子份额（acc 槽 2t+1）→ 代表 AG 网格 owner 槽（t+r）；
            // 2^N 层无配对边，no-op
            RhdPart1Info rhdBridgePart1{};
            CHK_RET(CalcRhdPart1Info(
                static_cast<u32>(rankSizeLevel1_), static_cast<u32>(rankIdxLevel1_), INVALID_VALUE_RANKID,
                rhdBridgePart1));
            if (rhdBridgePart1.inPart1 && rhdBridgePart1.isRepresentative) {
                const u64 bridgeSrcOff = (rhdBridgePart1.unitStart + 1) * subSliceSize;
                const u64 bridgeDstOff = loopResCtx.cclMem.size / 2 + rhdBridgePart1.pairedAlgRank * subSliceSize;
                CHK_RET(LocalCopy(
                    resCtx.threads[0], DataSlice(loopResCtx.cclMem.addr, bridgeSrcOff, subSliceSize, subSliceCount),
                    DataSlice(loopResCtx.cclMem.addr, bridgeDstOff, subSliceSize, subSliceCount)));
            }
        } else if (
            tempAlgParamsReduceScatterDpu.count != 0) { // 如果卡里没有数据，不参与框间，让该框有数据的其他卡参与跨框
            tempAlgParamsReduceScatterDpu.buffInfo.inBuffBaseOff = 0;   // ccl-out偏移量，每次更新，所以是0
            tempAlgParamsReduceScatterDpu.buffInfo.outBuffBaseOff = 0;  // ccl-out偏移量，每次更新，所以是0
            tempAlgParamsReduceScatterDpu.buffInfo.hcclBuffBaseOff = 0; // ccl-in
            tempAlgParamsReduceScatterDpu.sliceSize = 0;
            tempAlgParamsReduceScatterDpu.tailSize = 0;
            CHK_RET(SplitData(tempAlgParamsReduceScatterDpu.count, rankSizeLevel1_, tempAlgParamsReduceScatterDpu));
            CHK_PRT_RET(
                tempAlgParamsReduceScatterDpu.allRankSliceSize.size() != rankSizeLevel1_,
                HCCL_ERROR(
                    "[InsV2ReduceSequenceExecutor][tempAlgParamsReduceScatterDpu] slice num[%zu] is not "
                    "equal to rank size[%u].",
                    tempAlgParamsReduceScatterDpu.allRankSliceSize.size(), rankSizeLevel1_),
                HcclResult::HCCL_E_INTERNAL);
            // 这里的stride当成传统意义上的stride 间隔
            tempAlgParamsReduceScatterDpu.inputSliceStride = 0;  // 没用到
            tempAlgParamsReduceScatterDpu.outputSliceStride = 0; // 没用到
            //
            HCCL_INFO(
                "[InsV2ReduceSequenceExecutor] loop [%llu] tempAlgParamsReduceScatterDpu.inputSliceStride [%llu],"
                "tempAlgParamsReduceScatterDpu.outputSliceStride [%llu] "
                "tempAlgParamsReduceScatterDpu.sliceSize [%llu]",
                loop, tempAlgParamsReduceScatterDpu.inputSliceStride, tempAlgParamsReduceScatterDpu.outputSliceStride,
                tempAlgParamsReduceScatterDpu.sliceSize);
            HCCL_INFO(
                "[InsV2ReduceSequenceExecutor] loop [%llu] "
                "tempAlgParamsReduceScatterDpu.buffInfo.inBuffBaseOff [%llu],"
                "tempAlgParamsReduceScatterDpu.buffInfo.outBuffBaseOff [%llu]",
                loop, tempAlgParamsReduceScatterDpu.buffInfo.inBuffBaseOff,
                tempAlgParamsReduceScatterDpu.buffInfo.outBuffBaseOff);
            // 不需要重复
            tempAlgParamsReduceScatterDpu.repeatNum = 1;
            tempAlgParamsReduceScatterDpu.inputRepeatStride = 0;
            tempAlgParamsReduceScatterDpu.outputRepeatStride = 0;
            // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
            CHK_RET(algTemplateReduceScatterDpu->KernelRun(
                param, tempAlgParamsReduceScatterDpu, templateResourceReduceScatterDpu));
        }

        // ----------- 框间GatherDpu数据搬运 -----------
        // 框间的数据偏移和搬运量计算
        tempAlgParamsGatherDpu.count = tempAlgParamsReduceScatterMesh1D.allRankProcessedDataCount.at(rankIdxLevel0_);
        if constexpr (ReduceSeqIsRhdTemplate<InsAlgTemplate1>::value) {
            // subSlice = chunk/d1 均匀；装载 ISS=0 读 ccl-out[0] 本 rank 子份额，
            // 交换网格 [owner·SS]
            const u64 chunkCount = tempAlgParamsGatherDpu.count;
            const u64 subSliceCount = chunkCount / rankSizeLevel1_;
            const u64 subSliceSize = subSliceCount * dataTypeSize_;
            tempAlgParamsGatherDpu.count = subSliceCount;
            tempAlgParamsGatherDpu.sliceSize = subSliceSize;
            tempAlgParamsGatherDpu.tailSize = subSliceSize;
            tempAlgParamsGatherDpu.inputSliceStride = 0; // 装载源 = ccl-out[0] 本 rank 子份额（无 alg-rank 项）
            tempAlgParamsGatherDpu.outputSliceStride = subSliceSize; // 网格 [owner·SS]
            // 输入与 hcclBuff 同基址视图（cclMem+ccl/2），owner-0 时 inOff==scOff 使模板
            // 原装守卫跳过恒等装载
            tempAlgParamsGatherDpu.buffInfo.inputPtr = loopResCtx.cclMem.addr;
            tempAlgParamsGatherDpu.buffInfo.inBuffBaseOff = loopResCtx.cclMem.size / 2;
            // CCL 别名绑定：outBuffType=HCCL_BUFFER 时 PostLocalCopy 跳过，终态即交换网格；
            // 网格基址为 CCL 绝对偏移 ccl/2（对端 remoteCclMem 同偏移对称，收发同偏移直写）
            tempAlgParamsGatherDpu.buffInfo.outBuffBaseOff = loopResCtx.cclMem.size / 2;
            tempAlgParamsGatherDpu.buffInfo.hcclBuffBaseOff = loopResCtx.cclMem.size / 2;
            // RS_L1 终态已在网格 owner 槽，跳过 PRE 装载避免覆盖
            tempAlgParamsGatherDpu.skipPrePhase = true;
            tempAlgParamsGatherDpu.repeatNum = 1;
            tempAlgParamsGatherDpu.inputRepeatStride = 0;
            tempAlgParamsGatherDpu.outputRepeatStride = 0;
            CHK_RET(algTemplateGatherDpu->KernelRun(param, tempAlgParamsGatherDpu, templateResourceGatherDpu));
        } else if (tempAlgParamsReduceScatterDpu.count != 0) {
            tempAlgParamsGatherDpu.buffInfo.inBuffBaseOff = 0;   // ccl-out偏移量，每次更新，所以是0
            tempAlgParamsGatherDpu.buffInfo.outBuffBaseOff = 0;  // ccl-out偏移量，每次更新，所以是0
            tempAlgParamsGatherDpu.buffInfo.hcclBuffBaseOff = 0; // ccl-in
            tempAlgParamsGatherDpu.allRankDispls = tempAlgParamsReduceScatterDpu.allRankDispls;
            tempAlgParamsGatherDpu.allRankSliceSize = tempAlgParamsReduceScatterDpu.allRankSliceSize;
            tempAlgParamsGatherDpu.allRankProcessedDataCount = tempAlgParamsReduceScatterDpu.allRankProcessedDataCount;

            tempAlgParamsGatherDpu.sliceSize = 0;
            tempAlgParamsGatherDpu.tailSize = 0;
            // 这里的stride当成传统意义上的stride 间隔
            tempAlgParamsGatherDpu.inputSliceStride = 0;
            tempAlgParamsGatherDpu.outputSliceStride = 0;
            // 取 RS 相位 allRank* 首槽份额作 stride 契约
            tempAlgParamsGatherDpu.sliceSize = tempAlgParamsReduceScatterDpu.allRankSliceSize.front();
            tempAlgParamsGatherDpu.tailSize = tempAlgParamsReduceScatterDpu.allRankSliceSize.back();
            tempAlgParamsGatherDpu.inputSliceStride = tempAlgParamsGatherDpu.sliceSize;
            tempAlgParamsGatherDpu.outputSliceStride = tempAlgParamsGatherDpu.sliceSize;

            HCCL_INFO(
                "[InsV2ReduceSequenceExecutor] loop [%llu] tempAlgParamsGatherDpu.inputSliceStride [%llu],"
                "tempAlgParamsGatherDpu.outputSliceStride [%llu] tempAlgParamsGatherDpu.sliceSize [%llu]",
                loop, tempAlgParamsGatherDpu.inputSliceStride, tempAlgParamsGatherDpu.outputSliceStride,
                tempAlgParamsGatherDpu.sliceSize);
            HCCL_INFO(
                "[InsV2ReduceSequenceExecutor] loop [%llu] tempAlgParamsGatherDpu.buffInfo.inBuffBaseOff [%llu],"
                "tempAlgParamsGatherDpu.buffInfo.outBuffBaseOff [%llu]",
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
        if constexpr (ReduceSeqIsRhdTemplate<InsAlgTemplate1>::value) {
            // 输入=AG_L1 的 ccl-out 网格（本 rank chunk 在 [0]）：sliceSize=OSS=chunkSize、
            // ISS=0，输出网格 [owner·chunkSize]
            tempAlgParamsGatherMesh1D.buffInfo.inputPtr = loopResCtx.cclMem.addr;
            tempAlgParamsGatherMesh1D.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
            tempAlgParamsGatherMesh1D.buffInfo.inBuffBaseOff = loopResCtx.cclMem.size / 2;
            tempAlgParamsGatherMesh1D.buffInfo.hcclBuff = loopResCtx.cclMem;
            tempAlgParamsGatherMesh1D.buffInfo.hcclBuffBaseOff = loopResCtx.cclMem.size / 2; // 与 inBuffBaseOff 同址
            tempAlgParamsGatherMesh1D.sliceSize = currDataCount * dataTypeSize_ / rankSizeLevel0_; // chunkSize
            tempAlgParamsGatherMesh1D.tailSize = tempAlgParamsGatherMesh1D.sliceSize;
            tempAlgParamsGatherMesh1D.inputSliceStride = 0; // 本 rank chunk 在网格 [0]
            tempAlgParamsGatherMesh1D.outputSliceStride = tempAlgParamsGatherMesh1D.sliceSize; // 网格 [owner·chunkSize]
            tempAlgParamsGatherMesh1D.repeatNum = 1;
            tempAlgParamsGatherMesh1D.inputRepeatStride = 0;
            tempAlgParamsGatherMesh1D.outputRepeatStride = 0;
        } else {
            tempAlgParamsGatherMesh1D.buffInfo.inBuffBaseOff = 0; // CCL-OUT
            tempAlgParamsGatherMesh1D.sliceSize = 0;
            tempAlgParamsGatherMesh1D.tailSize = 0;
            // 这里的stride当成传统意义上的stride间隔
            tempAlgParamsGatherMesh1D.inputSliceStride = 0;
            tempAlgParamsGatherMesh1D.outputSliceStride = 0;
        }
        const bool rhdNonRootGather = ReduceSeqIsRhdTemplate<InsAlgTemplate1>::value && myRank_ != param.root;
        tempAlgParamsGatherMesh1D.buffInfo.outBuffBaseOff
            = rhdNonRootGather ? 0 : processedDataCount * dataTypeSize_; // root=USR-OUT；非 root 落 CCL 基址
        if constexpr (ReduceSeqIsRhdTemplate<InsAlgTemplate1>::value) {
            if (paddedTailLoop && !rhdNonRootGather) {
                tempAlgParamsGatherMesh1D.buffInfo.outputPtr = rhdTail.storage.addr;
                tempAlgParamsGatherMesh1D.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
                tempAlgParamsGatherMesh1D.buffInfo.outBuffBaseOff = rhdTail.outputOffset;
            }
        }
        tempAlgParamsGatherMesh1D.allRankDispls = tempAlgParamsReduceScatterMesh1D.allRankDispls;
        tempAlgParamsGatherMesh1D.allRankSliceSize = tempAlgParamsReduceScatterMesh1D.allRankSliceSize;
        tempAlgParamsGatherMesh1D.allRankProcessedDataCount
            = tempAlgParamsReduceScatterMesh1D.allRankProcessedDataCount;

        HCCL_INFO(
            "[InsV2ReduceSequenceExecutor] loop [%llu] tempAlgParamsGatherMesh.inputSliceStride [%llu],"
            "tempAlgParamsGatherMesh.outputSliceStride [%llu] tempAlgParamsGatherMesh.sliceSize [%llu]",
            loop, tempAlgParamsGatherMesh1D.inputSliceStride, tempAlgParamsGatherMesh1D.outputSliceStride,
            tempAlgParamsGatherMesh1D.sliceSize);
        HCCL_INFO(
            "[InsV2ReduceSequenceExecutor] loop [%llu] tempAlgParamsGatherMesh.buffInfo.inBuffBaseOff [%llu],"
            "tempAlgParamsGatherMesh.buffInfo.outBuffBaseOff [%llu]",
            loop, tempAlgParamsGatherMesh1D.buffInfo.inBuffBaseOff, tempAlgParamsGatherMesh1D.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsGatherMesh1D.repeatNum = 1;
        tempAlgParamsGatherMesh1D.inputRepeatStride = 0;
        tempAlgParamsGatherMesh1D.outputRepeatStride = 0;
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplateGatherMesh1D->KernelRun(param, tempAlgParamsGatherMesh1D, templateResourceGatherMesh1D));

        if (paddedTailLoop) {
            // 尾轮：GA_L0 输出已写入 tail 的 output tile，root 直拷 valid 前缀到 user
            if (!rhdNonRootGather) {
                const u64 validBytes = validDataCount * dataTypeSize_;
                CHK_RET(LocalCopy(
                    resCtx.threads[0],
                    DataSlice(rhdTail.storage.addr, rhdTail.outputOffset, validBytes, validDataCount),
                    DataSlice(param.outputPtr, processedDataCount * dataTypeSize_, validBytes, validDataCount)));
            }
            processedDataCount += validDataCount;
        } else {
            processedDataCount += currDataCount;
        }
        loop++;
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
            "[InsV2ReduceSequenceExecutor] SliceInfo: offset[%llu] size[%llu] count[%llu]",
            tempAlgParams.allRankDispls.at(i), tempAlgParams.allRankSliceSize.at(i),
            tempAlgParams.allRankProcessedDataCount.at(i));
    }

    return HcclResult::HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_FOUR_TEMPS(
    HcclCMDType::HCCL_CMD_REDUCE, DpuReduceSequenceMeshMesh, InsV2ReduceSequenceExecutor, TopoMatchTwoLevel,
    InsTempReduceScatterMesh1DIntra, InsTempReduceScatterMesh1dDpuInter, InsTempGatherDpuInter,
    InsTempGatherMesh1dIntra);
REGISTER_ALG_ATTRS(
    DpuReduceSequenceMeshMesh, topo.isSupportLevel0PcieMix = true; topo.minTopoLevelNum = TOPO_LEVEL_NUM_2;
    topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; topo.isHostDpuOnly = true; topo.isSupportLevel1Nhr = true;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS | LEVEL0_TOPO_CLOS;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        if (topo->level0Topo != Level0Shape::MESH_1D_CLOS || topo->level0PcieMix || topo->deviceNumPerModule == 1) {
            return true;
        }
        // UBX
        return topo->Level1Nhr || AutoSelectorBase::IsLayerAllConnetedWithTopo(topo, 0, CommTopo::COMM_TOPO_1DMESH);
    };
    op.opCustomCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails* topo) -> bool {
        return !CanUseDpuInlineReduce(opParam, topo);
    };);
REGISTER_EXECUTOR_BY_FOUR_TEMPS(
    HcclCMDType::HCCL_CMD_REDUCE, DpuReduceSequenceMeshNHR, InsV2ReduceSequenceExecutor, TopoMatchTwoLevel,
    InsTempReduceScatterMesh1DIntra, InsTempReduceScatterNhrDpu, InsTempGatherDpuInter, InsTempGatherMesh1dIntra);
REGISTER_ALG_ATTRS(
    DpuReduceSequenceMeshNHR, topo.isSupportLevel0PcieMix = true; topo.minTopoLevelNum = TOPO_LEVEL_NUM_2;
    topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; topo.isHostDpuOnly = true; topo.isSupportLevel1Nhr = true;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS | LEVEL0_TOPO_CLOS;
    topo.supportDevTypes = {HcclDevType::DEV_TYPE_950, HcclDevType::DEV_TYPE_960};
    topo.topoPrecondCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        if (!CheckDpuInlineReduceTopo(topo)) {
            return false;
        }
        if (topo->level0Topo != Level0Shape::MESH_1D_CLOS || topo->level0PcieMix || topo->deviceNumPerModule == 1) {
            return true;
        }
        return topo->Level1Nhr || AutoSelectorBase::IsLayerAllConnetedWithTopo(topo, 0, CommTopo::COMM_TOPO_1DMESH);
    };
    op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_64BIT;);
// NDA序列算法: 参考DpuReduceSequenceMeshNHR的Intra/Inter四级流水骨架,
// 框间inter模板与已合入的AR NDA保持一致(InsTempReduceScatterMesh1dNdaInter +
// InsTempAllGatherNhrNdaInter), 控制流保留在AICPU_TS执行。
REGISTER_EXECUTOR_BY_FOUR_TEMPS(
    HcclCMDType::HCCL_CMD_REDUCE, DpuReduceSequenceMeshNHRNDA, InsV2ReduceSequenceExecutor, TopoMatchTwoLevel,
    InsTempReduceScatterMesh1DIntra, InsTempReduceScatterMesh1dNdaInter, InsTempAllGatherNhrNdaInter,
    InsTempGatherMesh1dIntra);
REGISTER_ALG_ATTRS(DpuReduceSequenceMeshNHRNDA, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS;
                   topo.minTopoLevelNum = TOPO_LEVEL_NUM_2; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3;
                   topo.isHostDpuOnly = true; op.unsupportedDataTypes = UNSUPPORTED_64BIT; op.isSupportInplace = false);

// RHD（Recursive Halving-Doubling）二级（Dpu 引擎）：L0 z 轴绕路 + L1 host DPU RHD inter 相位
// 模板，模板实参 = RS 下沉链 L0/L1 + Gather 回升链 L1/L0。
// RHD 仅经 HCCL_ALGO 显式配置可选（preferred-only，topoCustomCheck 恒 false），默认选路零变化。
REGISTER_EXECUTOR_BY_FOUR_TEMPS(
    HcclCMDType::HCCL_CMD_REDUCE, DpuReduceSequenceMeshConcurRHD, InsV2ReduceSequenceExecutor, TopoMatchTwoLevel,
    InsTempReduceScatterMesh1DZAxisDetour, InsTempReduceScatterRhdDpuInter, InsTempAllGatherRHDDPU,
    InsTempAllGatherMesh1D1DZAxisDetour);
REGISTER_ALG_ATTRS(
    DpuReduceSequenceMeshConcurRHD, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D; topo.isHostDpuOnly = true;
    topo.isSupportLevel1Nhr = true; topo.minTopoLevelNum = TOPO_LEVEL_NUM_2; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2;
    op.unsupportedDataTypes = UNSUPPORTED_64BIT; op.isSupportInplace = false; op.isSupportProd = false;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails*) -> bool {
        return false;
    });

} // namespace ops_hccl
