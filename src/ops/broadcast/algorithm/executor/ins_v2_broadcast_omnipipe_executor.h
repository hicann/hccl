/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_BROADCAST_OMNIPIPE_EXECUTOR_H
#define HCCLV2_INS_V2_BROADCAST_OMNIPIPE_EXECUTOR_H

#include "alg_param.h"
#include "channel.h"
#include "alg_v2_template_base.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "utils.h"
#include "log.h"
#include "sal.h"
#include "config_log.h"
#include "topo_match_base.h"
#include "topo_match_multilevel.h"
#include "topo_match_ubx.h"
#include "topo_match_pcie_mix.h"
#include "topo_match_3_level.h"
#include "omnipipe_scatter_data_slice_calc.h"
#include "omnipipe_data_slice_calc.h"

namespace ops_hccl {

// 三级 aicpu Broadcast OmniPipe 执行器
// 实现思路: Broadcast = Scatter(root 按rank切分下发) + AllGather(所有rank聚合所有分片)
// 三级拓扑: Level0(mesh, 框内) + Level1(NHR, 框间) + Level2(NHR, 跨超节点)
// 模板参数:
//   InsScatterAlgTemplateX/Y/Z: 三级 Scatter 模板
//   InsAgAlgTemplateX/Y/Z:     三级 AllGather 模板
template <
    typename AlgTopoMatch, typename InsScatterAlgTemplateX, typename InsScatterAlgTemplateY,
    typename InsScatterAlgTemplateZ, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
class InsV2BroadcastOmniPipeExecutor : public InsCollAlgBase {
public:
    explicit InsV2BroadcastOmniPipeExecutor();
    ~InsV2BroadcastOmniPipeExecutor() override = default;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult RestoreChannelMap(
        const AlgResourceCtxSerializable& resCtx,
        std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo) const override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

protected:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);

    HcclResult InitCommInfo(
        const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo);

    HcclResult InitExecutorInfo(const OpParam& param, const AlgResourceCtxSerializable& resCtx);

    // 按level计算资源（每级一个主流+若干从流+1个channel）
    HcclResult CalcResLevel(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const std::shared_ptr<InsAlgTemplateBase> tempAlg, AlgResourceRequest& resourceRequest, bool addChannel) const;

    // 为各级 template 分配 thread 并建立同步索引
    HcclResult PrepareResForTemplateLevelScatter(u32 level, const std::shared_ptr<InsAlgTemplateBase>& tempBase);
    HcclResult PrepareResForTemplateLevelAllGather(u32 level, const std::shared_ptr<InsAlgTemplateBase>& tempBase);

    // 构建子通信域 ranks 与 template 实例 (scatter/ag 分别用具体类型指针)
    HcclResult BuildSubCommAndTempMap(
        const OpParam& param, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        std::vector<std::vector<u32>>& subCommRanks0, std::vector<std::vector<u32>>& subCommRanks1,
        std::vector<std::vector<u32>>& subCommRanks2, const TopoInfoWithNetLayerDetails* topoInfo);
    HcclResult BuildSubCommRanks(
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, std::vector<std::vector<u32>>& subCommRanks0,
        std::vector<std::vector<u32>>& subCommRanks1, std::vector<std::vector<u32>>& subCommRanks2,
        const TopoInfoWithNetLayerDetails* topoInfo);
    HcclResult BuildUbxSubCommRanks(
        std::vector<std::vector<u32>>& subCommRanks0, std::vector<std::vector<u32>>& subCommRanks1,
        std::vector<std::vector<u32>>& subCommRanks2, const TopoInfoWithNetLayerDetails* topoInfo);
    void InitRankIndex();
    HcclResult InitRankInfoAndTemp(
        const OpParam& param, std::vector<std::vector<u32>>& subCommRanks0,
        std::vector<std::vector<u32>>& subCommRanks1, std::vector<std::vector<u32>>& subCommRanks2);

    // 初始化各级 TemplateResource / TemplateDataParams
    HcclResult InitTemplateParams(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, std::map<u32, TemplateResource>& tempResMap,
        std::map<u32, TemplateDataParams>& tempAlgParamMap);
    void InitTemplateBufferInfo(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateDataParams& tempAlgParam);
    void InitTemplateParamByLevel(
        u32 templateLevel, u32 hierarchyLevel, const std::vector<std::vector<ThreadHandle>>& levelThreads,
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, std::map<u32, TemplateResource>& tempResMap,
        std::map<u32, TemplateDataParams>& tempAlgParamMap);

    // 单步数据切片信息生成templateParam: userIn -> HCCL_BUFFER
    HcclResult GenTempAlgParamsIn2HCCLBuff(
        TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo, u64 processedDataCount,
        const AlgResourceCtxSerializable& resCtx, const OpParam& param);

    // 单步数据切片信息生成templateParam: HCCL_BUFFER -> HCCL_BUFFER
    HcclResult GenTempAlgParamsHCCLBuff2HCCLBuff(
        TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo, u64 processedDataCount,
        const AlgResourceCtxSerializable& resCtx, const OpParam& param);

    // AllGather/Scatter阶段单步参数（ccl到ccl，仅复用偏移，参考allReduce 3-level）
    HcclResult GenTemplateAlgParamsByDimData(TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo) const;

    // 本地拷贝辅助方法（参考allReduce 3-level DoLocalCopy）
    HcclResult DoLocalCopy(
        const TemplateDataParams& tempAlgParams, const ThreadHandle& thread, const std::vector<u64>& allRankSplitData,
        const std::vector<u64>& curLoopAllRankSplitData) const;

    // 计算 Scatter/AllGather 在 Level0/1/2 的等效带宽
    HcclResult CalcEndpointBandwidth(
        const AlgResourceCtxSerializable& resCtx, std::vector<double>& endpointAttrBwAvgSC,
        std::vector<double>& endpointAttrBwAvgAG);

    // 计算数据切分: 每rank总量、每rank每loop量、单loop最大count、loop次数
    struct LoopSplitData {
        std::vector<u64> allRankSplitData;                       // 每个rank切分的总count
        std::vector<std::vector<u64>> multiLoopAllRankSplitData; // 每个rank每个loop切分的count
        u64 maxCountPerLoop{0};                                  // 每个loop最大count
        u32 loopTimes{0};                                        // loop次数
    };
    struct ScatterLevel01TaskParam {
        u64 root;
        u32 step;
        u32 stepCount;
        bool preferLevel1;
    };
    void SetScatterLevel01Task(const ScatterLevel01TaskParam& taskParam);
    HcclResult CalcLoopSplitData(u64 maxTmpMemSize, u64 root, LoopSplitData& loopSplitData);

    // 初始化OmniPipeSliceParam默认值
    HcclResult InitSliceParam(
        const OpParam& param, const std::vector<u64>& allRankSplitData,
        const std::vector<std::vector<u64>>& multiLoopAllRankSplitData, OmniPipeSliceParam& sliceParam);

    // 每轮loop按需重算 Scatter/AllGather 的 OmniPipeSliceInfo
    HcclResult PrepareSliceInfoForLoop(
        u64 loop, u64 root, const std::vector<u64>& allRankSplitData,
        const std::vector<std::vector<u64>>& multiLoopAllRankSplitData, const std::vector<double>& endpointAttrBwAvgSC,
        const std::vector<double>& endpointAttrBwAvgAG, OmniPipeSliceParam& sliceParam,
        OmniPipeSliceInfo& omniPipeSliceInfoSC, OmniPipeSliceInfo& omniPipeSliceInfoAG);
    HcclResult RunScatterStage(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, u64 processedDataCount,
        OmniPipeSliceInfo& omniPipeSliceInfoSC, const std::vector<double>& endpointAttrBwAvgSC,
        std::map<u32, TemplateResource>& tempResMap, std::map<u32, TemplateDataParams>& tempAlgParamMap);
    HcclResult RunScatterLevel2Step(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, u64 processedDataCount,
        OmniPipeSliceInfo& omniPipeSliceInfoSC, u32 stepZ, std::map<u32, TemplateDataParams>& tempAlgParamMap);
    HcclResult RunScatterLevel01Steps(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, u64 processedDataCount,
        OmniPipeSliceInfo& omniPipeSliceInfoSC, const std::vector<double>& endpointAttrBwAvgSC, u32 stepZ,
        u32 level0StepCountSC, std::map<u32, TemplateResource>& tempResMap,
        std::map<u32, TemplateDataParams>& tempAlgParamMap);
    HcclResult AdaptRootDataForAllGather(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, u64 processedDataCount,
        const std::vector<u64>& allRankSplitData, const std::vector<u64>& curLoopAllRankSplitData);
    HcclResult RunAllGatherStage(
        const OpParam& param, OmniPipeSliceInfo& omniPipeSliceInfoAG, std::map<u32, TemplateResource>& tempResMap,
        std::map<u32, TemplateDataParams>& tempAlgParamMap);
    HcclResult CopyAllGatherResult(
        const OpParam& param, u64 processedDataCount, u64 currDataCount, const std::vector<u64>& allRankSplitData,
        const std::vector<u64>& curLoopAllRankSplitData, TemplateDataParams& tempParamLocalcopy);
    void InitCommonTemplateParam(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateDataParams& tempAlgParamsCommon);
    void InitLocalCopyTemplateParam(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateDataParams& tempParamLocalcopy);
    void ResetScatterLevel1State();
    void ResetScatterState();

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};
    uint64_t rankSizeLevel2_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};
    uint64_t rankIdxLevel2_{0};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;

    ThreadHandle controlThread_ = 0;

    // Scatter 各级主线程 + 同步索引
    std::vector<ThreadHandle> tempMainThreadsLevel01SC_;
    std::vector<u32> ntfIdxCtrlToTempLevel01SC_;
    std::vector<u32> ntfIdxTempToCtrlLevel01SC_;
    std::vector<ThreadHandle> tempMainThreadsLevel2SC_;
    std::vector<u32> ntfIdxCtrlToTempLevel2SC_;
    std::vector<u32> ntfIdxTempToCtrlLevel2SC_;

    // AllGather 各级主线程 + 同步索引
    std::vector<ThreadHandle> tempMainThreadsLevel01AG_;
    std::vector<u32> ntfIdxCtrlToTempLevel01AG_;
    std::vector<u32> ntfIdxTempToCtrlLevel01AG_;
    std::vector<ThreadHandle> tempMainThreadsLevel2AG_;
    std::vector<u32> ntfIdxCtrlToTempLevel2AG_;
    std::vector<u32> ntfIdxTempToCtrlLevel2AG_;

    std::vector<std::vector<ThreadHandle>> levelThreadsSC_; // Scatter 各级从线程
    std::vector<std::vector<ThreadHandle>> levelThreadsAG_; // AllGather 各级从线程

    // Scatter 各级具体类型模板指针 (用于调用 SetRoot/SetDoTask/DoLocalCopy)
    std::shared_ptr<InsScatterAlgTemplateX> tempScatterLevel0_;
    std::shared_ptr<InsScatterAlgTemplateY> tempScatterLevel1_;
    std::shared_ptr<InsScatterAlgTemplateZ> tempScatterLevel2_;
    // AllGather 各级具体类型模板指针
    std::shared_ptr<InsAgAlgTemplateX> tempAgLevel0_;
    std::shared_ptr<InsAgAlgTemplateY> tempAgLevel1_;
    std::shared_ptr<InsAgAlgTemplateZ> tempAgLevel2_;

    OmniNeedSetStepNum omniNeedSetStepNum_ = OmniNeedSetStepNum::OMNIPIPE_DEFAULT;

    enum OmnipipeBCLevel {
        OMNIPIPE_SC_LEVEL0 = 0,
        OMNIPIPE_SC_LEVEL1 = 1,
        OMNIPIPE_SC_LEVEL2 = 2,
        OMNIPIPE_AG_LEVEL0 = 3,
        OMNIPIPE_AG_LEVEL1 = 4,
        OMNIPIPE_AG_LEVEL2 = 5,
        OMNIPIPE_BC_LEVEL_NUM = 6
    };

    enum class TopoType { UBX_2LEVEL, THREE_LEVEL };
    TopoType topoType_ = TopoType::UBX_2LEVEL;

    // root 三轴相对关系（用于编排阶段的路由判断）
    bool isSameXAxisAsRoot = false; // 框内与root同横轴
    bool isSameYAxisAsRoot = false; // 框内与root同纵轴
    bool isSameZAxisAsRoot = false; // 与root同Z轴(跨机同框间位置)
    bool isSameSerAsRoot = false;   // 与root同机
};

} // namespace ops_hccl

#endif // HCCLV2_INS_V2_BROADCAST_OMNIPIPE_EXECUTOR_H
