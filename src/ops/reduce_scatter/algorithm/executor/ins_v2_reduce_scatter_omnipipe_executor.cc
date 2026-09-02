/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_scatter_omnipipe_executor.h"
#include "topo_match_3_level.h"
#include "ins_temp_reduce_scatter_omnipipe_mesh_1D.h"
#include "ins_temp_reduce_scatter_omnipipe_mesh_1d_dpu.h"
#include "ins_temp_reduce_scatter_omnipipe_nhr.h"
#include "topo_match_pcie_mix.h"
#include "omnipipe_template_utils.h"
#include "alg_attrs_registry.h"
#include "auto_selector_base.h"
namespace ops_hccl {
constexpr uint32_t HIERARCHY_SIZE_3 = 3;
constexpr uint64_t RANK_SIZE_LEVEL_2 = 2;
constexpr uint64_t RANK_SIZE_LEVEL_4 = 4;
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2ReduceScatterOmniPipeExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InsV2ReduceScatterOmniPipeExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InitCommInfo(
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
    dataSize_ = dataCount_ * dataTypeSize_;

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO(
        "[InsV2ReduceScatterOmniPipeExecutor][InitCommInfo] initialize communication metadata, "
        "rank[%u], rankSize[%u], devType[%u], reduceOp[%u], dataType[%u], dataTypeSize[%u].",
        myRank_, rankSize_, devType_, reduceOp_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

// 实例化实际执行以来AutoMatchMeshNhr这个类的实现
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    BuildSubCommAndTempMap(
        const OpParam& param, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        std::vector<std::vector<u32>>& subCommRanks0, std::vector<std::vector<u32>>& subCommRanks1,
        std::vector<std::vector<u32>>& subCommRanks2, std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap,
        const TopoInfoWithNetLayerDetails* topoInfo)
{
    subCommRanks0.clear();
    subCommRanks1.clear();
    subCommRanks2.clear();
    tempMap.clear();

    HCCL_INFO(
        "[InsV2ReduceScatterOmniPipeExecutor][BuildSubCommAndTempMap] build sub-communicators from "
        "algorithm hierarchy, hierarchy[%s].",
        ThreeDVecToStrOmni(algHierarchyInfo_.infos).c_str());
    if (algHierarchyInfo_.infos.empty()) {
        HCCL_ERROR("[%s] algHierarchyInfo_.infos is empty.", __func__);
        return HCCL_E_PARA;
    }
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        if (algHierarchyInfo_.infos[0].size() < MIN_SUBGROUP_NUM) {
            HCCL_ERROR(
                "[%s] algHierarchyInfo_.infos[0] size[%zu] is less than 2.", __func__,
                algHierarchyInfo_.infos[0].size());
            return HCCL_E_PARA;
        }
        std::vector<u32> closRanks;
        if (!algHierarchyInfo_.infos[0].empty() && !algHierarchyInfo_.infos[0][0].empty()) {
            subCommRanks0 = {algHierarchyInfo_.infos[0][0]};
            u32 meshSize = algHierarchyInfo_.infos[0][0].size();
            if (!algHierarchyInfo_.infos[0][1].empty()) {
                for (auto rank : algHierarchyInfo_.infos[0][1]) {
                    if (rank % meshSize == topoInfo->userRank % meshSize) {
                        closRanks.push_back(rank);
                    }
                }
            }
        }
        subCommRanks1 = {closRanks};
        omniNeedSetStepNum_ = (subCommRanks1[0].size() == RANK_SIZE_LEVEL_4) ? OmniNeedSetStepNum::OMNIPIPE_UBX_16P :
                                                                               OmniNeedSetStepNum::OMNIPIPE_DEFAULT;
        if (!algHierarchyInfo_.infos[1].empty()) {
            subCommRanks2 = algHierarchyInfo_.infos[1];
            omniNeedSetStepNum_
                = (subCommRanks2[0].size() > 1) ? OmniNeedSetStepNum::OMNIPIPE_UBX_32P : omniNeedSetStepNum_;
        } else {
            subCommRanks2.emplace_back(std::vector<u32>{myRank_});
        }
    } else if (topoType_ == TopoType::THREE_LEVEL) {
        if (!algHierarchyInfo.infos[0].empty() && !algHierarchyInfo.infos[0][0].empty()) {
            subCommRanks0.push_back(algHierarchyInfo.infos[0][0]);
        } else {
            subCommRanks0.emplace_back(std::vector<u32>{myRank_});
        }
        if (!algHierarchyInfo.infos[1].empty() && !algHierarchyInfo.infos[1][0].empty()) {
            subCommRanks1.push_back(algHierarchyInfo.infos[1][0]);
        } else {
            subCommRanks1.emplace_back(std::vector<u32>{myRank_});
        }
        if (!algHierarchyInfo.infos[2].empty() && !algHierarchyInfo.infos[2][0].empty()) {
            subCommRanks2.push_back(algHierarchyInfo.infos[2][0]);
        } else {
            subCommRanks2.emplace_back(std::vector<u32>{myRank_});
        }
    } else {
        if (!algHierarchyInfo_.infos[0].empty()) {
            subCommRanks0 = algHierarchyInfo_.infos[0];
        }
        if (!algHierarchyInfo_.infos[1].empty()) {
            subCommRanks1 = algHierarchyInfo_.infos[1];
        } else {
            subCommRanks1.emplace_back(std::vector<u32>{myRank_});
        }
        subCommRanks2.emplace_back(std::vector<u32>{myRank_});
    }

    rankSizeLevel0_ = subCommRanks0[0].size();
    rankSizeLevel1_ = subCommRanks1[0].size();
    rankSizeLevel2_ = subCommRanks2[0].size();
    if (rankSizeLevel0_ == 0 || rankSizeLevel1_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel0_[%u] or rankSizeLevel1_[%u] is 0.", __func__, rankSizeLevel0_, rankSizeLevel1_);
        return HCCL_E_PARA;
    }

    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ % (rankSizeLevel0_ * rankSizeLevel1_) / rankSizeLevel0_;
    rankIdxLevel2_ = myRank_ / (rankSizeLevel0_ * rankSizeLevel1_);

    if (rankSizeLevel0_ > 1) {
        tempMap[OMNIPIPE_LEVEL0] = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel1_ > 1) {
        tempMap[OMNIPIPE_LEVEL1] = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1);
    }
    if (rankSizeLevel2_ > 1) {
        tempMap[OMNIPIPE_LEVEL2] = std::make_shared<InsAlgTemplate2>(param, myRank_, subCommRanks2);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    HCCL_DEBUG(
        "[InsV2ReduceScatterOmniPipeExecutor][CalcRes] start calculating template resources, rank[%u].",
        topoInfo->userRank);
    // 初始化一些基本成员变量
    InitCommInfo(param, topoInfo, algHierarchyInfo);

    if (algHierarchyInfo_.infos.size() == HIERARCHY_SIZE_3 && !algHierarchyInfo_.infos[2].empty()
        && !algHierarchyInfo_.infos[2][0].empty()) {
        topoType_ = TopoType::THREE_LEVEL;
    } else {
        topoType_ = TopoType::UBX_2LEVEL;
    }

    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;
    CHK_RET(BuildSubCommAndTempMap(
        param, algHierarchyInfo, subCommRanks0, subCommRanks1, subCommRanks2, tempMap, topoInfo));

    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    for (auto& temp : tempMap) {
        AlgResourceRequest resReqlevel;
        CHK_RET(temp.second->CalcRes(comm, param, topoInfo, resReqlevel));
        resourceRequest.slaveThreadNum += 1 + resReqlevel.slaveThreadNum;
        resourceRequest.notifyNumPerThread.emplace_back(resReqlevel.notifyNumOnMainThread + 1);
        resourceRequest.notifyNumPerThread.insert(
            resourceRequest.notifyNumPerThread.end(), resReqlevel.notifyNumPerThread.begin(),
            resReqlevel.notifyNumPerThread.end());
        resourceRequest.notifyNumOnMainThread++;
        // 对称路径将各层通道合并到 channels[0]，使公共资源层只发起一次 HcclChannelAcquire。
        // 对称内存句柄会随这次建链统一交换；普通路径仍按层保存通道，保持原有资源布局。
        if (!resReqlevel.channels.empty()) {
            if (param.supportSymmetricMemory) {
                if (resourceRequest.channels.empty()) {
                    resourceRequest.channels.resize(1);
                }
                resourceRequest.channels[0].insert(
                    resourceRequest.channels[0].end(), resReqlevel.channels[0].begin(), resReqlevel.channels[0].end());
            } else {
                resourceRequest.channels.emplace_back(resReqlevel.channels[0]);
            }
        }
    }

    return HCCL_SUCCESS;
}

// 该函数必须按照level0、level1、level2的顺序调用
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    PrepareResForTemplateLevel(u32 level, std::shared_ptr<InsAlgTemplateBase>& tempBase)
{
    u32 levelThreadNum = tempBase->GetThreadNum();
    if (level == OMNIPIPE_LEVEL0) {
        levelThreads_[OMNIPIPE_LEVEL0].assign(threads_.begin() + 1, threads_.begin() + 1 + levelThreadNum);
        tempMainThreadsLevel01_.push_back(levelThreads_[0].at(0));
    } else if (level == OMNIPIPE_LEVEL1) {
        levelThreads_[OMNIPIPE_LEVEL1].assign(
            threads_.begin() + 1 + levelThreads_[0].size(),
            threads_.begin() + 1 + levelThreads_[0].size() + levelThreadNum);
        tempMainThreadsLevel01_.push_back(levelThreads_[1].at(0));
    } else if (level == OMNIPIPE_LEVEL2) {
        levelThreads_[OMNIPIPE_LEVEL2].assign(
            threads_.begin() + 1 + levelThreads_[0].size() + levelThreads_[1].size(), threads_.end());
        tempMainThreadsLevel2_.push_back(levelThreads_[OMNIPIPE_LEVEL2].at(0));
    }

    // 获取当前template各自的主thread上有多少notify
    AlgResourceRequest levelTempRequest;
    CHK_RET(tempBase->GetRes(levelTempRequest));
    if (level < OMNIPIPE_LEVEL2) {
        notifyIdxCtrlToTempLevel01_.push_back(levelTempRequest.notifyNumOnMainThread);
        notifyIdxTempToCtrlLevel01_.push_back(tempMainThreadsLevel01_.size() + tempMainThreadsLevel2_.size() - 1);
    } else {
        notifyIdxCtrlToTempLevel2_.push_back(levelTempRequest.notifyNumOnMainThread);
        notifyIdxTempToCtrlLevel2_.push_back(tempMainThreadsLevel01_.size() + tempMainThreadsLevel2_.size() - 1);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO(
        "[InsV2ReduceScatterOmniPipeExecutor][Orchestrate] start reduce-scatter execution, "
        "rank[%u], symmetric[%d].",
        resCtx.topoInfo.userRank, param.supportSymmetricMemory);
    // 参数填充
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    threads_ = resCtx.threads;

    if (algHierarchyInfo_.infos.size() == HIERARCHY_SIZE_3 && !algHierarchyInfo_.infos[2].empty()
        && !algHierarchyInfo_.infos[2][0].empty()) {
        topoType_ = TopoType::THREE_LEVEL;
    } else {
        topoType_ = TopoType::UBX_2LEVEL;
    }

    // 计算subCommRanks和template
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;
    CHK_RET(BuildSubCommAndTempMap(
        param, algHierarchyInfo_, subCommRanks0, subCommRanks1, subCommRanks2, tempMap, &resCtx.topoInfo));

    // 为temp分配thread
    threads_ = resCtx.threads;
    controlThread_ = threads_.at(0);
    levelThreads_.resize(OMNIPIPE_LEVEL_NUM);

    // 对称路径的建链结果扁平存入 channels[0]，普通路径仍按层保存；遍历全部集合后，
    // 根据本 rank 与对端 rank 所属的子通信域重新归层，可同时兼容两种资源布局。
    const std::vector<const std::vector<std::vector<u32>>*> subCommsByLevel
        = {&subCommRanks0, &subCommRanks1, &subCommRanks2};
    const std::vector<uint64_t> rankSizesByLevel = {rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_};
    CHK_RET(ClassifyOmniPipeChannelsByLevel(
        myRank_, resCtx.channels, subCommsByLevel, rankSizesByLevel, remoteRankToChannelInfo_));
    if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix) {
        if (rankSizeLevel1_ > 1) {
            CHK_RET(tempMap[OMNIPIPE_LEVEL1]->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
        }
    }

    for (auto& temp : tempMap) {
        CHK_RET(PrepareResForTemplateLevel(temp.first, temp.second));
    }

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx, tempMap);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2ReduceScatterOmniPipeExecutor][Orchestrate] reduce-scatter execution failed, "
            "rank[%u], errorCode[0x%016llx].",
            myRank_, HCCL_ERROR_CODE(ret)),
        ret);
    HCCL_INFO("[InsV2ReduceScatterOmniPipeExecutor][Orchestrate] finish reduce-scatter execution, rank[%u].", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::RestoreChannelMap(
    const AlgResourceCtxSerializable& resCtx,
    std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo) const
{
    // 通道归层已在 Orchestrate 中通过 ClassifyOmniPipeChannelsByLevel 完成，正常路径不会调用本函数。
    // 此处仅满足基类虚函数契约；打印告警以防未来新增调用路径时静默落入基类默认归层（对称路径下结果错误）。
    HCCL_WARNING(
        "[InsV2ReduceScatterOmniPipeExecutor][RestoreChannelMap] unexpected call: channel classification is "
        "already done in Orchestrate, rank[%u].",
        myRank_);
    (void)resCtx;
    rankIdToChannelInfo.resize(OMNIPIPE_LEVEL_NUM);
    return HCCL_SUCCESS;
}

// 将计算出的单步slice信息初始化到templateParam中
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenTemplateAlgParamsByDimData(
        TemplateDataParams& tempAlgParams, const StepSliceInfo& stepSliceInfo, u64 processedDataCount,
        bool supportSymmetricMemory) const
{
    return FillOmniPipeTemplateAlgParams(
        tempAlgParams, stepSliceInfo, supportSymmetricMemory, processedDataCount, dataTypeSize_);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx,
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap)
{
    HCCL_INFO(
        "[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop] start reduce-scatter pipeline loops, "
        "rank[%u], symmetric[%d].",
        myRank_, param.supportSymmetricMemory);
    // 1.计算带宽
    double bw_rs_l0 = BW_OMNI_DEFAULT;
    double bw_rs_l1 = BW_OMNI_DEFAULT;
    double bw_rs_l2 = BW_OMNI_UBX_ROCE;

    if (resCtx.topoInfo.level0PcieMix) {
        if (rankSizeLevel1_ == RANK_SIZE_LEVEL_2) {
            bw_rs_l1 = BW_OMNI_PCIE_EIGHT_CLOS;
        } else if (rankSizeLevel1_ == RANK_SIZE_LEVEL_4) {
            bw_rs_l1 = BW_OMNI_PCIE_SIXTEEN_CLOS;
        }
    } else if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS) {
        bw_rs_l1 = BW_OMNI_UBX_RS_CLOS;
    }

    // 计算等价带宽
    double eqBw0 = bw_rs_l0; // L0 mesh
    double eqBw1 = bw_rs_l1; // L1 NHR
    double eqBw2 = bw_rs_l2; // L2 NHR

    // level0为mesh,等价mesh为其本身
    // level1为nhr
    // level2, ranksize = 1
    eqBw1 = rankSizeLevel1_ > 1 ? eqBw1 / (rankSizeLevel1_ - 1) : eqBw1;
    eqBw2 = rankSizeLevel2_ > 1 ? eqBw2 / (rankSizeLevel2_ - 1) : eqBw2;

    std::vector<double> endpointAttrBwNew{eqBw0, eqBw1, eqBw2};

    // 2、计算scratch 返回的数组0是maxCountPerloop, 1是loopTimes
    OmniPipeScratchParam scratchParam;
    scratchParam.endpointAttrBw = endpointAttrBwNew;
    scratchParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_};
    std::vector<u64> levelAlgType;
    (tempMap.count(OMNIPIPE_LEVEL0) > 0) ? levelAlgType.push_back(tempMap[OMNIPIPE_LEVEL0]->CalcScratchMultiple(
                                               BufferType::DEFAULT, BufferType::DEFAULT)) :
                                           levelAlgType.push_back(0);
    (tempMap.count(OMNIPIPE_LEVEL1) > 0) ? levelAlgType.push_back(tempMap[OMNIPIPE_LEVEL1]->CalcScratchMultiple(
                                               BufferType::DEFAULT, BufferType::DEFAULT)) :
                                           levelAlgType.push_back(0);
    (tempMap.count(OMNIPIPE_LEVEL2) > 0) ? levelAlgType.push_back(tempMap[OMNIPIPE_LEVEL2]->CalcScratchMultiple(
                                               BufferType::DEFAULT, BufferType::DEFAULT)) :
                                           levelAlgType.push_back(0);
    scratchParam.levelAlgType = levelAlgType;
    // 手动转成数组，这边只给reducescatter用
    std::vector<u64> dataSizeVec;
    for (int i = 0; i < rankSize_; i++) {
        dataSizeVec.push_back(dataSize_);
    }
    scratchParam.dataSize = dataSizeVec;
    scratchParam.dataTypeSize = dataTypeSize_;
    scratchParam.maxTmpMemSize = resCtx.cclMem.size;
    scratchParam.opMode = param.opMode;
    scratchParam.engine = param.engine;
    scratchParam.needSetStepNum = omniNeedSetStepNum_;
    if (param.opConfig.multipleDimensionSplitRatioSource != MultipleDimensionSplitRatioSource::BUILTIN_FORMULA) {
        scratchParam.multipleDimensionSplitRatio = param.opConfig.multipleDimensionSplitRatio;
    }
    HCCL_DEBUG(
        "[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop] scratch multipleDimensionSplitRatioSource=[%d], "
        "opConfigRatio=[%f], scratchParamRatio=[%f]",
        static_cast<int>(param.opConfig.multipleDimensionSplitRatioSource), param.opConfig.multipleDimensionSplitRatio,
        scratchParam.multipleDimensionSplitRatio);
    std::vector<u64> loopInfo = CalcOmniPipeScratchInfo(scratchParam);
    u64 maxCountPerLoop = loopInfo[0];
    u64 loopTimes = loopInfo[1];

    // 3、计算n-1次loop的slice信息
    OmniPipeSliceParam sliceParam;
    std::vector<u64> dataSizePerLoop;
    std::vector<u64> dataWholeSize;
    u64 perLoopSize = maxCountPerLoop * dataTypeSize_;

    // 普通路径按每个 loop 的紧凑 scratch 布局计算；对称路径按 user input 的完整分片跨度计算。
    for (int i = 0; i < rankSize_; i++) {
        dataSizePerLoop.push_back(perLoopSize);
        dataWholeSize.push_back(param.supportSymmetricMemory ? dataSize_ : perLoopSize);
    }
    sliceParam.dataSizePerLoop = dataSizePerLoop;
    sliceParam.dataWholeSize = dataWholeSize;
    sliceParam.endpointAttrBw = endpointAttrBwNew;
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, rankIdxLevel2_};
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_};
    sliceParam.levelAlgType = levelAlgType;
    sliceParam.dataTypeSize = dataTypeSize_;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = param.engine;
    sliceParam.needSetStepNum = omniNeedSetStepNum_;
    if (param.opConfig.multipleDimensionSplitRatioSource != MultipleDimensionSplitRatioSource::BUILTIN_FORMULA) {
        sliceParam.multipleDimensionSplitRatio = param.opConfig.multipleDimensionSplitRatio;
    }
    HCCL_DEBUG(
        "[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop] multipleDimensionSplitRatioSource=[%d], "
        "opConfigRatio=[%f], sliceParamRatio=[%f]",
        static_cast<int>(param.opConfig.multipleDimensionSplitRatioSource), param.opConfig.multipleDimensionSplitRatio,
        sliceParam.multipleDimensionSplitRatio);
    OmniPipeSliceInfo alignSliceInfo = CalcRSOmniPipeSliceInfo(sliceParam);

    // 4、计算第n次的loop的slice信息
    OmniPipeSliceInfo tailSliceInfo;
    if (dataCount_ % maxCountPerLoop != 0) {
        std::vector<u64> dataSizePerLoop;
        std::vector<u64> dataWholeSize;
        u64 perLoopSize = (dataCount_ % maxCountPerLoop) * dataTypeSize_;
        for (int i = 0; i < rankSize_; i++) {
            dataSizePerLoop.push_back(perLoopSize);
            // 尾 loop 仍沿用上述布局：普通路径使用尾块跨度，对称路径使用完整输入跨度。
            dataWholeSize.push_back(param.supportSymmetricMemory ? dataSize_ : perLoopSize);
        }
        sliceParam.dataSizePerLoop = dataSizePerLoop;
        sliceParam.dataWholeSize = dataWholeSize;
        tailSliceInfo = CalcRSOmniPipeSliceInfo(sliceParam);
    }

    u64 processedDataCount = 0;
    OmniPipeSliceInfo omnipipeSliceInfo;
    HCCL_INFO(
        "[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop] split operation into executor loops, "
        "loopCount[%llu], maxCountPerLoop[%llu].",
        loopTimes, maxCountPerLoop);
    std::map<u32, TemplateResource> tempResMap;
    std::map<u32, TemplateDataParams> tempAlgParamMap;
    for (auto& temp : tempMap) {
        tempResMap[temp.first].channels = remoteRankToChannelInfo_[temp.first];
        tempResMap[temp.first].threads = levelThreads_[temp.first];
        tempResMap[temp.first].npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
        tempResMap[temp.first].dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
        tempAlgParamMap[temp.first].buffInfo.hcclBuff = resCtx.cclMem;
        // 下发用户地址和对称内存开关；对称模板直接在 user input 上归约，普通模板使用 ccl scratch。
        tempAlgParamMap[temp.first].buffInfo.inputPtr = param.inputPtr;
        tempAlgParamMap[temp.first].buffInfo.outputPtr = param.outputPtr;
        tempAlgParamMap[temp.first].enableRemoteMemAccess = param.supportSymmetricMemory;
    }

    TemplateDataParams tempParamLocalcopy;
    tempParamLocalcopy.buffInfo.hcclBuff = resCtx.cclMem;
    tempParamLocalcopy.buffInfo.inputPtr = param.inputPtr;
    tempParamLocalcopy.buffInfo.outputPtr = param.outputPtr;
    // 5、进行一次loop的数据处理
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        auto loopSize = currDataCount * dataTypeSize_;
        if (!param.supportSymmetricMemory) {
            // 本地拷贝前同步。level0 和 level1 不会同时退化为单 rank，因此复用 level0/1 线程组。
            CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxCtrlToTempLevel01_));
            // 普通路径在每个 loop 前将所有 rank 的 user input 分片压紧到 ccl scratch。
            tempParamLocalcopy.buffInfo.inBuffType = BufferType::INPUT;
            tempParamLocalcopy.count = currDataCount;
            tempParamLocalcopy.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
            tempParamLocalcopy.inputSliceStride = dataCount_ * dataTypeSize_;
            tempParamLocalcopy.buffInfo.outBuffBaseOff = 0;
            tempParamLocalcopy.outputSliceStride = loopSize;
            tempParamLocalcopy.repeatNum = rankSize_;
            tempParamLocalcopy.sliceSize = loopSize;
            // 无论当前拓扑有几层，都复用第一个有效模板的线程执行本地拷贝。
            if (rankSizeLevel0_ > 1) {
                auto temp0 = std::dynamic_pointer_cast<InsAlgTemplate0>(tempMap.begin()->second);
                CHK_RET(temp0->DoLocalCopy(tempParamLocalcopy, tempResMap.begin()->second.threads));
            } else {
                auto temp1 = std::dynamic_pointer_cast<InsAlgTemplate1>(tempMap.begin()->second);
                CHK_RET(temp1->DoLocalCopy(tempParamLocalcopy, tempResMap.begin()->second.threads));
            }
            // 本地拷贝后同步。
            CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxTempToCtrlLevel01_));
        }
        // 对称路径的数据已位于 user input，无需执行 input 到 scratch 的头拷贝。

        // 5.2 确定当前是前n-1次loop的slice结果，还是存在尾块时最后一次loop的slice结果
        if (loop == loopTimes - 1 && !tailSliceInfo.isEmpty()) {
            omnipipeSliceInfo = tailSliceInfo;
        } else {
            omnipipeSliceInfo = alignSliceInfo;
        }
        u32 level2StepCount = omnipipeSliceInfo.dataSliceLevel2.size();
        u32 level0StepCount = omnipipeSliceInfo.dataSliceLevel0.size() / omnipipeSliceInfo.dataSliceLevel2.size();
        HCCL_INFO(
            "[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop] execute loop communication plan, "
            "loop[%llu], level2StepCount[%u], intraLevelStepCount[%u].",
            loop, level2StepCount, level0StepCount);

        // 5.3 遍历 level2 的通信步骤。
        u32 axisReduceId = 0; // 轴间reduce从计算slice的结果中获取
        std::vector<TemplateDataParams> axisReduceTempParams;
        for (int i = 0; i < level2StepCount; i++) {
            HCCL_INFO(
                "[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop] enter level-2 step, "
                "loop[%llu], stepZ[%d].",
                loop, i);
            if (rankSizeLevel2_ > 1) {
                HCCL_DEBUG(
                    "[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop] execute level-2 RS template, "
                    "loop[%llu], stepZ[%d], rankSize[%llu].",
                    loop, i, rankSizeLevel2_);
                CHK_RET(GenTemplateAlgParamsByDimData(
                    tempAlgParamMap[OMNIPIPE_LEVEL2], omnipipeSliceInfo.dataSliceLevel2[i], processedDataCount,
                    param.supportSymmetricMemory));
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel2_, notifyIdxCtrlToTempLevel2_));
            }
            // 5.4 遍历当前 level2 步骤内的 level0/level1 通信步骤。
            for (int j = 0; j < level0StepCount; j++) {
                // level0、1前同步
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxCtrlToTempLevel01_));
                // 初始化并执行机内template任务
                if (rankSizeLevel0_ > 1) {
                    HCCL_DEBUG(
                        "[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop] execute level-0 RS template, "
                        "loop[%llu], stepZ[%d], stepXY[%d], rankSize[%llu].",
                        loop, i, j, rankSizeLevel0_);
                    CHK_RET(GenTemplateAlgParamsByDimData(
                        tempAlgParamMap[0], omnipipeSliceInfo.dataSliceLevel0[i * level0StepCount + j],
                        processedDataCount, param.supportSymmetricMemory));
                    CHK_RET(
                        tempMap[0]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL0], tempResMap[OMNIPIPE_LEVEL0]));
                }
                if (rankSizeLevel1_ > 1) {
                    HCCL_DEBUG(
                        "[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop] execute level-1 RS template, "
                        "loop[%llu], stepZ[%d], stepXY[%d], rankSize[%llu].",
                        loop, i, j, rankSizeLevel1_);
                    CHK_RET(GenTemplateAlgParamsByDimData(
                        tempAlgParamMap[1], omnipipeSliceInfo.dataSliceLevel1[i * level0StepCount + j],
                        processedDataCount, param.supportSymmetricMemory));
                    CHK_RET(
                        tempMap[1]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL1], tempResMap[OMNIPIPE_LEVEL1]));
                }
                // level0、1尾同步
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxTempToCtrlLevel01_));
            }
            if (rankSizeLevel2_ > 1) {
                // z轴尾同步
                CHK_RET(tempMap[OMNIPIPE_LEVEL2]->KernelRun(
                    param, tempAlgParamMap[OMNIPIPE_LEVEL2], tempResMap[OMNIPIPE_LEVEL2]));
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel2_, notifyIdxTempToCtrlLevel2_));
                HCCL_DEBUG(
                    "[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop] finish level-2 RS step and "
                    "synchronize template threads, loop[%llu], stepZ[%d].",
                    loop, i);
            }
        }
        // 本地拷贝前同步。
        CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxCtrlToTempLevel01_));
        if (param.supportSymmetricMemory) {
            // 对称路径的归约结果位于 user input 的本 rank 分片：
            // myRank_ * dataCount_ + processedDataCount。这里只把当前 loop 结果搬到 user output。
            u64 symSrcOff = myRank_ * dataCount_ * dataTypeSize_ + processedDataCount * dataTypeSize_;
            DataSlice symSrc(param.inputPtr, symSrcOff, loopSize, currDataCount);
            DataSlice symDst(param.outputPtr, processedDataCount * dataTypeSize_, loopSize, currDataCount);
            CHK_RET(LocalCopy(tempResMap.begin()->second.threads[0], symSrc, symDst));
        } else {
            // 5.5 普通路径将当前 loop 的本 rank 结果从 ccl scratch 拷贝到 user output。
            tempParamLocalcopy.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
            tempParamLocalcopy.buffInfo.inBuffBaseOff = loopSize * myRank_;
            tempParamLocalcopy.inputSliceStride = 0;
            tempParamLocalcopy.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
            tempParamLocalcopy.outputSliceStride = 0;
            tempParamLocalcopy.sliceSize = loopSize; // 尾拷贝数据量变成1/rankSize
            // 只复制本 rank 的结果，模板内部无需再按 rank 展开。
            tempParamLocalcopy.repeatNum = 1;
            if (rankSizeLevel0_ > 1) {
                auto temp0 = std::dynamic_pointer_cast<InsAlgTemplate0>(tempMap.begin()->second);
                CHK_RET(temp0->DoLocalCopy(tempParamLocalcopy, tempResMap.begin()->second.threads));
            } else {
                auto temp1 = std::dynamic_pointer_cast<InsAlgTemplate1>(tempMap.begin()->second);
                CHK_RET(temp1->DoLocalCopy(tempParamLocalcopy, tempResMap.begin()->second.threads));
            }
        }
        // 本地拷贝后同步。
        CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsLevel01_, notifyIdxTempToCtrlLevel01_));
        processedDataCount += currDataCount;
    }
    HCCL_INFO(
        "[InsV2ReduceScatterOmniPipeExecutor][OrchestrateLoop] finish reduce-scatter pipeline loops, "
        "rank[%u].",
        myRank_);
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, DpuReduceScatterPipeLineMeshNHRMesh, InsV2ReduceScatterOmniPipeExecutor,
    TopoMatchMultilevel, InsTempReduceScatterOmniPipeMesh1D, InsTempReduceScatterOmniPipeNHR,
    InsTempReduceScatterOmniPipeMesh1dDpu);

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AicpuReduceScatterPipeLinePcie, InsV2ReduceScatterOmniPipeExecutor,
    TopoMatchPcieMix, InsTempReduceScatterOmniPipeMesh1D, InsTempReduceScatterOmniPipeNHR,
    InsTempReduceScatterOmniPipeMesh1dDpu);
REGISTER_ALG_ATTRS(AicpuReduceScatterPipeLinePcie, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS;
                   topo.maxTopoLevelNum = 1; op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_64BIT);
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AicpuReduceScatterPipeLineUBX, InsV2ReduceScatterOmniPipeExecutor,
    TopoMatchUBX, InsTempReduceScatterOmniPipeMesh1D, InsTempReduceScatterOmniPipeNHR,
    InsTempReduceScatterOmniPipeMesh1dDpu);
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, DpuReduceScatterPipeLineUBX, InsV2ReduceScatterOmniPipeExecutor, TopoMatchUBX,
    InsTempReduceScatterOmniPipeMesh1D, InsTempReduceScatterOmniPipeNHR, InsTempReduceScatterOmniPipeMesh1dDpu);
REGISTER_ALG_ATTRS(
    DpuReduceScatterPipeLineUBX, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS; topo.minTopoLevelNum = 2;
    topo.minTopoLevelNum = 2; topo.isHostDpuOnly = true;
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->level0Topo == Level0Shape::MESH_1D_CLOS && !topo->level0PcieMix;
    };
    op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_64BIT);

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AicpuReduceScatterPipeLine, InsV2ReduceScatterOmniPipeExecutor,
    TopoMatch3Level, InsTempReduceScatterOmniPipeMesh1D, InsTempReduceScatterOmniPipeNHR,
    InsTempReduceScatterOmniPipeMesh1D);
REGISTER_ALG_ATTRS(
    AicpuReduceScatterPipeLine, topo.minTopoLevelNum = 3; topo.maxTopoLevelNum = 3; op.isSupportProd = false;
    op.unsupportedDataTypes = UNSUPPORTED_64BIT;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->topLevelUboe && (topo->level0Symmetric && topo->level1Symmetric) && topo->deviceNumPerModule == 8;
    });
REGISTER_ALG_ATTRS(
    AicpuReduceScatterPipeLineUBX, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS; topo.maxTopoLevelNum = 1;
    op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_64BIT;
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        bool isMultiple = false;
        if (topo->level0Topo != Level0Shape::MESH_1D_CLOS) {
            return false;
        }
        AutoSelectorBase::CheckClosNumMultipleOfMeshNum(topo, isMultiple);
        return isMultiple;
    };
    op.opCustomCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails* topo) -> bool {
        return opParam.supportSymmetricMemory;
    });

} // namespace ops_hccl
