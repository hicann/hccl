/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include "scatter_ring_executor.h"

namespace ops_hccl {

ScatterRingExecutor::ScatterRingExecutor() : ScatterExecutorBase()
{
    // 只有ring算法使能了DMA消减
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    desc_.level1SupportedAlgos = {
        AlgTypeLevel1::ALG_LEVEL1_NHR,
        AlgTypeLevel1::ALG_LEVEL1_NB,
        AlgTypeLevel1::ALG_LEVEL1_RING
    };
    desc_.level2SupportedAlgos = {
        AlgTypeLevel2::ALG_LEVEL2_NHR,
        AlgTypeLevel2::ALG_LEVEL2_NB,
        AlgTypeLevel2::ALG_LEVEL2_RING
    };
}

HcclResult ScatterRingExecutor::CalcResRequest(HcclComm comm, const OpParam& param, TopoInfo* topoInfo,
    AlgHierarchyInfo& algHierarchyInfo, AlgResourceRequest& resourceRequest, AlgType& algType)
{
    CHK_RET(CalcGeneralTopoInfoForA3(comm, topoInfo, algHierarchyInfo));
    CHK_RET(RefreshAlgType(algType));

    u32 threadNum = (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING ?
        LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE : LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread = 1;
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    // level0 channel
    std::vector<ChannelDesc> level0Channels;
    CHK_RET(CalcLevel0ChannelRequest(param, topoInfo, algHierarchyInfo, algType, level0Channels));
    resourceRequest.channels.push_back(level0Channels);

    // level1 channel
    std::vector<ChannelDesc> level1Channels;
    CHK_RET(CalcLevel1ChannelRequest(param, topoInfo, algHierarchyInfo, algType, level1Channels));
    resourceRequest.channels.push_back(level1Channels);

    // level2 channel
    std::vector<ChannelDesc> level2Channels;
    CHK_RET(CalcLevel2ChannelRequest(param, topoInfo, algHierarchyInfo, algType, level2Channels));
    resourceRequest.channels.push_back(level2Channels);

    HCCL_INFO("[ScatterRingExecutor][CalcResRequest]slaveThreadNum[%u] notifyNumPerThread[%u] notifyNumOnMainThread[%u]"
        " level0Channels[%u] level1Channels[%u] level2Channels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread, resourceRequest.notifyNumOnMainThread,
        level0Channels.size(), level1Channels.size(), level2Channels.size());
    return HCCL_SUCCESS;
}

HcclResult ScatterRingExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[ScatterRingExecutor][KernelRun] starts.");

    CHK_RET(GetSubCommInfo(COMM_LEVEL0, level0CommInfo_));

    commIndex_ = level0CommInfo_.localRank;

    CHK_RET(GetSubCommInfo(COMM_LEVEL1, level1CommInfo_));
    CHK_RET(GetSubCommInfo(COMM_LEVEL2, level2CommInfo_));

    CHK_RET(KernelRunLevel2(param, execMem));
    CHK_RET(KernelRunLevel1(param, execMem));
    CHK_RET(KernelRunLevel0(param, execMem));

    if (!DMAReduceFlag_) {
        u8* src = static_cast<u8 *>(execMem.inputMem.addr) + level0SliceOffset_ + execMem.outputMem.size * commIndex_;
        CHK_RET(HcommLocalCopyOnThread(thread_, execMem.outputMem.addr, src, execMem.count * unitSize_));
    }
    HCCL_INFO("scatter ring run success");
    return HCCL_SUCCESS;
}

/* ***********超节点间scatter*********** */
HcclResult ScatterRingExecutor::KernelRunLevel2(const OpParam &param, ExecMem &execMem)
{
    u32 level2RankSize = level2CommInfo_.localRankSize;
    u32 level2Rank = level2CommInfo_.localRank;

    GetSubRootRank(param.root, COMM_LEVEL2, algResource_->algHierarchyInfo, subUserRankRootSupperPod_);

    if (level2RankSize > 1 && subUserRankRootSupperPod_ == topoInfo_->userRank) {
        u32 planeRootSupperPod = 0;
        CHK_RET(GetSubCommRankByUserRank(param.root, COMM_LEVEL2, algResource_->algHierarchyInfo, planeRootSupperPod));
        std::unique_ptr<AlgTemplateBase> level2TempAlg;
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level2TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_NB);
            HCCL_INFO("[ScatterRingExecutor][KernelRunLevel2] using NB algo inter-superPod.");
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level2TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_NHR);
            HCCL_INFO("[ScatterRingExecutor][KernelRunLevel2] using NHR algo inter-superPod.");
        } else {
            level2TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_RING);
            HCCL_INFO("[ScatterRingExecutor][KernelRunLevel2] using ring algo inter-superPod.");
        }
        CHK_SMART_PTR_NULL(level2TempAlg);

        u64 level2Count = execMem.inputMem.size / unitSize_;
        CHK_RET(level2TempAlg->Prepare(execMem.inputMem, execMem.inputMem, execMem.inputMem, level2Count,
            param.DataDes.dataType, thread_, HCCL_REDUCE_RESERVED, planeRootSupperPod, std::vector<Slice>(0)));
        CHK_RET(level2TempAlg->RunAsync(level2Rank, level2RankSize, channels_[COMM_LEVEL2]));
    }
    return HCCL_SUCCESS;
}

/* ***********节点间scatter*********** */
HcclResult ScatterRingExecutor::KernelRunLevel1(const OpParam &param, ExecMem &execMem)
{
    u32 level2RankSize = level2CommInfo_.localRankSize;
    u32 level2Rank = level2CommInfo_.localRank;
    u32 level1RankSize = level1CommInfo_.localRankSize;
    u32 level1Rank = level1CommInfo_.localRank;
    HCCL_DEBUG("level1RankSize:%u level1Rank:%u", level1RankSize, level1Rank);

    u64 level1SliceSize = execMem.inputMem.size / level2RankSize;
    u64 level1SliceCount = level1SliceSize / unitSize_;
    level1SliceOffset_ = level1SliceSize * level2Rank;

    GetSubRootRank(subUserRankRootSupperPod_, COMM_LEVEL1, algResource_->algHierarchyInfo, subRoot_);
    CHK_PRT_RET(subRoot_ == INVALID_VALUE_RANKID, \
        HCCL_ERROR("[ScatterRingExecutor][KernelRun]GetSubRootForScatter failed, " \
            "userRank[%u], root[%u], subRoot[%u]", topoInfo_->userRank, param.root, subRoot_), HCCL_E_INTERNAL);
    HCCL_DEBUG("[ScatterRingExecutor][KernelRun]GetSubRootForScatter, userRank[%u], root[%u], subRoot[%u]",
        topoInfo_->userRank, param.root, subRoot_);

    if (level1RankSize > 1 && subRoot_ == topoInfo_->userRank) {
        u32 rootRankLevel1 = 0;
        CHK_RET(GetSubCommRankByUserRank(subUserRankRootSupperPod_, COMM_LEVEL1, algResource_->algHierarchyInfo, rootRankLevel1));

        std::unique_ptr<AlgTemplateBase> level1TempAlg;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_NB);
            HCCL_INFO("[ScatterRingExecutor][KernelRunLevel1] using NB algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_NHR);
            HCCL_INFO("[ScatterRingExecutor][KernelRunLevel1] using NHR algo inter-server.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_RING);
            HCCL_INFO("[ScatterRingExecutor][KernelRunLevel1] using ring algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        HcclMem level1InputMem = HcclMemRange(execMem.inputMem, level1SliceOffset_, level1SliceSize);
        CHK_RET(level1TempAlg->Prepare(level1InputMem, level1InputMem, level1InputMem, level1SliceCount,
            param.DataDes.dataType, thread_, HCCL_REDUCE_RESERVED, rootRankLevel1, std::vector<Slice>(0),
            level1SliceOffset_));
        CHK_RET(level1TempAlg->RunAsync(level1Rank, level1RankSize, channels_[COMM_LEVEL1]));
    }
    return HCCL_SUCCESS;
}

/* ***********节点内scatter*********** */
HcclResult ScatterRingExecutor::KernelRunLevel0(const OpParam &param, ExecMem &execMem)
{
    // 每个server分配的slice大小
    u32 level0RankSize = level0CommInfo_.localRankSize;
    u32 level2RankSize = level2CommInfo_.localRankSize;
    u32 level1RankSize = level1CommInfo_.localRankSize;
    u32 level1Rank = level1CommInfo_.localRank;

    u64 level0SliceSize = execMem.inputMem.size / (level1RankSize * level2RankSize);
    level0SliceOffset_ = level0SliceSize * level1Rank + level1SliceOffset_;
    HCCL_DEBUG("[ScatterRingExecutor][KernelRunLevel0]inputMem.size=%llu, level0RankSize=%u, level0SliceSize=%llu, "
        "serverSliceOffset=%llu commIndex=%u commLevel1[commIndex]->rank=%u", execMem.inputMem.size, level0RankSize,
        level0SliceSize, level0SliceOffset_, commIndex_, level1Rank);

    HcclMem scatterRingInput = HcclMemRange(execMem.inputMem, level0SliceOffset_, level0SliceSize);

    // 将根节点数据切分成level0RankSize份
    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    // 根据数据量算每个环上数据的偏移和大小
    CHK_RET(PrepareDataSlice(execMem.count, unitSize_, level0RankSize, dataSegsSlice));

    HCCL_INFO("[ScatterRingExecutor][KernelRunLevel0] using multiring algo inner-server.");

    HcomCollOpInfo *scatterOpInfoPtr = nullptr;
    HcomCollOpInfo scatterOpInfo = {"", nullptr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType,
        subRoot_};
    if (DMAReduceFlag_) {
        scatterOpInfoPtr = &scatterOpInfo;
    }

    CHK_RET(MultiRingScatter(scatterRingInput, scatterRingInput, execMem.count, param.DataDes.dataType,
        dataSegsSlice, subRoot_, scatterOpInfoPtr, level0SliceOffset_));
    return HCCL_SUCCESS;
}

HcclResult ScatterRingExecutor::MultiRingScatter(HcclMem inputMem, HcclMem outputMem, const u64 count,
    const HcclDataType dataType, const std::vector<Slice> &dataSegsSlice, u32 root, const HcomCollOpInfo *opInfo,
    const u64 baseOffset)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = 0;
    std::vector<std::vector<Slice>> mutliRingsSlices;
    std::vector<std::vector<u32>> rankOrders;
    CHK_RET(PrepareMultiRingSlice(dataSegsSlice, ringNum, mutliRingsSlices, rankOrders));
    HCCL_INFO("[ScatterRingExecutor][MultiRingScatter] ringNum[%u]", ringNum);

    u32 rootRank = 0;
    CHK_RET(GetSubCommRankByUserRank(root, COMM_LEVEL0, algResource_->algHierarchyInfo, rootRank));

    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSlice = mutliRingsSlices[ringIndex];
        std::vector<u32> &rankOrder = rankOrders[ringIndex];
        std::vector<ChannelInfo> ringChannels;
        ringChannels.reserve(level0CommInfo_.localRankSize);
        for (u32 i = 0; i < level0CommInfo_.localRankSize; i++) {
            ringChannels.push_back(channels_[COMM_LEVEL0][rankOrder[i]]);
        }

        std::unique_ptr<AlgTemplateBase> tempAlg;
        if (opInfo == nullptr) {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_RING);
            CHK_SMART_PTR_NULL(tempAlg);
        } else {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_RING_DIRECT);
            CHK_SMART_PTR_NULL(tempAlg);
            CHK_RET(tempAlg->Prepare(
                const_cast<HcomCollOpInfo *>(opInfo), topoInfo_->userRank, rankOrder, singleRingSlice));
        }

        if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
            // 从环等主环通知开始
            CHK_RET(HcommInterThreadNotifyWaitOnThread(slaveThreads_[ringIndex], LOCAL_NOTIFY_IDX_ZERO, CUSTOM_TIMEOUT));

            ret = tempAlg->Prepare(inputMem, inputMem, outputMem, count, dataType,
                slaveThreads_[ringIndex], HCCL_REDUCE_RESERVED, rankOrder[rootRank], singleRingSlice,
                baseOffset);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u],scatter(ring) prepare failed, "\
                "return[%d]", ringIndex, ret), ret);

            CHK_RET(tempAlg->RunAsync(rankOrder[level0CommInfo_.localRank], level0CommInfo_.localRankSize,
                ringChannels));

            // 从环结束通知主环
            CHK_RET(HcommInterThreadNotifyRecordOnThread(slaveThreads_[ringIndex], thread_, ringIndex));

            // 主环启动从环
            CHK_RET(HcommInterThreadNotifyRecordOnThread(thread_, slaveThreads_[ringIndex], LOCAL_NOTIFY_IDX_ZERO));
        } else {  // 主环
            ret = tempAlg->Prepare(inputMem, inputMem, outputMem, count, dataType, thread_,
                HCCL_REDUCE_RESERVED, rankOrder[rootRank], singleRingSlice, baseOffset);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u],scatter(ring) prepare failed, "\
                "return[%d]", ringIndex, ret), ret);

            CHK_RET(tempAlg->RunAsync(rankOrder[level0CommInfo_.localRank], level0CommInfo_.localRankSize,
                ringChannels));

            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                // 主环等所有从环结束
                CHK_RET(HcommInterThreadNotifyWaitOnThread(thread_, ring, CUSTOM_TIMEOUT));
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult ScatterRingExecutor::MutliSegSlicePrepare(const std::vector<Slice> &dataSegsSlice, u32 ringNum,
    std::vector<std::vector<Slice>> &mutliSegsSlices)
{
    std::vector<Slice> singleSegSlices;
    singleSegSlices.reserve(ringNum);
    for (u32 rankId = 0; rankId < dataSegsSlice.size(); rankId++) {
        u32 ringIndex = 0;
        u64 rankDataSize = dataSegsSlice[rankId].size;
        u64 offsetStart = dataSegsSlice[rankId].offset;
        if (rankDataSize > 0 && ringNum != 0) {
            u64 sizeTemp = (rankDataSize + ringNum - 1) / ringNum; /* 1是为了向上取整 */
            u64 sizePerRing = RoundUpWithDivisor(sizeTemp, HCCL_MIN_SLICE_ALIGN);
            u64 residueSize = rankDataSize;
            while (residueSize > 0) {
                u64 singleRingSize = sizePerRing < residueSize ? sizePerRing : residueSize;
                Slice rankSliceTemp;
                rankSliceTemp.size = singleRingSize;
                rankSliceTemp.offset = offsetStart + rankDataSize - residueSize;
                ringIndex++;
                residueSize -= singleRingSize;
                singleSegSlices.push_back(rankSliceTemp);
            }
        }
        while (ringIndex < ringNum) {
            Slice rankSliceTemp;
            rankSliceTemp.size = 0;
            rankSliceTemp.offset = offsetStart;
            ringIndex++;
            singleSegSlices.push_back(rankSliceTemp);
        }
        mutliSegsSlices.push_back(singleSegSlices);
        singleSegSlices.clear();
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterRingExecutor::PrepareMultiRingSlice(const std::vector<Slice> &dataSegsSlice,
    u32 &ringNum, std::vector<std::vector<Slice>> &mutliRingsSlices, std::vector<std::vector<u32>> &rankOrders)
{
    u32 rankSize = level0CommInfo_.localRankSize;

    std::vector<u32> rankOrder0; // 环0
    rankOrder0.resize(rankSize);
    std::iota(rankOrder0.begin(), rankOrder0.end(), 0);

    if (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING) {
        ringNum = LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE;
        std::vector<u32> rankOrder1; // 环1
        rankOrder1.reserve(rankSize);
        rankOrder1.push_back(rankOrder0[0]);
        rankOrder1.insert(rankOrder1.end(), rankOrder0.rbegin(), rankOrder0.rend() - 1);
        rankOrders.push_back(rankOrder0);
        rankOrders.push_back(rankOrder1);
    } else {
        ringNum = LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
        mutliRingsSlices.push_back(dataSegsSlice);
        rankOrders.push_back(rankOrder0);
        return HCCL_SUCCESS;
    }
    // 打印多个环
    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
        for (u32 i = 0; i < rankOrders.size(); i++) {
            auto ring = rankOrders[i];
            std::ostringstream stringRepresentation;
            for (std::vector<uint32_t>::iterator it = ring.begin(); it != ring.end(); it++) {
                stringRepresentation << *it << " ";
            }
            std::string ringString = stringRepresentation.str();
            HCCL_DEBUG("[ScatterRingExecutor][PrepareMultiRingSlice]The No.%u ring: %s", i, ringString.c_str());
        }
    }

    // 将每块数据切分为ringNum份
    std::vector<std::vector<Slice> > mutliSegsSlices;
    mutliSegsSlices.reserve(dataSegsSlice.size());
    CHK_RET(MutliSegSlicePrepare(dataSegsSlice, ringNum, mutliSegsSlices));

    std::vector<Slice> singleRingSlices;
    singleRingSlices.reserve(rankSize);
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        for (u32 segsIndex = 0; segsIndex < rankSize; segsIndex++) {
            u32 rankPosition = rankOrders[ringIndex][segsIndex];
            Slice tempSlice = mutliSegsSlices[rankPosition][ringIndex];
            singleRingSlices.push_back(tempSlice);
        }
        mutliRingsSlices.push_back(singleRingSlices);
        singleRingSlices.clear();
    }

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ScatterRingFor91093Executor", ScatterRing, ScatterRingExecutor);
}
