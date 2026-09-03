/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "ccu_kernel_reduce_scatter_mesh1d_2die_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh_1D_2die_mem2mem.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

std::vector<CostModelParam> CcuTempReduceScatterMeshMem2Mem1D2Die::CalcCostCoeff(CalcCostCoeffParam param)
{
    int kernelNum = 19;
    int taskNum = 5 * (param.rankSize - 1);
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    int portNum0 = 7;
    int portNum1 = 6;
    float level0Ratio = 0.5f;
    float level1Ratio = 1.0f - level0Ratio;
    float nLevel0 = param.dataRatio * level0Ratio;
    float nLevel1 = param.dataRatio * level1Ratio;
    u32 ranksize0 = static_cast<u32>(param.rankSize * level0Ratio + 0.5f);
    u32 ranksize1 = param.rankSize - ranksize0;
    float A0 = 0.0f;
    float A1 = 0.0f;
    float B0 = 0.0f;
    float B1 = 0.0f;
    float D = 0.0f;
    (void)nLevel0;
    (void)nLevel1;
    CostModelManager::Global()->CalcMeshParam(
        param.dataRatio, CommTopo::COMM_TOPO_1DMESH, portNum0, ranksize0, A0, param.isPod);
    CostModelManager::Global()->CalcMeshParam(
        param.dataRatio, CommTopo::COMM_TOPO_CLOS, portNum1, ranksize1, A1, param.isPod);
    A = std::max(A0, A1);

    // ms reduce 一次做8张卡，所以一次reduce
    CostModelManager::Global()->CalcLocalReduceParams(param.dataRatio, EngineType::CCU, B0);
    CostModelManager::Global()->CalcLocalReduceParams(param.dataRatio, EngineType::AICPU, B1);
    // 跟实际标定，后面再改
    // B0 = B0 * 4.7;
    B = B0 + B1;

    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::CCU, C);
    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

constexpr u32 DIE_NUM = 2;

CcuTempReduceScatterMeshMem2Mem1D2Die::CcuTempReduceScatterMeshMem2Mem1D2Die(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempReduceScatterMeshMem2Mem1D2Die::~CcuTempReduceScatterMeshMem2Mem1D2Die() {}

HcclResult CcuTempReduceScatterMeshMem2Mem1D2Die::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.ccuKernelNum.push_back(DIE_NUM);
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG(
        "[CcuTempReduceScatterMeshMem2Mem1D2Die::CalcRes] "
        "notifyNumOnMainThread[%u] slaveThreadNum[%u] ccuKernelNum[%u]",
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum, resourceRequest.ccuKernelNum[0]);

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));

    uint32_t meshDieId = 0;
    CHK_RET(PartitionChannels(comm, channelDescs, meshDieId, rankIdToChannelDesc_));
    resourceRequest.channels.emplace_back(channelDescs);

    // 统一按 meshDieId 先、closDieId 后的顺序创建 kernel，避免 die 连线方向相反时跨 rank channel 同步不对齐
    uint32_t closDieId = 1 - meshDieId;
    uint32_t dieIdOrder[DIE_NUM] = {meshDieId, closDieId};

    for (uint32_t i = 0; i < DIE_NUM; i++) {
        uint32_t dieId = dieIdOrder[i];
        bool isReduceToOutput = (dieId == meshDieId);

        CcuKernelInfo kernelInfo;
        CHK_SAFETY_FUNC_RET(strcpy_s(
            kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuReduceScatterMesh1D2DieMem2MemKernel"));
        kernelInfo.kernelFunc = reinterpret_cast<void*>(CcuReduceScatterMesh1D2DieMem2MemKernel);

        auto kernelArg = std::make_shared<CcuKernelArgReduceScatterMesh1D2DieMem2Mem>();
        kernelArg->gRankSize = templateRankSize_;
        kernelArg->rankSize = rankGroup_[dieId].size();
        kernelArg->isReduceToOutput = isReduceToOutput;
        kernelArg->rankId = myRank_;
        kernelArg->opParam = param;
        kernelArg->subRankGroup = rankGroup_[dieId];
        kernelArg->subCommRanks = subCommRanks_;
        kernelInfo.setKernelArg(kernelArg);

        HCCL_INFO(
            "[CcuTempReduceScatterMeshMem2Mem1D2Die] kernelIdx[%u], dieId[%u], meshDieId[%u], gRankSize[%d], "
            "rankSize[%d], myRank[%d], isReduceToOutput[%d], channels[%u]",
            i, dieId, meshDieId, templateRankSize_, rankGroup_[dieId].size(), myRank_, isReduceToOutput,
            channels_[dieId].size());
        kernelInfo.channels = channels_[dieId];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }

    HCCL_DEBUG(
        "[CcuTempReduceScatterMeshMem2Mem1D2Die::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
        "ccuKernelInfos.size()=%llu",
        channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMeshMem2Mem1D2Die::ClassifyChannelsByDie(
    HcclComm comm, const std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc,
    std::map<uint32_t, std::vector<HcclChannelDesc>>& singleChByDie,
    std::map<uint32_t, std::vector<HcclChannelDesc>>& multiChByDie, bool& hasMultiChannel) const
{
    using DieIdType = uint32_t;
    const uint32_t dieIdTypeSize = sizeof(DieIdType);
    for (const auto& rankToChannels : rankIdToChannelDesc) {
        const std::vector<HcclChannelDesc>& channelList = rankToChannels.second;
        bool isMulti = channelList.size() > 1;
        if (isMulti) {
            hasMultiChannel = true;
        }
        for (const auto& channel : channelList) {
            DieIdType dieId = 0;
            EndpointDesc localEndpoint = channel.localEndpoint;
            CHK_RET(HcclRankGraphGetEndpointInfo(
                comm, myRank_, &localEndpoint, ENDPOINT_ATTR_DIE_ID, dieIdTypeSize, static_cast<void*>(&dieId)));
            (isMulti ? multiChByDie : singleChByDie)[dieId].emplace_back(channel);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMeshMem2Mem1D2Die::PartitionByMesh1dClos(
    const std::map<uint32_t, std::vector<HcclChannelDesc>>& singleChByDie,
    const std::map<uint32_t, std::vector<HcclChannelDesc>>& multiChByDie, uint32_t& meshDieId)
{
    // mesh1d + clos 二级拓扑：mesh 单链路给 meshDieId，clos 双链路只给非 meshDieId
    CHK_PRT_RET(
        singleChByDie.empty(),
        HCCL_ERROR(
            "[CcuTempReduceScatterMeshMem2Mem1D2Die][PartitionByMesh1dClos] Rank[%u] has clos channels but no "
            "mesh single channel.",
            myRank_),
        HcclResult::HCCL_E_INTERNAL);
    auto meshIt = singleChByDie.begin();
    meshDieId = meshIt->first;
    for (const auto& ch : meshIt->second) {
        channels_[meshDieId].emplace_back(ch);
        rankGroup_[meshDieId].push_back(ch.remoteRank);
    }
    for (const auto& pair : multiChByDie) {
        uint32_t dieId = pair.first;
        if (dieId == meshDieId) {
            continue;
        }
        for (const auto& ch : pair.second) {
            channels_[dieId].emplace_back(ch);
            rankGroup_[dieId].push_back(ch.remoteRank);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMeshMem2Mem1D2Die::PartitionByTwoDieRegular(
    const std::map<uint32_t, std::vector<HcclChannelDesc>>& singleChByDie, uint32_t& meshDieId)
{
    // two_die_regular 拓扑：按 channel 数量区分，少的是 mesh die（7 mesh < 8 clos）
    CHK_PRT_RET(
        singleChByDie.size() != DIE_NUM,
        HCCL_ERROR(
            "[CcuTempReduceScatterMeshMem2Mem1D2Die][PartitionByTwoDieRegular] Rank[%u] singleChByDie size[%u] "
            "!= DIE_NUM[%u].",
            myRank_, singleChByDie.size(), DIE_NUM),
        HcclResult::HCCL_E_INTERNAL);
    auto it0 = singleChByDie.begin();
    auto it1 = std::next(it0);
    if (it0->second.size() > it1->second.size()) {
        std::swap(it0, it1);
    }
    meshDieId = it0->first;
    HCCL_INFO(
        "[CcuTempReduceScatterMeshMem2Mem1D2Die][PartitionByTwoDieRegular] Rank[%u] two_die_regular, "
        "die[%u] channels[%u] -> meshDieId, die[%u] channels[%u] -> closDieId.",
        myRank_, it0->first, it0->second.size(), it1->first, it1->second.size());
    for (const auto& ch : it0->second) {
        channels_[it0->first].emplace_back(ch);
        rankGroup_[it0->first].push_back(ch.remoteRank);
    }
    for (const auto& ch : it1->second) {
        channels_[it1->first].emplace_back(ch);
        rankGroup_[it1->first].push_back(ch.remoteRank);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMeshMem2Mem1D2Die::PartitionChannels(
    HcclComm comm, const std::vector<HcclChannelDesc>& channelDescs, uint32_t& meshDieId,
    std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc)
{
    (void)channelDescs;
    std::map<uint32_t, std::vector<HcclChannelDesc>> singleChByDie;
    std::map<uint32_t, std::vector<HcclChannelDesc>> multiChByDie;
    bool hasMultiChannel = false;
    CHK_RET(ClassifyChannelsByDie(comm, rankIdToChannelDesc, singleChByDie, multiChByDie, hasMultiChannel));

    if (hasMultiChannel) {
        CHK_RET(PartitionByMesh1dClos(singleChByDie, multiChByDie, meshDieId));
    } else {
        CHK_RET(PartitionByTwoDieRegular(singleChByDie, meshDieId));
    }

    // myRank_ 加入 meshDieId 的 rankGroup_ 末尾，与 kernel 内 subRankGroup[myRankIdx] == rankId 判断对齐
    rankGroup_[meshDieId].push_back(myRank_);

    HCCL_INFO(
        "[CcuTempReduceScatterMeshMem2Mem1D2Die][PartitionChannels] Rank[%u], hasMultiChannel[%d], "
        "meshDieId[%u], die0 channels[%u] rankGroup[%u], die1 channels[%u] rankGroup[%u].",
        myRank_, hasMultiChannel, meshDieId, channels_[0].size(), rankGroup_[0].size(), channels_[1].size(),
        rankGroup_[1].size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMeshMem2Mem1D2Die::KernelRun(
    const OpParam& param, const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t repeatNumTmp = templateDataParams.repeatNum;
    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t scratchAddr = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    uint64_t inputSliceStride = templateDataParams.inputSliceStride;
    uint64_t inputRepeatStride = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize = templateDataParams.sliceSize;
    uint64_t lastSliceSize = templateDataParams.tailSize;

    uint64_t repeatNum = UINT64_MAX - repeatNumTmp;

    uint64_t currentRankSliceInputOffset = inputSliceStride * myRank_;

    LoopGroupConfig config{};
    config.msInterleave = REDUCE_MS_CNT;
    config.loopCount = REDUCE_SCATTER_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE;
    auto goSize = CalGoSize(normalSliceSize, config);

    std::vector<uint64_t> taskArgs
        = {inputAddr,         outputAddr,         token,           scratchAddr, currentRankSliceInputOffset,
           inputRepeatStride, outputRepeatStride, normalSliceSize, repeatNum,   goSize[0],
           goSize[1],         goSize[2],          goSize[3]};

    HCCL_INFO(
        "[CcuTempReduceScatterMeshMem2Mem1D2Die::KernelRun] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
        "scratchAddr[%llu], currentRankSliceInputOffset[%llu], inputRepeatStride[%llu], "
        "outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu], repeatNum[%llu]",
        inputAddr, outputAddr, scratchAddr, currentRankSliceInputOffset, inputRepeatStride, outputRepeatStride,
        normalSliceSize, lastSliceSize, repeatNumTmp);

    // 前流同步
    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    std::vector<u32> notifyIdxMainToSub(1, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));

    for (uint64_t dieIdx = 0; dieIdx < DIE_NUM; dieIdx++) {
        CcuResult launchRet = HcommCcuKernelLaunch(
            templateResource.threads[dieIdx], templateResource.ccuKernels[dieIdx], taskArgs.data(), taskArgs.size());
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR(
                "[CcuTempReduceScatterMeshMem2Mem1D2Die::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    // 后流同步
    std::vector<u32> notifyIdxSubToMain(1, 0);
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));

    // 使用SDMA inlinereduce做localcopy scratch -> output
    uint64_t hcclBuffOffset = buffInfo_.hcclBuffBaseOff;
    hcclBuffOffset += normalSliceSize * (templateRankSize_ / DIE_NUM);
    DataSlice srcSlice(
        buffInfo_.hcclBuff.addr, hcclBuffOffset, normalSliceSize,
        normalSliceSize / DATATYPE_SIZE_TABLE[param.DataDes.dataType]);
    DataSlice dstSlice(
        buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, normalSliceSize,
        normalSliceSize / DATATYPE_SIZE_TABLE[param.DataDes.dataType]);
    CHK_RET(LocalReduce(templateResource.threads[0], srcSlice, dstSlice, param.DataDes.dataType, param.reduceType));

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceScatterMeshMem2Mem1D2Die::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

u64 CcuTempReduceScatterMeshMem2Mem1D2Die::GetThreadNum() const { return DIE_NUM; }

HcclResult CcuTempReduceScatterMeshMem2Mem1D2Die::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
