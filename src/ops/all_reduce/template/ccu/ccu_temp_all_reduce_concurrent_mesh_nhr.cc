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
#include "alg_data_trans_wrapper.h"
#include "ccu_launch_dl.h"
#include "ccu_temp_all_reduce_concurrent_mesh_nhr.h"

namespace ops_hccl {

constexpr u32 CLOS_PORT_NUM_AR = 8;
constexpr u32 MESH_THREAD_NUM_AR = 1;
constexpr u32 CONCURRENT_LINK_TYPE_NUM = 2; // mesh + nhr 两种链路

CcuTempAllReduceConcurrentMeshNHR::CcuTempAllReduceConcurrentMeshNHR(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    if (subCommRanks.size() >= 1) {
        meshGroup_ = subCommRanks[0];
    }
    if (subCommRanks.size() >= 2) {
        nhrGroup_ = subCommRanks[1];
    }
    rankSize_ = meshGroup_.size();
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;

    auto itMesh = std::find(meshGroup_.begin(), meshGroup_.end(), rankId);
    if (itMesh != meshGroup_.end()) {
        myMeshRank_ = std::distance(meshGroup_.begin(), itMesh);
    }
    auto itNhr = std::find(nhrGroup_.begin(), nhrGroup_.end(), rankId);
    if (itNhr != nhrGroup_.end()) {
        myNhrRank_ = std::distance(nhrGroup_.begin(), itNhr);
    }
}

CcuTempAllReduceConcurrentMeshNHR::~CcuTempAllReduceConcurrentMeshNHR() {}

u64 CcuTempAllReduceConcurrentMeshNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempAllReduceConcurrentMeshNHR::GetThreadNum() const
{
    constexpr u64 MESH_THREADS = 1;
    constexpr u64 NHR_THREADS = 2;
    return MESH_THREADS + NHR_THREADS;
}

HcclResult CcuTempAllReduceConcurrentMeshNHR::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest = mergedReq_;
    return HCCL_SUCCESS;
}

// ==================== NHR 算法逻辑复刻 ====================

HcclResult CcuTempAllReduceConcurrentMeshNHR::CalcSlice(u64 dataSize, RankSliceInfo& sliceInfoVec) const
{
    sliceInfoVec.clear();
    sliceInfoVec.resize(nhrGroup_.size());
    if (dataSize == 0) {
        SliceInfo empty{0, 0};
        for (u32 i = 0; i < nhrGroup_.size(); i++) {
            sliceInfoVec[i].push_back(empty);
        }
        return HCCL_SUCCESS;
    }
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    u64 unitPerSlice = dataSize / dataSizePerVolume / nhrGroup_.size();
    u64 accumOff = 0;
    SliceInfo currSlice;
    for (u32 rankIdx = 0; rankIdx < nhrGroup_.size(); rankIdx++) {
        if (rankIdx == nhrGroup_.size() - 1) {
            currSlice.offset = accumOff;
            currSlice.size = dataSize - accumOff;
        } else {
            currSlice.offset = accumOff;
            currSlice.size = unitPerSlice * dataSizePerVolume;
        }
        sliceInfoVec[rankIdx].push_back(currSlice);
        accumOff += currSlice.size;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceConcurrentMeshNHR::GetReduceScatterStepInfo(u32 step, NHRStepInfo& stepInfo) const
{
    u32 virtRankIdx = myNhrRank_;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    const std::vector<u32>& ranks = nhrGroup_;
    u32 rs = nhrGroup_.size();
    stepInfo.step = step;
    stepInfo.myRank = virtRankIdx;

    u32 deltaRank = 1u << step;
    u32 sendTo = (virtRankIdx + rs - deltaRank) % rs;
    u32 recvFrom = (virtRankIdx + deltaRank) % rs;
    u32 nSlices = (rs - 1 + (1u << step)) / (1u << (step + 1));
    u32 deltaSliceIndex = 1u << (step + 1);
    u32 rxSliceIdx = virtRankIdx;
    u32 txSliceIdx = (virtRankIdx - (1u << step) + rs) % rs;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = ranks[sendTo];
    stepInfo.fromRank = ranks[recvFrom];

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
        txSliceIdx = (txSliceIdx + rs - deltaSliceIndex) % rs;
        rxSliceIdx = (rxSliceIdx + rs - deltaSliceIndex) % rs;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceConcurrentMeshNHR::GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo) const
{
    u32 virtRankIdx = myNhrRank_;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    const std::vector<u32>& ranks = nhrGroup_;
    u32 rs = nhrGroup_.size();
    stepInfo.step = step;
    stepInfo.myRank = virtRankIdx;

    u32 deltaRank = 1u << (nSteps - 1 - step);
    u32 recvFrom = (virtRankIdx + rs - deltaRank) % rs;
    u32 sendTo = (virtRankIdx + deltaRank) % rs;
    u32 nSlices = (rs - 1 + (1u << (nSteps - 1 - step))) / (1u << (nSteps - step));
    u32 deltaSliceIndex = 1u << (nSteps - step);
    u32 txSliceIdx = virtRankIdx;
    u32 rxSliceIdx = (virtRankIdx - (1u << (nSteps - 1 - step)) + rs) % rs;

    stepInfo.toRank = ranks[sendTo];
    stepInfo.nSlices = nSlices;
    stepInfo.fromRank = ranks[recvFrom];

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
        txSliceIdx = (txSliceIdx + rs - deltaSliceIndex) % rs;
        rxSliceIdx = (rxSliceIdx + rs - deltaSliceIndex) % rs;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceConcurrentMeshNHR::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo) const
{
    u32 nStepsNHR = nSteps / 2;
    if (step < nStepsNHR) {
        return GetReduceScatterStepInfo(step, stepInfo);
    }
    return GetAllGatherStepInfo(step % nStepsNHR, nStepsNHR, stepInfo);
}

HcclResult CcuTempAllReduceConcurrentMeshNHR::ProcessNHRStepInfo(
    HcclComm comm, u32 enableDieNum, u32 enableDieId, std::vector<NHRStepInfo>& stepInfoVector,
    std::map<u32, u32>& rank2ChannelIdx, std::vector<std::vector<HcclChannelDesc>>& channelsPerDie)
{
    constexpr u32 DIE_NUM_1 = 1;
    constexpr u32 STAG_NUM_2 = 2;
    u32 nSteps = STAG_NUM_2 * GetNHRStepNum(nhrGroup_.size());
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
        if (enableDieNum == DIE_NUM_1) {
            CHK_RET(SelectChannelToVec(
                comm, myRank_, stepInfo.fromRank, nhrRankIdToChannelDesc_, enableDieId, rank2ChannelIdx,
                channelsPerDie[0]));
            CHK_RET(SelectChannelToVec(
                comm, myRank_, stepInfo.toRank, nhrRankIdToChannelDesc_, enableDieId, rank2ChannelIdx,
                channelsPerDie[0]));
        } else {
            CHK_RET(SelectChannelToVec(
                comm, myRank_, stepInfo.fromRank, nhrRankIdToChannelDesc_, 0, rank2ChannelIdx, channelsPerDie[0]));
            CHK_RET(SelectChannelToVec(
                comm, myRank_, stepInfo.fromRank, nhrRankIdToChannelDesc_, 1, rank2ChannelIdx, channelsPerDie[1]));
            CHK_RET(SelectChannelToVec(
                comm, myRank_, stepInfo.toRank, nhrRankIdToChannelDesc_, 0, rank2ChannelIdx, channelsPerDie[0]));
            CHK_RET(SelectChannelToVec(
                comm, myRank_, stepInfo.toRank, nhrRankIdToChannelDesc_, 1, rank2ChannelIdx, channelsPerDie[1]));
        }
    }
    return HCCL_SUCCESS;
}

// ==================== CalcRes（直接创建 kernelInfo） ====================

HcclResult CcuTempAllReduceConcurrentMeshNHR::CalcMeshRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, CcuKernelInfo& meshKernelInfo)
{
    std::vector<HcclChannelDesc> myChannelDescs;
    CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(
        comm, param, topoInfo, std::vector<std::vector<u32>>{meshGroup_}, myChannelDescs, CommTopo::COMM_TOPO_1DMESH));
    std::vector<HcclChannelDesc> channelDescs;
    for (const auto& ch : myChannelDescs) {
        if (ch.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
            channelDescs.push_back(ch);
        }
    }
    CHK_PRT_RET(
        channelDescs.empty(), HCCL_ERROR("[CcuTempAllReduceConcurrentMeshNHR][CalcMeshRes] mesh channelDescs is empty"),
        HCCL_E_INTERNAL);

    CHK_SAFETY_FUNC_RET(
        strcpy_s(meshKernelInfo.kernelFuncName, sizeof(meshKernelInfo.kernelFuncName), "CcuKernelAllReduceMesh1D"));
    meshKernelInfo.kernelFunc = reinterpret_cast<void*>(CcuAllReduceMesh1DKernel);
    auto kernelArg = std::make_shared<CcuKernelArgAllReduceMesh1D>();
    kernelArg->rankSize = meshGroup_.size();
    kernelArg->rankId = myMeshRank_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = std::vector<std::vector<u32>>{meshGroup_};
    meshKernelInfo.setKernelArg(kernelArg);
    meshKernelInfo.channels = channelDescs;
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceConcurrentMeshNHR::CalcNhrRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, CcuKernelInfo& nhrKernelInfo)
{
    std::vector<HcclChannelDesc> myChannelDescs;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, std::vector<std::vector<u32>>{nhrGroup_}, myChannelDescs));
    std::vector<HcclChannelDesc> channelDescs;
    for (const auto& ch : myChannelDescs) {
        if (ch.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
            channelDescs.push_back(ch);
        }
    }
    CHK_RET(RestoreChannelMap(channelDescs, nhrRankIdToChannelDesc_));

    u32 enableDieNum = 0;
    u32 enableDieId = 0;
    CHK_RET(GetDieInfoFromChannelDescs(comm, nhrRankIdToChannelDesc_, myRank_, enableDieNum, enableDieId));
    CHK_PRT_RET(
        enableDieNum < 1 || enableDieNum > CCU_DIE_NUM_MAX_2,
        HCCL_ERROR("[CcuTempAllReduceConcurrentMeshNHR][CalcNhrRes] enableDieNum[%u] invalid", enableDieNum),
        HCCL_E_INTERNAL);

    std::vector<std::vector<HcclChannelDesc>> channelsPerDie(enableDieNum);
    std::map<u32, u32> rank2ChannelIdx;
    std::vector<NHRStepInfo> stepInfoVector;
    CHK_RET(ProcessNHRStepInfo(comm, enableDieNum, enableDieId, stepInfoVector, rank2ChannelIdx, channelsPerDie));
    if (enableDieNum > 1) {
        CHK_RET(ReverseChannelPerDieIfNeed(comm, myRank_, channelsPerDie));
    }

    CHK_SAFETY_FUNC_RET(
        strcpy_s(nhrKernelInfo.kernelFuncName, sizeof(nhrKernelInfo.kernelFuncName), "CcuKernelAllReduceNHR1D"));
    nhrKernelInfo.kernelFunc = reinterpret_cast<void*>(CcuAllReduceNHR1DKernel);
    auto kernelArg = std::make_shared<CcuKernelArgAllReduceNHR1D>();
    kernelArg->rankSize = nhrGroup_.size();
    kernelArg->rankId = myNhrRank_;
    kernelArg->axisId = 0;
    kernelArg->axisSize = enableDieNum;
    kernelArg->stepInfoVector = stepInfoVector;
    kernelArg->indexMap = rank2ChannelIdx;
    kernelArg->opParam = param;
    kernelArg->tempVTopo = std::vector<std::vector<u32>>{nhrGroup_};
    nhrKernelInfo.setKernelArg(kernelArg);
    nhrKernelInfo.channels = channelsPerDie[0];
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceConcurrentMeshNHR::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_PRT_RET(
        topoInfo == nullptr, HCCL_ERROR("[CcuTempAllReduceConcurrentMeshNHR][CalcRes] topoInfo is nullptr"),
        HCCL_E_PARA);

    // 合并资源：mesh 0 从流、nhr 1 从流 + 1 跨模板同步 → slaveThreadNum=2
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 2;
    resourceRequest.notifyNumPerThread = {2, 1};
    resourceRequest.ccuKernelNum.push_back(MESH_THREAD_NUM_AR);
    resourceRequest.ccuKernelNum.push_back(1);

    CcuKernelInfo meshKernelInfo;
    CHK_RET(CalcMeshRes(comm, param, topoInfo, meshKernelInfo));
    CcuKernelInfo nhrKernelInfo;
    CHK_RET(CalcNhrRes(comm, param, topoInfo, nhrKernelInfo));
    resourceRequest.ccuKernelInfos.push_back(meshKernelInfo);
    resourceRequest.ccuKernelInfos.push_back(nhrKernelInfo);

    mergedReq_ = resourceRequest;
    HCCL_INFO(
        "[CcuTempAllReduceConcurrentMeshNHR][CalcRes] success, meshRank[%u], nhrRank[%u]", myMeshRank_, myNhrRank_);
    return HCCL_SUCCESS;
}

// ==================== KernelRun ====================

HcclResult CcuTempAllReduceConcurrentMeshNHR::CalcDataSplit(
    const TemplateDataParams& templateDataParams, TemplateDataParams& meshParams, TemplateDataParams& nhrParams,
    u64& meshCount, u64& nhrCount) const
{
    u64 portNum0 = (rankSize_ > 0) ? (rankSize_ - 1) : 0;
    u64 portNum1 = CLOS_PORT_NUM_AR;

    const u64 sliceAlignCount = (dataTypeSize_ > 0) ? (HCCL_MIN_SLICE_ALIGN / dataTypeSize_) : 1;
    const u64 dataCount = templateDataParams.count;
    meshCount = (portNum0 + portNum1) > 0 ? (dataCount * portNum0 / (portNum0 + portNum1)) :
                                            (dataCount / CONCURRENT_LINK_TYPE_NUM);
    meshCount = meshCount / sliceAlignCount * sliceAlignCount;
    nhrCount = (dataCount > meshCount) ? (dataCount - meshCount) : 0;

    meshParams = templateDataParams;
    meshParams.count = meshCount;
    meshParams.buffInfo.inBuffBaseOff = templateDataParams.buffInfo.inBuffBaseOff;
    meshParams.buffInfo.outBuffBaseOff = templateDataParams.buffInfo.outBuffBaseOff;
    meshParams.sliceSize = meshCount * dataTypeSize_;
    meshParams.tailSize = meshParams.sliceSize;

    nhrParams = templateDataParams;
    nhrParams.count = nhrCount;
    u64 nhrDataOff = templateDataParams.buffInfo.inBuffBaseOff + meshCount * dataTypeSize_;
    nhrParams.buffInfo.inBuffBaseOff = nhrDataOff;
    nhrParams.buffInfo.outBuffBaseOff = nhrDataOff;
    nhrParams.sliceSize = nhrCount * dataTypeSize_;
    nhrParams.tailSize = nhrParams.sliceSize;

    HCCL_INFO(
        "[CcuTempAllReduceConcurrentMeshNHR][CalcDataSplit] dataCount[%llu], meshCount[%llu], nhrCount[%llu]",
        dataCount, meshCount, nhrCount);
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceConcurrentMeshNHR::LaunchMeshKernel(
    const TemplateDataParams& meshParams, TemplateResource& templateResource, u64 baseInputAddr, u64 baseOutputAddr,
    uint64_t token, const LoopGroupConfig& config)
{
    u64 inputAddr = baseInputAddr + meshParams.buffInfo.inBuffBaseOff;
    u64 outputAddr = baseOutputAddr + meshParams.buffInfo.outBuffBaseOff;
    RankSliceInfo meshSliceInfoVec;
    CHK_RET(CalcSlice(meshParams.sliceSize, meshSliceInfoVec));
    u64 offset = meshSliceInfoVec[myMeshRank_][0].offset;
    u64 meshPerRankSize = meshSliceInfoVec[myMeshRank_][0].size;
    auto goSize = CalGoSize(meshPerRankSize, config, GetCcuVersion());
    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, offset, goSize[0], goSize[1], goSize[2], goSize[3]};
    CcuResult launchRet = HcommCcuKernelLaunch(
        templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), taskArgs.size());
    CHK_PRT_RET(
        launchRet != CCU_SUCCESS,
        HCCL_ERROR("[CcuTempAllReduceConcurrentMeshNHR][LaunchMeshKernel] launch failed, ccuRet -> %d", launchRet),
        ConvertCcuToHccl(launchRet));
    CcuKernelSubmitInfo meshSubmit;
    meshSubmit.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(
        meshSubmit, inputAddr, outputAddr, token, offset, goSize[0], goSize[1], goSize[2], goSize[3],
        meshParams.buffInfo.inBuffBaseOff, meshParams.buffInfo.outBuffBaseOff));
    templateResource.submitInfos.push_back(meshSubmit);
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceConcurrentMeshNHR::LaunchNhrKernel(
    const TemplateDataParams& nhrParams, TemplateResource& templateResource, u64 baseInputAddr, u64 baseOutputAddr,
    uint64_t token)
{
    u64 inputAddr = baseInputAddr + nhrParams.buffInfo.inBuffBaseOff;
    u64 outputAddr = baseOutputAddr + nhrParams.buffInfo.outBuffBaseOff;
    u64 die0Size = nhrParams.sliceSize;
    u64 die1Size = 0;
    u64 isInputOutputEqual = (inputAddr == outputAddr) ? 1 : 0;
    RankSliceInfo die0SliceInfoVec;
    RankSliceInfo die1SliceInfoVec;
    CHK_RET(CalcSlice(die0Size, die0SliceInfoVec));
    CHK_RET(CalcSlice(die1Size, die1SliceInfoVec));
    u32 nhrRankSize = nhrGroup_.size();
    std::vector<uint64_t> taskArgs
        = {inputAddr,
           outputAddr,
           token,
           isInputOutputEqual,
           die0Size,
           die1Size,
           die0SliceInfoVec[0][0].size,
           die1SliceInfoVec[0][0].size,
           die0SliceInfoVec[nhrRankSize - 1][0].size,
           die1SliceInfoVec[nhrRankSize - 1][0].size};
    CcuResult launchRet = HcommCcuKernelLaunch(
        templateResource.threads[MESH_THREAD_NUM_AR], templateResource.ccuKernels[1], taskArgs.data(), taskArgs.size());
    CHK_PRT_RET(
        launchRet != CCU_SUCCESS,
        HCCL_ERROR("[CcuTempAllReduceConcurrentMeshNHR][LaunchNhrKernel] launch failed, ccuRet -> %d", launchRet),
        ConvertCcuToHccl(launchRet));
    CcuKernelSubmitInfo nhrSubmit;
    nhrSubmit.kernelHandle = templateResource.ccuKernels[1];
    CHK_RET(FillCachedArgs(
        nhrSubmit, inputAddr, outputAddr, token, isInputOutputEqual, die0Size, die1Size, die0SliceInfoVec[0][0].size,
        die1SliceInfoVec[0][0].size, die0SliceInfoVec[nhrRankSize - 1][0].size,
        die1SliceInfoVec[nhrRankSize - 1][0].size, nhrParams.buffInfo.inBuffBaseOff,
        nhrParams.buffInfo.outBuffBaseOff));
    templateResource.submitInfos.push_back(nhrSubmit);
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceConcurrentMeshNHR::KernelRun(
    const OpParam& param, const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    (void)param;
    TemplateDataParams meshParams;
    TemplateDataParams nhrParams;
    u64 meshCount = 0;
    u64 nhrCount = 0;
    CHK_RET(CalcDataSplit(templateDataParams, meshParams, nhrParams, meshCount, nhrCount));

    const u64 baseInputAddr = PointerToAddr(templateDataParams.buffInfo.inputPtr);
    const u64 baseOutputAddr = PointerToAddr(templateDataParams.buffInfo.outputPtr);
    u64 nhrDataOff = templateDataParams.buffInfo.inBuffBaseOff + meshCount * dataTypeSize_;
    HCCL_INFO(
        "[Concurrent][KernelRun] meshCount[%llu], nhrCount[%llu], nhrDataOff[%llu], meshOutputAddr[%llx], "
        "nhrOutputAddr[%llx]",
        meshCount, nhrCount, nhrDataOff, baseOutputAddr, baseOutputAddr + nhrDataOff);
    uint64_t token;
    CHK_RET(GetToken(templateDataParams.buffInfo, token));
    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_DEFAULT_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE;

    ThreadHandle meshMain = templateResource.threads[0];
    ThreadHandle nhrMain = templateResource.threads[MESH_THREAD_NUM_AR];
    std::vector<ThreadHandle> subThreads = {nhrMain};
    CHK_RET(PreSyncInterThreads(meshMain, subThreads, {1}));

    if (meshCount > 0 && meshParams.sliceSize > 0) {
        CHK_RET(LaunchMeshKernel(meshParams, templateResource, baseInputAddr, baseOutputAddr, token, config));
    }
    if (nhrCount > 0 && nhrParams.sliceSize > 0) {
        CHK_RET(LaunchNhrKernel(nhrParams, templateResource, baseInputAddr, baseOutputAddr, token));
    }

    CHK_RET(PostSyncInterThreads(meshMain, subThreads, {0}));

    HCCL_INFO(
        "[CcuTempAllReduceConcurrentMeshNHR][KernelRun] done, meshCount[%llu], nhrCount[%llu]", meshCount, nhrCount);
    return HCCL_SUCCESS;
}

// ==================== FastLaunch ====================

HcclResult
CcuTempAllReduceConcurrentMeshNHR::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    (void)param;
    const auto& submitInfos = tempFastLaunchCtx.ccuKernelSubmitInfos;
    if (submitInfos.size() < 2) {
        HCCL_INFO("[CcuTempAllReduceConcurrentMeshNHR][FastLaunch] submitInfos[%zu] < 2, skip", submitInfos.size());
        return HCCL_SUCCESS;
    }

    ThreadHandle meshMain = tempFastLaunchCtx.threads[0];
    ThreadHandle nhrMain = tempFastLaunchCtx.threads[MESH_THREAD_NUM_AR];
    std::vector<ThreadHandle> subThreads = {nhrMain};

    std::vector<u32> notifyIdxMainToSub = {1};
    std::vector<u32> notifyIdxSubToMain = {0};
    CHK_RET(PreSyncInterThreads(meshMain, subThreads, notifyIdxMainToSub));

    // mesh 重放（submitInfos[0]）：argSize=8, offset idx 8/9
    {
        uint64_t* args = const_cast<uint64_t*>(submitInfos[0].cachedArgs);
        args[0] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[8];
        args[1] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[9];
        CcuResult ret = HcommCcuKernelLaunch(meshMain, submitInfos[0].kernelHandle, reinterpret_cast<void*>(args), 8);
        CHK_PRT_RET(
            ret != CCU_SUCCESS,
            HCCL_ERROR("[CcuTempAllReduceConcurrentMeshNHR][FastLaunch] mesh launch failed, ccuRet -> %d", ret),
            ConvertCcuToHccl(ret));
    }

    // nhr 重放（submitInfos[1]）：argSize=10, offset idx 10/11
    {
        uint64_t* args = const_cast<uint64_t*>(submitInfos[1].cachedArgs);
        args[0] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[10];
        args[1] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[11];
        CcuResult ret = HcommCcuKernelLaunch(nhrMain, submitInfos[1].kernelHandle, reinterpret_cast<void*>(args), 10);
        CHK_PRT_RET(
            ret != CCU_SUCCESS,
            HCCL_ERROR("[CcuTempAllReduceConcurrentMeshNHR][FastLaunch] nhr launch failed, ccuRet -> %d", ret),
            ConvertCcuToHccl(ret));
    }

    CHK_RET(PostSyncInterThreads(meshMain, subThreads, notifyIdxSubToMain));

    HCCL_INFO("[CcuTempAllReduceConcurrentMeshNHR][FastLaunch] end");
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
