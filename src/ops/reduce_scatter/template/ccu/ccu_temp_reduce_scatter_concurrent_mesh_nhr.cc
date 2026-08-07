/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_launch_dl.h"
#include "ccu_temp_reduce_scatter_concurrent_mesh_nhr.h"

namespace ops_hccl {

constexpr u32 CLOS_PORT_NUM_SERVER_V2_CC = 8;
constexpr u32 MESH_THREAD_NUM = 1;
constexpr u32 NHR_THREAD_NUM = 2;

CcuTempReduceScatterConcurrentMeshNHR::CcuTempReduceScatterConcurrentMeshNHR(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    if (subCommRanks.size() >= 2) {
        nhrGroup_ = subCommRanks[1];
    }
    if (subCommRanks.size() >= 1) {
        meshGroup_ = subCommRanks[0];
    }

    rankSize_ = meshGroup_.size();
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];

    auto itNhr = std::find(nhrGroup_.begin(), nhrGroup_.end(), rankId);
    if (itNhr != nhrGroup_.end()) {
        myNhrRank_ = std::distance(nhrGroup_.begin(), itNhr);
    }

    auto itMesh = std::find(meshGroup_.begin(), meshGroup_.end(), rankId);
    if (itMesh != meshGroup_.end()) {
        myMeshRank_ = std::distance(meshGroup_.begin(), itMesh);
    }
}

CcuTempReduceScatterConcurrentMeshNHR::~CcuTempReduceScatterConcurrentMeshNHR() {}

u64 CcuTempReduceScatterConcurrentMeshNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempReduceScatterConcurrentMeshNHR::GetThreadNum() const { return MESH_THREAD_NUM + NHR_THREAD_NUM; }

HcclResult CcuTempReduceScatterConcurrentMeshNHR::FastLaunchMeshKernel(
    const CcuKernelSubmitInfo& submitInfo, ThreadHandle meshMain, const BuffInfo& buffInfo)
{
    uint64_t* args = const_cast<uint64_t*>(submitInfo.cachedArgs);
    args[0] = PointerToAddr(buffInfo.inputPtr) + args[8];
    args[1] = PointerToAddr(buffInfo.outputPtr) + args[9];
    CcuResult launchRet = HcommCcuKernelLaunch(meshMain, submitInfo.kernelHandle, reinterpret_cast<void*>(args), 8);
    CHK_PRT_RET(
        launchRet != CCU_SUCCESS,
        HCCL_ERROR(
            "[CcuTempReduceScatterConcurrentMeshNHR][FastLaunchMeshKernel] launch failed, ccuRet -> %d", launchRet),
        ConvertCcuToHccl(launchRet));
    return HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::FastLaunchNhrKernel(
    const CcuKernelSubmitInfo& submitInfo, ThreadHandle nhrMain, const BuffInfo& buffInfo)
{
    uint64_t* args = const_cast<uint64_t*>(submitInfo.cachedArgs);
    constexpr u32 inputIdx = 0;
    constexpr u32 outputIdx = 1;
    constexpr u32 currentRankSliceOutputOffsetIdx = 8;
    constexpr u32 isInputOutputEqualIdx = 12;
    constexpr u32 inputOffsetIdx = 13;
    constexpr u32 outputOffsetIdx = 14;
    constexpr u32 currentRankSliceInputOffsetIdx = 15;
    constexpr u64 argSize = 13;
    uint64_t inputAddr = PointerToAddr(buffInfo.inputPtr) + args[inputOffsetIdx];
    uint64_t outputAddr = PointerToAddr(buffInfo.outputPtr) + args[outputOffsetIdx];
    uint64_t currentRankSliceInputOffset = args[currentRankSliceInputOffsetIdx];
    uint64_t currentRankSliceOutputOffset = args[currentRankSliceOutputOffsetIdx];
    bool inputOutputEqual = (inputAddr + currentRankSliceInputOffset == outputAddr + currentRankSliceOutputOffset);
    args[inputIdx] = inputAddr;
    args[outputIdx] = outputAddr;
    args[isInputOutputEqualIdx] = static_cast<uint64_t>(inputOutputEqual);
    CcuResult launchRet
        = HcommCcuKernelLaunch(nhrMain, submitInfo.kernelHandle, reinterpret_cast<void*>(args), argSize);
    CHK_PRT_RET(
        launchRet != CCU_SUCCESS,
        HCCL_ERROR(
            "[CcuTempReduceScatterConcurrentMeshNHR][FastLaunchNhrKernel] launch failed, ccuRet -> %d", launchRet),
        ConvertCcuToHccl(launchRet));
    return HCCL_SUCCESS;
}

HcclResult
CcuTempReduceScatterConcurrentMeshNHR::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    (void)param;
    const auto& submitInfos = tempFastLaunchCtx.ccuKernelSubmitInfos;
    if (submitInfos.size() < 2) {
        HCCL_INFO("[CcuTempReduceScatterConcurrentMeshNHR][FastLaunch] submitInfos[%zu] < 2, skip", submitInfos.size());
        return HCCL_SUCCESS;
    }

    ThreadHandle meshMain = tempFastLaunchCtx.threads[0];
    ThreadHandle nhrMain = tempFastLaunchCtx.threads[MESH_THREAD_NUM];
    std::vector<ThreadHandle> subThreads = {nhrMain};

    CHK_RET(PreSyncInterThreads(meshMain, subThreads, {1}));
    CHK_RET(FastLaunchMeshKernel(submitInfos[0], meshMain, tempFastLaunchCtx.buffInfo));
    CHK_RET(FastLaunchNhrKernel(submitInfos[1], nhrMain, tempFastLaunchCtx.buffInfo));
    CHK_RET(PostSyncInterThreads(meshMain, subThreads, {0}));

    HCCL_INFO("[CcuTempReduceScatterConcurrentMeshNHR][FastLaunch] end");
    return HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest = mergedReq_;
    return HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::CalcMeshRes(
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
        channelDescs.empty(),
        HCCL_ERROR("[CcuTempReduceScatterConcurrentMeshNHR][CalcMeshRes] mesh channelDescs is empty"), HCCL_E_INTERNAL);

    CHK_SAFETY_FUNC_RET(
        strcpy_s(meshKernelInfo.kernelFuncName, sizeof(meshKernelInfo.kernelFuncName), "CcuKernelReduceScatterMesh1D"));
    meshKernelInfo.kernelFunc = reinterpret_cast<void*>(CcuReduceScatterMesh1DKernel);
    auto kernelArg = std::make_shared<CcuKernelArgReduceScatterMesh1D>();
    kernelArg->rankSize = meshGroup_.size();
    kernelArg->rankId = myMeshRank_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = std::vector<std::vector<u32>>{meshGroup_};
    meshKernelInfo.setKernelArg(kernelArg);
    meshKernelInfo.channels = channelDescs;
    return HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::GetNHRStepInfo(u32 step, NHRStepInfo& stepInfo)
{
    const std::vector<u32>& ranks = nhrGroup_;
    const u32 rankSize = nhrGroup_.size();
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = myNhrRank_;

    u32 deltaRank = 1u << step;
    u32 sendTo = (myNhrRank_ + rankSize - deltaRank) % rankSize;
    u32 recvFrom = (myNhrRank_ + deltaRank) % rankSize;
    u32 nSlices = (rankSize - 1 + (1u << step)) / (1u << (step + 1));
    u32 deltaSliceIndex = 1u << (step + 1);
    u32 txSliceIdx = sendTo;
    u32 rxSliceIdx = myNhrRank_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = ranks[sendTo];
    stepInfo.fromRank = ranks[recvFrom];

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
        txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::ProcessNHRStepInfo(
    HcclComm comm, u32 enableDieNum, u32 enableDieId, std::vector<NHRStepInfo>& stepInfoVector,
    std::map<u32, u32>& rank2ChannelIdx, std::vector<std::vector<HcclChannelDesc>>& channelsPerDie)
{
    constexpr u32 DIE_NUM_1 = 1;
    constexpr u32 DIE_NUM_2 = 2;
    u32 nSteps = GetNHRStepNum(nhrGroup_.size());
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetNHRStepInfo(step, stepInfo));
        stepInfoVector.push_back(stepInfo);
        if (enableDieNum == DIE_NUM_1) {
            CHK_RET(SelectChannelToVec(
                comm, myRank_, stepInfo.fromRank, nhrRankIdToChannelDesc_, enableDieId, rank2ChannelIdx,
                channelsPerDie[0]));
            CHK_RET(SelectChannelToVec(
                comm, myRank_, stepInfo.toRank, nhrRankIdToChannelDesc_, enableDieId, rank2ChannelIdx,
                channelsPerDie[0]));
        } else if (enableDieNum == DIE_NUM_2) {
            CHK_RET(SelectChannelToVec(
                comm, myRank_, stepInfo.fromRank, nhrRankIdToChannelDesc_, 0, rank2ChannelIdx, channelsPerDie[0]));
            CHK_RET(SelectChannelToVec(
                comm, myRank_, stepInfo.toRank, nhrRankIdToChannelDesc_, 0, rank2ChannelIdx, channelsPerDie[0]));
            CHK_RET(SelectChannelToVec(
                comm, myRank_, stepInfo.fromRank, nhrRankIdToChannelDesc_, 1, rank2ChannelIdx, channelsPerDie[1]));
            CHK_RET(SelectChannelToVec(
                comm, myRank_, stepInfo.toRank, nhrRankIdToChannelDesc_, 1, rank2ChannelIdx, channelsPerDie[1]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::CalcNhrRes(
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
        HCCL_ERROR("[CcuTempReduceScatterConcurrentMeshNHR][CalcNhrRes] enableDieNum[%u] invalid", enableDieNum),
        HCCL_E_INTERNAL);

    std::vector<std::vector<HcclChannelDesc>> channelsPerDie(enableDieNum);
    std::map<u32, u32> rank2ChannelIdx;
    std::vector<NHRStepInfo> stepInfoVector;
    CHK_RET(ProcessNHRStepInfo(comm, enableDieNum, enableDieId, stepInfoVector, rank2ChannelIdx, channelsPerDie));
    if (enableDieNum > 1) {
        CHK_RET(ReverseChannelPerDieIfNeed(comm, myRank_, channelsPerDie));
    }

    CHK_SAFETY_FUNC_RET(strcpy_s(
        nhrKernelInfo.kernelFuncName, sizeof(nhrKernelInfo.kernelFuncName), "CcuReduceScatterNHR1DMem2MemKernel"));
    nhrKernelInfo.kernelFunc = reinterpret_cast<void*>(CcuReduceScatterNHR1DMem2MemKernel);
    auto kernelArg = std::make_shared<CcuKernelArgReduceScatterNHR1D>();
    kernelArg->dimSize = nhrGroup_.size();
    kernelArg->rankId = myNhrRank_;
    kernelArg->mySubCommRankId = myNhrRank_;
    kernelArg->axisId = 0;
    kernelArg->axisSize = enableDieNum;
    kernelArg->stepInfoVector = stepInfoVector;
    kernelArg->rank2ChannelIdx = rank2ChannelIdx;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = std::vector<std::vector<u32>>{nhrGroup_};
    nhrKernelInfo.setKernelArg(kernelArg);
    nhrKernelInfo.channels = channelsPerDie[0];
    return HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_PRT_RET(
        topoInfo == nullptr, HCCL_ERROR("[CcuTempReduceScatterConcurrentMeshNHR][CalcRes] topoInfo is nullptr"),
        HCCL_E_PARA);

    // mesh 路：0 从流、主线程 0 notify；nhr 路：1 从流、主线程 1 notify。
    // 合并后：主线程(mesh main) 1 notify(给 PostSync)，从流 nhr main 2 notify(自身+mesh→nhr)，nhr sub 1 notify
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = MESH_THREAD_NUM - 1 + NHR_THREAD_NUM - 1 + 1; // =2
    resourceRequest.notifyNumPerThread = {2, 1};                                   // nhr main 2 个，nhr sub 1 个
    resourceRequest.ccuKernelNum.push_back(MESH_THREAD_NUM);                       // mesh 1 个 kernel
    resourceRequest.ccuKernelNum.push_back(NHR_THREAD_NUM - 1);                    // nhr 1 个 kernel

    CcuKernelInfo meshKernelInfo;
    CHK_RET(CalcMeshRes(comm, param, topoInfo, meshKernelInfo));
    CcuKernelInfo nhrKernelInfo;
    CHK_RET(CalcNhrRes(comm, param, topoInfo, nhrKernelInfo));
    resourceRequest.ccuKernelInfos.push_back(meshKernelInfo);
    resourceRequest.ccuKernelInfos.push_back(nhrKernelInfo);

    mergedReq_ = resourceRequest;
    HCCL_INFO(
        "[CcuTempReduceScatterConcurrentMeshNHR][CalcRes] success, meshRank[%u], nhrRank[%u]", myMeshRank_, myNhrRank_);
    return HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::CalcDataSplit(
    const OpParam& param, const TemplateDataParams& templateDataParams, TemplateDataParams& meshParams,
    TemplateDataParams& nhrParams, u64& meshCount, u64& nhrCount) const
{
    (void)param;
    u64 portNum0 = rankSize_ - 1;
    u64 portNum = CLOS_PORT_NUM_SERVER_V2_CC;

    const u64 sliceAlignCount = (dataTypeSize_ > 0) ? (HCCL_MIN_SLICE_ALIGN / dataTypeSize_) : 1;
    const u64 dataCount = templateDataParams.count;
    meshCount = (portNum0 + portNum) > 0 ? (dataCount * portNum0 / (portNum0 + portNum)) : (dataCount / 2);
    meshCount = meshCount / sliceAlignCount * sliceAlignCount;
    nhrCount = (dataCount > meshCount) ? (dataCount - meshCount) : 0;

    meshParams = templateDataParams;
    meshParams.count = meshCount;
    meshParams.buffInfo.inBuffBaseOff = templateDataParams.buffInfo.inBuffBaseOff;
    meshParams.buffInfo.outBuffBaseOff = templateDataParams.buffInfo.outBuffBaseOff;
    meshParams.sliceSize = meshCount * dataTypeSize_;
    meshParams.tailSize = meshParams.sliceSize;
    meshParams.outputSliceStride = 0;

    nhrParams = templateDataParams;
    nhrParams.count = nhrCount;
    u64 nhrDataOff = templateDataParams.buffInfo.inBuffBaseOff + meshCount * dataTypeSize_;
    nhrParams.buffInfo.inBuffBaseOff = nhrDataOff;
    nhrParams.buffInfo.outBuffBaseOff = nhrDataOff;
    nhrParams.sliceSize = nhrCount * dataTypeSize_;
    nhrParams.tailSize = nhrParams.sliceSize;
    nhrParams.outputSliceStride = 0;

    HCCL_INFO(
        "[CcuTempReduceScatterConcurrentMeshNHR][CalcDataSplit] dataCount[%llu], meshCount[%llu], nhrCount[%llu]",
        dataCount, meshCount, nhrCount);
    return HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::LaunchMeshKernel(
    const TemplateDataParams& meshParams, TemplateResource& templateResource, u64 baseInputAddr, u64 baseOutputAddr,
    uint64_t token, const LoopGroupConfig& config)
{
    u64 inputAddr = baseInputAddr + meshParams.buffInfo.inBuffBaseOff;
    u64 outputAddr = baseOutputAddr + meshParams.buffInfo.outBuffBaseOff;
    u64 offset = meshParams.inputSliceStride * myMeshRank_;
    auto goSize = CalGoSize(meshParams.sliceSize, config, GetCcuVersion());
    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, offset, goSize[0], goSize[1], goSize[2], goSize[3]};
    CcuResult launchRet = HcommCcuKernelLaunch(
        templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), taskArgs.size());
    CHK_PRT_RET(
        launchRet != CCU_SUCCESS,
        HCCL_ERROR("[CcuTempReduceScatterConcurrentMeshNHR][LaunchMeshKernel] launch failed, ccuRet -> %d", launchRet),
        ConvertCcuToHccl(launchRet));
    CcuKernelSubmitInfo meshSubmit;
    meshSubmit.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(
        meshSubmit, inputAddr, outputAddr, token, offset, goSize[0], goSize[1], goSize[2], goSize[3],
        meshParams.buffInfo.inBuffBaseOff, meshParams.buffInfo.outBuffBaseOff));
    templateResource.submitInfos.push_back(meshSubmit);
    return HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::LaunchNhrKernel(
    const TemplateDataParams& nhrParams, TemplateResource& templateResource, u64 baseInputAddr, u64 baseOutputAddr,
    uint64_t token)
{
    u64 inputAddr = baseInputAddr + nhrParams.buffInfo.inBuffBaseOff;
    u64 outputAddr = baseOutputAddr + nhrParams.buffInfo.outBuffBaseOff;
    u64 die0Size = nhrParams.sliceSize;
    u64 die1Size = 0;
    u64 die0LastSliceSize = nhrParams.sliceSize;
    u64 die1LastSliceSize = 0;
    u64 inputSliceStride = nhrParams.inputSliceStride;
    u64 currentRankSliceOutputOffset = 0;
    u64 inputRepeatStride = 0;
    u64 outputRepeatStride = 0;
    u64 repeatNumVar = UINT64_MAX - nhrParams.repeatNum;
    u64 isInputOutputEqual = (inputAddr == outputAddr) ? 1 : 0;
    u64 currentRankSliceInputOffset = inputSliceStride * myNhrRank_;
    std::vector<uint64_t> taskArgs = {inputAddr,         outputAddr,         token,
                                      die0Size,          die1Size,           die0LastSliceSize,
                                      die1LastSliceSize, inputSliceStride,   currentRankSliceOutputOffset,
                                      inputRepeatStride, outputRepeatStride, repeatNumVar,
                                      isInputOutputEqual};
    CcuResult launchRet = HcommCcuKernelLaunch(
        templateResource.threads[MESH_THREAD_NUM], templateResource.ccuKernels[1], taskArgs.data(), taskArgs.size());
    CHK_PRT_RET(
        launchRet != CCU_SUCCESS,
        HCCL_ERROR("[CcuTempReduceScatterConcurrentMeshNHR][LaunchNhrKernel] launch failed, ccuRet -> %d", launchRet),
        ConvertCcuToHccl(launchRet));
    CcuKernelSubmitInfo nhrSubmit;
    nhrSubmit.kernelHandle = templateResource.ccuKernels[1];
    CHK_RET(FillCachedArgs(
        nhrSubmit, inputAddr, outputAddr, token, die0Size, die1Size, die0LastSliceSize, die1LastSliceSize,
        inputSliceStride, currentRankSliceOutputOffset, inputRepeatStride, outputRepeatStride, repeatNumVar,
        isInputOutputEqual, nhrParams.buffInfo.inBuffBaseOff, nhrParams.buffInfo.outBuffBaseOff,
        currentRankSliceInputOffset));
    templateResource.submitInfos.push_back(nhrSubmit);
    return HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterConcurrentMeshNHR::KernelRun(
    const OpParam& param, const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    TemplateDataParams meshParams;
    TemplateDataParams nhrParams;
    u64 meshCount = 0;
    u64 nhrCount = 0;
    CHK_RET(CalcDataSplit(param, templateDataParams, meshParams, nhrParams, meshCount, nhrCount));

    const u64 baseInputAddr = PointerToAddr(templateDataParams.buffInfo.inputPtr);
    const u64 baseOutputAddr = PointerToAddr(templateDataParams.buffInfo.outputPtr);
    uint64_t token;
    CHK_RET(GetToken(templateDataParams.buffInfo, token));
    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_DEFAULT_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE;

    // 前同步：mesh 主线程(threads[0]) 通知 nhr 主线程(threads[1])
    std::vector<ThreadHandle> nhrSubThreads(
        templateResource.threads.begin() + MESH_THREAD_NUM + 1, templateResource.threads.end());
    std::vector<ThreadHandle> subThreads = {templateResource.threads[MESH_THREAD_NUM]};
    std::vector<u32> notifyIdxMainToSub = {static_cast<u32>(nhrSubThreads.size())};
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));

    // 双路并行 launch
    if (meshCount > 0 && meshParams.sliceSize > 0) {
        CHK_RET(LaunchMeshKernel(meshParams, templateResource, baseInputAddr, baseOutputAddr, token, config));
    }
    if (nhrCount > 0 && nhrParams.sliceSize > 0) {
        CHK_RET(LaunchNhrKernel(nhrParams, templateResource, baseInputAddr, baseOutputAddr, token));
    }

    // 后同步
    std::vector<u32> notifyIdxSubToMain = {0};
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));

    HCCL_INFO(
        "[CcuTempReduceScatterConcurrentMeshNHR][KernelRun] done, meshCount[%llu], nhrCount[%llu]", meshCount,
        nhrCount);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
