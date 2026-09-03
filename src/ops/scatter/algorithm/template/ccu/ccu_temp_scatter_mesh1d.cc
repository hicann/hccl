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
#include "ccu_kernel_scatter_mesh1d.h"
#include "ccu_temp_scatter_mesh1d.h"
#include "ccu_launch_dl.h"
#include "cost_model.h"

namespace ops_hccl {

std::vector<CostModelParam> CcuTempScatterMesh1D::CalcCostCoeff(CalcCostCoeffParam param)
{
    // Mesh 算法走 CLOS 时取 portNum[0]（单通道语义，不求和）；MESH 分支 portNum 不参与
    int portNum = static_cast<int>(param.portNum[0]);
    int kernelNum = 1; // 单 kernel 下发
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    CostModelManager::Global()->CalcMeshParam(param.dataRatio, param.netType, portNum, param.rankSize, A, param.isPod);
    // B=0：kernel 内 root 自留份 GroupCopy 与 7 份远端 Write 并发执行
    // （ccu_kernel_scatter_mesh1d.cc DoScatterOnce 注释"Write与GroupCopy并行执行"），
    // 本地 1 份搬运被远端传输掩盖，关键路径上无独立贡献
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::CCU, C);
    CostModelManager::Global()->CalcLaunchParams(
        CostModelManager::CalcTransTaskNum(param.rankSize), EngineType::CCU,
        D); // CCU 的 D 恒为 0（kernel launch 无展开）

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

CcuTempScatterMesh1D::CcuTempScatterMesh1D(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    HCCL_INFO("Start to run CcuTempScatterMesh1D");
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    auto itRoot = std::find(ranks.begin(), ranks.end(), param.root);
    if (itRoot != ranks.end()) {
        subCommRootId_ = std::distance(ranks.begin(), itRoot);
    }
}

CcuTempScatterMesh1D::~CcuTempScatterMesh1D() {}

void CcuTempScatterMesh1D::SetRoot(u32 root)
{
    HCCL_INFO("[CcuTempScatterMesh1D][SetRoot] myRank_ [%u], set root [%u] ", myRank_, root);
    std::vector<u32> ranks = subCommRanks_[0];
    std::string ranksStr = "";
    for (auto r : ranks) {
        ranksStr += std::to_string(r) + " ";
    }
    HCCL_INFO("[CcuTempScatterMesh1D][SetRoot] ranks = subCommRanks[0] is: %s", ranksStr.c_str());
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        subCommRootId_ = std::distance(ranks.begin(), itRoot);
    }
}

HcclResult CcuTempScatterMesh1D::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG(
        "[CcuTempScatterMesh1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    CHK_SAFETY_FUNC_RET(
        strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuScatterMesh1DKernel"));
    kernelInfo.kernelFunc = reinterpret_cast<void*>(CcuScatterMesh1DKernel);
    std::vector<HcclChannelDesc> channelDescs;
    if (topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(
            comm, param, topoInfo, subCommRanks_, myChannelDescs, CommTopo::COMM_TOPO_1DMESH));
        for (auto channel : myChannelDescs) {
            if (channel.channelProtocol == COMM_PROTOCOL_UB_CTP) {
                channelDescs.push_back(channel);
            }
        }
        HCCL_DEBUG("[CcuTempScatterMesh1D::CalcRes] Get Mesh Channel Success!");
    }
    auto kernelArg = std::make_shared<CcuKernelArgScatterMesh1D>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->rootId = subCommRootId_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG(
        "[CcuTempScatterMesh1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
        "ccuKernelInfos.size()=%llu",
        channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterMesh1D::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    (void)param;
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempScatterMesh1D::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempScatterMesh1D::FastLaunch] start");
    uint64_t* args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    buffInfo_ = tempFastLaunchCtx.buffInfo;
    constexpr u32 inputIdx = 0;
    constexpr u32 outputIdx = 1;
    args[inputIdx] = PointerToAddr(buffInfo_.inputPtr) + args[inputIdx];
    args[outputIdx] = PointerToAddr(buffInfo_.outputPtr) + args[outputIdx];
    void* taskArgs = reinterpret_cast<void*>(args);
    uint64_t argSize = 15;
    CcuResult launchRet = HcommCcuKernelLaunch(
        tempFastLaunchCtx.threads[0], tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempScatterMesh1D::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[CcuTempScatterMesh1D::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    (void)param;
    HCCL_INFO("[CcuTempScatterMesh1D] Template KernelRun start.");
    if (templateDataParams.sliceSize == 0 && templateDataParams.tailSize == 0) {
        HCCL_INFO("[CcuTempScatterMesh1D] sliceSize is 0, no need do, just success.");
        return HcclResult::HCCL_SUCCESS;
    }
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t repeatNumTmp = templateDataParams.repeatNum;
    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t inputSliceStride = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride = templateDataParams.outputSliceStride;
    uint64_t inputRepeatStride = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize = templateDataParams.sliceSize;
    uint64_t lastSliceSize = templateDataParams.tailSize;
    // root自身slice的大小：若root是最后一个rank则用lastSliceSize，否则用normalSliceSize
    uint64_t rootSliceSize = (subCommRootId_ == templateRankSize_ - 1) ? lastSliceSize : normalSliceSize;
    bool inputOutputEqual
        = (inputAddr + inputSliceStride * subCommRootId_ == outputAddr + outputSliceStride * subCommRootId_);
    uint64_t isInputOutputEqual = static_cast<uint64_t>(inputOutputEqual);
    uint64_t repeatNum = UINT64_MAX - repeatNumTmp;

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE * LOCAL_COPY_MS_PER_LOOP;
    auto goSize = CalGoSize(rootSliceSize, config);

    std::vector<uint64_t> taskArgs
        = {inputAddr,          outputAddr,         token,           inputSliceStride, outputSliceStride,
           inputRepeatStride,  outputRepeatStride, normalSliceSize, lastSliceSize,    repeatNum,
           isInputOutputEqual, goSize[0],          goSize[1],       goSize[2],        goSize[3]};
    CcuResult launchRet = HcommCcuKernelLaunch(
        templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), taskArgs.size());
    CHK_PRT_RET(
        (launchRet != CCU_SUCCESS),
        HCCL_ERROR("[CcuTempScatterMesh1D::KernelRun] kernel launch failed, ccuRet -> %d", launchRet),
        ConvertCcuToHccl(launchRet));

    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(
        submitInfo, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, token, inputSliceStride, outputSliceStride,
        inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, repeatNum, isInputOutputEqual, goSize[0],
        goSize[1], goSize[2], goSize[3]));
    templateResource.submitInfos.push_back(submitInfo);

    HCCL_INFO(
        "[CcuTempScatterMesh1D::KernelRun] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
        "inputSliceStride[%llu], outputSliceStride[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu], "
        "normalSliceSize[%llu], lastSliceSize[%llu], repeatNum[%llu], isInputOutputEqual[%llu]",
        inputAddr, outputAddr, inputSliceStride, outputSliceStride, inputRepeatStride, outputRepeatStride,
        normalSliceSize, lastSliceSize, repeatNumTmp, isInputOutputEqual);

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempScatterMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // scatter操作不需要scratch buffer
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempScatterMesh1D::GetThreadNum() const { return 1; }

HcclResult CcuTempScatterMesh1D::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
