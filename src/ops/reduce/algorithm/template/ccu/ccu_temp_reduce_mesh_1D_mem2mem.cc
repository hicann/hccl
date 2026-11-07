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
#include "ccu_kernel_reduce_mesh1d_mem2mem.h"
#include "ccu_temp_reduce_mesh_1D_mem2mem.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {
// cachedArgs 包含 argSize 和 4 个运行时偏移/规模字段。
constexpr size_t REDUCE_MESH_MEM2MEM_CACHED_ARG_OVERHEAD = 5;

std::vector<CostModelParam> CcuTempReduceMesh1DMem2Mem::CalcCostCoeff(CalcCostCoeffParam param)
{
    if (param.rankSize > REDUCE_MESH_MEM2MEM_MAX_RANK_SIZE) {
        return {};
    }
    int portNum = (param.isPod && param.netType == CommTopo::COMM_TOPO_CLOS && param.portNum.size() >= 2) ?
                      (param.portNum[0] + param.portNum[1]) :
                      param.portNum[0];
    int kernelNum = 2 * param.rankSize;
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;
    float n = param.dataRatio * param.rankSize;
    CostModelManager::Global()->CalcMeshParam(n, param.netType, portNum, param.rankSize, A, param.isPod);
    CostModelManager::Global()->CalcLocalCopyParams(n, EngineType::CCU, B);
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::CCU, C);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

CcuTempReduceMesh1DMem2Mem::CcuTempReduceMesh1DMem2Mem(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    templateRankSize_ = subCommRanks[0].size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
    auto rootIt = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), param.root);
    if (rootIt != subCommRanks[0].end()) {
        mySubCommRoot_ = std::distance(subCommRanks[0].begin(), rootIt);
    }
    dataType_ = param.DataDes.dataType;
}

CcuTempReduceMesh1DMem2Mem::~CcuTempReduceMesh1DMem2Mem() {}

void CcuTempReduceMesh1DMem2Mem::SetRoot(u32 root)
{
    std::vector<u32> ranks = subCommRanks_[0];
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        mySubCommRoot_ = std::distance(ranks.begin(), itRoot);
    }
    HCCL_INFO(
        "[CcuTempReduceMesh1DMem2Mem][SetRoot] myRank_ [%u], set root_ [%u] subCommRanks[%u]", mySubCommRank_, root,
        mySubCommRoot_);
}

HcclResult CcuTempReduceMesh1DMem2Mem::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_PRT_RET(
        templateRankSize_ > REDUCE_MESH_MEM2MEM_MAX_RANK_SIZE,
        HCCL_ERROR(
            "[CcuTempReduceMesh1DMem2Mem::CalcRes] rankSize[%u] exceeds max[%u]", templateRankSize_,
            REDUCE_MESH_MEM2MEM_MAX_RANK_SIZE),
        HcclResult::HCCL_E_NOT_SUPPORT);
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG(
        "[CcuTempReduceMesh1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] "
        "slaveThreadNum[%u]",
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    CHK_SAFETY_FUNC_RET(
        strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelReduceMesh1DMem2Mem"));
    kernelInfo.kernelFunc = reinterpret_cast<void*>(CcuReduceMesh1DMem2MemKernel);

    std::vector<HcclChannelDesc> channelDescs;
    if (topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(
            comm, param, topoInfo, subCommRanks_, myChannelDescs, CommTopo::COMM_TOPO_1DMESH));
        AppendChannelsByProtocol(myChannelDescs, COMM_PROTOCOL_UB_CTP, channelDescs);
        HCCL_DEBUG("[CcuTempReduceMesh1DMem2Mem::CalcRes] Get Mesh Channel Success!");
    }

    auto kernelArg = std::make_shared<CcuKernelArgReduceMesh1DMem2Mem>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->rootId = mySubCommRoot_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG(
        "[CcuTempReduceMesh1DMem2Mem::CalcRes] channelDescs.size()=%zu, dimsize=%zu, "
        "ccuKernelInfos.size()=%zu",
        channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

// 校验 N 个 scratch 槽能容纳当前 slice，避免地址越界。
static HcclResult CheckScratchStageable(
    const void* hcclBuffAddr, uint64_t hcclBuffSize, uint64_t hcclBuffBaseOff, uint64_t rankNum, uint64_t sliceSize,
    const char* caller)
{
    if (hcclBuffAddr == nullptr || rankNum == 0 || hcclBuffBaseOff > hcclBuffSize
        || sliceSize > (hcclBuffSize - hcclBuffBaseOff) / rankNum) {
        uint64_t maxSliceSize
            = (rankNum == 0 || hcclBuffBaseOff > hcclBuffSize) ? 0 : (hcclBuffSize - hcclBuffBaseOff) / rankNum;
        HCCL_ERROR(
            "[CcuTempReduceMesh1DMem2Mem][%s] invalid hcclBuff for scratch staging: addr[%p], size[%llu], "
            "baseOff[%llu], rankNum[%llu], sliceSize[%llu], maxSupportedSliceSize[%llu]",
            caller, hcclBuffAddr, hcclBuffSize, hcclBuffBaseOff, rankNum, sliceSize, maxSliceSize);
        return HcclResult::HCCL_E_MEMORY;
    }
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult CheckCachedArgCount(size_t taskArgCount)
{
    if (UNLIKELY(taskArgCount > CCU_MAX_TASK_ARG_NUM - REDUCE_MESH_MEM2MEM_CACHED_ARG_OVERHEAD)) {
        HCCL_ERROR(
            "[CcuTempReduceMesh1DMem2Mem] cached argument count[%zu] exceeds max[%d]",
            taskArgCount + REDUCE_MESH_MEM2MEM_CACHED_ARG_OVERHEAD, CCU_MAX_TASK_ARG_NUM);
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMesh1DMem2Mem::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    (void)param;
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempReduceMesh1DMem2Mem::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempReduceMesh1DMem2Mem::FastLaunch] start");
    uint64_t* args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    uint64_t argSize = args[0];
    // cachedArgs[0] 为 argSize，其余 taskArgs 索引顺延 1。
    constexpr u32 inputIdx = TASKARG_INPUT_ADDR + 1;
    constexpr u32 outputIdx = TASKARG_OUTPUT_ADDR + 1;
    constexpr u32 scratchIdx = TASKARG_SCRATCH_ADDR + 1;
    const u32 inputOffsetIdx = argSize + 1;
    const u32 outputOffsetIdx = argSize + 2;
    const u32 scratchOffsetIdx = argSize + 3;
    const u32 rankSizeIdx = argSize + 4;

    // FastLaunch 使用本次缓冲区，须重新校验 scratch 容量。
    CHK_RET(CheckScratchStageable(
        tempFastLaunchCtx.buffInfo.hcclBuff.addr, tempFastLaunchCtx.buffInfo.hcclBuff.size, args[scratchOffsetIdx],
        args[rankSizeIdx], args[TASKARG_NORMAL_SLICE_SIZE + 1], "FastLaunch"));

    args[inputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    args[outputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];
    // 使用本次 cclBuffer 地址重建 scratch 基址。
    args[scratchIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.hcclBuff.addr) + args[scratchOffsetIdx];

    void* taskArgs = reinterpret_cast<void*>(args + 1);
    CcuResult launchRet = HcommCcuKernelLaunch(
        tempFastLaunchCtx.threads[0], tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempReduceMesh1DMem2Mem::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[CcuTempReduceMesh1DMem2Mem::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMesh1DMem2Mem::KernelRun(
    const OpParam& param, const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    (void)param;
    CHK_PRT_RET(
        templateRankSize_ > REDUCE_MESH_MEM2MEM_MAX_RANK_SIZE,
        HCCL_ERROR(
            "[CcuTempReduceMesh1DMem2Mem::KernelRun] rankSize[%u] exceeds max[%u]", templateRankSize_,
            REDUCE_MESH_MEM2MEM_MAX_RANK_SIZE),
        HcclResult::HCCL_E_NOT_SUPPORT);
    if (templateDataParams.sliceSize == 0) {
        HCCL_INFO("[CcuTempReduceMesh1DMem2Mem] sliceSize is 0, no need do, just success.");
        return HCCL_SUCCESS;
    }

    buffInfo_ = templateDataParams.buffInfo;

    CHK_RET(CheckScratchStageable(
        buffInfo_.hcclBuff.addr, buffInfo_.hcclBuff.size, buffInfo_.hcclBuffBaseOff, templateRankSize_,
        templateDataParams.sliceSize, "KernelRun"));

    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t scratchAddr = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t repeatNum = templateDataParams.repeatNum;
    uint64_t inputRepeatStride = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize = templateDataParams.sliceSize;
    uint64_t repeatNumVar = UINT64_MAX - repeatNum;
    // 该模板不把此标志下发给 CCU kernel，仅用于主机侧排查输入输出地址关系
    const uint64_t inputOutputAddrEqual = (inputAddr == outputAddr) ? 1 : 0;

    HCCL_INFO(
        "[CcuTempReduceMesh1DMem2Mem::KernelRun] inputOutputAddrEqual[%llu], inputAddr[0x%llx], "
        "outputAddr[0x%llx], scratchAddr[0x%llx], repeatNum[%llu], inputRepeatStride[%llu], "
        "outputRepeatStride[%llu], normalSliceSize[%llu], repeatNumVar[%llu]",
        inputOutputAddrEqual, inputAddr, outputAddr, scratchAddr, repeatNum, inputRepeatStride, outputRepeatStride,
        normalSliceSize, repeatNumVar);
    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_M2M_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE;
    auto goSize = CalGoSize(normalSliceSize, config);

    std::vector<uint64_t> taskArgs(TASKARG_HEAD_SIZE, 0);
    taskArgs[TASKARG_INPUT_ADDR] = inputAddr;
    taskArgs[TASKARG_OUTPUT_ADDR] = outputAddr;
    taskArgs[TASKARG_SCRATCH_ADDR] = scratchAddr;
    taskArgs[TASKARG_TOKEN] = token;
    taskArgs.push_back(inputRepeatStride);
    taskArgs.push_back(outputRepeatStride);
    taskArgs.push_back(normalSliceSize);
    taskArgs.push_back(repeatNumVar);
    // CalGoSize 参数须按 kernel LoadArgs 的顺序追加。
    taskArgs.push_back(goSize[0]);
    taskArgs.push_back(goSize[1]);
    taskArgs.push_back(goSize[2]);
    taskArgs.push_back(goSize[3]);
    CHK_RET(CheckCachedArgCount(taskArgs.size()));
    CcuResult launchRet = HcommCcuKernelLaunch(
        templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), taskArgs.size());
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempReduceMesh1DMem2Mem::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    CHK_RET(SubmitKernelInfo(templateResource, taskArgs));

    HCCL_DEBUG("[CcuTempReduceMesh1DMem2Mem::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMesh1DMem2Mem::SubmitKernelInfo(
    TemplateResource& templateResource, const std::vector<uint64_t>& taskArgs) const
{
    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];

    // 缓存本次偏移及 rank 数，供 FastLaunch 使用。
    submitInfo.cachedArgs[0] = taskArgs.size();
    for (size_t i = 0; i < taskArgs.size(); i++) {
        submitInfo.cachedArgs[i + 1] = taskArgs[i];
    }
    submitInfo.cachedArgs[taskArgs.size() + 1] = buffInfo_.inBuffBaseOff;
    submitInfo.cachedArgs[taskArgs.size() + 2] = buffInfo_.outBuffBaseOff;
    submitInfo.cachedArgs[taskArgs.size() + 3] = buffInfo_.hcclBuffBaseOff;
    submitInfo.cachedArgs[taskArgs.size() + 4] = templateRankSize_;
    templateResource.submitInfos.push_back(submitInfo);
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceMesh1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // 每个 rank 对应一个 scratch 槽；executor 据此限制每轮 slice 大小。
    return templateRankSize_;
}

u64 CcuTempReduceMesh1DMem2Mem::GetThreadNum() const { return 1; }

HcclResult CcuTempReduceMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    return HCCL_SUCCESS;
}

} // namespace ops_hccl
