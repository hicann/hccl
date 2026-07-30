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
#include "ccu_kernel_reduce_mesh1d_twoshot_mem2mem.h"
#include "ccu_temp_reduce_mesh_1D_twoshot_mem2mem.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

constexpr uint32_t REDUCE_LOOP_COUNT = 16;

CcuTempReduceMesh1DTwoShotMem2Mem::CcuTempReduceMesh1DTwoShotMem2Mem(const OpParam& param,
                                       const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    templateRankSize_ = subCommRanks[0].size();
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

CcuTempReduceMesh1DTwoShotMem2Mem::~CcuTempReduceMesh1DTwoShotMem2Mem()
{
}

void CcuTempReduceMesh1DTwoShotMem2Mem::SetRoot(u32 root)
{
    std::vector<u32> ranks = subCommRanks_[0];
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        mySubCommRoot_ = std::distance(ranks.begin(), itRoot);
    }
    HCCL_INFO("[CcuTempReduceMesh1DTwoShotMem2Mem][SetRoot] myRank_ [%u], set root [%u] subCommRoot [%u]",
              mySubCommRank_, root, mySubCommRoot_);
}

HcclResult CcuTempReduceMesh1DTwoShotMem2Mem::CalcRes(HcclComm comm, const OpParam& param,
                                                      const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempReduceMesh1DTwoShotMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    CHK_SAFETY_FUNC_RET(strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelReduceMesh1DTwoShotMem2Mem"));
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuReduceMesh1DTwoShotMem2MemKernel);

    std::vector<HcclChannelDesc> channelDescs;
    if(topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, myChannelDescs,
                                                         CommTopo::COMM_TOPO_1DMESH));
        for(auto channel : myChannelDescs) {
            if(channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                channelDescs.push_back(channel);
            }
        }
        HCCL_DEBUG("[CcuTempReduceMesh1DTwoShotMem2Mem::CalcRes] Get Mesh Channel Success!");
    }

    auto kernelArg = std::make_shared<CcuKernelArgReduceMesh1DTwoShotMem2Mem>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->rootId = mySubCommRoot_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempReduceMesh1DTwoShotMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMesh1DTwoShotMem2Mem::FastLaunch(const OpParam& param,
                                                         const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempReduceMesh1DTwoShotMem2Mem::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempReduceMesh1DTwoShotMem2Mem::FastLaunch] start");
    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    uint64_t argSize = args[0];
    constexpr u32 inputIdx = 1;
    constexpr u32 outputIdx = 2;
    constexpr u32 scratchIdx = 3;
    const u32 inputOffsetIdx = argSize + 1;
    const u32 outputOffsetIdx = argSize + 2;
    const u32 scratchOffsetIdx = argSize + 3;

    args[inputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    args[outputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];
    args[scratchIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.hcclBuff.addr) + args[scratchOffsetIdx];

    void *taskArgs = reinterpret_cast<void*>(args + 1);
    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
                                               tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle,
                                               taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempReduceMesh1DTwoShotMem2Mem::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[CcuTempReduceMesh1DTwoShotMem2Mem::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMesh1DTwoShotMem2Mem::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    if (templateDataParams.sliceSize == 0) {
        HCCL_INFO("[CcuTempReduceMesh1DTwoShotMem2Mem] sliceSize is 0, no need do, just success.");
        return HCCL_SUCCESS;
    }

    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t scratchAddr        = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    uint64_t totalSize          = templateDataParams.sliceSize;
    uint64_t typeSize           = DataTypeSizeGet(param.DataDes.dataType);
    uint64_t totalCount         = totalSize / typeSize;
    uint64_t perRankCount       = totalCount / templateRankSize_;
    uint64_t normalSliceSize    = perRankCount * typeSize;
    uint64_t lastSliceSize      = totalSize - (templateRankSize_ - 1) * normalSliceSize;
    uint64_t repeatNumVar       = UINT64_MAX - templateDataParams.repeatNum;
    uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = REDUCE_LOOP_COUNT;
    config.memSlice     = CCU_MS_SIZE;
    auto goSize = (mySubCommRank_ == (templateRankSize_ - 1))
                  ? CalGoSize(lastSliceSize, config)
                  : CalGoSize(normalSliceSize, config);

    HCCL_INFO("[CcuTempReduceMesh1DTwoShotMem2Mem::KernelRun] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
              "scratchAddr[%llu], normalSliceSize[%llu], lastSliceSize[%llu], repeatNumVar[%llu], "
              "inputRepeatStride[%llu], outputRepeatStride[%llu]",
              inputAddr, outputAddr, scratchAddr, normalSliceSize, lastSliceSize, repeatNumVar,
              inputRepeatStride, outputRepeatStride);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, scratchAddr, token,
                                      normalSliceSize, lastSliceSize,
                                      repeatNumVar, inputRepeatStride, outputRepeatStride,
                                      goSize[0], goSize[1], goSize[2], goSize[3]};

    CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0],
                                               templateResource.ccuKernels[0],
                                               taskArgs.data(), taskArgs.size());
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempReduceMesh1DTwoShotMem2Mem::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    CHK_RET(SubmitKernelInfo(templateResource, taskArgs));

    HCCL_DEBUG("[CcuTempReduceMesh1DTwoShotMem2Mem::KernelRun] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMesh1DTwoShotMem2Mem::SubmitKernelInfo(TemplateResource& templateResource,
                                                               const std::vector<uint64_t>& taskArgs) const
{
    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];

    size_t argNum = taskArgs.size() + 4;
    if (UNLIKELY(argNum > CCU_MAX_TASK_ARG_NUM)) {
        HCCL_ERROR("[CcuTempReduceMesh1DTwoShotMem2Mem::KernelRun] argNum is bigger than CCU_MAX_TASK_ARG_NUM[%d]",
                   CCU_MAX_TASK_ARG_NUM);
        return HcclResult::HCCL_E_INTERNAL;
    }

    submitInfo.cachedArgs[0] = taskArgs.size();
    for (size_t i = 0; i < taskArgs.size(); i++) {
        submitInfo.cachedArgs[i + 1] = taskArgs[i];
    }
    submitInfo.cachedArgs[taskArgs.size() + 1] = buffInfo_.inBuffBaseOff;
    submitInfo.cachedArgs[taskArgs.size() + 2] = buffInfo_.outBuffBaseOff;
    submitInfo.cachedArgs[taskArgs.size() + 3] = buffInfo_.hcclBuffBaseOff;
    templateResource.submitInfos.push_back(submitInfo);
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceMesh1DTwoShotMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

u64 CcuTempReduceMesh1DTwoShotMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempReduceMesh1DTwoShotMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
