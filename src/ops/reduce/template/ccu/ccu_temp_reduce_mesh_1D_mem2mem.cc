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

CcuTempReduceMesh1DMem2Mem::CcuTempReduceMesh1DMem2Mem(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
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

CcuTempReduceMesh1DMem2Mem::~CcuTempReduceMesh1DMem2Mem()
{
}

void CcuTempReduceMesh1DMem2Mem::SetRoot(u32 root)
{
    std::vector<u32> ranks = subCommRanks_[0];
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        mySubCommRoot_  = std::distance(ranks.begin(), itRoot);
    }
    HCCL_INFO("[CcuTempBroadcastMesh1DMem2Mem][SetRoot] myRank_ [%u], set root_ [%u] subCommRanks[%u]", mySubCommRank_, root, mySubCommRoot_);
}

HcclResult CcuTempReduceMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempReduceMesh1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelReduceMesh1DMem2Mem");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuReduceMesh1DMem2MemKernel);

    std::vector<HcclChannelDesc> channelDescs;
    if(topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, myChannelDescs, CommTopo::COMM_TOPO_1DMESH));
        for(auto channel : myChannelDescs) {
            if(channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                channelDescs.push_back(channel);
            }
        }
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

    HCCL_DEBUG("[CcuTempReduceMesh1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMesh1DMem2Mem::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempReduceMesh1DMem2Mem::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempReduceMesh1DMem2Mem::FastLaunch] start");
    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    uint64_t argSize = args[0];
    constexpr u32 inputIdx = 1;
    constexpr u32 outputIdx = 2;
    const u32 inputOffsetIdx = argSize + 1;
    const u32 outputOffsetIdx = argSize + 2;

    args[inputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    args[outputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];

    void *taskArgs = reinterpret_cast<void*>(args + 1);
    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
                                               tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle,
                                               taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempReduceMesh1DMem2Mem::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[CcuTempReduceMesh1DMem2Mem::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMesh1DMem2Mem::KernelRun(const OpParam& param,
                                                 const TemplateDataParams& templateDataParams,
                                                 TemplateResource& templateResource)
{
    if (templateDataParams.sliceSize == 0) {
        HCCL_INFO("[CcuTempReduceMesh1DMem2Mem] sliceSize is 0, no need do, just success.");
        return HCCL_SUCCESS;
    }

    buffInfo_ = templateDataParams.buffInfo;

    uint64_t                               inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t                               outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t                               token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t                               repeatNum          = templateDataParams.repeatNum;
    uint64_t                               inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t                               outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t                               normalSliceSize    = templateDataParams.sliceSize;
    uint64_t                               lastSliceSize      = templateDataParams.tailSize;
    uint64_t                               repeatNumVar       = UINT64_MAX - repeatNum;

    // 数据切分为sliceNum块，当数据量不能均匀切分时，后面smallDataSliceNum个数据块比前面bigDataSliceNum个数据块每块少1个数据
    uint64_t sliceNum   = templateRankSize_ - 1; // 最终切分的chunk数量
    uint64_t sliceCount = normalSliceSize / DataTypeSizeGet(dataType_);
    // bigDataSliceSize、smallDataSliceSize 为 chunkSize
    uint64_t bigDataSliceNum    = sliceCount % sliceNum;
    uint64_t bigDataSliceSize   = (sliceCount / sliceNum + 1) * DataTypeSizeGet(dataType_);
    uint64_t smallDataSliceNum  = sliceNum - sliceCount % sliceNum;
    uint64_t smallDataSliceSize = sliceCount / sliceNum * DataTypeSizeGet(dataType_);
    uint64_t isInputOutputEqual = (inputAddr == outputAddr) ? 1 : 0;
    HCCL_INFO("[CcuTempReduceMesh1DMem2Mem::KernelRun] TaskArgs: inputAddr[%llu], outputAddr[%llu], repeatNum[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu], repeatNumVar[%llu], bigDataSliceNum[%llu], bigDataSliceSize[%llu], smallDataSliceNum[%llu], smallDataSliceSize[%llu]", inputAddr, outputAddr, repeatNum, inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar, bigDataSliceNum, bigDataSliceSize, smallDataSliceNum, smallDataSliceSize);
    LoopGroupConfig  config{};
    config.msInterleave = LOCAL_COPY_MS;
    config.loopCount    = LOOP_NUM;
    config.memSlice     = LOCAL_COPY_MS * CCU_MS_SIZE;
    auto     goSize     = CalGoSize(normalSliceSize, config);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, isInputOutputEqual, inputRepeatStride,
                                      outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar};
    for (uint64_t i = 0; i < bigDataSliceNum; i++) {
        taskArgs.push_back(bigDataSliceSize);
    }
    for (uint64_t i = 0; i < smallDataSliceNum; i++) {
        taskArgs.push_back(smallDataSliceSize);
    }
    taskArgs.push_back(goSize[0]);
    taskArgs.push_back(goSize[1]);
    taskArgs.push_back(goSize[2]);
    taskArgs.push_back(goSize[3]);
    CcuResult launchRet =  HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), taskArgs.size());
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempReduceMesh1DMem2Mem::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    CHK_RET(SubmitKernelInfo(templateResource, taskArgs));
    
    HCCL_DEBUG("[CcuTempReduceMesh1DMem2Mem::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMesh1DMem2Mem::SubmitKernelInfo(TemplateResource& templateResource,
                                                        const std::vector<uint64_t>& taskArgs) const
{
    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];

    size_t argNum = taskArgs.size() + 3;
    if (UNLIKELY(argNum > CCU_MAX_TASK_ARG_NUM)) {
        HCCL_ERROR("[CcuTempReduceMesh1DMem2Mem::KernelRun] argNum is bigger than CCU_MAX_TASK_ARG_NUM[%d]", CCU_MAX_TASK_ARG_NUM);
        return HcclResult::HCCL_E_INTERNAL;
    }

    submitInfo.cachedArgs[0] = taskArgs.size();
    for (size_t i = 0; i < taskArgs.size(); i++) {
        submitInfo.cachedArgs[i + 1] = taskArgs[i];
    }
    submitInfo.cachedArgs[taskArgs.size() + 1] = buffInfo_.inBuffBaseOff;
    submitInfo.cachedArgs[taskArgs.size() + 2] = buffInfo_.outBuffBaseOff;
    templateResource.submitInfos.push_back(submitInfo);
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceMesh1DMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempReduceMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    return HCCL_SUCCESS;
}

} // namespace ops_hccl 