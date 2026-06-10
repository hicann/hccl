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
#include "ccu_kernel_reduce_mesh1d.h"
#include "ccu_temp_reduce_mesh_1D.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

CcuTempReduceMesh1D::CcuTempReduceMesh1D(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    templateRankSize_ = subCommRanks[0].size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
    dataType_ = param.DataDes.dataType;
    auto rootIt = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), param.root);
    if (rootIt != subCommRanks[0].end()) {
        mySubCommRoot_ = std::distance(subCommRanks[0].begin(), rootIt);
    }
}

CcuTempReduceMesh1D::~CcuTempReduceMesh1D()
{
}

HcclResult CcuTempReduceMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempReduceMesh1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelReduceMesh1D");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuReduceMesh1DKernel);

    std::vector<HcclChannelDesc> channelDescs;
    if(topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
        HCCL_DEBUG("[CcuTempReduceMesh1D::CalcRes] Get Mesh Channel Success!");
    } else {
        std::vector<HcclChannelDesc> tempChannelDescs;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, tempChannelDescs, CommTopo::COMM_TOPO_1DMESH));
        for(auto channel : tempChannelDescs) {
            if(channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                channelDescs.push_back(channel);
            }
        }
        HCCL_DEBUG("[CcuTempReduceMesh1D::CalcRes] Get Mesh Channel Success!");
    }

    auto kernelArg = std::make_shared<CcuKernelArgReduceMesh1D>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->rootId = mySubCommRoot_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempReduceMesh1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMesh1D::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempReduceMesh1D::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempReduceMesh1D::FastLaunch] start");
    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    constexpr u32 inputIdx = 0;
    constexpr u32 outputIdx = 1;
    constexpr u32 inputOffsetIdx = 15;
    constexpr u32 outputOffsetIdx = 16;
    uint64_t argSize = 15;

    args[inputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    args[outputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];

    void *taskArgs = reinterpret_cast<void*>(args);
    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
                                               tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle,
                                               taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempReduceMesh1D::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[CcuTempReduceMesh1D::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMesh1D::KernelRun(const OpParam& param,
                                          const TemplateDataParams& templateDataParams,
                                          TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t                      inputAddr               = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t                      outputAddr              = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t                      token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t                      inputSliceStride        = templateDataParams.inputSliceStride;
    uint64_t                      outputSliceStride       = templateDataParams.outputSliceStride;
    uint64_t                      repeatNum               = templateDataParams.repeatNum;
    uint64_t                      inputRepeatStride       = templateDataParams.inputRepeatStride;
    uint64_t                      outputRepeatStride      = templateDataParams.outputRepeatStride;
    uint64_t                      normalSliceSize         = templateDataParams.sliceSize;
    uint64_t                      lastSliceSize           = templateDataParams.tailSize;
    uint64_t                      repeatNumVar            = UINT64_MAX - repeatNum;

    LoopGroupConfig  config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_DEFAULT_LOOP_COUNT;
    config.memSlice     = CCU_MS_SIZE;
    auto goSize         = CalGoSize(normalSliceSize, config);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, inputSliceStride, outputSliceStride, repeatNum, inputRepeatStride, outputRepeatStride,
                                      normalSliceSize, lastSliceSize, repeatNumVar, goSize[0], goSize[1], goSize[2], goSize[3]};
    uint64_t argSize = 15;

    CcuResult launchRet =  HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempReduceMesh1D::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(submitInfo, inputAddr, outputAddr, token, inputSliceStride, outputSliceStride,
    repeatNum, inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar, goSize[0], goSize[1], goSize[2], goSize[3], buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff));
    templateResource.submitInfos.push_back(submitInfo);
    
    HCCL_DEBUG("[CcuTempReduceMesh1D::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl 