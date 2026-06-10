
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_all_reduce_mesh_1D_one_shot.h"
#include "log.h"
#include "channel.h"
#include "ccu_kernel_all_reduce_mesh1d_one_shot.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

CcuTempAllReduceMesh1DOneShot::CcuTempAllReduceMesh1DOneShot(const OpParam& param, const u32 rankId,
                                                             const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
}

CcuTempAllReduceMesh1DOneShot::~CcuTempAllReduceMesh1DOneShot()
{
}

HcclResult CcuTempAllReduceMesh1DOneShot::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                  AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllReduceMesh1DOneShot::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllReduceMesh1DOneShotKernel");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllReduceMesh1DOneShotKernel);
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
        HCCL_DEBUG("[CcuTempAllReduceMesh1DOneShot::CalcRes] Get Mesh Channel Success!");
    }    
    auto kernelArg = std::make_shared<CcuKernelArgAllReduceMesh1DOneShot>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllReduceMesh1DOneShot::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1DOneShot::KernelRun(const OpParam& param,
                                                    const TemplateDataParams& templateDataParams,
                                                    TemplateResource& templateResource)
{
    (void)param;
    buffInfo_ = templateDataParams.buffInfo;
    HCCL_INFO("CcuTempAllReduceMesh1DOneShot KernelRun inputPtr(%p), inputSize(%d)", buffInfo_.inputPtr, buffInfo_.inputSize);
    if (buffInfo_.inputPtr == nullptr || buffInfo_.outputPtr == nullptr) {
        HCCL_ERROR("[CcuTempAllReduceMesh1DOneShot::KernelRun] inputPtr or outputPtr is nullptr");
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t sliceSize = templateDataParams.sliceSize;

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_DEFAULT_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE;
    auto goSize = CalGoSize(sliceSize, config);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, goSize[0], goSize[1], goSize[2], goSize[3]};
    uint32_t argSize = static_cast<uint32_t>(taskArgs.size());
    CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0],
        taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAllReduceMesh1DOneShot::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(submitInfo, inputAddr, outputAddr, token, goSize[0], goSize[1], goSize[2],
        goSize[3], buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff));
    templateResource.submitInfos.push_back(submitInfo);

    HCCL_DEBUG("[CcuTempAllReduceMesh1DOneShot::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1DOneShot::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    (void)param;
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAllReduceMesh1DOneShot::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempAllReduceMesh1DOneShot::FastLaunch] start");
    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    constexpr u32 inputIdx = 0;
    constexpr u32 outputIdx = 1;
    constexpr u32 inputOffsetIdx = 7;
    constexpr u32 outputOffsetIdx = 8;
    constexpr uint32_t argSize = 7;
    args[inputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    args[outputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];

    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
        tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, reinterpret_cast<void*>(args), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAllReduceMesh1DOneShot::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[CcuTempAllReduceMesh1DOneShot::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllReduceMesh1DOneShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}
} // namespace Hccl
