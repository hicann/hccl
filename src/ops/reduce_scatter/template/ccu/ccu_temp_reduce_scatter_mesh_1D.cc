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
#include "ccu_kernel_reduce_scatter_mesh1d.h"
#include "ccu_temp_reduce_scatter_mesh_1D.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

CcuTempReduceScatterMesh1D::CcuTempReduceScatterMesh1D(const OpParam& param, const u32 rankId,
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
}

CcuTempReduceScatterMesh1D::~CcuTempReduceScatterMesh1D()
{
}

HcclResult CcuTempReduceScatterMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                               AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempReduceScatterMesh1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelReduceScatterMesh1D");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuReduceScatterMesh1DKernel);
    
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
        HCCL_DEBUG("[CcuTempReduceScatterMesh1D::CalcRes] Get Mesh Channel Success!");
    }
    auto kernelArg = std::make_shared<CcuKernelArgReduceScatterMesh1D>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempReduceScatterMesh1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMesh1D::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempReduceScatterMesh1D::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempReduceScatterMesh1D::FastLaunch] start");
    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    constexpr u32 inputIdx = 0;
    constexpr u32 outputIdx = 1;
    constexpr u32 inputOffsetIdx = 8;
    constexpr u32 outputOffsetIdx = 9;
    uint64_t argSize = 8;

    args[inputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    args[outputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];

    void *taskArgs = reinterpret_cast<void*>(args);
    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
                                               tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle,
                                               taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempReduceScatterMesh1D::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[CcuTempReduceScatterMesh1D::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMesh1D::KernelRun(const OpParam& param,
                                                 const TemplateDataParams& templateDataParams,
                                                 TemplateResource& templateResource)
{
    opMode_ = param.opMode;
    buffInfo_ = templateDataParams.buffInfo;

    HcclDataType dataType = param.DataDes.dataType;
    HcclDataType outputDataType = param.DataDes.outputType;
    if (outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType = dataType;
    }

    uint64_t expandingtimes = DataTypeSizeGet(outputDataType) /
                              DataTypeSizeGet(dataType);  // 膨胀的倍数是输出类型/输入类型
    HCCL_INFO("[CcuTempReduceScatterMesh1D::KernelRun] dataType[%d] outputDatatype[%d]", param.DataDes.dataType,
              param.DataDes.outputType);
    uint64_t baseInputAddr      = PointerToAddr(buffInfo_.inputPtr);
    uint64_t inputAddr          = baseInputAddr + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t sliceSize          = templateDataParams.sliceSize;
    uint64_t inputSliceStride   = templateDataParams.inputSliceStride;
    uint64_t offset             = inputSliceStride * mySubCommRank_;
    LoopGroupConfig  config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_DEFAULT_LOOP_COUNT;
    config.memSlice     = CCU_MS_SIZE;
    auto     goSize             = CalGoSize(sliceSize, config);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, offset, goSize[0], goSize[1], goSize[2], goSize[3]};
    uint64_t argSize = 8;

    HCCL_INFO("[CcuTempReduceScatterMesh1D::KernelRun] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
               "offset[%llu], sliceSize[%llu]",
               inputAddr, outputAddr, offset, sliceSize);
    CcuResult launchRet =  HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempReduceScatterMesh1D::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }
    
    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(submitInfo, inputAddr, outputAddr, token, offset,
        goSize[0], goSize[1], goSize[2], goSize[3],
        buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff)); // input和output的offset追加在最后
    templateResource.submitInfos.push_back(submitInfo);

    HCCL_DEBUG("[CcuTempReduceScatterMesh1D::KernelRun] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceScatterMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempReduceScatterMesh1D::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempReduceScatterMesh1D::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    return HCCL_SUCCESS;
}

} // namespace ops_hccl
