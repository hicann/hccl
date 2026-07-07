/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "ccu_kernel_all_gather_omnipipe_mesh1d_mem2mem.h"
#include "ccu_temp_all_gather_omnipipe_mesh1d_mem2mem.h"
#include "ccu_launch_dl.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

CcuTempAllGatherOmniPipeMesh1DMem2Mem::CcuTempAllGatherOmniPipeMesh1DMem2Mem(const OpParam& param, const u32 rankId,
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

    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank[%u] templateRankSize[%u]", __func__, rankId, mySubCommRank_, templateRankSize_);
}

CcuTempAllGatherOmniPipeMesh1DMem2Mem::~CcuTempAllGatherOmniPipeMesh1DMem2Mem()
{
}

HcclResult CcuTempAllGatherOmniPipeMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    // 不需要从流
    GetRes(resourceRequest);
    // 需要1个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[%s] notifyNumOnMainThread[%u] slaveThreadNum[%u]", __func__,
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllGatherOmniPipeMesh1DMem2MemKernel");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllGatherOmniPipeMesh1DMem2MemKernel);
    std::vector<HcclChannelDesc> channelDescs;

    if(topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1DFullMesh(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_1DMESH));
        for(auto channel : channelDescs){
            if(channel.channelProtocol != COMM_PROTOCOL_UBC_CTP){
                HCCL_ERROR("[%s] channelProtocol: %u", __func__, channel.channelProtocol);
                return HcclResult::HCCL_E_INTERNAL;
            }
        }
    }
    HCCL_INFO("[%s] Get Mesh Channel Success!", __func__);

    auto kernelArg = std::make_shared<CcuKernelArgAllGatherOmniPipeMesh1DMem2Mem>();
    kernelArg->opParam = param;
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->subCommRanks = subCommRanks_;

    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[%s] myRank[%u] mySubCommRank[%u] channelSize[%u] dimsize[%u] ccuKernelInfos.size[%u]", __func__,
        myRank_, mySubCommRank_, channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGatherOmniPipeMesh1DMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempAllGatherOmniPipeMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherOmniPipeMesh1DMem2Mem::KernelRun(const OpParam& param,
            const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;
    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    uint32_t rankId = myRank_;
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);

    uint64_t inBuffBaseOff = templateDataParams.buffInfo.inBuffBaseOff;
    uint64_t outBuffBaseOff = templateDataParams.buffInfo.outBuffBaseOff;

    uint64_t inputAddr = inputAddrBase + inBuffBaseOff;
    uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    if (localCopyFlag == 1) {
        buffInfo_.outBuffBaseOff += templateDataParams.inputSliceStride * myRank_;
        DataSlice srcSlice(buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        DataSlice dstSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy inputAddrBase[%llu] inputAddrOffset[%llu] outputAddrBase[%llu]"
                   "outputAddrOffset[%llu] sliceSize[%llu]", __func__, myRank_, inputAddrBase, buffInfo_.inBuffBaseOff,
                   outputAddrBase, buffInfo_.outBuffBaseOff, templateDataParams.sliceSize);
        if (templateDataParams.sliceSize != 0) {
            CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
        }
    } else {
        uint64_t sliceStride = stepSliceInfo.stepInputSliceStride[mySubCommRank_];
        uint32_t repeatNum = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_].size();
        for (uint32_t rpt = 0; rpt < repeatNum; ++rpt) {
            uint64_t sliceSize = stepSliceInfo.stepSliceSize[mySubCommRank_][rpt];
            LoopGroupConfig  config{};
            config.msInterleave = CCU_MS_INTERLEAVE;
            config.loopCount    = CCU_MS_LOCAL_COPY_LOOP_COUNT;
            config.memSlice     = LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
            auto   goSize       = CalGoSize(sliceSize, config);
            uint64_t inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_][rpt];

            std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, sliceSize, sliceStride, localCopyFlag,
                    inputOmniPipeSliceStride, goSize[0], goSize[1], goSize[2], goSize[3]};
            HCCL_DEBUG("[%s] myRank[%u] mySubCommRank[%u] rpt[%u] inputAddrBase[%llu] outputAddrBase[%llu] inBuffBaseOff[%llu] "
                        "outBuffBaseOff[%llu] inputAddr[%llu] outputAddr[%llu] sliceSize[%llu] sliceStride[%llu] localCopyFlag[%llu]",
                        __func__, myRank_, mySubCommRank_, rpt, inputAddrBase, outputAddrBase, inBuffBaseOff, outBuffBaseOff,
                        inputAddr, outputAddr, sliceSize, sliceStride, localCopyFlag);

            CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0],
                                                        taskArgs.data(), taskArgs.size());
            if (launchRet != CCU_SUCCESS) {
                HCCL_ERROR("[%s] kernel launch failed, ccuRet -> %d", __func__, launchRet);
                return ConvertCcuToHccl(launchRet);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGatherOmniPipeMesh1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // No Scratch buff
    (void)inBuffType;
    (void)outBuffType;
    return 1;
}
} // namespace ops_hccl