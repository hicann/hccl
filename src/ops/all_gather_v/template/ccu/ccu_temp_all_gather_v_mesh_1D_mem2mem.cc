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
#include "ccu_kernel_all_gather_v_mesh1d_mem2mem.h"
#include "ccu_temp_all_gather_v_mesh_1D_mem2mem.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

CcuTempAllGatherVMesh1DMem2Mem::CcuTempAllGatherVMesh1DMem2Mem(const OpParam& param, const u32 rankId,
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

CcuTempAllGatherVMesh1DMem2Mem::~CcuTempAllGatherVMesh1DMem2Mem()
{
}

HcclResult CcuTempAllGatherVMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    GetRes(resourceRequest);
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllGatherVMesh1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelAllGatherVMesh1DMem2Mem");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllGatherVMesh1DMem2MemKernel);

    std::vector<HcclChannelDesc> channelDescs;
    if(topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_1DMESH));
        for(auto channel : channelDescs){
            if(channel.channelProtocol != COMM_PROTOCOL_UBC_CTP){
                HCCL_ERROR("[CcuTempAllGatherVMesh1DMem2Mem][CalcRes] channelProtocol: %u", channel.channelProtocol);
                return HCCL_E_INTERNAL;
            }
        }
    }
    HCCL_DEBUG("[CcuTempAllGatherVMesh1DMem2Mem::CalcRes] Get Mesh Channel Success!");
    auto kernelArg = std::make_shared<CcuKernelArgAllGatherVMesh1DMem2Mem>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllGatherVMesh1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherVMesh1DMem2Mem::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;
    uint32_t rankId = mySubCommRank_;
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[param.vDataDes.dataType];
    uint64_t mySliceSize = templateDataParams.allRankSliceSize[rankId];
    uint64_t mySliceSizeOutputOffset = templateDataParams.allRankDispls[rankId] * dataTypeSize;

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + templateDataParams.tailSize * dataTypeSize;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + templateDataParams.tailSize * dataTypeSize;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice     = CCU_MS_SIZE * LOCAL_COPY_MS_PER_LOOP;
    auto goSize = CalGoSize(mySliceSize, config);
    // 代替GeneArgs
    std::vector<uint64_t> taskArgs = {
        inputAddr, outputAddr, token, mySliceSize, mySliceSizeOutputOffset, goSize[0], goSize[1], goSize[2], goSize[3]};
    uint64_t argSize = 9;

    HCCL_INFO("[CcuTempAllGatherVMesh1DMem2Mem::KernelRun] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
               "mySliceSize[%llu], mySliceSizeOutputOffset[%llu]",
               inputAddr, outputAddr, mySliceSize, mySliceSizeOutputOffset);

    CcuResult launchRet
        = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAllGatherVMesh1DMem2Mem::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGatherVMesh1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempAllGatherVMesh1DMem2Mem::GetThreadNum()
{
    return 1;
}
 
HcclResult CcuTempAllGatherVMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HCCL_SUCCESS;
}
} // namespace ops_hccl