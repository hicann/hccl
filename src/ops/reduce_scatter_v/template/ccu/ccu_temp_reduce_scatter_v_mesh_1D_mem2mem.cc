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
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "ccu_kernel_reduce_scatter_v_mesh1d_mem2mem.h"
#include "ccu_temp_reduce_scatter_v_mesh_1D_mem2mem.h"

namespace ops_hccl {

CcuTempReduceScatterVMesh1DMem2Mem::CcuTempReduceScatterVMesh1DMem2Mem(const OpParam& param, const u32 rankId,
                                                                       const std::vector<std::vector<u32>>& subCommRanks)
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

CcuTempReduceScatterVMesh1DMem2Mem::~CcuTempReduceScatterVMesh1DMem2Mem()
{
}

HcclResult CcuTempReduceScatterVMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    GetRes(resourceRequest);
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempReduceScatterVMesh1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;

    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelReduceScatterVMesh1DMem2Mem>(arg);
                         };
    std::vector<HcclChannelDesc> channelDescs;
    if(topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_1DMESH));
        for(auto channel : channelDescs){
            if(channel.channelProtocol != COMM_PROTOCOL_UBC_CTP){
                HCCL_ERROR("[CcuTempReduceScatterVMesh1DMem2Mem][CalcRes] channelProtocol: %u", channel.channelProtocol);
                return HCCL_E_INTERNAL;
            }
        }
    }
    HCCL_DEBUG("[CcuTempReduceScatterVMesh1DMem2Mem::CalcRes] Get Mesh Channel Success!");

    kernelInfo.kernelArg = std::make_shared<CcuKernelArgReduceScatterVMesh1DMem2Mem>(subCommRanks_[0].size(),
                                                                                    mySubCommRank_,
                                                                                    param,
                                                                                    subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempReduceScatterVMesh1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}


uint64_t CcuTempReduceScatterVMesh1DMem2Mem::GetTokenWithFallback(const BuffInfo& buffInfo) {
    // 由于 input 中可能有气泡导致 token 计算保持，暂时不适用 input 计算 token
    if (buffInfo.outputPtr != nullptr) {
        HCCL_INFO("Generate token using output buffer: ptr=%p, size=%llu",
                  buffInfo.outputPtr, buffInfo.outputSize);
        return hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo.outputPtr),
                                           static_cast<uint64_t>(buffInfo.outputSize));
    } else if (buffInfo.hcclBuff.addr != nullptr) {
        HCCL_INFO("Generate token using scratch buffer: ptr=%p, size=%llu",
                  buffInfo.hcclBuff.addr, buffInfo.hcclBuff.size);
        return hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo.hcclBuff.addr),
                                           static_cast<uint64_t>(buffInfo.hcclBuff.size));
    } else {
        HCCL_INFO("All buffers are null, using 0 as token");
        return 0;
    }
}

HcclResult CcuTempReduceScatterVMesh1DMem2Mem::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token              = GetTokenWithFallback(buffInfo_);
    uint64_t scratchAddr        = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;

    uint64_t sliceSize          = templateDataParams.allRankSliceSize[mySubCommRank_];
    uint64_t offset             = templateDataParams.allRankDispls[mySubCommRank_];
    uint64_t scratchInterval    = sliceSize;  // 每个rank的数据在scratch中连续存放，间隔为sliceSize

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgReduceScatterVMesh1DMem2Mem>(
        inputAddr, outputAddr, token, scratchAddr, scratchInterval, sliceSize, offset);

    void* taskArgPtr = static_cast<void*>(taskArg.get());
    HCCL_INFO("templateResource.threads.size[%zu], templateResource.ccuKernels.size[%zu]", templateResource.threads.size(), templateResource.ccuKernels.size());
    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceScatterVMesh1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // reduce_scatter_v场景，scratch Buffer需要存储所有rank的slice
    return templateRankSize_;
}

u64 CcuTempReduceScatterVMesh1DMem2Mem::GetThreadNum()
{
    return 1;
}

HcclResult CcuTempReduceScatterVMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
