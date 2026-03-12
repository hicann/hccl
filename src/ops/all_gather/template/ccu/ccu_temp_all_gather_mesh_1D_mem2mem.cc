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
#include "ccu_kernel_all_gather_mesh1d_mem2mem.h"
#include "ccu_temp_all_gather_mesh_1D_mem2mem.h"

namespace ops_hccl {

CcuTempAllGatherMesh1DMem2Mem::CcuTempAllGatherMesh1DMem2Mem(const OpParam& param, const u32 rankId,
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

CcuTempAllGatherMesh1DMem2Mem::~CcuTempAllGatherMesh1DMem2Mem()
{
}

HcclResult CcuTempAllGatherMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    GetRes(resourceRequest);
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelAllGatherMesh1DMem2Mem>(arg);
                         };
    std::vector<HcclChannelDesc> channelDescs;
    if(topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_1DMESH));
        for(auto channel : channelDescs){
            if(channel.channelProtocol != COMM_PROTOCOL_UBC_CTP){
                HCCL_ERROR("[CcuTempAllGatherMesh1DMem2Mem][CalcRes] channelProtocol: %u", channel.channelProtocol);
                return HCCL_E_INTERNAL;
            }
        }
    }
    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2Mem::CalcRes] Get Mesh Channel Success!");

    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllGatherMesh1DMem2Mem>(subCommRanks_[0].size(),
                                                                                    mySubCommRank_,
                                                                                    param,
                                                                                    subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherMesh1DMem2Mem::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        const TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));

    uint64_t inputSliceStride   = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride  = templateDataParams.outputSliceStride;
    uint32_t repeatNum          = templateDataParams.repeatNum;
    uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize    = templateDataParams.sliceSize;
    uint64_t lastSliceSize      = templateDataParams.tailSize;
    uint64_t isInputOutputEqual = (inputAddr == outputAddr) ? 1 : 0;

    HcclDataType dataType       = param.DataDes.dataType;
    uint64_t dataTypeSize       = DataTypeSizeGet(dataType);
    uint64_t dataCount          = normalSliceSize / dataTypeSize;
    if (dataCount == 0 && lastSliceSize == 0) {
        HCCL_INFO("[CcuTempAllGatherMesh1D] DataCount == 0 && lastSliceSize == 0, Template Run Ends.");
        return HcclResult::HCCL_SUCCESS;
    }

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllGatherMesh1DMem2Mem>(
        inputAddr, outputAddr, token, inputSliceStride, outputSliceStride, repeatNum, inputRepeatStride, outputRepeatStride,
        normalSliceSize, lastSliceSize, isInputOutputEqual);
    
    void* taskArgPtr = static_cast<void*>(taskArg.get());
    HCCL_INFO("templateResource.threads.size[%zu], templateResource.ccuKernels.size[%zu]", templateResource.threads.size(), templateResource.ccuKernels.size());
    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGatherMesh1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempAllGatherMesh1DMem2Mem::GetThreadNum() const
{
    return 1;
}
 
HcclResult CcuTempAllGatherMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HCCL_SUCCESS;
}
} // namespace ops_hccl