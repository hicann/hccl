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
#include "ccu_kernel_broadcast_mesh1d.h"
#include "ccu/ccu_temp_broadcast_mesh_1D.h"

namespace ops_hccl {

CcuTempBroadcastMesh1D::CcuTempBroadcastMesh1D(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    templateRankSize_ = subCommRanks[0].size();
}

CcuTempBroadcastMesh1D::~CcuTempBroadcastMesh1D()
{
}

HcclResult CcuTempBroadcastMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempBroadcastMesh1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;

    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelBroadcastMesh1D>(arg);
                         };
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgBroadcastMesh1D>(subCommRanks_[0].size(),
                                                                                    myRank_,
                                                                                    param.root,
                                                                                    param,
                                                                                    subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempBroadcastMesh1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HCCL_SUCCESS;
}

HcclResult CcuTempBroadcastMesh1D::KernelRun(const OpParam& param,
                                                    const TemplateDataParams& templateDataParams,
                                                    const TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t offSet = 0;
    uint64_t sliceSize = templateDataParams.sliceSize;

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgBroadcastMesh1D>(
        inputAddr, outputAddr, token, offSet, sliceSize);

    void* taskArgPtr = static_cast<void*>(taskArg.get());

    HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr);

    HCCL_DEBUG("[CcuTempBroadcastMesh1D::KernelRun] end");

    return HCCL_SUCCESS;
}
} // namespace ops_hccl