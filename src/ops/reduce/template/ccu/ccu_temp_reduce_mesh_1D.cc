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
#include "ccu_kernel_reduce_mesh1d.h"
#include "ccu_temp_reduce_mesh_1D.h"

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
    
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelReduceMesh1D>(arg);
                         };
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgReduceMesh1D>(subCommRanks_[0].size(),
                                                                             mySubCommRank_,
                                                                             mySubCommRoot_,
                                                                             param,
                                                                             subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempReduceMesh1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMesh1D::KernelRun(const OpParam& param,
                                          const TemplateDataParams& templateDataParams,
                                          const TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t                      inputAddr               = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t                      outputAddr              = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t                      token                   = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                            static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t                      inputSliceStride        = templateDataParams.inputSliceStride;
    uint64_t                      outputSliceStride       = templateDataParams.outputSliceStride;
    uint64_t                      repeatNum               = templateDataParams.repeatNum;
    uint64_t                      inputRepeatStride       = templateDataParams.inputRepeatStride;
    uint64_t                      outputRepeatStride      = templateDataParams.outputRepeatStride;
    uint64_t                      normalSliceSize         = templateDataParams.sliceSize;
    uint64_t                      lastSliceSize           = templateDataParams.tailSize;
    uint64_t                      repeatNumVar            = UINT64_MAX - repeatNum;
    
    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgReduceMesh1D>(
        inputAddr, outputAddr, token, inputSliceStride, outputSliceStride, repeatNum, inputRepeatStride,
        outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar);

    void* taskArgPtr = static_cast<void*>(taskArg.get());

    HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr);
    
    HCCL_DEBUG("[CcuTempReduceMesh1D::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl 