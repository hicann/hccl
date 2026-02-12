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
#include "ccu_kernel_reduce_scatter_mesh1d.h"
#include "ccu_temp_reduce_scatter_mesh_1D.h"

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

HcclResult CcuTempReduceScatterMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
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
    
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelReduceScatterMesh1D>(arg);
                         };
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgReduceScatterMesh1D>(subCommRanks_[0].size(),
                                                                                    mySubCommRank_,
                                                                                    param,
                                                                                    subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempReduceScatterMesh1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}


HcclResult CcuTempReduceScatterMesh1D::KernelRun(const OpParam& param,
                                                 const TemplateDataParams& templateDataParams,
                                                 const TemplateResource& templateResource)
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
    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t sliceSize          = templateDataParams.sliceSize;
    uint64_t inputSliceStride   = templateDataParams.inputSliceStride;
    uint64_t offset             = inputSliceStride * mySubCommRank_;
    
    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgReduceScatterMesh1D>(
        inputAddr, outputAddr, sliceSize, offset, token);

    void* taskArgPtr = static_cast<void*>(taskArg.get());

    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));

    HCCL_DEBUG("[CcuTempReduceScatterMesh1D::KernelRun] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceScatterMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

} // namespace ops_hccl
