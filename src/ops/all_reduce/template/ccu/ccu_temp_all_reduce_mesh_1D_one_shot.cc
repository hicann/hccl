
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
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"

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
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                            return std::make_unique<CcuKernelAllReduceMesh1DOneShot>(arg);
                         };
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllReduceMesh1DOneShot>(subCommRanks_[0].size(),
                                                                                mySubCommRank_,
                                                                                param,
                                                                                subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllReduceMesh1DOneShot::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1DOneShot::KernelRun(const OpParam& param,
                                                    const TemplateDataParams& templateDataParams,
                                                    const TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;
    HCCL_INFO("CcuTempAllReduceMesh1DOneShot KernelRun inputPtr(%p), inputSize(%d)", buffInfo_.inputPtr, buffInfo_.inputSize);
    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                 static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t sliceSize = templateDataParams.sliceSize;

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllReduceMesh1DOneShot>(
        inputAddr, outputAddr, sliceSize, token);

    void* taskArgPtr = static_cast<void*>(taskArg.get());

    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));

    HCCL_DEBUG("[CcuTempReduceScatterMeshMem2Mem1D::KernelRun] end");

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
