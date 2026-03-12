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
#include "ccu_kernel_scatter_mesh1d.h"
#include "ccu_temp_scatter_mesh1d.h"

namespace ops_hccl {

CcuTempScatterMesh1D::CcuTempScatterMesh1D(const OpParam &param, const u32 rankId,
                                                         const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    HCCL_INFO("Start to run CcuTempScatterMesh1D");
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    auto itRoot = std::find(ranks.begin(), ranks.end(), param.root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
}

CcuTempScatterMesh1D::~CcuTempScatterMesh1D() {}

void CcuTempScatterMesh1D::SetRoot(u32 root)
{
    HCCL_INFO("[CcuTempScatterMesh1D][SetRoot] myRank_ [%u], set root [%u] ", myRank_, root);
    std::vector<u32> ranks = subCommRanks_[0];
    std::string ranksStr = "";
    for (auto r : ranks) { ranksStr += std::to_string(r) + " "; }
    HCCL_INFO("[CcuTempScatterNHR1DMem2Mem][SetSubCommRoot] ranks = subCommRanks[0] is: %s", ranksStr.c_str());
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
}

HcclResult CcuTempScatterMesh1D::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                                                AlgResourceRequest &resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempScatterMesh1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;

    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelScatterMesh1D>(arg);
    };
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgScatterMesh1D>(subCommRanks_[0].size(), mySubCommRank_,
                                                                              subCommRootId_, param, subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempScatterMesh1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterMesh1D::KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
                                                  const TemplateResource &templateResource)
{
    if (templateDataParams.sliceSize == 0) {
        HCCL_INFO("[CcuTempScatterMesh1D] sliceSize is 0, no need do, just success.");
        return HcclResult::HCCL_SUCCESS;
    }
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t repeatNumTmp = templateDataParams.repeatNum;
    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                 static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t inputSliceStride = templateDataParams.inputSliceStride;
    uint64_t inputRepeatStride = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize = templateDataParams.sliceSize;
    uint64_t lastSliceSize = templateDataParams.tailSize;

    uint64_t repeatNum = UINT64_MAX - repeatNumTmp;

    HCCL_INFO("[CcuTempScatterMesh1D] create CcuTaskArgScatterMesh1D, normalSliceSize [%u] ", normalSliceSize);
    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgScatterMesh1D>(
        inputAddr, outputAddr, token, inputSliceStride, inputRepeatStride, outputRepeatStride, normalSliceSize,
        lastSliceSize, repeatNum);

    void *taskArgPtr = static_cast<void *>(taskArg.get());

    HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr);

    HCCL_DEBUG("[CcuTempScatterMesh1D::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempScatterMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // scatter操作不需要scratch buffer
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempScatterMesh1D::GetThreadNum() const
{
    return 1;
}

}  // namespace ops_hccl
