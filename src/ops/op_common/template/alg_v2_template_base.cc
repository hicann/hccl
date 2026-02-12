/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_v2_template_base.h"

namespace ops_hccl {

InsAlgTemplateBase::InsAlgTemplateBase(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                       const std::vector<std::vector<u32>>& subCommRanks)
{
    opMode_            = param.opMode;
    root_              = param.root;

    myRank_            = rankId;
    if (subCommRanks.size() > 1) {
        templateRankSize_  = subCommRanks[0].size() * subCommRanks[1].size();
    } else {
        templateRankSize_  = subCommRanks[0].size();
    }
    subCommRanks_      = subCommRanks;

    reduceOp_          = param.reduceType;
    // 从OpParam中获取
    enableDetour_      = param.enableDetour;
}

InsAlgTemplateBase::~InsAlgTemplateBase()
{
}

HcclResult InsAlgTemplateBase::KernelRun(const OpParam& param,
                                         const TemplateDataParams& tempAlgParams,
                                         const TemplateResource& templateResource)
{
    (void)param;
    (void)tempAlgParams;
    (void)templateResource;
    HCCL_ERROR("[InsAlgTemplateBase] Unsupported interface of kernel run!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult InsAlgTemplateBase::DPUKernelRun(const TemplateDataParams& tempAlgParam,
    const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>>& subCommRanks)
{
    (void)tempAlgParam;
    (void)channels;
    (void)myRank;
    (void)subCommRanks;
    HCCL_ERROR("[InsAlgTemplateBase] Unsupported interface of dpu kernel run!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult InsAlgTemplateBase::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                                       AlgResourceRequest& resourceRequest)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)resourceRequest;
    HCCL_ERROR("[InsAlgTemplateBase] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult InsAlgTemplateBase::GetRes(AlgResourceRequest& resourceRequest)
{
    (void)resourceRequest;
    HCCL_ERROR("[InsAlgTemplateBase] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

u64 InsAlgTemplateBase::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    return 0;
}

u64 InsAlgTemplateBase::GetThreadNum()
{
    return 0;
}

}