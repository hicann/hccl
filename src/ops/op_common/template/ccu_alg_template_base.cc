/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_alg_template_base.h"
#include "log.h"

namespace ops_hccl {
CcuAlgTemplateBase::CcuAlgTemplateBase()
{
}

CcuAlgTemplateBase::CcuAlgTemplateBase(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                       const std::vector<std::vector<u32>> &subCommRanks)
    : myRank_(rankId), subCommRanks_(subCommRanks)
{
    opMode_ = param.opMode;
    root_ = param.root;
}

void CcuAlgTemplateBase::InitCcuAlgTemplate(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                const std::vector<std::vector<u32>> &subCommRanks)
{
    opMode_ = param.opMode;
    root_ = param.root;
}

CcuAlgTemplateBase::~CcuAlgTemplateBase()
{
}

HcclResult CcuAlgTemplateBase::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                                       AlgResourceRequest& resourceRequest)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)resourceRequest;
    HCCL_ERROR("[CcuAlgTemplateBase] Unsupported interface of CcuAlgTemplateBase::CalcRes!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult CcuAlgTemplateBase::KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
                                         const TemplateResource& templateResource)
{
    (void)param;
    (void)templateDataParams;
    (void)templateResource;
    HCCL_ERROR("[CcuAlgTemplateBase] Unsupported interface of CcuAlgTemplateBase::KernelRun!");
    return HcclResult::HCCL_E_INTERNAL;
}

u64 CcuAlgTemplateBase::GetThreadNum()
{
    return 0;
}

HcclResult CcuAlgTemplateBase::GetRes(AlgResourceRequest& resourceRequest)
{
    (void)resourceRequest;
    HCCL_ERROR("[CcuAlgTemplateBase] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

u64 CcuAlgTemplateBase::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}


uint64_t CcuAlgTemplateBase::PointerToAddr(void* pointer)
{
    if (pointer != nullptr) {
        return reinterpret_cast<uint64_t>(pointer);
    } else {
        return 0;
    }
}

HcclResult CcuAlgTemplateBase::RestoreChannelMap(const std::vector<HcclChannelDesc>& channelDescs,
                                                 std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc)
{
    for (auto &channel: channelDescs) {
        u32 remoteRank = channel.remoteRank;
        rankIdToChannelDesc[remoteRank].push_back(channel);
    }
    return HCCL_SUCCESS;
}

HcclResult CcuAlgTemplateBase::GetChannelDieId(HcclComm comm, uint32_t rankId, const HcclChannelDesc& channelDesc,
                                               uint32_t& dieId)
{
    EndpointAttrDieId tmpDieId{};
    uint32_t infoLen = sizeof(EndpointAttrDieId);
    CHK_RET(HcclRankGraphGetEndpointInfo(comm, rankId, &(channelDesc.localEndpoint), ENDPOINT_ATTR_DIE_ID, infoLen,
                                         &tmpDieId));
    dieId = tmpDieId;
    HCCL_DEBUG("[CcuAlgTemplateBase::GetChannelDieId] rank[%d]: get channel die id [%d]", rankId, dieId);
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl