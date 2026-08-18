/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_ALG_TEMPLATE_BASE_H
#define COMMON_ALG_TEMPLATE_BASE_H

#include "template_utils.h"
#include "alg_param.h"

namespace ops_hccl {

// Template 属性结构体，子类按需覆盖成员值（用于 cost model 等场景判断 template 特征）
struct TemplateProp {
    bool isNhr = false; // 是否为 NHR 算法（影响 cost model 的 netType 选择）
};

class CommonAlgTemplateBase {
public:
    explicit CommonAlgTemplateBase() = default;
    explicit CommonAlgTemplateBase(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    virtual ~CommonAlgTemplateBase() = default;

    // Template 属性，子类按需覆盖（如 NHR 算法覆盖为 {.isNhr = true}）
    static constexpr TemplateProp props = {};

    virtual std::string Describe() const = 0;
    virtual HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest)
        = 0;
    virtual HcclResult GetRes(AlgResourceRequest& resourceRequest) const = 0;
    virtual u64 GetThreadNum() const = 0;
    virtual u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) = 0;

    virtual HcclResult
    KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
        = 0;
    virtual HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) = 0;
    virtual HcclResult CalcDataSplitByPortGroup(
        const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
        std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset)
    {
        CalcDataSplitByPortGroupCommon(
            totalDataCount, dataTypeSize, channels, elemCountOut, sizeOut, elemOffset, channelsPerRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    virtual HcclResult SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels)
    {
        CHK_PRT_RET(channels.empty(), HCCL_ERROR("[SetchannelsPerRank] channels is empty."), HCCL_E_INTERNAL);
        channelsPerRank_ = CalcChannelsPerRank(channels);
        return HCCL_SUCCESS;
    }

protected:
    u32 channelsPerRank_ = 1;
};

} // namespace ops_hccl

#endif // COMMON_ALG_TEMPLATE_BASE_H
