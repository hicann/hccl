/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSR_HOST_TEST_AIV_ALG_TEMPLATE_BASE_H
#define BSR_HOST_TEST_AIV_ALG_TEMPLATE_BASE_H

// AivAlgTemplateBase 的桩：生产基类的构造负责把 rankId / subCommRanks 落成 myRank_ / tempRankSize_，
// 而 BSR 模板的 KernelRun 正是把这两个值原样下发（rank / rankSize），所以这里的赋值必须与生产一致。

#include <string>
#include <vector>

#include "alg_param.h"
#include "cost_model.h"
#include "hccl_aiv_utils.h"
#include "template_utils.h"

namespace ops_hccl {

class AivAlgTemplateBase {
public:
    static constexpr TemplateProp props = {};
    AivAlgTemplateBase() = default;
    AivAlgTemplateBase(const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
        : opMode_(param.opMode),
          root_(param.root),
          myRank_(rankId),
          tempRankSize_(static_cast<u32>(subCommRanks[0].size())),
          subCommRanks_(subCommRanks),
          reduceOp_(static_cast<HcclReduceOp>(param.reduceType))
    {}
    virtual ~AivAlgTemplateBase() = default;

    virtual std::string Describe() const = 0;

    virtual HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest)
    {
        (void)comm;
        (void)param;
        (void)topoInfo;
        (void)resourceRequest;
        return HcclResult::HCCL_SUCCESS;
    }

    virtual HcclResult CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
    {
        (void)dataSize;
        numBlocks = numBlocksLimit;
        return HcclResult::HCCL_SUCCESS;
    }

    virtual HcclResult
    KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
    {
        (void)param;
        (void)tempAlgParams;
        (void)templateResource;
        return HcclResult::HCCL_SUCCESS;
    }

protected:
    OpMode opMode_ = OpMode::OPBASE;
    u32 root_ = 0;
    u32 myRank_ = INVALID_VALUE_RANKID;
    u32 tempRankSize_ = 0;
    std::vector<std::vector<u32>> subCommRanks_;
    HcclReduceOp reduceOp_ = HCCL_REDUCE_SUM;
    HcclDataType dataType_ = HCCL_DATA_TYPE_FP32;
    u32 sliceId_ = 0;
};

} // namespace ops_hccl

#endif // BSR_HOST_TEST_AIV_ALG_TEMPLATE_BASE_H
