/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_TEMPLATE_SELECTOR_TEST_BASE_H
#define AIV_TEMPLATE_SELECTOR_TEST_BASE_H

#include <memory>
#include <string>
#include <vector>
#include "alg_param.h"
#include "cost_model.h"
#include "hccl_aiv_utils.h"
#include "template_utils.h"

namespace ops_hccl {

// 与产品头保持一致：PCIe单算子场景启用写语义，第0层全Mesh连通时使用读语义。
inline bool IsAivWriteEnabled(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo)
{
    return param.opMode == OpMode::OPBASE && topoInfo != nullptr && topoInfo->level0PcieMix
           && !IsAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH);
}

// 仅对模板声明的依赖做桩，构造函数与虚函数行为由下方桩类提供。
class AivAlgTemplateBase {
public:
    static constexpr TemplateProp props = {};
    AivAlgTemplateBase() = default;
    AivAlgTemplateBase(const OpParam&, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
        : myRank_(rankId),
          tempRankSize_(subCommRanks.empty() ? 0 : static_cast<u32>(subCommRanks.front().size())),
          subCommRanks_(subCommRanks)
    {}
    virtual ~AivAlgTemplateBase() = default;

    virtual std::string Describe() const { return "Test template"; }
    virtual u64 CalcScratchMultiple(BufferType, BufferType) { return 1; }
    virtual HcclResult CalNumBlocks(u32& numBlocks, u64, u32 numBlocksLimit)
    {
        numBlocks = numBlocksLimit;
        return HCCL_SUCCESS;
    }
    virtual HcclResult KernelRun(const OpParam&, const TemplateDataParams&, const TemplateResource&)
    {
        return HCCL_SUCCESS;
    }
    virtual HcclResult CalcRes(HcclComm, const OpParam&, const TopoInfoWithNetLayerDetails*, AlgResourceRequest&)
    {
        return HCCL_SUCCESS;
    }

protected:
    // 与生产实现一致：只填写公共数据字段，cmdType/argsType/核数由调用方管理。
    void FillCommonAivOpArgs(
        AivOpArgs& args, const OpParam& param, const TemplateDataParams& tempAlgParams,
        const TemplateResource& templateResource) const
    {
        args.input = tempAlgParams.buffInfo.inBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
        args.output = tempAlgParams.buffInfo.outBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
        args.rank = u32(myRank_);
        args.rankSize = tempRankSize_;
        args.count = tempAlgParams.sliceSize / HCCL_SIZE_TABLE[dataType_];
        args.dataType = dataType_;
        args.op = param.reduceType;
        args.root = root_;
        args.sliceId = static_cast<uint32_t>(sliceId_);
        args.buffersIn = templateResource.aivCommInfoPtr;
        args.stream = param.stream;
        args.hcclComm = static_cast<HcclComm>(param.hcclComm);
        args.isOpBase = (param.opMode == OpMode::OPBASE);
    }

    void IncSliceId() { ++sliceId_; }

    u32 myRank_ = 0;
    u32 tempRankSize_ = 0;
    std::vector<std::vector<u32>> subCommRanks_;
    u32 root_ = 0;
    u32 sliceId_ = 0;
    HcclDataType dataType_ = HCCL_DATA_TYPE_FP32;
};

} // namespace ops_hccl

#endif // AIV_TEMPLATE_SELECTOR_TEST_BASE_H
