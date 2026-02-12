/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_ALG_TEMPLATE_BASE
#define HCCL_CCU_ALG_TEMPLATE_BASE

#include "template_utils.h"
#include "alg_param.h"

namespace ops_hccl {
class CcuAlgTemplateBase {
public:
    explicit CcuAlgTemplateBase();
    explicit CcuAlgTemplateBase(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                const std::vector<std::vector<u32>> &subCommRanks);
    virtual void InitCcuAlgTemplate(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                const std::vector<std::vector<u32>> &subCommRanks);

    virtual ~CcuAlgTemplateBase();

    virtual std::string Describe() const = 0;

    virtual HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                               AlgResourceRequest& resourceRequest);
    virtual HcclResult KernelRun(const OpParam& param,
                                 const TemplateDataParams& templateDataParams,
                                 const TemplateResource& templateResource);
                                 
    virtual HcclResult GetRes(AlgResourceRequest& resourceRequest);
    virtual u64 GetThreadNum();

    virtual u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType);

    uint64_t PointerToAddr(void* pointer);

    HcclResult GetChannelDieId(HcclComm comm, uint32_t rankId, const HcclChannelDesc& channelDesc, uint32_t& dieId);

protected:
    OpMode                        opMode_;              // 单算子还是图模式
    u32                           root_         = 0;    // 一般是scatter、broadcast需要
    u32                           myRank_       = INVALID_VALUE_RANKID;
    u32                           templateRankSize_ = 0;
    std::vector<std::vector<u32>> subCommRanks_;
    BuffInfo buffInfo_;
    HcclReduceOp reduceOp_;
    HcclDataType dataType_;
    uint64_t scratchBufferSize_ = 0;
    HcclResult RestoreChannelMap(const std::vector<HcclChannelDesc>& channelDescs,
                                 std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc);
};
}
#endif // HCCLV2_CCU_ALG_TEMPLATE_BASE