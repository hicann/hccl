/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALG_V2_TEMPLATE_BASE
#define ALG_V2_TEMPLATE_BASE

#include <vector>
#include <memory>
#include "template_utils.h"
#include "alg_template_base.h"

#ifdef __cplusplus
extern "C" {
#endif
int32_t __attribute__((weak)) HcommThreadJoin(ThreadHandle thread, uint32_t timeout);
#ifdef __cplusplus
}
#endif

namespace ops_hccl {

class InsAlgTemplateBase {
public:
    explicit InsAlgTemplateBase() {};
    explicit InsAlgTemplateBase(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                const std::vector<std::vector<u32>> &subCommRanks);

    virtual ~InsAlgTemplateBase();

    virtual std::string Describe() const = 0;

    // 将原来的 InsQuePtr替换为ThreadHandle, 将tempLinks换位channels
    virtual HcclResult KernelRun(const OpParam& param,
                                 const TemplateDataParams& tempAlgParams,
                                 const TemplateResource& templateResource);

    virtual HcclResult DPUKernelRun(const TemplateDataParams& tempAlgParam,
        const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
        const std::vector<std::vector<uint32_t>>& subCommRanks);

    // calculate resources
    virtual HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                               AlgResourceRequest& resourceRequest);
    virtual HcclResult GetRes(AlgResourceRequest& resourceRequest) const;
    virtual u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType);

    virtual u64 GetThreadNum() const;

    virtual void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) = 0;

    virtual void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) = 0;

protected:

    OpMode                           opMode_; // 单算子还是图模式
    u32                              root_ = 0;  // 一般是scatter、broadcast需要

    u32                              myRank_       = INVALID_VALUE_RANKID;
    u32                              templateRankSize_ = 0;
    std::vector<std::vector<u32>>    subCommRanks_;

    BuffInfo                         buffInfo_;

    u32                              threadNum_ = 0;
    HcclReduceOp                     reduceOp_;
    HcclDataType                     dataType_;

    // 从OpParam中获取
    bool                             enableDetour_    = false;
    // 用于记录主thread向从thread发送record的时候使用从thread的哪个notify
    std::vector<u32>                 notifyIdxMainToSub_;
    // 用于记录从thread向主thread发送record的时候使用主thread的哪个notify
    std::vector<u32>                 notifyIdxSubToMain_;
};
} // namespace Hccl

#endif // !HCCLV2_INS_ALG_TEMPLATE_BASE