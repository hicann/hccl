/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_RING_DIRECT_CUSTOM_H
#define SCATTER_RING_DIRECT_CUSTOM_H

#include "alg_template_base.h"

namespace ops_hccl {
class ScatterRingDirect : public AlgTemplateBase {
public:
    explicit ScatterRingDirect();
    ~ScatterRingDirect() override;

    // should be called soon after template ScatterRingDirect instance created
    HcclResult Prepare(HcomCollOpInfo *opInfo, const u32 userRank,
        const std::vector<u32> &ringsOrders, const std::vector<Slice> &userMemInputSlices) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels) override;

protected:
private:
    HcclResult CheckParameters(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels);
    HcclResult OneRankMemcpy();
    HcclResult GetInitializedNeighborLinks(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels);
    HcclResult SetSlices(const u32 rank, const u32 rankSize);
    HcclResult RunScatter(const u32 rank, const u32 rankSize);
    HcclResult RunScatterOnOtherRank(const u32 stepsFromRank2Root, const u32 step, const Slice &txSlice,
                                     const Slice &rxSlice, const u32 rankSize);
    HcclResult RunScatterOnRootRank(const u32 step, const Slice &subSlice, const Slice &cclSlice, const u32 rank,
                                    const u32 rankSize);

    HcomCollOpInfo                     *opInfo_{nullptr};
    u32                                 userRank_;
    std::vector<u32>                    ringsOrder_;
    std::vector<Slice>                  userMemInputSlices_;
    u64                                       lastStepOffset_ = 0;
    ChannelInfo leftChannel_;
    ChannelInfo rightChannel_;
};
}

#endif /* SCATTER_RING_DIRECT_CUSTOM_H */