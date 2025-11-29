/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_NB_CUSTOM_H
#define SCATTER_NB_CUSTOM_H

#include "alg_template_base.h"

namespace ops_hccl {
class ScatterNB : public AlgTemplateBase {
public:
    explicit ScatterNB();

    ~ScatterNB() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels) override;
    u32 CalcCeilLog2(const u32 num)
    {
        u32 ans = 0;
        for (u32 tmp = num - 1; tmp != 0; tmp >>= 1, ++ans) {}
        return ans;
    }

protected:
private:
    std::vector<bool> slicesFlag_;

    HcclResult RunScatterTx(const u32 step, std::vector<ChannelInfo> &channels);
    HcclResult RunScatterRx(const u32 step, std::vector<ChannelInfo> &channels);

    void PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const;
    HcclResult RunScatterNB(std::vector<ChannelInfo> &channels);
    HcclResult RdmaTx(ChannelInfo &channel, const std::vector<Slice> &txSlices);
    HcclResult SdmaRx(ChannelInfo &channel, const std::vector<Slice> &rxSlices);

    u32 interRank_;       // comm内的rank排序
    u32 interRankSize_; // 本comm内ranksize总数
};
}

#endif /* SCATTER_NB_CUSTOM_H */
