/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_NHR_CUSTOM_H
#define SCATTER_NHR_CUSTOM_H

#include "alg_template_base.h"
#include "nhr_base.h"

namespace ops_hccl {
class ScatterNHR : public NHRBase {
public:
    explicit ScatterNHR();

    ~ScatterNHR() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels) override;
    HcclResult RunScatterNHR(std::vector<ChannelInfo> &channels);

protected:
private:
    void PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const;
    HcclResult SdmaRx(ChannelInfo &channelLeft, ChannelInfo &channelRight, InterServerAlgoStep &stepInfo);
    HcclResult RdmaTxRx(ChannelInfo &channelLeft, ChannelInfo &channelRight, InterServerAlgoStep &stepInfo);
    HcclResult RdmaTxRx();
    HcclResult Tx(const ChannelInfo &channel, std::vector<Slice> &txSlices);
    HcclResult Rx(const ChannelInfo &channel, std::vector<Slice> &rxSlices);

    HcclResult GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, InterServerAlgoStep &stepInfo);
    // 有点奇怪，这边为什么不用基类里面的函数呢
    HcclResult ExecuteBarrier(ChannelInfo &channelLeft, ChannelInfo &channelRight);

    u32 interRank_;       // comm内的rank排序
    u32 interRankSize_;  // 本comm内ranksize总数

    // 开源开放新接口资源
    void* engineCtx_ = nullptr;
};
}
#endif /* SCATTER_NHR_CUSTOM_H */
