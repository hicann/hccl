/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_BIRS_INTER_H
#define REDUCE_SCATTER_BIRS_INTER_H

#include "reduce_scatter_birs.h"
#include "alg_template_base_experimental.h"

namespace ops_hccl_experimental {

class ReduceScatterBIRSInter : public ReduceScatterBIRS {
public:
    explicit ReduceScatterBIRSInter();

    ~ReduceScatterBIRSInter() override;

    HcclResult Prepare(u32 serverNum, u32 interRankSize) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels) override;

protected:
    HcclResult LocalCopyPreproc(ThreadHandle& stream, const u32 rank, u64 sliceSize, u64 localStrideSize);

    HcclResult Preprocess(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels) override;
    HcclResult
    HCCSIntraStep(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize);
    HcclResult
    SIOIntraStep(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize);
    HcclResult LocalCopyIntraStep(
        u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize);
    HcclResult PreprocInterServer(
        const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize,
        std::vector<ChannelInfo>& channels);
    HcclResult InterServer(
        const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize,
        std::vector<ChannelInfo>& channels);

private:
    u32 serverNum_;
    u32 intraRankSize_;
    std::vector<u32> vec_offsets;
};
} // namespace ops_hccl_experimental

#endif /* * REDUCE_SCATTER_BIRS_INTER_H */
