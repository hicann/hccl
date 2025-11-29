/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_MESH_CUSTOM_H
#define SCATTER_MESH_CUSTOM_H

#include "alg_template_base.h"

namespace ops_hccl {
class ScatterMesh : public AlgTemplateBase {
public:
    explicit ScatterMesh();

    ~ScatterMesh() override;

    // should be called soon after template ScatterMesh instance created
    HcclResult Prepare(u32 interRank, u32 interRankSize) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels) override;

protected:
private:
    HcclResult RunSendScatter(const u32 dstRank, const Slice &slice, std::vector<ChannelInfo> &channels);
    HcclResult RunRecvScatter(const u32 srcRank, const Slice &slice, std::vector<ChannelInfo> &channels);
    void PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const;

    u32 interRank_;       // comm内的rank排序
    u32 interRankSize_; // 本comm内ranksize总数
};
}

#endif /* * SCATTER_MESH_CUSTOM_H */
