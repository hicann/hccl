/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSR_HOST_TEST_CHANNEL_REQUEST_H
#define BSR_HOST_TEST_CHANNEL_REQUEST_H

// 通道申请桩：记录 CalcRes 传下来的对端集合（增量建链的判据就是它），并按需返回指定协议的通道。

#include <vector>

#include "alg_param.h"

namespace ops_hccl {
namespace bsr_host_test {
    inline std::vector<std::vector<u32>> channelRequestRanks;
    inline u32 meshChannelRequests = 0;
    inline u32 meshClosChannelRequests = 0;
    // CLOS 分支返回的通道协议序列，用例可注入非 UB_MEM 项验证过滤
    inline std::vector<u32> closProtocolsReturned;
} // namespace bsr_host_test

inline HcclResult CalcChannelRequestMeshClosMultiJetty(
    HcclComm, const OpParam&, const TopoInfoWithNetLayerDetails*, const std::vector<std::vector<u32>>& subCommRanks,
    std::vector<HcclChannelDesc>& channelDescs, bool)
{
    bsr_host_test::channelRequestRanks = subCommRanks;
    ++bsr_host_test::meshClosChannelRequests;
    channelDescs.clear();
    for (u32 protocol : bsr_host_test::closProtocolsReturned) {
        HcclChannelDesc desc;
        desc.channelProtocol = protocol;
        channelDescs.push_back(desc);
    }
    return HcclResult::HCCL_SUCCESS;
}

inline HcclResult CalcChannelRequestMesh1D(
    HcclComm, const OpParam&, const TopoInfoWithNetLayerDetails*, const std::vector<std::vector<u32>>& subCommRanks,
    std::vector<HcclChannelDesc>& channelDescs)
{
    bsr_host_test::channelRequestRanks = subCommRanks;
    ++bsr_host_test::meshChannelRequests;
    channelDescs.clear();
    for (u32 i = 0; i < subCommRanks[0].size(); ++i) {
        channelDescs.push_back(HcclChannelDesc{});
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl

#endif // BSR_HOST_TEST_CHANNEL_REQUEST_H
