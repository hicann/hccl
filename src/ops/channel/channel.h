/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_CHANNEL
#define OPS_HCCL_SRC_OPS_CHANNEL

#include "hccl/base.h"
#include "alg_param.h"

namespace ops_hccl {

enum CommPlane {
    COMM_LEVEL0 = 0,
    COMM_LEVEL1,
    COMM_LEVEL2,
    COMM_LEVEL_RESERVED
};
constexpr u32 NORMAL_NOTIFY_NUM = 3;

HcclResult CalcLevel0ChannelRequest(const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    AlgType& algType, std::vector<ChannelDesc> &channels);
HcclResult CalcLevel1ChannelRequest(const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    AlgType& algType, std::vector<ChannelDesc> &channels);
HcclResult CalcLevel2ChannelRequest(const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    AlgType& algType, std::vector<ChannelDesc> &channels);

}

#endif