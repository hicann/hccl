/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_P2P_COMMON_UTILS_H
#define OPS_HCCL_P2P_COMMON_UTILS_H

#include <hccl/hccl_res.h>
#include <hccl/hccl_rank_graph.h>
#include "common.h"

namespace ops_hccl_p2p {
HcclResult GetDeviceType(DeviceType *deviceType);
HcclResult AcquireChannel(HcclComm comm, CommEngine engine, DeviceType devType,
                          uint32_t srcRank, uint32_t dstRank, ChannelHandle *channel);
}
#endif
