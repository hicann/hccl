/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_REDUCE_SCATTER_CCU_COMMON_UTILS_H
#define OPS_HCCL_REDUCE_SCATTER_CCU_COMMON_UTILS_H

#include <hccl/hccl_res.h>
#include <hccl/hccl_rank_graph.h>
#include "common.h"
#include "ccu_kernel.h"

namespace ops_hccl_rs {

HcclResult GetDeviceType(DeviceType* deviceType);

HcclResult GetThreadForCcu(HcclComm comm, const OpParam& param, AlgResourceCtxSerializable& resCtxHost);

HcclResult GetChannelForCcu(HcclComm comm, const OpParam& param, std::vector<ChannelHandle>& kernelChannels);

HcclResult GetCcuKernel(
    HcclComm comm, const OpParam& param, AlgResourceCtxSerializable& resCtxHost,
    const std::vector<ChannelHandle>& kernelChannels, CcuKernelInfo& kernelInfo);

HcclResult AllocAlgResource(HcclComm comm, const OpParam& param, AlgResourceCtxSerializable& resCtxHost);

constexpr uint64_t SetBits(uint16_t start, uint16_t end);

constexpr uint64_t SetBits(uint16_t end);

uint64_t GetMaxLoopIterNum();

uint64_t GetLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum);

uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum);

uint64_t GetOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset);

std::vector<uint64_t> CalGoSize(uint64_t size, const LoopGroupConfig& config);

} // namespace ops_hccl_rs

#endif // OPS_HCCL_REDUCE_SCATTER_CCU_COMMON_UTILS_H
