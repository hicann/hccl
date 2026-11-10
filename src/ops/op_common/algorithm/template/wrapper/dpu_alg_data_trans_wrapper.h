/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_ALG_DATA_TRANS_WRAPPER
#define DPU_ALG_DATA_TRANS_WRAPPER

#include "hcomm_primitives.h"
#include "hccl_types.h"
#include "alg_param.h"
#include "template_utils.h"

namespace ops_hccl {

HcclResult
HcommChannelDrainOnThreadWithCompat(ThreadHandle thread, ChannelHandle channel, bool isDrainSupported = false);
HcclResult DpuSendWrite(const DataInfo& sendInfo, ThreadHandle thread = 0, bool isDrainSupported = false);
HcclResult DpuRecvWrite(const DataInfo& recvInfo, ThreadHandle thread = 0, bool isDrainSupported = false);
HcclResult DpuSendRecvWrite(const SendRecvInfo& sendRecvInfo, ThreadHandle thread = 0, bool isDrainSupported = false);
HcclResult SendWriteForNda(const DataInfo& sendInfo, const ThreadHandle& thread);
HcclResult RecvWriteForNda(const DataInfo& recvInfo, const ThreadHandle& thread);
HcclResult SendRecvWriteForNda(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread);
HcclResult SendNdaSlices(
    const ChannelInfo& channel, const std::vector<DataSlice>& srcSlices, const std::vector<DataSlice>& dstSlices,
    const ThreadHandle& thread);
HcclResult RecvNdaSlices(const ChannelInfo& channel, const ThreadHandle& thread);

} // namespace ops_hccl
#endif // DPU_ALG_DATA_TRANS_WRAPPER
