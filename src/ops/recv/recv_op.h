/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_RECV_OP
#define OPS_HCCL_SRC_OPS_RECV_OP

#include <string>
#include "alg_param.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclRecv(
    void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream);
HcclResult HcclRecvGraphMode(
    void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, const char* group, aclrtStream stream,
    const char* tag, void **streams, size_t streamCount, void *scratchMemAddr, uint64_t scratchMemSize);

#ifdef __cplusplus
}
#endif

namespace ops_hccl {
    HcclResult GetAndCheckRecvPara(
        const HcclComm comm, const void *recvBuf, const uint64_t count, const HcclDataType dataType,
        const uint32_t srcRank, u32 &rankSize, u32 &userRank, std::string &tag);
    HcclResult GenerateRecvOpParam(
        OpParam &param, void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
        const HcclComm comm, const aclrtStream stream, const std::string &tag);

    HcclResult RecvExec(
        void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
        const HcclComm comm, const aclrtStream stream, const u32 &rankSize,
        const OpMode &opMode, const std::string &tag, const ResPackGraphMode &resPack = ResPackGraphMode());
} // namespace ops_hccl

#endif // ifndef OPS_HCCL_SRC_OPS_RECV_OP
