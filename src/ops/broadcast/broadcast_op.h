/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_REDUCE_SCATTER_OP
#define OPS_HCCL_SRC_OPS_REDUCE_SCATTER_OP

#include <string>
#include <memory>
#include "hccl.h"

#include "alg_param.h"
#include "executor_v2_base.h"
#include "alg_type.h"
#include "execute_selector.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream);
HcclResult HcclBroadcastGraphMode(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, const char* group, 
                                  aclrtStream stream, const char *tag, void **streams, size_t streamCount, void *scratchMemAddr, uint64_t scratchMemSize);

#ifdef __cplusplus
}
#endif

namespace ops_hccl {
HcclResult BroadcastOutPlace(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream, const std::string &tag);
HcclResult BroadcastOutPlaceGraphMode(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream, const std::string &tag,
                             const ResPackGraphMode &resPack);
HcclResult BroadcastOutPlaceCommon(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream, const std::string &tag,
                             OpMode opMode, const ResPackGraphMode &resPack);

HcclResult BroadcastInitAndCheck(HcclComm comm, void *buf, uint64_t count, HcclDataType dataType, uint32_t root, aclrtStream stream, std::string &opTag);

HcclResult CheckBroadcastInputPara(const HcclComm comm, const void *buf);

}

#endif