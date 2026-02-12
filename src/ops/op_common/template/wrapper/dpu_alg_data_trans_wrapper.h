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

HcclResult SendWrite(const DataInfo &snedInfo);
HcclResult RecvWrite(const DataInfo &recvInfo);
HcclResult SendRecvWrite(const SendRecvInfo &sendRecvInfo);

}
#endif // DPU_ALG_DATA_TRANS_WRAPPER