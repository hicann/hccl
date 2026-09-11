/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RE_DATA_TRANSFER_H
#define RE_DATA_TRANSFER_H

#include "alg_param.h"
#include "data_types.h"

namespace ops_hccl {

struct TransferContext {
    bool enableRemoteMemAccess = true;
    BufferType buffType = BufferType::OUTPUT;
    DataSlicesList txRxSlicesList;
    TemplateResource templateRes;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HCCL_REDUCE_RESERVED;
};

HcclResult DataTransferSend(const TransferContext& ctx);

} // namespace ops_hccl

#endif // RE_DATA_TRANSFER_H
