/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_base.h"
#include "alg_template_base_experimental.h"
#include "exec_timeout_manager.h"

namespace ops_hccl_experimental {

HcclResult AlgTemplateBaseExperimental::Prepare(
    HcclMem& inputMem, HcclMem& outputMem, HcclMem& scratchMem, const u64 count, const HcclDataType dataType,
    ThreadHandle thread, const std::vector<ThreadHandle>& slaveThreads, const HcclReduceOp reductionOp, const u32 root,
    const std::vector<Slice>& slices, const u64 baseOffset, const bool disableDMAReduce)
{
    HCCL_ERROR("Unexpected base class function fallback. Missing override in derived class");
    return HCCL_E_PARA;
}

} // namespace ops_hccl_experimental
