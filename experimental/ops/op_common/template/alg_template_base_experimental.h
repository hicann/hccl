/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALG_TEMPLATE_BASE_EXPERIMENTAL_PUB_H
#define ALG_TEMPLATE_BASE_EXPERIMENTAL_PUB_H

#include <vector>
#include <memory>
#include <list>
#include "alg_param.h"
#include "utils.h"
#include "alg_template_base.h"

namespace ops_hccl_experimental {
using ops_hccl::Slice;
using ops_hccl::TemplateType;

constexpr TemplateType TEMPLATE_REDUCE_SCATTER_BIRS = static_cast<TemplateType>(1001);
constexpr TemplateType TEMPLATE_REDUCE_SCATTER_BIRS_INTER = static_cast<TemplateType>(1002);

class AlgTemplateBaseExperimental : public ops_hccl::AlgTemplateBase {
public:
    virtual HcclResult Prepare(
        HcclMem& inputMem, HcclMem& outputMem, HcclMem& scratchMem, const u64 count, const HcclDataType dataType,
        ThreadHandle thread, const std::vector<ThreadHandle>& slaveThreads, const HcclReduceOp reductionOp,
        const u32 root, const std::vector<Slice>& slices, const u64 baseOffset, const bool disableDMAReduce);
};
} // namespace ops_hccl_experimental

#endif /* ALG_TEMPLATE_BASE_EXPERIMENTAL_PUB_H */
