/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_register.h"
#include "reduce_scatter_mesh_local_reduce_template.h"

namespace ops_hccl {

ReduceScatterLocalReduceTemplate::ReduceScatterLocalReduceTemplate() : AlgTemplateBase()
{
}

ReduceScatterLocalReduceTemplate::~ReduceScatterLocalReduceTemplate()
{
}

HcclResult ReduceScatterLocalReduceTemplate::RunAsync(
    const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels)
{
    // ranksize ==1 的处理
    if (rankSize == 1) {
        if (inputMem_.addr != outputMem_.addr) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, outputMem_.addr, inputMem_.addr, inputMem_.size)));
        }
        return HCCL_SUCCESS;
    }
    for (u32 i = 1; i < rankSize; i++) {
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread_, scratchMem_.addr,
            static_cast<void *>(static_cast<s8 *>(inputMem_.addr) + count_ * sizeof(float) * i),
            count_, static_cast<HcommDataType>(dataType_), static_cast<HcommReduceOp>(reductionOp_))));
    }
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, outputMem_.addr, scratchMem_.addr, slices_[0].size)));

    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCE_SCATTER_LOCAL_REDUCE, ReduceScatterLocalReduceTemplate);
}