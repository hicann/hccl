/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "algo_desc.h"

#include "log.h"
#include "executor/ops_executor.h"

namespace ops_hccl {

std::unique_ptr<OpsExecutor> HcclAlgorithm::GetExecutor(const OpParam& param)
{
    return std::make_unique<OpsExecutor>(*this, param);
}

void HcclAlgorithm::Dump()
{
    HCCL_INFO(
        "[HcclAlgorithm][Dump] engineType[%d], hcclCmdType[%d], execPolicy[%d], childrenNum[%zu]",
        static_cast<int>(engineType), static_cast<int>(hcclCmdType), static_cast<int>(algoExecDesc.execPolicy),
        algoExecDesc.children.size());
}

} // namespace ops_hccl
