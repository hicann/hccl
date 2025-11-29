/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_run_params.h"

using namespace std;

namespace HcclSim {

SimRunParams& SimRunParams::GetInstance()
{
    static SimRunParams instance;
    return instance;
}

HcclWorkflowMode ConvertWorkflowMode(const CheckerOpParam& opParam) {
    switch (opParam.opMode) {
        case CheckerOpMode::OPBASE:
            return HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
        case CheckerOpMode::OFFLOAD:
            return HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB;
        default:
            return HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;    // 默认OP_BASE模式
    }
}

void SimRunParams::Init(const CheckerOpParam& opParam)
{
    hcclAicpuUnfold_ = opParam.aicpuUnfoldMode;
    workflowMode_ = ConvertWorkflowMode(opParam);
}

}