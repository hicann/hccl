/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_RUN_PARAMS_H
#define SIM_RUN_PARAMS_H

#include "hccl_sim_pub.h"

namespace HcclSim {

class SimRunParams {
public:
    static SimRunParams& GetInstance();

    void Init(const CheckerOpParam& opParam);
    const bool& GetExternalInputHcclAicpuUnfold() { return hcclAicpuUnfold_; };
    HcclWorkflowMode GetWorkflowMode() { return workflowMode_; };

private:
    SimRunParams() = default;
    ~SimRunParams() = default;

private:
    bool hcclAicpuUnfold_{false};
    HcclWorkflowMode workflowMode_{HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE};
};

}
#endif