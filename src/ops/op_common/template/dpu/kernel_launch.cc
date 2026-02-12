/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_launch.h"
#include "log.h"
#include "template_utils.h"
#include "alg_v2_template_register.h"

namespace ops_hccl {
int32_t HcclLaunchDPUKernel(uint64_t ptr, int32_t size)
{
    if ((ptr == 0) || (size <= 0)) {
        HCCL_ERROR("%s get nullptr or error size", __func__);
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    // 反序列化共享内存
    auto shmemPtr = reinterpret_cast<char *>(ptr);
    std::vector<char> sequenceData(shmemPtr, shmemPtr + size);
    DPURunInfo dpuRunInfo;
    dpuRunInfo.DeSerialize(sequenceData);

    // 根据名字获取template
    auto templateIns = InsAlgTemplateRegistry::Instance().GetAlgTemplate(dpuRunInfo.templateName);
    if (templateIns.get() == nullptr) {
        HCCL_ERROR("Fail to find template of %s", dpuRunInfo.templateName.c_str());
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    // dpu算法展开
    if (templateIns->DPUKernelRun(dpuRunInfo.tempAlgParams, dpuRunInfo.channels, dpuRunInfo.myRank, dpuRunInfo.subCommRanks) != HCCL_SUCCESS) {
        HCCL_ERROR("Template[%s] DPUKernelRun failed", dpuRunInfo.templateName.c_str());
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    return static_cast<int32_t>(HCCL_SUCCESS);
}
}