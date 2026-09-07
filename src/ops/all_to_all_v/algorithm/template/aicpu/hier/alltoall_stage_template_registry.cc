/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoall_stage_template_registry.h"

namespace ops_hccl {

AlltoAllStageTemplateRegistry& AlltoAllStageTemplateRegistry::Instance()
{
    static AlltoAllStageTemplateRegistry instance;
    return instance;
}

HcclResult AlltoAllStageTemplateRegistry::Register(const std::string& algoName, StageTemplateCreator creator)
{
    std::lock_guard<std::mutex> lock(mu_);
    auto it = creators_.find(algoName);
    if (it != creators_.end()) {
        HCCL_WARNING("[AlltoAllStageTemplateRegistry] algoName[%s] already registered, overwrite.", algoName.c_str());
    }
    creators_[algoName] = creator;
    return HcclResult::HCCL_SUCCESS;
}

std::shared_ptr<InsTempAlltoAllHierStageBase> AlltoAllStageTemplateRegistry::Create(
    const std::string& algoName, const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks,
    u32 stageIndex)
{
    std::lock_guard<std::mutex> lock(mu_);
    auto it = creators_.find(algoName);
    if (it == creators_.end()) {
        HCCL_ERROR("[AlltoAllStageTemplateRegistry] algoName[%s] not found in registry.", algoName.c_str());
        return nullptr;
    }
    InsTempAlltoAllHierStageBase* raw = it->second(param, rankId, subCommRanks, stageIndex);
    if (raw == nullptr) {
        HCCL_ERROR(
            "[AlltoAllStageTemplateRegistry] Failed to create stage template for algoName[%s].", algoName.c_str());
        return nullptr;
    }
    return std::shared_ptr<InsTempAlltoAllHierStageBase>(raw);
}

} // namespace ops_hccl
