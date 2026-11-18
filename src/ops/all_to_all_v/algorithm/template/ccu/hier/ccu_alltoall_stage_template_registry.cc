/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_alltoall_stage_template_registry.h"

namespace ops_hccl {

CcuAlltoAllStageTemplateRegistry& CcuAlltoAllStageTemplateRegistry::Instance()
{
    static CcuAlltoAllStageTemplateRegistry instance;
    return instance;
}

HcclResult CcuAlltoAllStageTemplateRegistry::Register(const std::string& algoName, CcuStageTemplateCreator creator)
{
    std::lock_guard<std::mutex> lock(mu_);
    auto it = creators_.find(algoName);
    if (it != creators_.end()) {
        HCCL_WARNING(
            "[CcuAlltoAllStageTemplateRegistry][%s] algoName[%s] already registered, overwrite.", __func__,
            algoName.c_str());
    }
    creators_[algoName] = creator;
    return HcclResult::HCCL_SUCCESS;
}

std::shared_ptr<CcuTempAlltoAllHierStageBase> CcuAlltoAllStageTemplateRegistry::Create(
    const std::string& algoName, const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks,
    u32 stageIndex)
{
    std::lock_guard<std::mutex> lock(mu_);
    auto it = creators_.find(algoName);
    if (it == creators_.end()) {
        HCCL_ERROR("[CcuAlltoAllStageTemplateRegistry][%s] algoName[%s] not found.", __func__, algoName.c_str());
        return nullptr;
    }
    CcuTempAlltoAllHierStageBase* raw = it->second(param, rankId, subCommRanks, stageIndex);
    if (raw == nullptr) {
        HCCL_ERROR("[CcuAlltoAllStageTemplateRegistry][%s] failed for algoName[%s].", __func__, algoName.c_str());
        return nullptr;
    }
    return std::shared_ptr<CcuTempAlltoAllHierStageBase>(raw);
}

} // namespace ops_hccl
