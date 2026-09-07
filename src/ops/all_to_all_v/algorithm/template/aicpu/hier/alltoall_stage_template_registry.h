/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_STAGE_TEMPLATE_REGISTRY_H
#define ALLTOALL_STAGE_TEMPLATE_REGISTRY_H

#include <functional>
#include <map>
#include <mutex>
#include <memory>
#include <string>
#include "ins_temp_all_to_all_hier_stage_base.h"

namespace ops_hccl {

using StageTemplateCreator
    = std::function<InsTempAlltoAllHierStageBase*(const OpParam&, u32, const std::vector<std::vector<u32>>&, u32)>;

class AlltoAllStageTemplateRegistry {
public:
    static AlltoAllStageTemplateRegistry& Instance();

    HcclResult Register(const std::string& algoName, StageTemplateCreator creator);
    std::shared_ptr<InsTempAlltoAllHierStageBase> Create(
        const std::string& algoName, const OpParam& param, u32 rankId,
        const std::vector<std::vector<u32>>& subCommRanks, u32 stageIndex);

private:
    AlltoAllStageTemplateRegistry() = default;
    ~AlltoAllStageTemplateRegistry() = default;
    AlltoAllStageTemplateRegistry(const AlltoAllStageTemplateRegistry&) = delete;
    AlltoAllStageTemplateRegistry& operator=(const AlltoAllStageTemplateRegistry&) = delete;

    std::map<std::string, StageTemplateCreator> creators_;
    std::mutex mu_;
};

} // namespace ops_hccl

#define REGISTER_A2A_STAGE_TEMPLATE(algoName, ClassName)                                        \
    static HcclResult g_reg_##ClassName = AlltoAllStageTemplateRegistry::Instance().Register(   \
        #algoName, [](const OpParam& p, u32 r, const std::vector<std::vector<u32>>& s, u32 i) { \
            return new ClassName(p, r, s, i);                                                   \
        });

#endif // !ALLTOALL_STAGE_TEMPLATE_REGISTRY_H
