/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_ALLTOALL_STAGE_TEMPLATE_REGISTRY_H_
#define CCU_ALLTOALL_STAGE_TEMPLATE_REGISTRY_H_

#include <functional>
#include <map>
#include <mutex>
#include <memory>
#include <string>
#include "ccu_temp_all_to_all_hier_stage_base.h"

namespace ops_hccl {

/* CCU 二级序列 AllToAll 的 stage 模板注册表（名字注册 + 工厂创建 + 自注册宏） */
using CcuStageTemplateCreator
    = std::function<CcuTempAlltoAllHierStageBase*(const OpParam&, u32, const std::vector<std::vector<u32>>&, u32)>;

class CcuAlltoAllStageTemplateRegistry {
public:
    static CcuAlltoAllStageTemplateRegistry& Instance();

    HcclResult Register(const std::string& algoName, CcuStageTemplateCreator creator);
    std::shared_ptr<CcuTempAlltoAllHierStageBase> Create(
        const std::string& algoName, const OpParam& param, u32 rankId,
        const std::vector<std::vector<u32>>& subCommRanks, u32 stageIndex);

private:
    CcuAlltoAllStageTemplateRegistry() = default;
    ~CcuAlltoAllStageTemplateRegistry() = default;
    CcuAlltoAllStageTemplateRegistry(const CcuAlltoAllStageTemplateRegistry&) = delete;
    CcuAlltoAllStageTemplateRegistry& operator=(const CcuAlltoAllStageTemplateRegistry&) = delete;

    std::map<std::string, CcuStageTemplateCreator> creators_;
    std::mutex mu_;
};

} // namespace ops_hccl

#define REGISTER_CCU_A2A_STAGE_TEMPLATE(algoName, ClassName)                                        \
    static HcclResult g_ccuReg_##ClassName = CcuAlltoAllStageTemplateRegistry::Instance().Register( \
        #algoName, [](const OpParam& p, u32 r, const std::vector<std::vector<u32>>& s, u32 i) {     \
            return new ClassName(p, r, s, i);                                                       \
        });

#endif // CCU_ALLTOALL_STAGE_TEMPLATE_REGISTRY_H_
