/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALGO_DESC_H
#define ALGO_DESC_H

#include <memory>
#include <variant>
#include <vector>
#include <string>

#include <hccl/hccl_types.h>
#include "topo_match_base_v2.h"
#include "alg_attrs.h"
#include "alg_type.h"
#include "alg_param.h"

namespace ops_hccl {

using HcclAlgEngineType = CommEngine;

enum class HcclAlgExecPolicy {
    SEQUENCE,
    PARALLEL,
    OMNIPIPE,
};

struct TemplateDesc {
    HcclCMDType hcclCmdType{HcclCMDType::HCCL_CMD_INVALID};
    HcclAlgoType algType{HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT};
};

enum SubCommIndexType : int {
    SUB_COMM_INDEX_0 = 0,
    SUB_COMM_INDEX_1 = 1,
    SUB_COMM_INDEX_2 = 2,
    SUB_COMM_INDEX_3 = 3,
    SUB_COMM_INDEX_4 = 4,
    SUB_COMM_INDEX_5 = 5,
};

struct TemplateExecDesc {
    TemplateDesc templateDesc;
    int subCommIndex = 0;
    int netLayer = -1;
};

struct AlgoExecDesc;
using VariantType = std::variant<TemplateExecDesc, std::shared_ptr<AlgoExecDesc>>;
struct AlgoExecDesc {
    HcclAlgExecPolicy execPolicy = HcclAlgExecPolicy::SEQUENCE;
    std::vector<VariantType> children;
    std::vector<u32> dataSplitRatio;
};

class OpsExecutor;
class BaseEngine;

class HcclAlgorithm {
public:
    HcclAlgorithm() = default;
    ~HcclAlgorithm() = default;

    std::unique_ptr<OpsExecutor> GetExecutor(const OpParam& param);
    void Dump();

    HcclCMDType hcclCmdType{HcclCMDType::HCCL_CMD_INVALID};
    HcclAlgEngineType engineType{HcclAlgEngineType::COMM_ENGINE_RESERVED};
    std::shared_ptr<TopoMatchBaseV2> topoMatch;
    AlgAttrs algAttrs;
    AlgoExecDesc algoExecDesc;
    std::string algName;
};

} // namespace ops_hccl

#endif // ALGO_DESC_H
