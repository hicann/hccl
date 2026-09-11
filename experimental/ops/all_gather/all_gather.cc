/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "algo_desc.h"
#include "executor/adaptor_executor.h"
#include "topo/topo_match_four_level.h"
#include "alg_attrs_registry.h"

namespace ops_hccl {

// 4级串行：Mesh(layer3) -> NHR(layer2) -> NHR(layer1) -> Mesh(layer0)
static AlgoExecDesc MakeAllGather4LevelAlgoExecDesc()
{
    TemplateDesc meshDesc{HcclCMDType::HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH};
    TemplateDesc nhrDesc{HcclCMDType::HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_NHR};

    AlgoExecDesc desc;
    desc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    desc.children = {
        TemplateExecDesc{meshDesc, SUB_COMM_INDEX_3},
        TemplateExecDesc{nhrDesc, SUB_COMM_INDEX_2},
        TemplateExecDesc{nhrDesc, SUB_COMM_INDEX_1},
        TemplateExecDesc{meshDesc, SUB_COMM_INDEX_0},
    };
    desc.dataSplitRatio = {1, 1, 1, 1};
    return desc;
}

// AicpuAllGatherSequenceMeshNHRNHRMesh (TopoMatchFourLevel, Sequence 4层)
static HcclAlgorithm MakeAicpuAllGatherSequenceMeshNHRNHRMesh()
{
    HcclAlgorithm algo;
    algo.hcclCmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    algo.engineType = HcclAlgEngineType::COMM_ENGINE_AICPU;
    algo.topoMatch = std::make_shared<TopoMatchFourLevel>();
    AlgAttrsRegistry::ParseAlgName("AicpuAllGatherSequenceMeshNHRNHRMesh", algo.algAttrs);
    algo.algoExecDesc = MakeAllGather4LevelAlgoExecDesc();
    algo.algName = "AicpuAllGatherSequenceMeshNHRNHRMesh";
    return algo;
}

REGISTER_ALG(
    HcclCMDType::HCCL_CMD_ALLGATHER, AicpuAllGatherSequenceMeshNHRNHRMesh, MakeAicpuAllGatherSequenceMeshNHRNHRMesh());

} // namespace ops_hccl
