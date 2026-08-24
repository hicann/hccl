/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cost_table.h"

#include "hccl_algo_dims.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <functional>
#include <map>
#include <new>
#include <set>

#include "auto_selector_base.h"
#include "hccl_aiv_utils.h"
#include "order_preserved_common.h"
#include "selector_engine.h"

namespace ops_hccl {

// ---------------------------------------------------------------------------
// 匿名命名空间：规则构建函数 + helper
// ---------------------------------------------------------------------------
namespace {

    inline bool IsAgInputOutputOverlap(const OpParam& op)
    {
        if (op.inputPtr == nullptr || op.outputPtr == nullptr || op.inputSize == 0 || op.outputSize == 0) {
            return false;
        }
        uintptr_t inStart = reinterpret_cast<uintptr_t>(op.inputPtr);
        uintptr_t outStart = reinterpret_cast<uintptr_t>(op.outputPtr);
        return inStart <= outStart + op.outputSize - 1 && outStart <= inStart + op.inputSize - 1;
    }

    std::vector<AlgFilterRule> BuildAllReduceRules()
    {
        static const std::set<std::string> aivAlgos = {"AivAllReduceSoleMeshOneShot", "AivAllReduceSoleMeshTwoShot"};
        static const std::set<std::string> ccuMsAlgos
            = {"CcuMSAllReduceSoleMeshOneShot",        "CcuMSAllReduceSoleMesh",
               "CcuMSAllReduceSoleMesh2Die",           "CcuMSAllReduceSequenceMesh2Die",
               "CcuMSAllReduceConcurMeshNHRMultiLink", "CcuMSAllReduceSoleMeshMsConcur",
               "CcuMSAllReducePipeLineMeshNHR"};
        static const std::set<std::string> ccuSchedAlgos = {"CcuSchedAllReduceSoleNHR",
                                                            "CcuSchedAllReduceSequenceMeshMesh",
                                                            "CcuSchedAllReduceSoleMesh",
                                                            "CcuSchedAllReduceParallelMeshNHR",
                                                            "CcuSchedAllReduceConcurMeshNHRMultiLink",
                                                            "CcuAllReduceParallelNHR1DMutiJetty",
                                                            "CcuSchedAllReducePipeLineMeshNHR",
                                                            "CcuSchedAllReduceSoleNHRMultiLink",
                                                            "CcuSchedAllReduceSoleMesh2Die",
                                                            "CcuSchedAllReduceSequenceMesh2Die"};

        std::set<std::string> ccuAll = ccuMsAlgos;
        ccuAll.insert(ccuSchedAlgos.begin(), ccuSchedAlgos.end());
        std::set<std::string> ccuAivAll = ccuAll;
        ccuAivAll.insert(aivAlgos.begin(), aivAlgos.end());

        return {
            // 必不选：int8 排除 ccu sched 的 SequenceMesh1D/Mesh1DMem2Mem
            {"int8_skip_ccu_seq",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails*) {
                 return op.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8;
             },
             false,
             {"CcuSchedAllReduceSequenceMeshMesh", "CcuSchedAllReduceSoleMesh"}},
            // 必不选：PROD 排除 ccu + aiv
            {"prod_skip_ccu_aiv",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails*) {
                 return op.reduceType == HcclReduceOp::HCCL_REDUCE_PROD;
             },
             false, ccuAivAll},
            // 必不选：64bit(INT64/UINT64/FP64) 排除 ccu + aiv
            {"64bit_skip_ccu_aiv",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails*) {
                 return Is64BitDataType(op.DataDes.dataType);
             },
             false, ccuAivAll},
            // 必不选：保序模式排除 ccu + aiv（回退 aicpu）
            {"order_preserved_skip_ccu_aiv",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails* topo) {
                 return IsNeedStrictModeForOrderPreserved(op, topo->userRankSize);
             },
             false, ccuAivAll},
            // 必不选：aiv + level2Uboe 排除 aiv
            {"aiv_skip_level2uboe",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->topoLevelNums == TOPO_LEVEL_NUM_3 && topo->level2Uboe;
             },
             false, aivAlgos},
            // 必选：保序模式 + rankSize > 32 → AllReduceOrderPreservedGroup
            {"order_preserved_group",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails* topo) {
                 return IsNeedStrictModeForOrderPreserved(op, topo->userRankSize)
                        && topo->userRankSize > MAX_RANK_NUM_FOR_ORDER_PRESERVED;
             },
             true,
             {"AllReduceOrderPreservedGroup"}},
            // 必选：保序模式 + rankSize <= 32 → AicpuAllReduceStrictOrderedMesh
            {"order_preserved",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails* topo) {
                 return IsNeedStrictModeForOrderPreserved(op, topo->userRankSize)
                        && topo->userRankSize <= MAX_RANK_NUM_FOR_ORDER_PRESERVED;
             },
             true,
             {"AicpuAllReduceStrictOrderedMesh"}},
            // 必选：3级拓扑 → AicpuAllReduceSequenceMeshConcurNHRNHR
            {"three_level_only",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->topoLevelNums == TOPO_LEVEL_NUM_3;
             },
             true,
             {"AicpuAllReduceSequenceMeshConcurNHRNHR"}},
            // 必选：level0 本地实例仅1卡（每框仅1卡参与通信）→ SoleNHR
            {"single_card_per_server_sole_nhr",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return !topo->netLayerDetails.localNetInsSizeOfLayer.empty()
                        && topo->netLayerDetails.localNetInsSizeOfLayer[0] == 1;
             },
             true,
             {"CcuSchedAllReduceSoleNHR", "AicpuAllReduceSoleNHR", "AivAllReduceSoleMeshOneShot",
              "AivAllReduceSoleMeshTwoShot"}},
            // 必不选：2Die 算法仅在 TWO_DIE_REGULAR 拓扑下可选，其他拓扑排除
            {"two_die_regular_only",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->level0MeshType != Level0MeshType::TWO_DIE_REGULAR;
             },
             false,
             {"CcuSchedAllReduceSoleMesh2Die", "CcuSchedAllReduceSequenceMesh2Die"}},
            // 必不选：parallel/sequence/concurrent 算法仅在多级拓扑(topoLevelNums>1)下可选
            {"multilevel_only",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->topoLevelNums == 1;
             },
             false,
             {"AicpuAllReduceConcurMeshTwoShotNHR", "AicpuAllReduceParallelMeshNHR",
              "AicpuAllReduceSequenceMeshConcurNHR", "AicpuAllReduceSequenceMeshConcurNHRNHR",
              "CcuAllReduceParallelNHR1DMutiJetty", "CcuMSAllReduceConcurMeshNHRMultiLink",
              "CcuMSAllReduceSequenceMesh2Die", "CcuSchedAllReduceConcurMeshNHRMultiLink",
              "CcuSchedAllReduceParallelMeshNHR", "CcuSchedAllReduceSequenceMesh2Die",
              "CcuSchedAllReduceSequenceMeshMesh", "DpuAllReduceSequenceMeshNHR", "InsAllReduceParallelMesh1DNHRPcie",
              "InsAllReduceParallelRSAGDpu", "InsAllReduceParallelRSAGUBX", "InsAllReduceParallelRSAGUboe"}},
        };
    }

    std::vector<AlgFilterRule> BuildAllGatherRules()
    {
        static const std::set<std::string> aivAlgos = {"AivAllGatherSoleMesh"};
        static const std::set<std::string> ccuMsAlgos
            = {"CcuMSAllGatherSoleMesh", "CcuMSAllGatherSoleMesh2Die", "CcuSchedAllGatherSoleMesh2Die",
               "CcuSchedAllGatherSoleNHRMultiLink", "CcuSchedAllGatherSoleMeshConcur"};
        static const std::set<std::string> ccuSchedAlgos
            = {"CcuSchedAllGatherSoleMesh",
               "CcuSchedAllGatherSoleNHR",
               "CcuSchedAllGatherSequenceMeshMesh",
               "CcuSchedAllGatherParallelMeshNHR",
               "CcuSchedAllGatherParallelMeshNHRMultiLink",
               "CcuMSAllGatherConcurMeshNHRMultiLink",
               "CcuSchedAllGatherConcurMeshNHRMultiLink",
               "CcuSchedAllGatherPipeLineMeshNHR"};
        static const std::set<std::string> ccuAll = [&]() {
            std::set<std::string> s = ccuMsAlgos;
            s.insert(ccuSchedAlgos.begin(), ccuSchedAlgos.end());
            return s;
        }();
        static const std::set<std::string> aivCcuSched = [&]() {
            std::set<std::string> s = aivAlgos;
            s.insert(ccuSchedAlgos.begin(), ccuSchedAlgos.end());
            return s;
        }();
        static const std::set<std::string> twoDieAlgos
            = {"CcuMSAllGatherSoleMesh2Die", "CcuSchedAllGatherSoleMesh2Die"};
        static const std::set<std::string> concurrentAlgos
            = {"CcuMSAllGatherConcurMeshNHRMultiLink", "CcuSchedAllGatherConcurMeshNHRMultiLink",
               "AicpuAllGatherConcurMeshNHR"};
        static const std::set<std::string> multilevelAlgos
            = {"CcuSchedAllGatherSequenceMeshMesh",
               "CcuSchedAllGatherParallelMeshNHR",
               "AicpuAllGatherParallelMeshNHR",
               "AicpuAllGatherParallelNHRNHR",
               "AicpuAllGatherSequenceMeshConcurNHR",
               "AicpuAllGatherSequenceMeshConcurNHRNHR",
               "AicpuAllGatherPipeLine",
               "DpuAllGatherPipeLineMeshNHRNHR",
               "DpuAllGatherSequenceMeshNHR"};
        // 仅单级拓扑可用的 sole 算法，多级时排除
        static const std::set<std::string> singleLevelOnlyAlgos
            = {"AicpuAllGatherSoleMeshConcur", "AicpuAllGatherSoleMesh"};
        // 仅在非 MESH_1D 拓扑(CLOS/MESH_1D_CLOS)下才会被 selector 选中的算法
        // 依据 all_gather_auto_selector.cc：这些算法在 level0Topo==MESH_1D 的任何分支都不会被选中
        static const std::set<std::string> nonMesh1dAlgos
            = {"AicpuAllGatherSoleNHRMultiLink",
               "InsAllGatherParallelMesh1DNHRPcie",
               "AicpuAllGatherPipeLinePcie",
               "AicpuAllGatherConcurMeshNHR",
               "InsAllGatherParallelMesh1DNHRMultiJetty",
               "AicpuAllGatherPipeLineUBX",
               "DpuAllGatherPipeLineUBX",
               "CcuSchedAllGatherConcurMeshNHRMultiLink",
               "CcuSchedAllGatherParallelMeshNHRMultiLink",
               "CcuSchedAllGatherPipeLineMeshNHR",
               "CcuSchedAllGatherSoleNHRMultiLink"};

        return {
            // 必不选：inplace(input/output overlap) 排除 ccu(ccu_ms和ccu_sched都不支持)
            {"inplace_skip_ccu",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails*) {
                 return IsAgInputOutputOverlap(op);
             },
             false, ccuAll},
            // 必不选：level2UbRtp 排除 aiv + ccu_sched
            {"level2ub_rtp_skip_aiv_ccu_sched",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->level2UbRtp;
             },
             false, aivCcuSched},
            // 必不选：3级拓扑 + level2Uboe 排除 aiv + ccu_sched
            {"level2uboe_skip_aiv_ccu_sched",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->topoLevelNums == TOPO_LEVEL_NUM_3 && topo->level2Uboe;
             },
             false, aivCcuSched},
            // 必不选：ccu_ms 仅支持单级拓扑，多级拓扑排除
            {"ccu_ms_single_level_only",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->topoLevelNums > 1;
             },
             false, ccuMsAlgos},
            // 必不选：2Die 算法仅在 TWO_DIE_REGULAR 拓扑下可选，其他拓扑排除
            {"two_die_regular_only",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->level0MeshType != Level0MeshType::TWO_DIE_REGULAR;
             },
             false, twoDieAlgos},
            // 必不选：concurrent 算法仅在 MESH_1D_CLOS(UBX) 拓扑下可选，其他拓扑排除
            {"concurrent_ubx_only",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->level0Topo != Level0Shape::MESH_1D_CLOS;
             },
             false, concurrentAlgos},
            // 必不选：仅在非 MESH_1D 拓扑下可选的算法，在 level0 topo 为 MESH_1D 时排除
            {"non_mesh1d_only",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->level0Topo == Level0Shape::MESH_1D;
             },
             false, nonMesh1dAlgos},
            // 必不选：parallel/sequence/concurrent/omnipipe/DPU 算法仅在多级拓扑(topoLevelNums>1)下可选
            {"multilevel_only",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->topoLevelNums == 1;
             },
             false, multilevelAlgos},
            // 必不选：仅单级拓扑可用的 sole 算法，多级时排除
            {"single_level_only",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->topoLevelNums > 1;
             },
             false, singleLevelOnlyAlgos},
        };
    }

    std::vector<AlgFilterRule> BuildReduceScatterRules()
    {
        static const std::set<std::string> aivAlgos = {"AivReduceScatterSoleMesh"};
        static const std::set<std::string> ccuMsAlgos
            = {"CcuMSReduceScatterSoleMesh", "CcuMSReduceScatterSoleMesh2Die", "CcuMSReduceScatterSoleMeshConcur",
               "CcuMSReduceScatterConcurMeshNHRMultiLink", "CcuMSReduceScatterPipeLineMeshNHR"};
        static const std::set<std::string> ccuSchedAlgos
            = {"CcuSchedReduceScatterSoleNHR",
               "CcuSchedReduceScatterSequenceMeshMesh",
               "CcuSchedReduceScatterParallelMeshNHR",
               "CcuSchedReduceScatterParallelMeshNHRMultiLink",
               "CcuSchedReduceScatterConcurMeshNHRMultiLink",
               "CcuSchedReduceScatterSoleNHRMultiLink",
               "CcuSchedReduceScatterPipeLineMeshNHR",
               "CcuSchedReduceScatterSoleMesh",
               "CcuSchedReduceScatterSoleMesh2Die"};
        static const std::set<std::string> ccuAll = [&]() {
            std::set<std::string> s = ccuMsAlgos;
            s.insert(ccuSchedAlgos.begin(), ccuSchedAlgos.end());
            return s;
        }();
        static const std::set<std::string> ccuAivAll = [&]() {
            std::set<std::string> s = ccuAll;
            s.insert(aivAlgos.begin(), aivAlgos.end());
            return s;
        }();
        static const std::set<std::string> aivCcuSched = [&]() {
            std::set<std::string> s = aivAlgos;
            s.insert(ccuSchedAlgos.begin(), ccuSchedAlgos.end());
            return s;
        }();
        static const std::set<std::string> twoDieAlgos
            = {"CcuMSReduceScatterSoleMesh2Die", "CcuSchedReduceScatterSoleMesh2Die"};
        static const std::set<std::string> multilevelAlgos
            = {"AicpuReduceScatterParallelMeshNHR",
               "InsReduceScatterParallelMesh1DNHRUBX",
               "InsReduceScatterParallelMesh1DNHRPcie",
               "InsReduceScatterParallelNHRNHRUboe",
               "DpuReduceScatterSequenceMeshMesh",
               "AicpuReduceScatterSequenceMeshConcurNHRNHR",
               "AicpuReduceScatterSequenceMeshConcurNHR",
               "AicpuReduceScatterConcurMeshNHR",
               "CcuSchedReduceScatterSequenceMeshMesh",
               "CcuSchedReduceScatterParallelMeshNHR",
               "CcuSchedReduceScatterParallelMeshNHRMultiLink",
               "CcuSchedReduceScatterConcurMeshNHRMultiLink",
               "CcuMSReduceScatterConcurMeshNHRMultiLink",
               "CcuSchedReduceScatterSoleNHRMultiLink",
               "DpuReduceScatterPipeLineMeshNHRMesh",
               "AicpuReduceScatterPipeLinePcie",
               "AicpuReduceScatterPipeLineUBX",
               "DpuReduceScatterPipeLineUBX",
               "AicpuReduceScatterPipeLine",
               "CcuSchedReduceScatterPipeLineMeshNHR",
               "CcuMSReduceScatterPipeLineMeshNHR"};
        // 仅在 UBX(MESH_1D_CLOS) 拓扑下才可选的算法
        static const std::set<std::string> ubxOnlyAlgos
            = {"InsReduceScatterParallelMesh1DNHRUBX", "InsReduceScatterParallelNHRNHRUboe",
               "InsReduceScatterParallelMesh1DNHRPcie"};
        // CCU mesh 类算法（受 frameNum 和数据量硬约束），不含 SoleNHR/SoleNHRMultiLink
        static const std::set<std::string> ccuMeshAlgos
            = {"CcuSchedReduceScatterSequenceMeshMesh",
               "CcuSchedReduceScatterParallelMeshNHR",
               "CcuSchedReduceScatterParallelMeshNHRMultiLink",
               "CcuSchedReduceScatterConcurMeshNHRMultiLink",
               "CcuSchedReduceScatterPipeLineMeshNHR",
               "CcuSchedReduceScatterSoleMesh",
               "CcuSchedReduceScatterSoleMesh2Die",
               "CcuMSReduceScatterSoleMesh",
               "CcuMSReduceScatterSoleMesh2Die",
               "CcuMSReduceScatterSoleMeshConcur",
               "CcuMSReduceScatterConcurMeshNHRMultiLink",
               "CcuMSReduceScatterPipeLineMeshNHR"};

        return {
            // 必选：保序模式 + rankSize > 32 → ReduceScatterOrderPreservedGroup
            {"order_preserved_group",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails* topo) {
                 return IsNeedStrictModeForOrderPreserved(op, topo->userRankSize)
                        && topo->userRankSize > MAX_RANK_NUM_FOR_ORDER_PRESERVED;
             },
             true,
             {"ReduceScatterOrderPreservedGroup"}},
            // 必选：保序模式 + rankSize <= 32 → AicpuReduceScatterStrictOrderedMesh
            {"order_preserved",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails* topo) {
                 return IsNeedStrictModeForOrderPreserved(op, topo->userRankSize)
                        && topo->userRankSize <= MAX_RANK_NUM_FOR_ORDER_PRESERVED;
             },
             true,
             {"AicpuReduceScatterStrictOrderedMesh"}},
            // 必选：每机出1卡(localNetInsSizeOfLayer[0]==1) → 选 SoleNHR / AIV SoleMesh
            {"one_card_per_server_nhr",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->topoLevelNums > 1 && !topo->netLayerDetails.localNetInsSizeOfLayer.empty()
                        && topo->netLayerDetails.localNetInsSizeOfLayer[0] == 1 && !topo->hostDpuOnly
                        && !IsNeedStrictModeForOrderPreserved(op, topo->userRankSize);
             },
             true,
             {"CcuSchedReduceScatterSoleNHR", "AicpuReduceScatterSoleNHR", "AivReduceScatterSoleMesh"}},
            // 必选：Level1Nhr/Level0Nhr(GCD==1) → 选 SoleNHR
            // 旧路径 reduce_scatter_auto_selector.cc:222(ccu_sched) / :431(aicpu Level1Nhr) / :434(aicpu Level0Nhr)
            {"nhr_by_gcd",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails* topo) {
                 return (topo->Level1Nhr || topo->Level0Nhr) && !topo->hostDpuOnly
                        && !IsNeedStrictModeForOrderPreserved(op, topo->userRankSize);
             },
             true,
             {"CcuSchedReduceScatterSoleNHR", "AicpuReduceScatterSoleNHR", "AivReduceScatterSoleMesh"}},
            // 必选：3级拓扑 → AicpuReduceScatterSequenceMeshConcurNHRNHR
            // 旧路径优先级：Level1Nhr/Level0Nhr/localNetIns[0]==1 先于3级MESH_1D分支
            // 因此排除 GCD==1 和每机1卡场景，避免与 nhr_by_gcd/one_card_per_server_nhr 同时命中
            {"three_level_only",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails* topo) {
                 bool oneCardPerServer = !topo->netLayerDetails.localNetInsSizeOfLayer.empty()
                                         && topo->netLayerDetails.localNetInsSizeOfLayer[0] == 1;
                 return topo->topoLevelNums == TOPO_LEVEL_NUM_3 && !topo->hostDpuOnly
                        && !IsNeedStrictModeForOrderPreserved(op, topo->userRankSize) && !topo->Level1Nhr
                        && !topo->Level0Nhr && !oneCardPerServer;
             },
             true,
             {"AicpuReduceScatterSequenceMeshConcurNHRNHR"}},
            // 必不选：inplace(input/output overlap) 排除 ccu
            {"inplace_skip_ccu",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails*) {
                 return IsAgInputOutputOverlap(op);
             },
             false, ccuAll},
            // 必不选：int8 排除 ccu（ccu_ms 和 ccu_sched 均不支持 INT8 的 ms reduce）
            // 旧路径 reduce_scatter_auto_selector.cc:70(ccu_ms) / :228(多级ccu_sched) / :332(单级ccu_sched)
            {"int8_skip_ccu",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails*) {
                 return op.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8;
             },
             false, ccuAll},
            // 必不选：PROD 排除 ccu + aiv
            {"prod_skip_ccu_aiv",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails*) {
                 return op.reduceType == HcclReduceOp::HCCL_REDUCE_PROD;
             },
             false, ccuAivAll},
            // 必不选：64bit(INT64/UINT64/FP64) 排除 ccu + aiv
            {"64bit_skip_ccu_aiv",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails*) {
                 return Is64BitDataType(op.DataDes.dataType);
             },
             false, ccuAivAll},
            // 必不选：保序模式排除 ccu + aiv
            {"order_preserved_skip_ccu_aiv",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails* topo) {
                 return IsNeedStrictModeForOrderPreserved(op, topo->userRankSize);
             },
             false, ccuAivAll},
            // 必不选：level2UbRtp 排除 aiv + ccu_sched
            {"level2ub_rtp_skip_aiv_ccu_sched",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->level2UbRtp;
             },
             false, aivCcuSched},
            // 必不选：3级拓扑 + level2Uboe 排除 aiv + ccu_sched
            {"level2uboe_skip_aiv_ccu_sched",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->topoLevelNums == TOPO_LEVEL_NUM_3 && topo->level2Uboe;
             },
             false, aivCcuSched},
            // 必不选：ccu_ms 仅支持单级拓扑，多级拓扑排除
            {"ccu_ms_single_level_only",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->topoLevelNums > 1;
             },
             false, ccuMsAlgos},
            // 必不选：2Die 算法仅在 TWO_DIE_REGULAR 拓扑下可选
            {"two_die_regular_only",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->level0MeshType != Level0MeshType::TWO_DIE_REGULAR;
             },
             false, twoDieAlgos},
            // 必不选：parallel/sequence/concurrent/omnipipe 算法仅在多级拓扑(topoLevelNums>1)下可选
            {"multilevel_only",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->topoLevelNums == 1;
             },
             false, multilevelAlgos},
            // 必不选：UBX/Pcie/Uboe 专用算法仅在 MESH_1D_CLOS(UBX) 拓扑下可选
            {"ubx_only",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->level0Topo != Level0Shape::MESH_1D_CLOS;
             },
             false, ubxOnlyAlgos},
            // ── 资源硬约束（旧路径 NOT_MATCH 降级，新路径 must-not-select 排除）──
            // 必不选：frameNum > 16 时排除 CCU mesh 类算法（kernel repeatNum 上限）
            // 旧路径 reduce_scatter_auto_selector.cc:259-264 回退到 CcuSchedReduceScatterSoleNHR
            {"ccu_frame_num_limit",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return AutoSelectorBase::CalcFrameNum(topo) > 16;
             },
             false, ccuMeshAlgos},
            // 必不选：AIV 数据量超限排除 AIV 算法
            // 旧路径 reduce_scatter_auto_selector.cc:594-605 (totalSize >= 8MB*rankSize 或 > cclBuffer*16)
            {"aiv_data_size_limit",
             [](const OpParam& op, const TopoInfoWithNetLayerDetails* topo) {
                 u64 perDataSize = DATATYPE_SIZE_TABLE[op.DataDes.dataType];
                 u64 totalSize = op.DataDes.count * perDataSize * topo->userRankSize;
                 if (op.opExecuteConfig != OpExecuteConfig::AIV_ONLY
                     && totalSize >= 8 * 1024 * 1024 * topo->userRankSize) {
                     return true;
                 }
                 void* cclBufferAddr = nullptr;
                 uint64_t cclBufferSize = 0;
                 if (HcclGetHcclBuffer(op.hcclComm, &cclBufferAddr, &cclBufferSize) == HCCL_SUCCESS) {
                     return totalSize > cclBufferSize * 16;
                 }
                 return false;
             },
             false, aivAlgos},
            // 必不选：AIV rankSize 超限排除 AIV 算法（kernel 硬编码 MAX_RANK_SIZE=2048）
            // 旧路径 reduce_scatter_auto_selector.cc:579-583
            {"aiv_rank_size_limit",
             [](const OpParam&, const TopoInfoWithNetLayerDetails* topo) {
                 return topo->userRankSize > 2048;
             },
             false, aivAlgos},
        };
    }

} // namespace

// ---------------------------------------------------------------------------
// CostTableManager 方法实现
// ---------------------------------------------------------------------------

HcclResult CostTableManager::FilterCMByConfig(
    CostModel& cm, CostTable& ct, const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam)
{
    HCCL_DEBUG(
        "[FilterCMByConfig] filter cost model by config, opType=%d, algCount=%d.", static_cast<int>(opParam.opType),
        cm.count);
    switch (opParam.opType) {
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
            return FilterAllReduce(cm, ct, topoInfo, opParam);
        case HcclCMDType::HCCL_CMD_ALLGATHER:
            return FilterAllGather(cm, ct, topoInfo, opParam);
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
            return FilterReduceScatter(cm, ct, topoInfo, opParam);
        default:
            HCCL_WARNING(
                "[FilterCMByConfig] opType=%d not supported yet, keep all algorithms.",
                static_cast<int>(opParam.opType));
            return HcclResult::HCCL_SUCCESS;
    }
}

void CostTableManager::DumpCostTable(const CostTable& ct)
{
    HCCL_INFO("====== [DFX_CostTableDump] algoCount=%d ======", ct.count);
    for (int i = 0; i < ct.count; ++i) {
        const char* name = (ct.costs[i].algName != nullptr) ? ct.costs[i].algName : "";
        HCCL_INFO("  [DFX_CostTableDump] [%d/%d] algName=%s, cost=%.6f", i + 1, ct.count, name, ct.costs[i].cost);
    }
    HCCL_INFO("====== [DFX_CostTableDump] dump end ======");
}

HcclResult CostTableManager::FilterByRules(
    CostModel& cm, CostTable& ct, const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::vector<AlgFilterRule>& rules, const std::string& tag)
{
    HCCL_INFO("[%s] filter, algCount=%d.", tag.c_str(), cm.count);
    ct.costs = nullptr;
    ct.count = 0;
    if (cm.count <= 0) {
        return HcclResult::HCCL_SUCCESS;
    }
    ct.costs = new (std::nothrow) AlgoCost[cm.count]();
    if (ct.costs == nullptr) {
        HCCL_ERROR("[%s] alloc AlgoCost failed, count=%d.", tag.c_str(), cm.count);
        return HcclResult::HCCL_E_PARA;
    }

    u64 dataSize = opParam.DataDes.count * DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];

    const char* opTypePascal = HcclOpTypeToPascal(opParam.opType);

    std::set<std::string> mustSelect;
    for (const auto& rule : rules) {
        if (rule.isMustSelect && rule.condition(opParam, topoInfo)) {
            mustSelect.insert(rule.algos.begin(), rule.algos.end());
        }
    }
    if (!mustSelect.empty()) {
        for (int i = 0; i < cm.count; ++i) {
            if (cm.costAlgoParams[i].count <= 0) {
                continue;
            }
            const char* algName = cm.costAlgoParams[i].algName;
            std::string name = (algName != nullptr) ? algName : "";
            if (mustSelect.count(name) == 0) {
                continue;
            }
            float cost = CalcAlgCost(name, dataSize, cm.costAlgoParams[i], opParam.opType);
            ct.costs[ct.count].algName = algName;
            ct.costs[ct.count].cost = cost;
            ++ct.count;
        }
        HCCL_INFO("[%s] mustSelect matched, count=%d.", tag.c_str(), ct.count);
        DumpCostTable(ct);
        return HcclResult::HCCL_SUCCESS;
    }

    std::set<std::string> mustNotSelect;
    std::map<std::string, std::string> filteredByRule;
    for (const auto& rule : rules) {
        if (!rule.isMustSelect && rule.condition(opParam, topoInfo)) {
            for (const auto& algo : rule.algos) {
                mustNotSelect.insert(algo);
                filteredByRule[algo] = rule.name;
            }
        }
    }
    bool isAivOnly = (opParam.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY);
    for (int i = 0; i < cm.count; ++i) {
        if (cm.costAlgoParams[i].count <= 0) {
            continue;
        }
        const char* algName = cm.costAlgoParams[i].algName;
        std::string name = (algName != nullptr) ? algName : "";
        if (opTypePascal != nullptr && name.find(opTypePascal) == std::string::npos) {
            HCCL_INFO(
                "[%s] algName=%s filtered out by op-type mismatch (expected %s).", tag.c_str(), name.c_str(),
                opTypePascal);
            continue;
        }
        if (mustNotSelect.count(name) > 0) {
            std::string ruleName = filteredByRule.count(name) > 0 ? filteredByRule[name] : "unknown";
            if (isAivOnly && SelectorEngine::GetEngineByAlgName(name) == OpExecuteConfig::AIV) {
                HCCL_ERROR(
                    "[%s] AIV_ONLY: algName=%s filtered out by rule[%s].", tag.c_str(), name.c_str(), ruleName.c_str());
            } else {
                HCCL_DEBUG("[%s] algName=%s filtered out by rule[%s].", tag.c_str(), name.c_str(), ruleName.c_str());
            }
            continue;
        }
        float cost = CalcAlgCost(name, dataSize, cm.costAlgoParams[i], opParam.opType);
        ct.costs[ct.count].algName = algName;
        ct.costs[ct.count].cost = cost;
        ++ct.count;
        HCCL_INFO("[%s] algName=%s cost=%f.", tag.c_str(), name.c_str(), cost);
    }
    DumpCostTable(ct);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CostTableManager::FilterAllReduce(
    CostModel& cm, CostTable& ct, const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam)
{
    return FilterByRules(cm, ct, topoInfo, opParam, BuildAllReduceRules(), "FilterAllReduce");
}

HcclResult CostTableManager::FilterAllGather(
    CostModel& cm, CostTable& ct, const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam)
{
    return FilterByRules(cm, ct, topoInfo, opParam, BuildAllGatherRules(), "FilterAllGather");
}

HcclResult CostTableManager::FilterReduceScatter(
    CostModel& cm, CostTable& ct, const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam)
{
    return FilterByRules(cm, ct, topoInfo, opParam, BuildReduceScatterRules(), "FilterReduceScatter");
}

float CostTableManager::CalcAlgCost(
    const std::string& algName, u64 dataSize, const CostAlgoParams& algoParams, HcclCMDType opType) const
{
    AlgNetMeta meta;
    AlgNetMetaRegistry::Global()->Query(algName, meta);

    OpExecuteConfig engine = SelectorEngine::GetEngineByAlgName(algName);

    const CostModelParam* params = algoParams.param;
    std::vector<u32> groups = meta.groupSizes;
    if (groups.empty()) {
        groups.assign(static_cast<size_t>(algoParams.count), 1);
    }

    std::vector<float> utils(static_cast<size_t>(algoParams.count), 1.0f);
    float cost = 0.0f;
    u32 idx = 0;
    for (u32 g = 0; g < groups.size() && idx < static_cast<u32>(algoParams.count); ++g) {
        float groupCost = 0.0f;
        for (u32 k = 0; k < groups[g] && idx < static_cast<u32>(algoParams.count); ++k, ++idx) {
            AlgNetType nt = (idx < meta.netTypes.size()) ? meta.netTypes[idx] : AlgNetType::MESH;
            float util = 1.0f;
            if (QueryUbUtil(nt, dataSize, engine, util, opType) != HcclResult::HCCL_SUCCESS) {
                util = 1.0f;
            }
            utils[idx] = util;
            float abCost = (params[idx].A / util + params[idx].B) * static_cast<float>(dataSize);
            float segCost = abCost + params[idx].C;
            groupCost = (meta.intraGroupMode == CostAggMode::MAX) ? std::max(groupCost, segCost) : groupCost + segCost;
        }
        cost += groupCost;
    }
    if (engine == OpExecuteConfig::AICPU_TS && opType != HcclCMDType::HCCL_CMD_ALLGATHER) {
        cost = std::max(cost, 0.0005f);
    }
    std::string paramInfo;
    char buf[64];
    for (int j = 0; j < algoParams.count; ++j) {
        if (snprintf_s(buf, sizeof(buf), sizeof(buf) - 1, "%e", params[j].A) < 0) {
            buf[0] = '\0';
        }
        std::string aStr = buf;
        if (snprintf_s(buf, sizeof(buf), sizeof(buf) - 1, "%e", params[j].B) < 0) {
            buf[0] = '\0';
        }
        std::string bStr = buf;
        if (snprintf_s(buf, sizeof(buf), sizeof(buf) - 1, "%e", params[j].C) < 0) {
            buf[0] = '\0';
        }
        std::string cStr = buf;
        paramInfo += " [A=" + aStr + " B=" + bStr + " C=" + cStr + " util=" + std::to_string(utils[j]) + "]";
    }
    HCCL_INFO(
        "[CalcAlgCost] algName=%s segCount=%d dataSize=%llu cost=%f params:%s.", algName.c_str(), algoParams.count,
        dataSize, cost, paramInfo.c_str());
    return cost;
}

HcclResult CostTableManager::CostTableGen(
    CostModel& cm, CostTable& ct, const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam)
{
    HCCL_INFO("[CostTableGen] generate cost table, algCount=%d.", cm.count);
    HcclResult ret = FilterCMByConfig(cm, ct, topoInfo, opParam);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CostTableGen] FilterCMByConfig failed, ret=%d.", static_cast<int>(ret));
    }
    return ret;
}

const std::vector<UbUtilEntry> CostTableManager::closUbUtilTable_
    = {{0.125 * 1024 * 1024ULL, 0.02755f}, {0.25 * 1024 * 1024ULL, 0.05357f}, {0.5 * 1024 * 1024ULL, 0.10388f},
       {1 * 1024 * 1024ULL, 0.1855f},      {2 * 1024 * 1024ULL, 0.3f},        {4 * 1024 * 1024ULL, 0.4288f},
       {8 * 1024 * 1024ULL, 0.5302f},      {16 * 1024 * 1024ULL, 0.568f},     {32 * 1024 * 1024ULL, 0.6549f},
       {64 * 1024 * 1024ULL, 0.7184f},     {128 * 1024 * 1024ULL, 0.7408f},   {256 * 1024 * 1024ULL, 0.7644f}};

const std::vector<UbUtilEntry> CostTableManager::meshUbUtilTable_
    = {{1 * 1024 * 1024ULL, 0.7135f},  {2 * 1024 * 1024ULL, 0.7758f},   {4 * 1024 * 1024ULL, 0.8112f},
       {8 * 1024 * 1024ULL, 0.8301f},  {16 * 1024 * 1024ULL, 0.84f},    {32 * 1024 * 1024ULL, 0.8449f},
       {64 * 1024 * 1024ULL, 0.8475f}, {128 * 1024 * 1024ULL, 0.8487f}, {256 * 1024 * 1024ULL, 0.8494f}};

CostTableManager::~CostTableManager() {}

HcclResult CostTableManager::QueryUbUtil(
    AlgNetType netType, u64 dataSize, OpExecuteConfig engine, float& utilization, HcclCMDType opType) const
{
    const std::vector<UbUtilEntry>& table = (netType == AlgNetType::CLOS) ? closUbUtilTable_ : meshUbUtilTable_;
    if (table.empty()) {
        HCCL_WARNING(
            "[CostTableManager] ub util table empty, netType=%d dataSize=%llu.", static_cast<int>(netType), dataSize);
        return HcclResult::HCCL_E_PARA;
    }
    // AllGather: CLOS 小数据量(< 1MB)统一用 1MB 的 util
    if (opType == HcclCMDType::HCCL_CMD_ALLGATHER && netType == AlgNetType::CLOS && dataSize < 1 * 1024 * 1024ULL) {
        dataSize = 1 * 1024 * 1024ULL;
    }
    auto it = std::lower_bound(table.begin(), table.end(), dataSize, [](const UbUtilEntry& e, u64 ds) {
        return e.upperBound < ds;
    });
    if (it == table.end()) {
        utilization = table.back().utilization;
    } else {
        utilization = it->utilization;
    }
    if (engine == OpExecuteConfig::AIV) {
        utilization = utilization / 0.85f * 0.65f;
    }
    HCCL_DEBUG(
        "[CostTableManager] QueryUbUtil netType=%d dataSize=%llu engine=%d utilization=%f.", static_cast<int>(netType),
        dataSize, static_cast<int>(engine), utilization);
    return HcclResult::HCCL_SUCCESS;
}

CostTableManager* CostTableManager::Global()
{
    static CostTableManager* globalCostTableManager = new CostTableManager;
    return globalCostTableManager;
}

} // namespace ops_hccl
