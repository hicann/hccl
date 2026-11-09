/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "selector_test_common.h"
#include "selector_engine.h"
#include "alg_attrs_registry.h"
#include "cost_table.h"
#include "alg_parse.h"

namespace inline_reduce_ut {
namespace {
    using SelectorEngineTest = SelectorTest;

    void ExpectSelection(OpParam param, TopoInfoWithNetLayerDetails topo, bool useInline)
    {
        std::string name;
        ASSERT_EQ(SelectorEngine::Global()->Run(param.hcclComm, param, &topo, name), HCCL_SUCCESS);
        EXPECT_EQ(name, AlgorithmName(param.opType, useInline));
        EXPECT_EQ(param.opExecuteConfig, OpExecuteConfig::HOSTCPU);
    }

    TEST_F(SelectorEngineTest, SelectsRegisteredPerOpInlineMatrix)
    {
        for (auto op : OPS) {
            for (auto deviceType : {HcclDevType::DEV_TYPE_950, HcclDevType::DEV_TYPE_960}) {
                for (auto type : DATA_TYPES) {
                    for (auto reduce : REDUCE_OPS) {
                        if (reduce != HCCL_REDUCE_SUM && type != HCCL_DATA_TYPE_FP32)
                            continue;
                        // Device generation is independent of the type/reduction matrix; cover 960 with FP32 SUM.
                        if (deviceType == HcclDevType::DEV_TYPE_960
                            && (type != HCCL_DATA_TYPE_FP32 || reduce != HCCL_REDUCE_SUM))
                            continue;
                        ResetSelectorState();
                        SCOPED_TRACE(
                            testing::Message() << static_cast<int>(op) << ":" << static_cast<int>(deviceType) << ":"
                                               << type << ":" << reduce);
                        auto param = MakeParam(op);
                        param.deviceType = deviceType;
                        param.DataDes.dataType = type;
                        param.reduceType = reduce;
                        auto topo = MakeTopo();
                        topo.deviceType = deviceType;
                        ExpectSelection(param, topo, true);
                    }
                }
            }
        }
    }

    TEST_F(SelectorEngineTest, FiltersCapabilitiesBeforeCachingAndTypesPerOp)
    {
        for (auto op : OPS) {
            for (u32 factor = 0; factor < 6; ++factor) {
                ResetSelectorState();
                SCOPED_TRACE(testing::Message() << static_cast<int>(op) << ":" << factor);
                auto param = MakeParam(op);
                auto topo = MakeTopo();
                if (factor == 0)
                    SetDomainCapability(topo, false);
                SetPrimitiveAvailable(factor != 1);
                if (factor >= 2 && factor <= 4)
                    param.DataDes.dataType = std::array<HcclDataType, 3>{
                        HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64}[factor - 2];
                if (factor == 5)
                    param.reduceType = HCCL_REDUCE_PROD;
                ExpectSelection(param, topo, false);
            }
        }
    }

    TEST_F(SelectorEngineTest, AppliesStrictConditionIdenticallyToLegacy)
    {
        for (auto op : OPS) {
            for (const auto& c : DETERMINISTIC_CASES) {
                ResetSelectorState();
                SCOPED_TRACE(testing::Message() << static_cast<int>(op) << ":" << c.name);
                SetDeterministicLevel(c.level);
                auto param = MakeParam(op);
                param.DataDes.dataType = c.type;
                param.reduceType = c.reduce;
                auto topo = MakeTopo();
                topo.userRankSize = c.ranks;
                const auto* attrs = AlgAttrsRegistry::Instance().Get(AlgorithmName(op, true));
                ASSERT_NE(attrs, nullptr);
                EXPECT_EQ(CheckAlgoMatchOpWithReason(*attrs, param, &topo).matched, c.supported);
                if (c.supported)
                    ExpectSelection(param, topo, true);
                else {
                    std::string name;
                    EXPECT_EQ(SelectorEngine::Global()->Run(nullptr, param, &topo, name), HCCL_E_NOT_SUPPORT);
                }
            }
        }
    }

    TEST_F(SelectorEngineTest, InitialEngineAllowsHostDpuFallback)
    {
        for (auto op : OPS) {
            for (auto engine :
                 {OpExecuteConfig::HOSTCPU, OpExecuteConfig::AICPU_TS, OpExecuteConfig::CCU_MS, OpExecuteConfig::AIV}) {
                ResetSelectorState();
                auto param = MakeParam(op);
                param.opExecuteConfig = engine;
                ExpectSelection(param, MakeTopo(), true);
            }
        }
    }

    TEST_F(SelectorEngineTest, PreferredAndTunerCannotBypassCapabilityHardConstraint)
    {
        for (auto op : OPS) {
            const auto preferredName = AlgorithmName(op, true);
            SetPreferredAlgorithm(preferredName);
            CostAlgoParams entry{};
            entry.algName = preferredName.c_str();
            entry.count = 1;
            CostModel preferred{&entry, 1};
            ASSERT_EQ(MarkHcclAlgoPriority(nullptr, preferred, {"Dpu"}), HCCL_SUCCESS);
            ASSERT_EQ(entry.hcclAlgoPriority, 1);

            for (bool tuner : {false, true}) {
                for (bool symbolMissing : {false, true}) {
                    ResetSelectorState();
                    SetPreferredAlgorithm(AlgorithmName(op, true));
                    SetTunerLoaded(tuner);
                    SetPrimitiveAvailable(!symbolMissing);
                    auto topo = MakeTopo();
                    SetDomainCapability(topo, symbolMissing);
                    const auto* attrs = AlgAttrsRegistry::Instance().Get(AlgorithmName(op, true));
                    ASSERT_NE(attrs, nullptr);
                    EXPECT_TRUE(attrs->topo.topoPrecondCheck);
                    EXPECT_FALSE(CheckAlgoMatchTopoWithReason(attrs->name, &topo, false).matched);
                    ExpectSelection(MakeParam(op), topo, false);
                }
            }
        }
    }

    TEST_F(SelectorEngineTest, KeepsOpDecisionsDynamicWithinCachedModel)
    {
        for (auto op : OPS) {
            ResetSelectorState();
            for (u32 stage = 0; stage < 5; ++stage) {
                auto param = MakeParam(op);
                if (stage == 1)
                    param.reduceType = HCCL_REDUCE_PROD;
                if (stage == 2)
                    param.DataDes.dataType = HCCL_DATA_TYPE_FP64;
                if (stage == 3)
                    param.DataDes.dataType = HCCL_DATA_TYPE_INT16;
                ExpectSelection(param, MakeTopo(), stage != 1 && stage != 2);
            }
            EXPECT_EQ(CostModelCreationCount(nullptr), 1);
            auto first = MakeParam(op);
            first.hcclComm = reinterpret_cast<HcclComm>(0x1010);
            auto second = MakeParam(op);
            second.hcclComm = reinterpret_cast<HcclComm>(0x2020);
            auto incapable = MakeTopo();
            SetDomainCapability(incapable, false);
            ExpectSelection(first, MakeTopo(), true);
            ExpectSelection(second, incapable, false);
            ExpectSelection(first, MakeTopo(), true);
            EXPECT_EQ(CostModelCreationCount(first.hcclComm), 1);
            EXPECT_EQ(CostModelCreationCount(second.hcclComm), 1);
        }
    }

    TEST_F(SelectorEngineTest, RejectsNonHostDpuDeviceAndInapplicableTopology)
    {
        for (auto op : OPS) {
            for (u32 factor = 0; factor < 5; ++factor) {
                ResetSelectorState();
                auto topo = MakeTopo();
                if (factor == 0)
                    topo.deviceType = HcclDevType::DEV_TYPE_910B;
                if (factor == 1)
                    topo.hostDpuOnly = false;
                if (factor == 2)
                    topo.topoLevelNums = 1;
                if (factor == 3)
                    topo.topoLevelNums = 4;
                if (factor == 4)
                    topo.level0Topo = Level0Shape::MESH_1D_CLOS;
                EXPECT_FALSE(CheckAlgoMatchTopoWithReason(AlgorithmName(op, true), &topo).matched);
                if (factor == 0)
                    ExpectSelection(MakeParam(op), topo, false);
            }
        }
    }

    TEST_F(SelectorEngineTest, KeepsExactlyOneSequenceCandidateForAutomaticSelection)
    {
        constexpr u32 NON_TYPE_CASES = 4;
        constexpr std::array<HcclDataType, 3> UNSUPPORTED_TYPES
            = {HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64};
        for (auto op : OPS) {
            for (u32 factor = 0; factor < NON_TYPE_CASES + UNSUPPORTED_TYPES.size(); ++factor) {
                SCOPED_TRACE(testing::Message() << "op=" << static_cast<int>(op) << " factor=" << factor);
                ResetSelectorState();
                SetPrimitiveAvailable(factor != 2);
                auto topo = MakeTopo();
                SetDomainCapability(topo, factor != 1);
                auto param = MakeParam(op);
                if (factor == 3)
                    param.reduceType = HCCL_REDUCE_PROD;
                if (factor >= NON_TYPE_CASES) {
                    param.DataDes.dataType = UNSUPPORTED_TYPES[factor - NON_TYPE_CASES];
                    const auto* attrs = AlgAttrsRegistry::Instance().Get(AlgorithmName(op, true));
                    ASSERT_NE(attrs, nullptr);
                    for (bool needSoftCheck : {false, true}) {
                        const auto result = CheckAlgoMatchOpWithReason(*attrs, param, &topo, needSoftCheck);
                        EXPECT_FALSE(result.matched);
                        EXPECT_EQ(result.reason, "unsupportedDataTypes");
                    }
                }
                struct ScopedCosts {
                    CostModel model{nullptr, 0};
                    CostTable table{nullptr, 0};
                    ~ScopedCosts()
                    {
                        CostTableManager::FreeCostTable(table);
                        CostModelManager::FreeCostModel(model);
                    }
                } costs;
                auto& cm = costs.model;
                ASSERT_EQ(
                    CostModelManager::GenerateCostModel(nullptr, cm, param, &topo, {OpExecuteConfig::HOSTCPU}, {"Dpu"}),
                    HCCL_SUCCESS);
                auto& ct = costs.table;
                ASSERT_EQ(CostTableManager::Global()->GenerateCostTable(cm, ct, &topo, param), HCCL_SUCCESS);
                ASSERT_EQ(ct.count, 1);
                EXPECT_EQ(ct.costs[0].algName, AlgorithmName(op, factor == 0));
            }
        }
    }

    TEST_F(SelectorEngineTest, RegistersSharedNhrTemplateAndSeparateFallbacks)
    {
        const AllAlgos& all = *GetAllAlgos();
        for (auto op : OPS) {
            for (bool useInline : {false, true}) {
                const auto name = AlgorithmName(op, useInline);
                const auto* attrs = AlgAttrsRegistry::Instance().Get(name);
                ASSERT_NE(attrs, nullptr);
                EXPECT_EQ(attrs->opType, op);
                EXPECT_EQ(attrs->engine, OpExecuteConfig::HOSTCPU);
                const AlgElement* found = nullptr;
                for (int i = 0; i < all.count; ++i)
                    if (name == all.algElements[i].algName)
                        found = &all.algElements[i];
                ASSERT_NE(found, nullptr);
                ASSERT_GE(found->templateNum, 2);
                EXPECT_STREQ(
                    found->templateName[1], useInline ? "InsTempReduceScatterNhrDpu" :
                                            op == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ?
                                                        "InsTempReduceScatterMesh1dDpu" :
                                                        "InsTempReduceScatterMesh1dDpuInter");
            }
        }
    }

} // namespace
} // namespace inline_reduce_ut
