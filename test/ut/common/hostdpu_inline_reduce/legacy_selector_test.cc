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

namespace inline_reduce_ut {
namespace {
    using LegacySelectorTest = SelectorTest;

    TEST_F(LegacySelectorTest, SelectsPerOpInlineMatrix)
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
                        SCOPED_TRACE(
                            testing::Message() << static_cast<int>(op) << ":" << static_cast<int>(deviceType) << ":"
                                               << type << ":" << reduce);
                        auto param = MakeParam(op);
                        param.deviceType = deviceType;
                        param.DataDes.dataType = type;
                        param.reduceType = reduce;
                        auto topo = MakeTopo(Level0Shape::CLOS);
                        topo.deviceType = deviceType;
                        std::string name;
                        ASSERT_EQ(SelectLegacy(param, topo, name), SelectorStatus::MATCH);
                        EXPECT_EQ(name, AlgorithmName(op, true));
                    }
                }
            }
        }
    }

    TEST_F(LegacySelectorTest, FallsBackForEachUnsupportedCondition)
    {
        for (auto op : OPS) {
            for (u32 factor = 0; factor < 7; ++factor) {
                SCOPED_TRACE(testing::Message() << static_cast<int>(op) << ":" << factor);
                auto param = MakeParam(op);
                auto topo = MakeTopo(Level0Shape::CLOS);
                SetPrimitiveAvailable(factor != 1);
                if (factor == 0)
                    SetDomainCapability(topo, false);
                if (factor >= 2 && factor <= 4)
                    param.DataDes.dataType = std::array<HcclDataType, 3>{
                        HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64}[factor - 2];
                if (factor == 5)
                    param.reduceType = HCCL_REDUCE_PROD;
                if (factor == 6)
                    topo.deviceType = HcclDevType::DEV_TYPE_910B;
                std::string name;
                ASSERT_EQ(SelectLegacy(param, topo, name), SelectorStatus::MATCH);
                EXPECT_EQ(name, AlgorithmName(op, false));
            }
        }
    }

    TEST_F(LegacySelectorTest, AppliesStrictSumConditionAndItsIndependentExceptions)
    {
        for (auto op : OPS) {
            for (const auto& c : DETERMINISTIC_CASES) {
                SCOPED_TRACE(testing::Message() << static_cast<int>(op) << ":" << c.name);
                auto param = MakeParam(op);
                param.DataDes.dataType = c.type;
                param.reduceType = c.reduce;
                auto topo = MakeTopo(Level0Shape::CLOS);
                topo.userRankSize = c.ranks;
                SetDeterministicLevel(c.level);
                std::string name;
                ASSERT_EQ(SelectLegacy(param, topo, name), SelectorStatus::MATCH);
                EXPECT_EQ(name, AlgorithmName(op, c.supported));
            }
        }
    }

    TEST_F(LegacySelectorTest, PreservesCapabilityAcrossSerializationAndRepeatedCalls)
    {
        SetDpuInlineReduceLogCaptureEnabled(true);
        ClearDpuInlineReduceCapturedLogs();

        for (const auto& state :
             {std::pair<bool, bool>{false, false},
              {true, false},
              {false, true},
              {true, true},
              {true, false},
              {true, true}}) {
            const bool domain = state.first;
            const bool primitive = state.second;
            SetPrimitiveAvailable(primitive);
            EXPECT_EQ(HcommIsSupportHcommWriteReduceWithNotifyOnThread(), primitive);
            auto topo = MakeTopo(Level0Shape::CLOS);
            SetDomainCapability(topo, domain);
            TopoInfoWithNetLayerDetails restored{};
            auto serialized = topo.Serialize();
            restored.DeSerialize(serialized);
            EXPECT_EQ(restored.allDpuSupportInlineReduce, domain);
            for (auto op : OPS) {
                std::string name;
                ASSERT_EQ(SelectLegacy(MakeParam(op), restored, name), SelectorStatus::MATCH);
                EXPECT_EQ(name, AlgorithmName(op, domain && primitive));
            }
        }
        const std::string warning = "write-reduce primitive symbol is unavailable.";
        const auto& logs = GetDpuInlineReduceCapturedLogs();
        const auto first = logs.find(warning);
        if (first != std::string::npos) {
            EXPECT_EQ(logs.find(warning, first + warning.size()), std::string::npos);
        }
    }

    TEST_F(LegacySelectorTest, PreservesSequenceAndPipelineTopologyBoundaries)
    {
        for (auto op : OPS) {
            for (auto shape : {Level0Shape::MESH_1D, Level0Shape::CLOS, Level0Shape::MESH_1D_CLOS}) {
                for (bool pcieMix : {false, true}) {
                    auto topo = MakeTopo(shape);
                    topo.level0PcieMix = pcieMix;
                    std::string name;
                    ASSERT_EQ(SelectLegacy(MakeParam(op), topo, name), SelectorStatus::MATCH);
                    const bool pipeline = shape == Level0Shape::MESH_1D_CLOS && !pcieMix;
                    const std::string expected
                        = pipeline ? (op == HcclCMDType::HCCL_CMD_ALLREDUCE ? "DpuAllReducePipeLineMeshNHRNHR" :
                                      op == HcclCMDType::HCCL_CMD_REDUCE    ? "DpuReducePipeLineMeshNHRNHR" :
                                                                              "DpuReduceScatterPipeLineMeshNHRMesh") :
                                     AlgorithmName(op, true);
                    EXPECT_EQ(name, expected);
                }
            }
            auto topo = MakeTopo(Level0Shape::CLOS);
            topo.topoLevelNums = 1;
            std::string name;
            EXPECT_EQ(SelectLegacy(MakeParam(op), topo, name), SelectorStatus::NOT_MATCH);
            EXPECT_TRUE(name.empty());
        }
    }

    TEST_F(LegacySelectorTest, ReportsOrderedDecisionReasons)
    {
        const std::array<DpuInlineReduceDecisionReason, 7> reasons
            = {DpuInlineReduceDecisionReason::SUPPORTED,
               DpuInlineReduceDecisionReason::CAPABILITY_NOT_SUPPORTED,
               DpuInlineReduceDecisionReason::DEVICE_TYPE_NOT_SUPPORTED,
               DpuInlineReduceDecisionReason::DATA_TYPE_NOT_SUPPORTED,
               DpuInlineReduceDecisionReason::REDUCE_OP_NOT_SUPPORTED,
               DpuInlineReduceDecisionReason::DETERMINISTIC_STRICT,
               DpuInlineReduceDecisionReason::PRIMITIVE_NOT_SUPPORTED};
        const std::array<const char*, 7> reasonTexts
            = {"SUPPORTED",
               "CAPABILITY_NOT_SUPPORTED",
               "DEVICE_TYPE_NOT_SUPPORTED",
               "DATA_TYPE_NOT_SUPPORTED",
               "REDUCE_OP_NOT_SUPPORTED",
               "DETERMINISTIC_STRICT",
               "PRIMITIVE_NOT_SUPPORTED"};
        for (u32 i = 0; i < reasons.size(); ++i) {
            EXPECT_STREQ(GetDpuInlineReduceDecisionReason(reasons[i]), reasonTexts[i]);
            auto param = MakeParam();
            auto topo = MakeTopo();
            SetPrimitiveAvailable(i != 6);
            SetDeterministicLevel(
                i == 5 ? DeterministicEnableLevel::DETERMINISTIC_STRICT :
                         DeterministicEnableLevel::DETERMINISTIC_DISABLE);
            if (i == 1)
                SetDomainCapability(topo, false);
            if (i == 2)
                topo.deviceType = HcclDevType::DEV_TYPE_910B;
            if (i == 3)
                param.DataDes.dataType = HCCL_DATA_TYPE_FP64;
            if (i == 4)
                param.reduceType = HCCL_REDUCE_PROD;
            SetDpuInlineReduceLogCaptureEnabled(true);
            ClearDpuInlineReduceCapturedLogs();
            EXPECT_EQ(CanUseDpuInlineReduce(param, &topo), i == 0);
            EXPECT_NE(
                GetDpuInlineReduceCapturedLogs().find("reason=" + std::string(reasonTexts[i]) + "."),
                std::string::npos);
        }
        SetPrimitiveAvailable(true);
        for (bool supported : {false, true}) {
            auto topo = MakeTopo();
            SetDomainCapability(topo, supported);
            ClearDpuInlineReduceCapturedLogs();
            const auto decision = DpuInlineReduceCustomCheck(&topo);
            EXPECT_EQ(decision.supported, supported);
            EXPECT_TRUE(GetDpuInlineReduceCapturedLogs().empty());
            EXPECT_EQ(CheckDpuInlineReduceTopo(&topo), supported);
            EXPECT_NE(
                GetDpuInlineReduceCapturedLogs().find(
                    supported ? "matched: reason=SUPPORTED." : "filtered: reason=CAPABILITY_NOT_SUPPORTED."),
                std::string::npos);
        }
    }
} // namespace
} // namespace inline_reduce_ut
