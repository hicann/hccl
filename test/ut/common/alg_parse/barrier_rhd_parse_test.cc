/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "alg_parse.h"
#include "barrier_rhd_config.h"

using namespace ops_hccl;

TEST(BarrierRhdParser, ParsesTwoAndThreeLevelSequences)
{
    for (const std::string& config :
         {"barrier:sequence{level0=mesh,level1=rhd}", "barrier:sequence{level0=mesh,level1=rhd,level2=rhd}"}) {
        HcclAlgoParser parser;
        ASSERT_EQ(parser.Parser(config), HCCL_SUCCESS);
        ASSERT_EQ(parser.executorList.size(), 1U);
        const auto& entry = parser.executorList[0];
        EXPECT_EQ(entry.opType, "barrier");
        EXPECT_EQ(entry.executorType, "sequence");
        ASSERT_GE(entry.algoList.size(), 2U);
        EXPECT_EQ(entry.algoList[0].algoType, "mesh");
        for (size_t i = 1; i < entry.algoList.size(); ++i) {
            EXPECT_EQ(entry.algoList[i].algoType, "rhd");
        }
    }
}

TEST(BarrierRhdParser, MapsRhdWithoutChangingSoleShorthand)
{
    EXPECT_EQ(GetAlgoNameToTypeMap().at("RHD"), AlgoType::RHD);
    EXPECT_EQ(GetAlgoTypeToNameMap().at(AlgoType::RHD), "RHD");
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("barrier:rhd"), HCCL_SUCCESS);
    ASSERT_EQ(parser.executorList.size(), 1U);
    EXPECT_EQ(parser.executorList[0].executorType, "sole");
}

TEST(BarrierRhdParser, MatchesExactRegisteredNamesForEachEngineAndLevelCount)
{
    const char* names[]
        = {"AicpuBarrierSequenceMeshRHD",  "AicpuBarrierSequenceMeshRHDRHD",     "DpuBarrierSequenceMeshRHD",
           "DpuBarrierSequenceMeshRHDRHD", "DpuBarrierSequenceMeshNHR",          "AicpuAllReduceSequenceMeshRHD",
           "AicpuBarrierSoleRHD",          "AicpuBarrierSequenceMeshRHDRHDExtra"};
    const char* configs[]
        = {"barrier:sequence{level0=mesh,level1=rhd}", "barrier:sequence{level0=mesh,level1=rhd,level2=rhd}"};
    const char* engines[] = {"Aicpu", "Dpu"};
    constexpr size_t ENTRY_COUNT = sizeof(names) / sizeof(names[0]);
    for (size_t engine = 0; engine < 2; ++engine) {
        for (size_t level = 0; level < 2; ++level) {
            SCOPED_TRACE(std::string(engines[engine]) + ":" + configs[level]);
            HcclAlgoParser parser;
            ASSERT_EQ(parser.Parser(configs[level]), HCCL_SUCCESS);
            CostAlgoParams entries[ENTRY_COUNT]{};
            for (size_t index = 0; index < ENTRY_COUNT; ++index) {
                entries[index].algName = names[index];
                entries[index].count = 1;
            }
            CostModel model{entries, static_cast<int>(ENTRY_COUNT)};
            // 通过公共匹配入口验证 ComposeAlgoName 生成的名字能命中算法注册名。
            ASSERT_EQ(UpdateCostModelWithAlgo(parser, model, {engines[engine]}), HCCL_SUCCESS);
            for (size_t index = 0; index < ENTRY_COUNT; ++index) {
                EXPECT_EQ(entries[index].hcclAlgoPriority, index == engine * 2 + level ? 1 : 0) << names[index];
                EXPECT_EQ(entries[index].count, 1);
            }
        }
    }
}

TEST(BarrierRhdParser, BareRhdDoesNotSelectSequenceRegistrations)
{
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("barrier:rhd"), HCCL_SUCCESS);
    CostAlgoParams entries[4]{};
    const char* names[]
        = {"AicpuBarrierSequenceMeshRHD", "AicpuBarrierSequenceMeshRHDRHD", "DpuBarrierSequenceMeshRHD",
           "DpuBarrierSequenceMeshRHDRHD"};
    for (size_t index = 0; index < 4; ++index) {
        entries[index].algName = names[index];
        entries[index].count = 1;
    }
    CostModel model{entries, 4};
    ASSERT_EQ(UpdateCostModelWithAlgo(parser, model, {"Aicpu", "Dpu"}), HCCL_SUCCESS);
    for (const auto& entry : entries) {
        EXPECT_EQ(entry.hcclAlgoPriority, 0);
    }
}

TEST(BarrierRhdParser, OnlyLastMatchingExplicitClauseAdmitsRhd)
{
    const std::vector<const char*> barrierNames = {"AicpuBarrierSoleNHR",         "DpuBarrierSequenceMeshNHR",
                                                   "AicpuBarrierSequenceMeshRHD", "AicpuBarrierSequenceMeshRHDRHD",
                                                   "DpuBarrierSequenceMeshRHD",   "DpuBarrierSequenceMeshRHDRHD"};
    const std::string twoLevels = "barrier:sequence{level0=mesh,level1=rhd}";
    const std::string threeLevels = "barrier:sequence{level0=mesh,level1=rhd,level2=rhd}";
    const struct {
        std::string config;
        int expectedLevel;
        int expectedLevelDpuOnly;
    } cases[] = {
        {twoLevels, 2, 2},
        {threeLevels, 3, 3},
        {"barrier:sequence{level0=mesh,level1=not(nhr)}", 0, 0},
        {"barrier:sequence{}", 0, 0},
        {twoLevels + ";" + threeLevels, 3, 3},
        {threeLevels + ";" + twoLevels, 2, 2},
        {twoLevels + ";barrier:sole{rhd}", 2, 2},
        {twoLevels + ";barrier:sequence{}", 0, 0},
        {twoLevels + ";barrier:sole{nhr}", 0, 2},
        {twoLevels + ";barrier:not(sequence{level0=mesh,level1=rhd})", 0, 0},
    };
    for (const auto& testCase : cases) {
        SCOPED_TRACE(testCase.config);
        HcclAlgoParser parser;
        ASSERT_EQ(parser.Parser(testCase.config), HCCL_SUCCESS);
        const std::vector<std::string> bothEngines = {"Aicpu", "Dpu"};
        const std::vector<std::string> dpuOnly = {"Dpu"};
        for (const auto& engine : bothEngines) {
            EXPECT_EQ(
                IsBarrierRhdExplicitlyConfigured(parser, engine + "BarrierSequenceMeshRHD", barrierNames, bothEngines),
                testCase.expectedLevel == 2);
            EXPECT_EQ(
                IsBarrierRhdExplicitlyConfigured(
                    parser, engine + "BarrierSequenceMeshRHDRHD", barrierNames, bothEngines),
                testCase.expectedLevel == 3);
        }
        EXPECT_EQ(
            IsBarrierRhdExplicitlyConfigured(parser, "DpuBarrierSequenceMeshRHD", barrierNames, dpuOnly),
            testCase.expectedLevelDpuOnly == 2);
        EXPECT_EQ(
            IsBarrierRhdExplicitlyConfigured(parser, "DpuBarrierSequenceMeshRHDRHD", barrierNames, dpuOnly),
            testCase.expectedLevelDpuOnly == 3);
    }
}
