/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hostdpu_inline_reduce_st_common.h"

using namespace hostdpu_st;

class ST_HOSTDPU_INLINE_REDUCE_TEST : public testing::Test {};

TEST_F(ST_HOSTDPU_INLINE_REDUCE_TEST, AutomaticallyExecutesSupportedSum)
{
    FlowCase test;
    RunCollective(HcclCMDType::HCCL_CMD_REDUCE, test);
}

TEST_F(ST_HOSTDPU_INLINE_REDUCE_TEST, LegacyEntryExecutesDeterministicSum)
{
    FlowCase test;
    test.legacy = true;
    test.deterministic = true;
    RunCollective(HcclCMDType::HCCL_CMD_REDUCE, test);
}

TEST_F(ST_HOSTDPU_INLINE_REDUCE_TEST, NonPowerOfTwoRanksReduceTheTailExactlyOnce)
{
    FlowCase test;
    test.topology = {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    test.count = 1025;
    test.dataType = HCCL_DATA_TYPE_INT32;
    test.reduceOp = HCCL_REDUCE_MIN;
    test.root = 8;
    RunCollective(HcclCMDType::HCCL_CMD_REDUCE, test);
}

TEST_F(ST_HOSTDPU_INLINE_REDUCE_TEST, TinyInputCompletesWithoutLosingContributions)
{
    FlowCase test;
    test.topology = {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    test.count = 1;
    test.reduceOp = HCCL_REDUCE_MAX;
    test.root = 0;
    RunCollective(HcclCMDType::HCCL_CMD_REDUCE, test);
}

TEST_F(ST_HOSTDPU_INLINE_REDUCE_TEST, MultipleBufferLoopsPreserveTheFinalShortChunk)
{
    FlowCase test;
    test.count = 131073;
    test.dataType = HCCL_DATA_TYPE_FP16;
    test.cclBufferBytes = 1024 * 1024;
    test.root = 0;
    RunCollective(HcclCMDType::HCCL_CMD_REDUCE, test);
}

TEST_F(ST_HOSTDPU_INLINE_REDUCE_TEST, AutomaticFallbacksCompleteTheCollective)
{
    const std::array<FlowCase, 3> fallbackCases = []() {
        std::array<FlowCase, 3> cases;
        cases[0].capability = false;
        cases[1].dataType = HCCL_DATA_TYPE_FP64;
        cases[2].reduceOp = HCCL_REDUCE_PROD;
        for (auto& test : cases) {
            test.expectInline = false;
        }
        return cases;
    }();
    for (const auto& test : fallbackCases) {
        SCOPED_TRACE(
            testing::Message() << "capability=" << test.capability << " dataType=" << test.dataType
                               << " reduceOp=" << test.reduceOp);
        RunCollective(HcclCMDType::HCCL_CMD_REDUCE, test);
    }
}
