/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_testcase_common.h"

class ST_REDUCE_TEST_2 :
    public ::testing::TestWithParam<std::tuple<TopoMeta, u64, HcclDataType, HcclReduceOp, uint32_t>> {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }

    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_BUFFSIZE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }

    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_P(ST_REDUCE_TEST_2, st_reduce_aicpu_test)
{
    auto params = GetParam();
    const auto& topoMeta = std::get<0>(params);
    u64 recvCount = std::get<1>(params);
    HcclDataType dataType = std::get<2>(params);
    HcclReduceOp reduceOp = std::get<3>(params);
    uint32_t root = std::get<4>(params);

    RunReduceTest(topoMeta, recvCount, dataType, reduceOp, root);
}

// 参数化实例化
INSTANTIATE_TEST_SUITE_P(
    ReduceVariants2, ST_REDUCE_TEST_2,
    ::testing::Values(
        // NHR
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 2), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 3), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 4), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 2), 1 << 20, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 2), (1 << 30) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 5), 400 * (1 << 20) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 2),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 6), 111 * (1 << 20) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 5),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 7), 50 * (1 << 20) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 4),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 8), 17 * (1 << 20) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 4),
        // 1DMeshNHR
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 2, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 3, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1 << 30, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1 << 30, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 1),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1 << 30, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 2),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1 << 30, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 3),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 1),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 4097, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 2),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), (1 << 30) - 1, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1 << 30, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_MIN, 1),
        std::make_tuple(GenerateMeshTopoMeta(3, 1, 2), 100, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, 3),
        std::make_tuple(GenerateMeshTopoMeta(3, 1, 3), 100, HCCL_DATA_TYPE_BFP16, HCCL_REDUCE_SUM, 8),
        std::make_tuple(GenerateMeshTopoMeta(4, 1, 2), 210 * (1 << 20), HCCL_DATA_TYPE_INT32, HCCL_REDUCE_MIN, 6),
        std::make_tuple(GenerateMeshTopoMeta(4, 1, 4), 4095, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 11),
        std::make_tuple(GenerateMeshTopoMeta(8, 1, 2), 100, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_MAX, 5),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 24 * (1 << 20) - 1, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 96 * (1 << 20) - 1, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 16 * (1 << 20) - 1, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0)
        // 2DMeshNHR
        ));
