/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HOSTDPU_SELECTOR_TEST_COMMON_H
#define HOSTDPU_SELECTOR_TEST_COMMON_H

#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "alg_attrs.h"
#include "auto_selector_base.h"
#include "dpu_inline_reduce_selector.h"
#include "hcomm_primitives_dl.h"
#include "order_preserved_common.h"

void SetDpuInlineReduceLogCaptureEnabled(bool enabled);
void ClearDpuInlineReduceCapturedLogs();
const std::string& GetDpuInlineReduceCapturedLogs();

namespace inline_reduce_ut {
using namespace ops_hccl;
inline constexpr std::array<HcclDataType, 6> DATA_TYPES
    = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT32,
       HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32,  HCCL_DATA_TYPE_BFP16};
inline constexpr std::array<HcclReduceOp, 3> REDUCE_OPS = {HCCL_REDUCE_SUM, HCCL_REDUCE_MAX, HCCL_REDUCE_MIN};
inline constexpr std::array<HcclCMDType, 3> OPS
    = {HcclCMDType::HCCL_CMD_ALLREDUCE, HcclCMDType::HCCL_CMD_REDUCE, HcclCMDType::HCCL_CMD_REDUCE_SCATTER};

struct DeterministicCase {
    const char* name;
    HcclDataType type;
    HcclReduceOp reduce;
    u32 ranks;
    DeterministicEnableLevel level;
    bool supported;
};
inline constexpr auto STRICT = DeterministicEnableLevel::DETERMINISTIC_STRICT;
inline constexpr std::array<DeterministicCase, 12> DETERMINISTIC_CASES
    = {{{"strict_float_sum", HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, 3, STRICT, false},
        {"disabled", HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, 3, DeterministicEnableLevel::DETERMINISTIC_DISABLE, true},
        {"ordinary_deterministic", HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, 3,
         DeterministicEnableLevel::DETERMINISTIC_ENABLE, true},
        {"integer", HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 3, STRICT, true},
        {"max", HCCL_DATA_TYPE_FP32, HCCL_REDUCE_MAX, 3, STRICT, true},
        {"min", HCCL_DATA_TYPE_FP32, HCCL_REDUCE_MIN, 3, STRICT, true},
        {"threshold", HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, 2, STRICT, true},
        {"single_rank", HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, 1, STRICT, true},
        {"fp16", HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, 3, STRICT, false},
        {"bfp16", HCCL_DATA_TYPE_BFP16, HCCL_REDUCE_SUM, 3, STRICT, false},
        {"int8", HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 3, STRICT, true},
        {"int16", HCCL_DATA_TYPE_INT16, HCCL_REDUCE_SUM, 3, STRICT, true}}};

void ResetSelectorState();
void SetPrimitiveAvailable(bool available);
// Inject the result of whole-domain capability negotiation independently of dlsym.
inline void SetDomainCapability(TopoInfoWithNetLayerDetails& topo, bool supported)
{
    topo.allDpuSupportInlineReduce = supported;
}
void SetDeterministicLevel(DeterministicEnableLevel level);
void SetEnvironmentDeterministicLevel(DeterministicEnableLevel level);
u32 CostModelCreationCount(HcclComm comm);
void SetPreferredAlgorithm(const std::string& name);
void SetTunerLoaded(bool loaded);
OpParam MakeParam(HcclCMDType op = HcclCMDType::HCCL_CMD_ALLREDUCE);
TopoInfoWithNetLayerDetails MakeTopo(Level0Shape shape = Level0Shape::MESH_1D);
std::string AlgorithmName(HcclCMDType op, bool useInline);
SelectorStatus SelectLegacy(const OpParam& param, const TopoInfoWithNetLayerDetails& topo, std::string& name);

class SelectorTest : public testing::Test {
protected:
    void SetUp() override
    {
        const char* configuredAlgo = std::getenv("HCCL_ALGO");
        if (configuredAlgo != nullptr)
            originalAlgo_ = configuredAlgo;
        ASSERT_EQ(unsetenv("HCCL_ALGO"), 0);
        ResetSelectorState();
        ASSERT_TRUE(HcommIsSupportHcommWriteReduceWithNotifyOnThread());
    }
    void TearDown() override
    {
        ResetSelectorState();
        if (originalAlgo_)
            EXPECT_EQ(setenv("HCCL_ALGO", originalAlgo_->c_str(), 1), 0);
        else
            EXPECT_EQ(unsetenv("HCCL_ALGO"), 0);
    }

private:
    std::optional<std::string> originalAlgo_;
};
} // namespace inline_reduce_ut
#endif
