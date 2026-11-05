/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_alg_template_base.h"
#include "selector_test_common.h"

namespace ops_hccl {
namespace aiv_selector_test {
    namespace {

        TEST_F(AivTemplateSelectorTest, PcieOpBaseEnablesWriteSemantics)
        {
            EXPECT_TRUE(IsAivWriteEnabled(param_, &topo_));
        }

        TEST_F(AivTemplateSelectorTest, NonPcieTopologyDisablesWriteSemantics)
        {
            topo_.level0PcieMix = false;
            EXPECT_FALSE(IsAivWriteEnabled(param_, &topo_));
        }

        TEST_F(AivTemplateSelectorTest, FullMeshDisablesWriteEvenWithPcieLinks)
        {
            topo_.level0AllMesh = true;
            EXPECT_FALSE(IsAivWriteEnabled(param_, &topo_));
            topo_.level0AllMesh = false;
            EXPECT_TRUE(IsAivWriteEnabled(param_, &topo_));
        }

        TEST_F(AivTemplateSelectorTest, GraphModeDisablesWriteSemantics)
        {
            param_.opMode = OpMode::OFFLOAD;
            EXPECT_FALSE(IsAivWriteEnabled(param_, &topo_));
        }

        TEST_F(AivTemplateSelectorTest, MissingTopologyDisablesWriteSemantics)
        {
            EXPECT_FALSE(IsAivWriteEnabled(param_, nullptr));
        }

    } // namespace
} // namespace aiv_selector_test
} // namespace ops_hccl
