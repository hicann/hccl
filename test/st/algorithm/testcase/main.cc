/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"

// rm -rf ../st/* && cmake ../cmake/superbuild/ -DHOST_PACKAGE=st -DBUILD_MOD=hccl_checker_ops && make -j16
GTEST_API_ int main(int argc, char **argv)
{
    std::cout << "Start to run demo for hccl_checker_ops_stest." << std::endl;
    // testing::GTEST_FLAG(filter) = "ST_SCATTER_TEST.st_scatter_opbase_test_origin";
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}