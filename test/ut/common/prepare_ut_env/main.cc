/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include "gtest/gtest.h"

GTEST_API_ int main(int argc, char **argv) 
{
  printf("Running main() from gtest_main.cc\n");
  //testing::GTEST_FLAG(filter) = "HcomKernelInfoTest.st_LoadTask_comm";
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

using namespace std;
class UtPrepareEnv : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        //std::cout << "CleanEnv SetUP" << std::endl;
    }
    static void TearDownTestCase()
    {
        //std::cout << "CleanEnv TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        //std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        //std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(UtPrepareEnv, hccl_prepare_ut_env)
{
    printf("prepare hccl ut llt env start\n");
    // 忽略文件不存在，无权限等异常，不检查返回值
    system("find /dev/shm/ -name 'hccl*' | xargs -i rm {}");    
    sleep(3);
    printf("prepare hccl ut llt env finished\n");
    return;
}
