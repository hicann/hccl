# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(hccl_checker_ops_utest)
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../)

# 编译HCCL业务so文件
execute_process(
    COMMAND bash ${TOP_DIR}/hccl/test/utils/vm-plugins/verifier/script/build.sh
    RESULT_VARIABLE HCCL_BUILD_RESULT
)
if(NOT HCCL_BUILD_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to execute hccl build script. Exit code: ${HCCL_BUILD_RESULT}")
endif()

set(src_list)
add_executable(hccl_checker_ops_utest ${src_list})
add_subdirectory(testcase)
target_link_libraries(hccl_checker_ops_utest PRIVATE
    checker_fwk
)

target_compile_options(hccl_checker_ops_utest PRIVATE
    -std=c++14
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O0
    -g
    -D_GLIBCXX_USE_CXX11_ABI=0
)

run_llt_test(
    TARGET hccl_checker_ops_utest
)