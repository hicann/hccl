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
project(hccl_checker_ops_stest)
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

# 编译HCCL业务so文件
if (NOT BUILD_OPEN_PROJECT)
    execute_process(
        COMMAND bash ${TOP_DIR}/hccl/test/st/algorithm/utils/script/build.sh
        RESULT_VARIABLE HCCL_BUILD_RESULT
    )
    if(NOT HCCL_BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to execute hccl build script. Exit code: ${HCCL_BUILD_RESULT}")
    endif()
endif()

if (BUILD_OPEN_PROJECT)
    set(UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils/src)
    set(TESTCASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/testcase)
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    set(HCCL_DEV_BASE ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
    set(THIRD_PARTH_PATH ${HCCL_DEV_BASE}/output/third_party)
endif()

add_subdirectory(testcase)
add_subdirectory(utils)