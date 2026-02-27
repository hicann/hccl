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

if (BUILD_OPEN_PROJECT)
    set(UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils/src)
    set(TESTCASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/testcase)
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    set(HCCL_DEV_BASE ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
    set(THIRD_PARTH_PATH ${HCCL_DEV_BASE}/output/third_party)

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/x86_64-linux")
    else()
        set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/aarch64-linux")
    endif()
    message(STATUS "Final Ascend Path: ${ASCEND_HOME_PATH}")
endif()

# 下载并编译gtest
set(CANN_3RD_LIB_PATH ${HCCL_DEV_BASE}/third_party)
include(${HCCL_DEV_BASE}/cmake/third_party/gtest.cmake)

add_subdirectory(testcase)
add_subdirectory(utils)