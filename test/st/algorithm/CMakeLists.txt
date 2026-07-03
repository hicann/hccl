# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(hccl_checker_ops_stest)
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
set(HCCL_DEV_BASE ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

enable_testing()

option(ENABLE_GCOV "Enable gcov/lcov coverage instrumentation" OFF)
include(${HCCL_DEV_BASE}/cmake/intf_pub_gccnative.cmake)

if(NOT DEFINED ST_TASKS)
    set(ST_TASKS "all")
endif()
message(STATUS "ST_TASKS=${ST_TASKS}")

if (BUILD_OPEN_PROJECT)
    set(UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils/src)
    set(TESTCASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/testcase)
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    set(THIRD_PARTH_PATH ${HCCL_DEV_BASE}/output/third_party)

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/x86_64-linux")
    else()
        set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/aarch64-linux")
    endif()
    message(STATUS "Final Ascend Path: ${ASCEND_HOME_PATH}")
endif()

# 前向兼容：探测 CANN 版本号，与主构建保持一致
set(_st_cann_ver_header "${ASCEND_HOME_PATH}/include/version/cann_version.h")
if(EXISTS "${_st_cann_ver_header}")
    file(STRINGS "${_st_cann_ver_header}" _st_cann_ver_line
         REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
    if(_st_cann_ver_line)
        string(REGEX MATCH
               "\\(([0-9]+) \\* 10000000\\) \\+ \\(([0-9]+) \\* 100000\\) \\+ \\(([0-9]+) \\* 1000\\)"
               _ "${_st_cann_ver_line}")
        if(NOT CMAKE_MATCH_1 STREQUAL "" AND NOT CMAKE_MATCH_2 STREQUAL "" AND NOT CMAKE_MATCH_3 STREQUAL "")
            math(EXPR _ST_CANN_VERSION_NUM
                 "${CMAKE_MATCH_1} * 10000000 + ${CMAKE_MATCH_2} * 100000 + ${CMAKE_MATCH_3} * 1000")
            add_compile_definitions(CANN_VERSION_NUM=${_ST_CANN_VERSION_NUM})
            message(STATUS "[ST] Detected CANN_VERSION_NUM = ${_ST_CANN_VERSION_NUM}")
        endif()
    endif()
endif()

# 下载并编译gtest
set(CANN_3RD_LIB_PATH ${HCCL_DEV_BASE}/third_party)
include(${HCCL_DEV_BASE}/cmake/third_party/gtest.cmake)

add_subdirectory(testcase)
add_subdirectory(utils)