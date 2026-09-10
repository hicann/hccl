# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

file(GLOB_RECURSE UT_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

# 被测源文件：V2 base（公共工具函数）+ 4 个 matcher + 日志桩
set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match/topo_match_base_v2.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match/topo_match_one_level.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match/topo_match_two_level.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match/topo_match_three_level.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match/topo_match_concurrent_v2.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/log_stub.cc
)

add_executable(topo_match_test ${UT_FILES} ${SOURCE_FILES})

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/x86_64-linux")
else()
    set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/aarch64-linux")
endif()

target_include_directories(topo_match_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/inc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym/ccu
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/base
    ${ASCEND_HOME_PATH}/pkg_inc/hccl
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
)

target_compile_options(topo_match_test PRIVATE
    -U_FORTIFY_SOURCE
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
)

# 前向兼容：探测 CANN 版本号，与主构建/ST 保持一致。缺少该宏时 dlsym_common.h
# 中的桩类型块会启用，与 CANN 公共头重复定义导致 conflicting declaration。
set(_ut_cann_ver_header "${ASCEND_HOME_PATH}/include/version/cann_version.h")
if(EXISTS "${_ut_cann_ver_header}")
    file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line
         REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
    if(_ut_cann_ver_line)
        string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
            _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
        math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
        target_compile_definitions(topo_match_test PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
        message(STATUS "[UT] topo_match Detected CANN_VERSION_NUM = ${_UT_CANN_VERSION_NUM}")
    else()
        message(FATAL_ERROR "[UT] Failed to parse CANN_VERSION_NUM from ${_ut_cann_ver_header}: "
            "no '#define CANN_VERSION_NUM' line found.")
    endif()
else()
    message(WARNING "[UT] cann_version.h not found at ${_ut_cann_ver_header}, CANN_VERSION_NUM left undefined.")
endif()

target_link_libraries(topo_match_test PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    GTest::gtest
    GTest::gtest_main
    -lpthread
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -lgcov
    -lrt
    -lpthread
)

run_llt_test(TARGET topo_match_test)
