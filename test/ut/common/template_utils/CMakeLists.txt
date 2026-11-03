# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

file(GLOB_RECURSE UT_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

# 被测源文件：template_utils 公共层（RHD 步骤生成函数族）；桩经 UT_FILES 拾取
set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/template_utils.cc
)

add_executable(template_utils_test ${UT_FILES} ${SOURCE_FILES})

set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT ASCEND_HOME_PATH OR NOT IS_DIRECTORY "${ASCEND_HOME_PATH}")
    message(FATAL_ERROR "[UT] template_utils ASCEND_HOME_PATH is not set or does not exist: "
        "'${ASCEND_HOME_PATH}' (source CANN set_env.sh before configuring UT, see docs/zh/build/build.md)")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/x86_64-linux")
else()
    set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/aarch64-linux")
endif()

target_include_directories(template_utils_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/channel
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/registry
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/inc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/topo_info
    ${PROJECT_SOURCE_DIR}/src/ops
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym/ccu
    ${PROJECT_SOURCE_DIR}/include
    "${ASCEND_HOME_PATH}/pkg_inc"
    "${ASCEND_HOME_PATH}/pkg_inc/base"
    "${ASCEND_HOME_PATH}/pkg_inc/hccl"
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/include/hccl"
)

target_compile_options(template_utils_test PRIVATE
    -U_FORTIFY_SOURCE
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
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
        target_compile_definitions(template_utils_test PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
        message(STATUS "[UT] template_utils Detected CANN_VERSION_NUM = ${_UT_CANN_VERSION_NUM}")
    else()
        message(FATAL_ERROR "[UT] Failed to parse CANN_VERSION_NUM from ${_ut_cann_ver_header}: "
            "no '#define CANN_VERSION_NUM' line found.")
    endif()
else()
    message(WARNING "[UT] cann_version.h not found at ${_ut_cann_ver_header}, CANN_VERSION_NUM left undefined.")
endif()

target_link_libraries(template_utils_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
    -lgcov
    -lrt
)

run_llt_test(TARGET template_utils_test TASK_NUM 1)
