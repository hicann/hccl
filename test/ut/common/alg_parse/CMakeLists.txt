# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

file(GLOB_RECURSE UT_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
# 添加需要测试的源文件
set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/common/alg_parse.cc
)

message(STATUS "Final Ascend Path: ${UT_FILES}")
add_executable(alg_parse_test ${UT_FILES} ${SOURCE_FILES})

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/x86_64-linux")
else()
    set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/aarch64-linux")
endif()
message(STATUS "Final Ascend Path: ${ASCEND_HOME_PATH}")

target_include_directories(alg_parse_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${PROJECT_SOURCE_DIR}/src/common
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/base
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
)

get_target_property(MY_INC_DIRS alg_parse_test INCLUDE_DIRECTORIES)
message(STATUS "Include directories for my_target: ${MY_INC_DIRS}")


target_compile_options(alg_parse_test   PRIVATE 
    -U_FORTIFY_SOURCE
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
)

target_link_libraries(alg_parse_test  PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    GTest::gtest
    -lpthread
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -lgcov
    -lrt
    -lpthread
)

run_llt_test(TARGET  alg_parse_test)