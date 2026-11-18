# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

# 对称内存公共层门禁矩阵 UT：需要 hcomm 头提供 HcclCommSymWindow（CANN 9.1.0 起），
# 低版本环境跳过本专项，不影响其他 UT 目录。
set(_sym_mem_cann_root "${ASCEND_CANN_PACKAGE_PATH}")
if(IS_DIRECTORY "${_sym_mem_cann_root}/${CMAKE_SYSTEM_PROCESSOR}-linux")
    set(_sym_mem_cann_root "${_sym_mem_cann_root}/${CMAKE_SYSTEM_PROCESSOR}-linux")
endif()
set(_sym_mem_version_header "${_sym_mem_cann_root}/include/version/cann_version.h")
if(NOT EXISTS "${_sym_mem_version_header}")
    message(WARNING "[UT] Skip sym_mem_gate_test: ${_sym_mem_version_header} not found.")
    return()
endif()
file(STRINGS "${_sym_mem_version_header}" _sym_mem_version_line
    REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
if(NOT _sym_mem_version_line)
    message(WARNING "[UT] Skip sym_mem_gate_test: CANN_VERSION_NUM not found.")
    return()
endif()
string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
    _sym_mem_version_expr "${_sym_mem_version_line}")
if(NOT _sym_mem_version_expr MATCHES "^[0-9() \t+*/-]+$")
    message(WARNING "[UT] Skip sym_mem_gate_test: unrecognized CANN_VERSION_NUM expression.")
    return()
endif()
math(EXPR _sym_mem_version_num "${_sym_mem_version_expr}")
if(_sym_mem_version_num LESS 90100000)
    message(STATUS "[UT] Skip sym_mem_gate_test: requires CANN 9.1 or later.")
    return()
endif()

set(SYMEM_UT_FILES
    sym_mem_gate_test.cc
    stub/hcomm_stub.cc
    stub/op_common_stub.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/op_common.cc
    ${PROJECT_SOURCE_DIR}/src/common/log.cc
)

add_executable(sym_mem_gate_test ${SYMEM_UT_FILES})

target_compile_definitions(sym_mem_gate_test PRIVATE
    HOST_COMPILE ORION_MODE OPEN_BUILD_PROJECT CANN_VERSION_NUM=${_sym_mem_version_num})

# op_common 还包含其他引擎/算法路径，按函数裁剪未调用的依赖。
target_compile_options(sym_mem_gate_test PRIVATE
    -O0 -g -ffunction-sections -fdata-sections -fno-inline -U_FORTIFY_SOURCE)
target_link_options(sym_mem_gate_test PRIVATE -Wl,--gc-sections)

file(GLOB_RECURSE HCCL_HEADERS CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/common/*.h" "${PROJECT_SOURCE_DIR}/src/common/*.hpp"
    "${PROJECT_SOURCE_DIR}/src/ops/op_common/*.h" "${PROJECT_SOURCE_DIR}/src/ops/op_common/*.hpp")
set(HCCL_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
foreach(header IN LISTS HCCL_HEADERS)
    get_filename_component(header_dir "${header}" DIRECTORY)
    list(APPEND HCCL_INCLUDE_DIRS "${header_dir}")
endforeach()
list(REMOVE_DUPLICATES HCCL_INCLUDE_DIRS)
target_include_directories(sym_mem_gate_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/stub"
    ${HCCL_INCLUDE_DIRS})

target_include_directories(sym_mem_gate_test SYSTEM PRIVATE
    "${_sym_mem_cann_root}/include" "${_sym_mem_cann_root}/include/hccl" "${_sym_mem_cann_root}/include/hcomm"
    "${_sym_mem_cann_root}/include/acl" "${_sym_mem_cann_root}/include/acl/error_codes"
    "${_sym_mem_cann_root}/include/mmpa" "${_sym_mem_cann_root}/include/mmpa/sub_inc"
    "${_sym_mem_cann_root}/pkg_inc" "${_sym_mem_cann_root}/pkg_inc/base" "${_sym_mem_cann_root}/pkg_inc/hccl"
    "${_sym_mem_cann_root}/pkg_inc/hcomm" "${_sym_mem_cann_root}/pkg_inc/runtime"
    "${_sym_mem_cann_root}/pkg_inc/runtime/runtime" "${_sym_mem_cann_root}/pkg_inc/runtime/runtime/rts"
    "${_sym_mem_cann_root}/pkg_inc/aicpu" "${_sym_mem_cann_root}/pkg_inc/aicpu/common"
    "${_sym_mem_cann_root}/pkg_inc/profiling")

target_link_directories(sym_mem_gate_test PRIVATE "${_sym_mem_cann_root}/lib64")
set_target_properties(sym_mem_gate_test PROPERTIES
    BUILD_RPATH "${_sym_mem_cann_root}/lib64")

target_link_libraries(sym_mem_gate_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
    hcomm acl_rt c_sec unified_dlog runtime mmpa error_manager)

run_llt_test(TARGET sym_mem_gate_test TASK_NUM 1)
