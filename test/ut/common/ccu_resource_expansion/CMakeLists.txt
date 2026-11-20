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

# 仅在具备所需头文件版本时接入本专项，避免影响其他UT目录的配置。
set(_ccu_cann_root "${ASCEND_CANN_PACKAGE_PATH}")
if(IS_DIRECTORY "${_ccu_cann_root}/${CMAKE_SYSTEM_PROCESSOR}-linux")
    set(_ccu_cann_root "${_ccu_cann_root}/${CMAKE_SYSTEM_PROCESSOR}-linux")
endif()
set(_ccu_cann_version_header "${_ccu_cann_root}/include/version/cann_version.h")
if(NOT EXISTS "${_ccu_cann_version_header}")
    message(WARNING "[UT] Skip ccu_resource_expansion_test: ${_ccu_cann_version_header} not found.")
    return()
endif()
file(STRINGS "${_ccu_cann_version_header}" _ccu_cann_version_line
    REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
if(NOT _ccu_cann_version_line)
    message(WARNING "[UT] Skip ccu_resource_expansion_test: CANN_VERSION_NUM not found.")
    return()
endif()
string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
    _ccu_cann_version_expr "${_ccu_cann_version_line}")
if(NOT _ccu_cann_version_expr MATCHES "^[0-9() \t+*/-]+$")
    message(WARNING "[UT] Skip ccu_resource_expansion_test: unrecognized CANN_VERSION_NUM expression.")
    return()
endif()
math(EXPR _ccu_cann_version_num "${_ccu_cann_version_expr}")
if(_ccu_cann_version_num LESS 90200000)
    message(STATUS "[UT] Skip ccu_resource_expansion_test: requires CANN 9.2 or later.")
    return()
endif()

find_library(ASCEND_HAL_LIBRARY ascend_hal HINTS
    "${ASCEND_CANN_PACKAGE_PATH}/devlib/linux/${CMAKE_SYSTEM_PROCESSOR}"
    "${_ccu_cann_root}/devlib/linux/${CMAKE_SYSTEM_PROCESSOR}"
    "${_ccu_cann_root}/lib64")
if(NOT ASCEND_HAL_LIBRARY)
    message(WARNING "[UT] Skip ccu_resource_expansion_test: ascend_hal not found.")
    return()
endif()
get_filename_component(ASCEND_HAL_DIR "${ASCEND_HAL_LIBRARY}" DIRECTORY)

set_property(DIRECTORY PROPERTY EP_BASE "${CMAKE_BINARY_DIR}/third_party/ccu_resource_expansion")
add_cann_third_party(boost)
add_cann_third_party(mockcpp)

add_executable(ccu_resource_expansion_test
    ccu_resource_expansion_test.cc
    stub/hcomm_stub.cc
    stub/op_common_stub.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/op_common.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/stream_capture.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/ccu_log.cc
    ${PROJECT_SOURCE_DIR}/src/common/log.cc
)

target_compile_definitions(ccu_resource_expansion_test PRIVATE
    HOST_COMPILE ORION_MODE OPEN_BUILD_PROJECT)

# op_common还包含其他引擎路径，按函数裁剪未调用的依赖。
target_compile_options(ccu_resource_expansion_test PRIVATE
    -O0 -g -ffunction-sections -fdata-sections -fno-inline -U_FORTIFY_SOURCE)
target_link_options(ccu_resource_expansion_test PRIVATE -Wl,--gc-sections)

file(GLOB_RECURSE HCCL_HEADERS CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/common/*.h" "${PROJECT_SOURCE_DIR}/src/common/*.hpp"
    "${PROJECT_SOURCE_DIR}/src/ops/op_common/*.h" "${PROJECT_SOURCE_DIR}/src/ops/op_common/*.hpp")
set(HCCL_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
foreach(header IN LISTS HCCL_HEADERS)
    get_filename_component(header_dir "${header}" DIRECTORY)
    list(APPEND HCCL_INCLUDE_DIRS "${header_dir}")
endforeach()
list(REMOVE_DUPLICATES HCCL_INCLUDE_DIRS)
target_include_directories(ccu_resource_expansion_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/stub"
    ${HCCL_INCLUDE_DIRS})

target_include_directories(ccu_resource_expansion_test SYSTEM PRIVATE
    "${_ccu_cann_root}/include" "${_ccu_cann_root}/include/hccl" "${_ccu_cann_root}/include/hcomm"
    "${_ccu_cann_root}/include/hcomm/ccu" "${_ccu_cann_root}/include/acl"
    "${_ccu_cann_root}/include/acl/error_codes" "${_ccu_cann_root}/include/mmpa"
    "${_ccu_cann_root}/include/mmpa/sub_inc"
    "${_ccu_cann_root}/pkg_inc" "${_ccu_cann_root}/pkg_inc/base" "${_ccu_cann_root}/pkg_inc/hccl"
    "${_ccu_cann_root}/pkg_inc/hcomm" "${_ccu_cann_root}/pkg_inc/hcomm/ccu"
    "${_ccu_cann_root}/pkg_inc/runtime" "${_ccu_cann_root}/pkg_inc/runtime/runtime"
    "${_ccu_cann_root}/pkg_inc/runtime/runtime/rts" "${_ccu_cann_root}/pkg_inc/aicpu"
    "${_ccu_cann_root}/pkg_inc/aicpu/common" "${_ccu_cann_root}/pkg_inc/toolchain"
    "${_ccu_cann_root}/pkg_inc/profiling")

target_compile_definitions(ccu_resource_expansion_test PRIVATE CANN_VERSION_NUM=${_ccu_cann_version_num})

target_link_directories(ccu_resource_expansion_test PRIVATE "${_ccu_cann_root}/lib64")
set_target_properties(ccu_resource_expansion_test PROPERTIES
    BUILD_RPATH "${_ccu_cann_root}/lib64;${ASCEND_HAL_DIR}")

target_link_libraries(ccu_resource_expansion_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    mockcpp
    -lpthread
    hcomm acl_rt c_sec unified_dlog runtime mmpa error_manager "${ASCEND_HAL_LIBRARY}")

run_llt_test(TARGET ccu_resource_expansion_test TASK_NUM 1)
