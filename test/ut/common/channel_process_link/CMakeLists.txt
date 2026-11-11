# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

# 版本探测前置（在定义任何target之前）：本用例编译真实channel.cc，其9.1分支引用的
# 头文件/符号按9.1工具链编写桩接口。探测失败或工具链低于9.1时跳过本用例构建，
# 不阻断其余UT（用message(STATUS)+return()，不用message(FATAL_ERROR)）。
set(_ccu_cann_root "${ASCEND_CANN_PACKAGE_PATH}")
if(IS_DIRECTORY "${_ccu_cann_root}/${CMAKE_SYSTEM_PROCESSOR}-linux")
    set(_ccu_cann_root "${_ccu_cann_root}/${CMAKE_SYSTEM_PROCESSOR}-linux")
endif()
set(_ccu_cann_ver_header "${_ccu_cann_root}/include/version/cann_version.h")
set(_ccu_cann_version_line "")
if(EXISTS "${_ccu_cann_ver_header}")
    file(STRINGS "${_ccu_cann_ver_header}" _ccu_cann_version_line
        REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
endif()
if(_ccu_cann_version_line STREQUAL "")
    message(STATUS "[UT] channel_process_link skipped: no CANN_VERSION_NUM found in ${_ccu_cann_ver_header}.")
    return()
endif()
string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
    _ccu_cann_version_expr "${_ccu_cann_version_line}")
# math(EXPR) 支持纯数字与+-*/()算术表达式（如((9*10000000)+(2*100000))），
# 但宏引用等其他形式会FATAL_ERROR，先校验字符集再计算
if(NOT _ccu_cann_version_expr MATCHES "^[0-9() \t+*/-]+$")
    message(STATUS "[UT] channel_process_link skipped: CANN_VERSION_NUM is not a plain arithmetic "
        "expression, expr = '${_ccu_cann_version_expr}'.")
    return()
endif()
math(EXPR _ccu_cann_version_num "${_ccu_cann_version_expr}" OUTPUT_FORMAT DECIMAL)
if(_ccu_cann_version_num LESS 90100000)
    message(STATUS "[UT] channel_process_link skipped: requires CANN >= 9.1, "
        "detected CANN_VERSION_NUM = ${_ccu_cann_version_num}.")
    return()
endif()
message(STATUS "[UT] channel_process_link detected CANN_VERSION_NUM = ${_ccu_cann_version_num}.")

# ascend_hal探测前置到target定义之前：未命中时NOTFOUND值进入target_link_libraries会阻断
# 整个UT构建，按本文件"跳过不阻断"的约定，与版本探测同风格STATUS+return()跳过本用例。
find_library(ASCEND_HAL_LIBRARY ascend_hal HINTS
    "${ASCEND_CANN_PACKAGE_PATH}/devlib/linux/${CMAKE_SYSTEM_PROCESSOR}"
    "${_ccu_cann_root}/devlib/linux/${CMAKE_SYSTEM_PROCESSOR}"
    "${_ccu_cann_root}/lib64")
if(NOT ASCEND_HAL_LIBRARY)
    message(STATUS "[UT] channel_process_link skipped: ascend_hal not found.")
    return()
endif()
get_filename_component(ASCEND_HAL_DIR "${ASCEND_HAL_LIBRARY}" DIRECTORY)

add_executable(channel_process_link_test
    channel_process_link_test.cc
    stub/hcomm_stub.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/channel/channel.cc
    # 用例直接调用 CalcChannelRequestNhrMultiJettyUbx，该函数引用以下两个实现；
    # 其余未引用函数由 -ffunction-sections + --gc-sections 裁剪。
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/channel/channel_request.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/template_utils.cc
    ${PROJECT_SOURCE_DIR}/src/common/log.cc
)

target_compile_definitions(channel_process_link_test PRIVATE
    HOST_COMPILE ORION_MODE OPEN_BUILD_PROJECT)

# channel.cc还包含其他路径，按函数裁剪未调用的依赖。
target_compile_options(channel_process_link_test PRIVATE
    -O0 -g -ffunction-sections -fdata-sections -fno-inline -U_FORTIFY_SOURCE)
target_link_options(channel_process_link_test PRIVATE -Wl,--gc-sections)

file(GLOB_RECURSE HCCL_HEADERS CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/common/*.h" "${PROJECT_SOURCE_DIR}/src/common/*.hpp"
    "${PROJECT_SOURCE_DIR}/src/ops/op_common/*.h" "${PROJECT_SOURCE_DIR}/src/ops/op_common/*.hpp")
set(HCCL_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
foreach(header IN LISTS HCCL_HEADERS)
    get_filename_component(header_dir "${header}" DIRECTORY)
    list(APPEND HCCL_INCLUDE_DIRS "${header_dir}")
endforeach()
list(REMOVE_DUPLICATES HCCL_INCLUDE_DIRS)
target_include_directories(channel_process_link_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/stub"
    ${HCCL_INCLUDE_DIRS})

target_include_directories(channel_process_link_test SYSTEM PRIVATE
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

target_compile_definitions(channel_process_link_test PRIVATE CANN_VERSION_NUM=${_ccu_cann_version_num})

target_link_directories(channel_process_link_test PRIVATE "${_ccu_cann_root}/lib64")
set_target_properties(channel_process_link_test PROPERTIES
    BUILD_RPATH "${_ccu_cann_root}/lib64;${ASCEND_HAL_DIR}")

target_link_libraries(channel_process_link_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
    hcomm acl_rt c_sec unified_dlog runtime mmpa error_manager "${ASCEND_HAL_LIBRARY}")

run_llt_test(TARGET channel_process_link_test TASK_NUM 1)
