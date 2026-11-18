# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# order_launch UT：入图主链路、双层门控、降级与错误路径（ST 保序链路不可达，回归保护由本 UT 承担）；
# 对齐 ccu_resource_expansion：生产源码直编，hcomm 依赖经强符号桩覆写。

# 按安装包 CANN_VERSION_NUM 注入版本宏
set(_ol_cann_root "${ASCEND_CANN_PACKAGE_PATH}")
if(IS_DIRECTORY "${_ol_cann_root}/${CMAKE_SYSTEM_PROCESSOR}-linux")
    set(_ol_cann_root "${_ol_cann_root}/${CMAKE_SYSTEM_PROCESSOR}-linux")
endif()
set(_ol_cann_version_header "${_ol_cann_root}/include/version/cann_version.h")
if(NOT EXISTS "${_ol_cann_version_header}")
    message(WARNING "[UT] Skip order_launch_test: ${_ol_cann_version_header} not found.")
    return()
endif()
file(STRINGS "${_ol_cann_version_header}" _ol_cann_version_line
    REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
if(NOT _ol_cann_version_line)
    message(WARNING "[UT] Skip order_launch_test: CANN_VERSION_NUM not found.")
    return()
endif()
string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
    _ol_cann_version_expr "${_ol_cann_version_line}")
if(NOT _ol_cann_version_expr MATCHES "^[0-9() \t+*/-]+$")
    message(WARNING "[UT] Skip order_launch_test: unrecognized CANN_VERSION_NUM expression.")
    return()
endif()
math(EXPR _ol_cann_version_num "${_ol_cann_version_expr}")

add_executable(order_launch_test
    order_launch_test.cc
    stub/hcomm_stub.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/order_launch.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/dlhcomm_function.cc
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym/hccl_dl.cc
    ${PROJECT_SOURCE_DIR}/src/common/log.cc
)

target_compile_definitions(order_launch_test PRIVATE
    HOST_COMPILE ORION_MODE OPEN_BUILD_PROJECT CANN_VERSION_NUM=${_ol_cann_version_num})

target_compile_options(order_launch_test PRIVATE
    -O0 -g -ffunction-sections -fdata-sections -fno-inline -U_FORTIFY_SOURCE)
target_link_options(order_launch_test PRIVATE -Wl,--gc-sections)

file(GLOB_RECURSE HCCL_HEADERS CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/common/*.h" "${PROJECT_SOURCE_DIR}/src/common/*.hpp"
    "${PROJECT_SOURCE_DIR}/src/ops/op_common/*.h" "${PROJECT_SOURCE_DIR}/src/ops/op_common/*.hpp")
set(HCCL_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
foreach(header IN LISTS HCCL_HEADERS)
    get_filename_component(header_dir "${header}" DIRECTORY)
    list(APPEND HCCL_INCLUDE_DIRS "${header_dir}")
endforeach()
list(REMOVE_DUPLICATES HCCL_INCLUDE_DIRS)
target_include_directories(order_launch_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/stub"
    ${HCCL_INCLUDE_DIRS})

target_include_directories(order_launch_test SYSTEM PRIVATE
    "${_ol_cann_root}/include" "${_ol_cann_root}/include/hccl" "${_ol_cann_root}/include/hcomm"
    "${_ol_cann_root}/include/hcomm/ccu" "${_ol_cann_root}/include/acl"
    "${_ol_cann_root}/include/acl/error_codes" "${_ol_cann_root}/include/mmpa"
    "${_ol_cann_root}/include/mmpa/sub_inc"
    "${_ol_cann_root}/pkg_inc" "${_ol_cann_root}/pkg_inc/base" "${_ol_cann_root}/pkg_inc/hccl"
    "${_ol_cann_root}/pkg_inc/hcomm" "${_ol_cann_root}/pkg_inc/hcomm/ccu"
    "${_ol_cann_root}/pkg_inc/runtime" "${_ol_cann_root}/pkg_inc/runtime/runtime"
    "${_ol_cann_root}/pkg_inc/runtime/runtime/rts" "${_ol_cann_root}/pkg_inc/aicpu"
    "${_ol_cann_root}/pkg_inc/aicpu/common" "${_ol_cann_root}/pkg_inc/toolchain"
    "${_ol_cann_root}/pkg_inc/profiling")

target_link_directories(order_launch_test PRIVATE "${_ol_cann_root}/lib64")
set_target_properties(order_launch_test PROPERTIES
    BUILD_RPATH "${_ol_cann_root}/lib64")

# 不链接真实 libhcomm（符号由桩提供）；hccl_dl.cc 需 dl 库（CMAKE_DL_LIBS 在 glibc>=2.34 为空）
target_link_libraries(order_launch_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
    ${CMAKE_DL_LIBS}
    acl_rt c_sec unified_dlog runtime mmpa error_manager)

run_llt_test(TARGET order_launch_test TASK_NUM 1)
