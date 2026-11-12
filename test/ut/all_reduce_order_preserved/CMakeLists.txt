# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

add_executable(all_reduce_order_preserved_count_test
    all_reduce_order_preserved_count_test.cc
    ../common/topo_match/stub/log_stub.cc
    ${_repo_root}/src/ops/op_common/algorithm/executor/executor_v2_base.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/alg_v2_template_base.cc
)

target_compile_definitions(all_reduce_order_preserved_count_test PRIVATE AICPU_COMPILE)
target_include_directories(all_reduce_order_preserved_count_test PRIVATE
    ${_repo_root}/include
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
    ${_repo_root}/src/ops/op_common
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/ops/op_common/selector
    ${_repo_root}/src/ops/op_common/topo_info
    ${_repo_root}/src/ops/op_common/algorithm/executor
    ${_repo_root}/src/ops/op_common/algorithm/executor/channel
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry
    ${_repo_root}/src/ops/op_common/algorithm/template
    ${_repo_root}/src/ops/op_common/algorithm/template/registry
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper
    ${_repo_root}/src/ops/op_common/algorithm/topo_match
    ${_repo_root}/src/ops/all_reduce/algorithm/executor
    ${_repo_root}/src/ops/reduce_scatter/algorithm/executor
    ${_repo_root}/src/ops/reduce_scatter/algorithm/template/aicpu
    ${_repo_root}/src/ops/all_gather/algorithm/template/aicpu
)

file(GLOB _cann_arch_dir LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
if(NOT _cann_arch_dir)
    message(FATAL_ERROR "[UT] No CANN arch directory (*-linux) found under: ${ASCEND_CANN_PACKAGE_PATH}")
endif()
list(GET _cann_arch_dir 0 _cann_arch_dir)
target_include_directories(all_reduce_order_preserved_count_test PRIVATE
    ${_cann_arch_dir}/include
    ${_cann_arch_dir}/include/hccl
    ${_cann_arch_dir}/pkg_inc
    ${_cann_arch_dir}/pkg_inc/base
    ${_cann_arch_dir}/pkg_inc/hccl
    ${_cann_arch_dir}/pkg_inc/hcomm/ccu
)

# 与其余UT保持一致，使用配套CANN版本以避免dlsym兼容类型重复定义。
set(_version_header "${_cann_arch_dir}/include/version/cann_version.h")
if(EXISTS "${_version_header}")
    file(STRINGS "${_version_header}" _version_line
        REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
    string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1" _version_expr "${_version_line}")
    string(STRIP "${_version_expr}" _version_expr)
    if(NOT _version_line OR NOT _version_expr)
        message(FATAL_ERROR "[UT] Failed to parse CANN_VERSION_NUM from ${_version_header}: missing or empty definition.")
    endif()
    math(EXPR _version_num "${_version_expr}")
    target_compile_definitions(all_reduce_order_preserved_count_test PRIVATE CANN_VERSION_NUM=${_version_num})
else()
    message(WARNING "[UT] cann_version.h not found at ${_version_header}, CANN_VERSION_NUM left undefined.")
endif()

target_link_libraries(all_reduce_order_preserved_count_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET all_reduce_order_preserved_count_test TASK_NUM 1)
