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

set(ut_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoall_hier_test.cc
)

add_executable(alltoall_hier_test ${ut_src_list})

target_compile_options(alltoall_hier_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(alltoall_hier_test PRIVATE
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/aicpu/hier
    ${_repo_root}/src/ops/op_common
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/ops/op_common/algorithm/template
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper
    ${_repo_root}/src/ops/op_common/algorithm/executor
    ${_repo_root}/src/ops/op_common/algorithm/template/registry
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry
    ${_repo_root}/src/ops/op_common/algorithm/executor/channel
    ${_repo_root}/src/ops/op_common/selector
    ${_repo_root}/src/ops/op_common/algorithm/topo_match
    ${_repo_root}/src/ops/op_common/topo_info
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
)

file(GLOB _cann_arch_dir LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
if(_cann_arch_dir)
    list(GET _cann_arch_dir 0 _cann_arch_dir)
    target_include_directories(alltoall_hier_test PRIVATE
        ${_cann_arch_dir}/include
        ${_cann_arch_dir}/include/hccl
        ${_cann_arch_dir}/include/hcomm/ccu
        ${_cann_arch_dir}/pkg_inc
        ${_cann_arch_dir}/pkg_inc/hccl
        ${_cann_arch_dir}/pkg_inc/base
        ${_cann_arch_dir}/pkg_inc/hcomm/ccu
    )

    foreach(_arch_dir ${_cann_arch_dir})
        set(_ut_cann_ver_header "${_arch_dir}/include/version/cann_version.h")
        if(EXISTS "${_ut_cann_ver_header}")
            file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line
                 REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
            if(_ut_cann_ver_line)
                string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
                    _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
                math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
                target_compile_definitions(alltoall_hier_test
                    PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
                message(STATUS "[UT] Detected CANN_VERSION_NUM = ${_UT_CANN_VERSION_NUM}")
            else()
                message(FATAL_ERROR "[UT] Failed to parse CANN_VERSION_NUM from ${_ut_cann_ver_header}.")
            endif()
            break()
        endif()
    endforeach()
endif()

target_link_libraries(alltoall_hier_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET alltoall_hier_test TASK_NUM 1)
