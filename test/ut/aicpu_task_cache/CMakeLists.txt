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
set(_cache_src ${_repo_root}/src/ops/op_common/algorithm/template/aicpu/task_cache)

add_executable(aicpu_task_cache_test
    # 窗口能力与地址查询由测试桩提供, 不加载真实HCOMM动态库。
    aicpu_task_cache_test.cc
    stub/securec_stub.cc
    ${_cache_src}/aicpu_task_cache_key.cc
    ${_cache_src}/aicpu_task_cache_policy.cc
    ${_cache_src}/aicpu_task_cache_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/topo_match/stub/log_stub.cc
)

target_include_directories(aicpu_task_cache_test PRIVATE
    ${_cache_src}
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
)

file(GLOB _cann_arch_dir LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
if(_cann_arch_dir)
    list(GET _cann_arch_dir 0 _cann_arch_dir)
    target_include_directories(aicpu_task_cache_test PRIVATE
        ${_cann_arch_dir}/include
        ${_cann_arch_dir}/include/hccl
        ${_cann_arch_dir}/pkg_inc
        ${_cann_arch_dir}/pkg_inc/base
        ${_cann_arch_dir}/pkg_inc/hccl
        ${_cann_arch_dir}/pkg_inc/hcomm/ccu
    )

    set(_cann_ver_header "${_cann_arch_dir}/include/version/cann_version.h")
    if(EXISTS "${_cann_ver_header}")
        file(STRINGS "${_cann_ver_header}" _cann_ver_line REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
        if(_cann_ver_line)
            string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
                _cann_ver_expr "${_cann_ver_line}")
            math(EXPR _cann_ver_num "${_cann_ver_expr}" OUTPUT_FORMAT DECIMAL)
            target_compile_definitions(aicpu_task_cache_test PRIVATE CANN_VERSION_NUM=${_cann_ver_num})
        endif()
    endif()
endif()

target_compile_options(aicpu_task_cache_test PRIVATE -U_FORTIFY_SOURCE)
target_link_libraries(aicpu_task_cache_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET aicpu_task_cache_test TASK_NUM 1)
