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

# 被测源文件：DlHcommFunction真实单例 + 桩；
# ccu_fallback.cc本身由ccu_fallback_test.cc以源码包含方式引入（白盒直调static函数CompareCcuParam），
# 不再单独编译，避免同一目标内重复符号
set(ccu_fallback_test_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_fallback_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_fallback_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/log_stub.cc
    ${_repo_root}/src/ops/op_common/dlhcomm_function.cc
    ${_repo_root}/src/common/hcomm_dlsym/hccl_dl.cc
)

add_executable(ccu_fallback_test ${ccu_fallback_test_src_list})

target_compile_options(ccu_fallback_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(ccu_fallback_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
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
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
)

file(GLOB _cann_arch_dir LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
if(_cann_arch_dir)
    list(GET _cann_arch_dir 0 _cann_arch_dir)
    target_include_directories(ccu_fallback_test PRIVATE
        ${_cann_arch_dir}/include
        ${_cann_arch_dir}/include/hccl
        ${_cann_arch_dir}/include/acl
        ${_cann_arch_dir}/include/hcomm/ccu
        ${_cann_arch_dir}/pkg_inc
        ${_cann_arch_dir}/pkg_inc/hccl
        ${_cann_arch_dir}/pkg_inc/base
        ${_cann_arch_dir}/pkg_inc/hcomm/ccu
    )

    # 前向兼容：探测 CANN 版本号，与主构建保持一致。缺少该宏时 dlsym_common.h
    # 中的桩类型块会启用，与 CANN 公共头重复定义导致 conflicting declaration。
    set(_ut_cann_ver_header "${_cann_arch_dir}/include/version/cann_version.h")
    if(EXISTS "${_ut_cann_ver_header}")
        file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line
             REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
        if(_ut_cann_ver_line)
            string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
                _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
            math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
            target_compile_definitions(ccu_fallback_test PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
            message(STATUS "[UT] ccu_fallback Detected CANN_VERSION_NUM = ${_UT_CANN_VERSION_NUM}")
        else()
            message(FATAL_ERROR "[UT] Failed to parse CANN_VERSION_NUM from ${_ut_cann_ver_header}: "
                "no '#define CANN_VERSION_NUM' line found.")
        endif()
    else()
        message(WARNING "[UT] cann_version.h not found at ${_ut_cann_ver_header}, CANN_VERSION_NUM left undefined.")
    endif()
else()
    message(FATAL_ERROR "[UT] No CANN arch directory (*-linux) found under: ${ASCEND_CANN_PACKAGE_PATH}")
endif()

# ccu_fallback.cc 依赖 memcpy_s/strncpy_s/sprintf_s/memset_s，链接 CANN 包内 libc_sec
find_library(UT_CSEC_LIB NAMES c_sec
    PATHS ${ASCEND_CANN_PACKAGE_PATH}/lib64 ${_cann_arch_dir}/lib64
    NO_DEFAULT_PATH
)
if(NOT UT_CSEC_LIB)
    message(FATAL_ERROR "[UT] libc_sec not found under CANN package: ${ASCEND_CANN_PACKAGE_PATH}")
endif()
get_filename_component(UT_CSEC_LIB_DIR "${UT_CSEC_LIB}" DIRECTORY)

target_link_libraries(ccu_fallback_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    ${UT_CSEC_LIB}
    -ldl
    -lpthread
)

# 运行时优先从 CANN 包目录加载 libc_sec.so，避免依赖外部 LD_LIBRARY_PATH
set_target_properties(ccu_fallback_test PROPERTIES BUILD_RPATH "${UT_CSEC_LIB_DIR}")

run_llt_test(TARGET ccu_fallback_test TASK_NUM 1)
