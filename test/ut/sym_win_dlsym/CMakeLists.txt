# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# sym_win_dlsym: host 侧「通信域 rank → LSA WorldTeam memberId」登记与逐 peer 取址 UT。
# 直接编译生产源码 hccl_sym_win_dl.cc；dlopen/dlsym 与 legacy 取址接口以强符号桩替换，
# 覆盖映射命中、LSA 未覆盖全部 rank、逐 peer 取址透传通信域 rank、空入参早返回。

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

add_executable(sym_win_dlsym_test
    sym_win_dlsym_test.cc
    sym_win_dlsym_stub.cc
    ${_repo_root}/src/common/hcomm_dlsym/hccl_sym_win_dl.cc
)

target_include_directories(sym_win_dlsym_test PRIVATE
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
)

# 与 reduce_scatter_seq64 一致：优先选当前架构对应的 CANN 目录，并注入 CANN_VERSION_NUM；
# 缺少该宏时 dlsym_common.h 的桩类型块会与 CANN 公共头重复定义。
file(GLOB _cann_arch_dirs LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
if(_cann_arch_dirs)
    set(_cann_arch_dir "")
    foreach(_dir ${_cann_arch_dirs})
        if(_dir MATCHES "/${CMAKE_SYSTEM_PROCESSOR}-linux$")
            set(_cann_arch_dir ${_dir})
            break()
        endif()
    endforeach()
    if(NOT _cann_arch_dir)
        list(GET _cann_arch_dirs 0 _cann_arch_dir)
    endif()
    target_include_directories(sym_win_dlsym_test PRIVATE
        ${_cann_arch_dir}/include
        ${_cann_arch_dir}/include/hccl
        ${_cann_arch_dir}/include/hcomm
        ${_cann_arch_dir}/include/hcomm/ccu
        ${_cann_arch_dir}/pkg_inc
        ${_cann_arch_dir}/pkg_inc/base
        ${_cann_arch_dir}/pkg_inc/hccl
        ${_cann_arch_dir}/pkg_inc/hcomm/ccu
    )

    set(_cann_ver_header "${_cann_arch_dir}/include/version/cann_version.h")
    if(EXISTS "${_cann_ver_header}")
        file(STRINGS "${_cann_ver_header}" _cann_ver_line REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
        if(NOT _cann_ver_line)
            message(FATAL_ERROR "[UT] no '#define CANN_VERSION_NUM' found in ${_cann_ver_header}")
        endif()
        string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1" _cann_ver_expr "${_cann_ver_line}")
        math(EXPR _cann_ver_num "${_cann_ver_expr}" OUTPUT_FORMAT DECIMAL)
        target_compile_definitions(sym_win_dlsym_test PRIVATE CANN_VERSION_NUM=${_cann_ver_num})
    endif()
endif()

target_compile_options(sym_win_dlsym_test PRIVATE -U_FORTIFY_SOURCE)

target_link_libraries(sym_win_dlsym_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
    -ldl
)

run_llt_test(TARGET sym_win_dlsym_test TASK_NUM 1)
