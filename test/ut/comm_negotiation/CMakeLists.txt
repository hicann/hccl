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

set(_ut_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_negotiation_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_negotiation_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/log_stub.cc
    # 被测真实源码：协商模块 + CommWorker（真实线程模型）
    ${_repo_root}/src/ops/op_common/comm_negotiation.cc
    ${_repo_root}/src/ops/op_common/comm_worker.cc
    ${_repo_root}/src/ops/op_common/comm_worker_mgr.cc
)

set(_ut_include_dirs
    ${_repo_root}/include
    ${_repo_root}/src/ops/op_common
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    # hccl_ccu_res_dl.h 引用 ccu_types_dl.h（位于 ccu 子目录，与主构建 include 集合一致）
    ${_repo_root}/src/common/hcomm_dlsym/ccu
)

# CANN 版本探测宏（沿用 recursive_executor UT 惯例，同时注入 CANN 头文件路径）
macro(_detect_cann_version _target)
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
        target_include_directories(${_target} PRIVATE
            ${_cann_arch_dir}/include
            ${_cann_arch_dir}/include/hccl
            ${_cann_arch_dir}/include/hcomm/ccu
            # adapter_acl.h 裸引用 acl_base.h，与主构建 intf_pub 的 include 集合保持一致
            ${_cann_arch_dir}/include/acl
            ${_cann_arch_dir}/include/acl/error_codes
            ${_cann_arch_dir}/pkg_inc
            ${_cann_arch_dir}/pkg_inc/hccl
            ${_cann_arch_dir}/pkg_inc/base
            ${_cann_arch_dir}/pkg_inc/hcomm/ccu
        )
        foreach(_arch_dir ${_cann_arch_dirs})
            set(_ut_cann_ver_header "${_arch_dir}/include/version/cann_version.h")
            if(EXISTS "${_ut_cann_ver_header}")
                file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line
                     REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
                if(_ut_cann_ver_line)
                    string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
                        _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
                    math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
                    target_compile_definitions(${_target}
                        PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
                    message(STATUS "[UT] Detected CANN_VERSION_NUM = ${_UT_CANN_VERSION_NUM}")
                endif()
                break()
            endif()
        endforeach()
    endif()
endmacro()

add_executable(comm_negotiation_test ${_ut_src_list})

target_compile_options(comm_negotiation_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(comm_negotiation_test PRIVATE ${_ut_include_dirs})

_detect_cann_version(comm_negotiation_test)

target_link_libraries(comm_negotiation_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    # 被测源码使用 memcpy_s/memset_s 等 securec 函数，与主构建一致链接 CANN 的 libc_sec.so
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
    -lpthread
)

run_llt_test(TARGET comm_negotiation_test TASK_NUM 1)

# 弱符号缺失降级用例：桩不定义 HcclCreateSubCommConfig（模拟老版本 HCOMM 未提供该接口），
# SUT 弱引用链接期解析为空地址，验证 HCCL_E_UNAVAIL 降级路径与资源回滚
set(_ut_symbol_missing_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_negotiation_symbol_missing_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_negotiation_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/log_stub.cc
    ${_repo_root}/src/ops/op_common/comm_negotiation.cc
    ${_repo_root}/src/ops/op_common/comm_worker.cc
    ${_repo_root}/src/ops/op_common/comm_worker_mgr.cc
)

add_executable(comm_negotiation_symbol_missing_test ${_ut_symbol_missing_src_list})

target_compile_definitions(comm_negotiation_symbol_missing_test PRIVATE
    UT_NEG_NO_SUBCOMM_SYMBOL
)

target_compile_options(comm_negotiation_symbol_missing_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(comm_negotiation_symbol_missing_test PRIVATE ${_ut_include_dirs})

_detect_cann_version(comm_negotiation_symbol_missing_test)

target_link_libraries(comm_negotiation_symbol_missing_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
    -lpthread
)

run_llt_test(TARGET comm_negotiation_symbol_missing_test TASK_NUM 1)
