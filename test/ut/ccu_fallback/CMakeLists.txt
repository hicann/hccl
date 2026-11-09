# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# CCU 协商降级策略层 UT（ccu_fallback）：
# 与公共协商模块UT（comm_negotiation）解耦——comm_negotiation 接口打桩，不链接其实现与 CommWorker；
# 与回退动作层（op_common.cc 的 ReSelector/SetCommEngine）解耦——回退动作打桩；
# 被测真实源码 = 策略层（ccu_fallback.cc）。

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

set(_ut_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_fallback_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_fallback_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/log_stub.cc
    # 被测真实源码
    ${_repo_root}/src/ops/op_common/ccu_fallback.cc
)

# include 集合覆盖 ccu_fallback.h -> op_common.h -> executor/selector/template 全链
# （集合与 selector_engine UT 保持一致，另补仓对外头 include/ 供 cccl.h 解析）
set(_ut_include_dirs
    ${_repo_root}/include
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/ops/op_common
    ${_repo_root}/src/ops/op_common/selector
    ${_repo_root}/src/ops/op_common/selector/inc
    ${_repo_root}/src/ops/op_common/topo_info
    ${_repo_root}/src/ops/op_common/algorithm/template
    ${_repo_root}/src/ops/op_common/algorithm/template/registry
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper
    ${_repo_root}/src/ops/op_common/algorithm/executor
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry
    ${_repo_root}/src/ops/op_common/algorithm/executor/channel
    ${_repo_root}/src/ops/op_common/algorithm/topo_match
    ${_repo_root}/src/common
    ${_repo_root}/src/common/tuner
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
)

# CANN 版本探测宏（沿用 comm_negotiation UT 惯例，同时注入 CANN 头文件路径）：
# 缺少该宏时 dlsym_common.h 中的桩类型块会启用，与 CANN 公共头重复定义导致 conflicting declaration。
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

add_executable(ccu_fallback_test ${_ut_src_list})

target_compile_options(ccu_fallback_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(ccu_fallback_test PRIVATE ${_ut_include_dirs})

_detect_cann_version(ccu_fallback_test)

target_link_libraries(ccu_fallback_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    # 被测源码使用 strncpy_s/sprintf_s 等 securec 函数，与主构建一致链接 CANN 的 libc_sec.so
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
    -lpthread
)

run_llt_test(TARGET ccu_fallback_test TASK_NUM 1)
