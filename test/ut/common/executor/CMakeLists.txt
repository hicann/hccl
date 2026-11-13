# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 被测实现 omnipipe_executor_utils.h 是 header-only 的内联工具集（IsOnlyThirdAxisTopo 等），
# UT 只需 include 头文件，无需额外编译/链接生产源文件。该头文件间接聚合了 executor_v2_base.h，
# 因此需要补齐 executor/template/selector/topo_info 等公共目录的 include 路径。
set(_omnipipe_executor_utils_ut_include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym/ccu
    ${PROJECT_SOURCE_DIR}/src/ops
    ${PROJECT_SOURCE_DIR}/src/ops/op_common
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/inc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector/pub_inc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/topo_info
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/channel
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/registry
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/registry
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/aicpu
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/wrapper
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/ccu
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match
)

# 前向兼容：探测 CANN 版本号，与主构建/ST 保持一致（宏逻辑与 test/ut/recursive_executor 一致）。
# 缺少该宏时 dlsym_common.h 中的桩类型块会启用，与 CANN 公共头重复定义导致 conflicting declaration。
macro(_detect_cann_version _target)
    file(GLOB _cann_arch_dirs LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
    if(_cann_arch_dirs)
        # 按当前平台架构选择匹配的目录
        set(_cann_arch_dir "")
        foreach(_dir ${_cann_arch_dirs})
            if(_dir MATCHES "/${CMAKE_SYSTEM_PROCESSOR}-linux$")
                set(_cann_arch_dir ${_dir})
                break()
            endif()
        endforeach()
        # 兜底：未匹配到则取第一个
        if(NOT _cann_arch_dir)
            list(GET _cann_arch_dirs 0 _cann_arch_dir)
        endif()
        target_include_directories(${_target} PRIVATE
            ${_cann_arch_dir}/include
            ${_cann_arch_dir}/include/hccl
            ${_cann_arch_dir}/include/hcomm
            ${_cann_arch_dir}/include/hcomm/ccu
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

# ---------- omnipipe_executor_utils_test: OmniPipe 执行器公共工具（纯判定/换算函数） ----------
set(omnipipe_executor_utils_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/omnipipe_executor_utils_test.cc
)

add_executable(omnipipe_executor_utils_test ${omnipipe_executor_utils_src_list})

target_compile_options(omnipipe_executor_utils_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(omnipipe_executor_utils_test PRIVATE ${_omnipipe_executor_utils_ut_include_dirs})

_detect_cann_version(omnipipe_executor_utils_test)

target_link_libraries(omnipipe_executor_utils_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET omnipipe_executor_utils_test TASK_NUM 1)
