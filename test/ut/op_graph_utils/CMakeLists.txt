# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# hccl_op_utils (GE 自定义算子公共工具) 纯函数 UT: 真实编译被测源, hcomm/rt/log 依赖由本目录桩提供
set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

set(ut_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_op_utils_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_op_utils_read_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_custom_op_parse_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_op_v_parse_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/log_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/op_deps_stub.cc
    ${_repo_root}/src/common/op_graph/hccl_op_utils.cc
    ${_repo_root}/src/common/op_graph/hccl_op_v_utils.cc
)

add_executable(op_graph_utils_test ${ut_src_list})

target_compile_options(op_graph_utils_test PRIVATE
    -U_FORTIFY_SOURCE
)

# 头路径: 本目录桩优先 (log.h), 其余为被测源与 toolkit 公共头 (hccl.h/hcom.h/graph/gert 均为 header-only 使用)
target_include_directories(op_graph_utils_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${_repo_root}/src/common/op_graph
    ${_repo_root}/src/common
    ${_repo_root}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
)

# CANN 版本号探测: 与主构建口径一致 (宏逻辑照搬 test/ut/selector_engine)
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
            ${_cann_arch_dir}/include/hcomm
            ${_cann_arch_dir}/pkg_inc
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

_detect_cann_version(op_graph_utils_test)

target_link_libraries(op_graph_utils_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET op_graph_utils_test TASK_NUM 1)
