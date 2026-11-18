# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_broadcast_sequence_executor_ut_include_dirs
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym/ccu
    ${PROJECT_SOURCE_DIR}/src/ops
    ${PROJECT_SOURCE_DIR}/src/ops/op_common
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/inc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector/inc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/topo_info
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/channel
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/registry
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/registry
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/aicpu
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/wrapper
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match
    ${PROJECT_SOURCE_DIR}/src/ops/broadcast/algorithm/executor
)

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
                endif()
                break()
            endif()
        endforeach()
    endif()
endmacro()

add_executable(broadcast_sequence_executor_test broadcast_sequence_executor_test.cc)

target_compile_options(broadcast_sequence_executor_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(broadcast_sequence_executor_test PRIVATE
    ${_broadcast_sequence_executor_ut_include_dirs}
)

_detect_cann_version(broadcast_sequence_executor_test)

target_link_libraries(broadcast_sequence_executor_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET broadcast_sequence_executor_test TASK_NUM 1)
