# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------

# AlltoAllAutoSelector::SelectNdaAlgo 生产实现 UT：
# 覆盖 NDA 选路的空拓扑 / 层级边界（TOPO_LEVEL_NUM_2~3）/ 输入输出重叠 / level0Topo 形状守卫。
# 直连编译 alltoall_auto_selector.cc + auto_selector_base.cc，依赖复用 selector_engine 的桩。

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

set(_selector_ut_include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${_repo_root}/test/ut/selector_engine/stub
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/ops/op_common
    ${_repo_root}/src/ops/op_common/selector
    ${_repo_root}/src/ops/op_common/selector/pub_inc
    ${_repo_root}/src/ops/op_common/topo_info
    ${_repo_root}/src/ops/op_common/algorithm/template
    ${_repo_root}/src/ops/op_common/algorithm/template/registry
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper
    ${_repo_root}/src/ops/op_common/algorithm/template/aiv
    ${_repo_root}/src/ops/op_common/algorithm/executor
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry
    ${_repo_root}/src/ops/op_common/algorithm/executor/channel
    ${_repo_root}/src/ops/op_common/algorithm/topo_match
    ${_repo_root}/src/ops/all_to_all_v/selector
    ${_repo_root}/src/common
    ${_repo_root}/src/common/tuner
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
)

# 前向兼容：探测 CANN 版本号（对齐 selector_engine，避免 dlsym_common.h 桩类型块与公共头冲突）
macro(_detect_cann_version _target)
    if(NOT ASCEND_CANN_PACKAGE_PATH)
        message(FATAL_ERROR "[UT] ASCEND_CANN_PACKAGE_PATH is not set, source CANN set_env.sh first.")
    endif()
    file(GLOB _cann_arch_dirs LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
    if(NOT _cann_arch_dirs)
        message(FATAL_ERROR "[UT] no *-linux directory under ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")
    endif()
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
    if(DEFINED CANN_VERSION_NUM)
        if(NOT "${CANN_VERSION_NUM}" MATCHES "^[1-9][0-9]*$")
            message(FATAL_ERROR "[UT] CANN_VERSION_NUM must be a positive decimal integer, e.g. 90200000.")
        endif()
        target_compile_definitions(${_target} PRIVATE CANN_VERSION_NUM=${CANN_VERSION_NUM})
        message(STATUS "[UT] Using explicit CANN_VERSION_NUM = ${CANN_VERSION_NUM}")
    else()
        set(_ut_cann_found "")
        foreach(_arch_dir ${_cann_arch_dirs})
            set(_ut_cann_ver_header "${_arch_dir}/include/version/cann_version.h")
            if(EXISTS "${_ut_cann_ver_header}")
                file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line
                     REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
                if(_ut_cann_ver_line)
                    string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
                        _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
                    math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
                    target_compile_definitions(${_target} PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
                    message(STATUS "[UT] Detected CANN_VERSION_NUM = ${_UT_CANN_VERSION_NUM}")
                    set(_ut_cann_found "1")
                else()
                    message(FATAL_ERROR "[UT] Failed to parse CANN_VERSION_NUM from ${_ut_cann_ver_header}: "
                        "no '#define CANN_VERSION_NUM' line found. "
                        "dlsym_common.h stub blocks would collide with CANN public headers.")
                endif()
                break()
            endif()
        endforeach()
        if(NOT _ut_cann_found)
            message(FATAL_ERROR "[UT] cann_version.h not found under ${ASCEND_CANN_PACKAGE_PATH}, "
                "pass -DCANN_VERSION_NUM=<SDK version integer>, e.g. 90200000 for 9.2.0.")
        endif()
    endif()
endmacro()

set(ut_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoall_nda_selector_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/selector_nda_deps_stub.cc
    ${_repo_root}/test/ut/selector_engine/stub/log_stub.cc
    ${_repo_root}/test/ut/selector_engine/stub/securec_stub.cc
    ${_repo_root}/test/ut/selector_engine/stub/selector_deps_stub.cc
    ${_repo_root}/src/ops/op_common/selector/auto_selector_base.cc
    ${_repo_root}/src/ops/all_to_all_v/selector/alltoall_auto_selector.cc
    ${_repo_root}/src/ops/all_to_all_v/selector/alltoallv_auto_selector.cc
    ${_repo_root}/src/ops/all_to_all_v/selector/alltoallvc_auto_selector.cc
)

add_executable(alltoall_nda_selector_test ${ut_src_list})

target_compile_options(alltoall_nda_selector_test PRIVATE
    -U_FORTIFY_SOURCE
    -ffunction-sections
    -fdata-sections
    -fno-access-control
)

target_include_directories(alltoall_nda_selector_test PRIVATE ${_selector_ut_include_dirs})

_detect_cann_version(alltoall_nda_selector_test)

target_link_libraries(alltoall_nda_selector_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
    -Wl,--gc-sections
)

run_llt_test(TARGET alltoall_nda_selector_test TASK_NUM 1)
