# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

add_executable(reduce_scatter_nhr_dpu_test
    test_nhr_geometry.cc
    test_nhr_serialization.cc
    test_nhr_layout.cc
    test_nhr_guard.cc
    test_nhr_cost.cc
    nhr_test_stubs.cc
    log_stub.cc
    ${_repo_root}/src/ops/op_common/exec_timeout_manager.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/template_utils.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/alg_v2_template_base.cc
    ${_repo_root}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_nhr_dpu.cc
)

target_compile_options(reduce_scatter_nhr_dpu_test PRIVATE
    -U_FORTIFY_SOURCE
    -ffunction-sections
    -fdata-sections
    -fno-access-control
)

# RunNHR 依赖 DPU wrapper/参数校验，模板单测只覆盖 host 侧几何与寻址逻辑
target_compile_definitions(reduce_scatter_nhr_dpu_test PRIVATE AICPU_COMPILE)

target_include_directories(reduce_scatter_nhr_dpu_test PRIVATE
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/ops/op_common
    ${_repo_root}/src/ops/op_common/selector
    ${_repo_root}/src/ops/op_common/topo_info
    ${_repo_root}/src/ops/op_common/algorithm/template
    ${_repo_root}/src/ops/op_common/algorithm/template/registry
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper
    ${_repo_root}/src/ops/op_common/algorithm/executor
    ${_repo_root}/src/ops/op_common/algorithm/executor/channel
    ${_repo_root}/src/ops/reduce_scatter/algorithm/template/aicpu
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
)

file(GLOB _cann_arch_dirs LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
if(_cann_arch_dirs)
    # 按当前平台架构选择匹配的目录，避免多架构包时探测到别的架构的头文件
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
    target_include_directories(reduce_scatter_nhr_dpu_test PRIVATE
        ${_cann_arch_dir}/include
        ${_cann_arch_dir}/include/hccl
        ${_cann_arch_dir}/include/hcomm/ccu
        ${_cann_arch_dir}/pkg_inc
        ${_cann_arch_dir}/pkg_inc/hccl
        ${_cann_arch_dir}/pkg_inc/base
        ${_cann_arch_dir}/pkg_inc/hcomm/ccu
    )
    set(_ut_cann_ver_header "${_cann_arch_dir}/include/version/cann_version.h")
    if(EXISTS "${_ut_cann_ver_header}")
        file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
        if(_ut_cann_ver_line)
            string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
                _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
            math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
            target_compile_definitions(reduce_scatter_nhr_dpu_test PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
        endif()
    endif()
endif()

# RHD DPU inter 模板与 guard 用例仅在 >= 9.0 段编译（对齐同 PR ST 侧
# test/st/algorithm/utils/src/aicpu/CMakeLists.txt 的 _ST_CANN_VERSION_NUM 门控）：
# 该源依赖新版 CANN 才提供的 DPU wrapper 接口，低版本包上无条件编译会整体红
if(_UT_CANN_VERSION_NUM GREATER_EQUAL 90000000)
    target_sources(reduce_scatter_nhr_dpu_test PRIVATE
        test_rhd_dpu_inter_guard.cc
        ${_repo_root}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_rhd_dpu_inter.cc
    )
endif()

target_link_libraries(reduce_scatter_nhr_dpu_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
    -Wl,--gc-sections
)

run_llt_test(TARGET reduce_scatter_nhr_dpu_test TASK_NUM 1)
