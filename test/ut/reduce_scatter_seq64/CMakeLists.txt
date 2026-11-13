# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# reduce_scatter_seq64: RS/AR AicpuReduce 模板 cost 建模 UT（轻量目标，不链 executor/注册闭包）。
# 仅编入 4 个被测模板 + cost 框架的静态方法链；HCOMM 原语以真实 dlsym 层闭合链接，
# log/tuner/环境配置/securec 等以 ut_stub.cc 桩提供，残余未定义符号时按其注释追加桩定义。

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

# 公共 include 路径（cost 闭包按传递解析实际用到的 18 个目录）
set(_ut_include_dirs
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/ops/op_common
    ${_repo_root}/src/ops/op_common/algorithm/template
    ${_repo_root}/src/ops/op_common/algorithm/template/registry
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper
    ${_repo_root}/src/ops/op_common/algorithm/executor
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry
    ${_repo_root}/src/ops/op_common/algorithm/executor/channel
    ${_repo_root}/src/ops/op_common/selector
    ${_repo_root}/src/ops/op_common/selector/pub_inc
    ${_repo_root}/src/ops/op_common/topo_info
    ${_repo_root}/src/ops/reduce_scatter/algorithm/template
    ${_repo_root}/src/ops/reduce_scatter/algorithm/template/aicpu
    ${_repo_root}/src/ops/all_reduce/algorithm/template/aicpu
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
    ${_repo_root}/src/common/tuner
)

# CANN 版本探测宏（与 all_reduce_biar/recursive_executor 保持一致）：
# 缺少该宏时 dlsym_common.h 的桩类型块会启用，与 CANN 公共头重复定义导致 conflicting declaration。
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
                    message(STATUS "[UT] Detected CANN_VERSION_NUM = ${_UT_CANN_VERSION_NUM}")
                else()
                    message(FATAL_ERROR "[UT] Failed to parse CANN_VERSION_NUM from ${_ut_cann_ver_header}: "
                        "no '#define CANN_VERSION_NUM' line found.")
                endif()
                break()
            endif()
        endforeach()
    endif()
endmacro()

# 桩 + dlsym 层 + 被测产品源码（HCOMM/symwin 原语定义齐全，被测路径不会真正调用）
set(_ut_src
    ${CMAKE_CURRENT_SOURCE_DIR}/ut_stub.cc
    ${_repo_root}/src/common/hcomm_dlsym/hcomm_primitives_dl.cc
    ${_repo_root}/src/common/hcomm_dlsym/hccl_sym_win_dl.cc
    ${_repo_root}/src/common/hcomm_dlsym/hccl_dl.cc
    # 被测: 4 个模板 + cost 框架
    ${_repo_root}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_aicpu_reduce_nhr.cc
    ${_repo_root}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_aicpu_reduce_nhr_pcie.cc
    ${_repo_root}/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_aicpu_reduce_nhr.cc
    ${_repo_root}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_mesh_1D.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/alg_v2_template_base.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/template_utils.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc
    ${_repo_root}/src/ops/op_common/exec_timeout_manager.cc
    ${_repo_root}/src/ops/op_common/selector/algo_name_mapper.cc
    ${_repo_root}/src/ops/op_common/selector/cost_model.cc
    ${_repo_root}/src/ops/op_common/selector/alg_attrs_registry.cc
    ${_repo_root}/src/ops/op_common/selector/auto_selector_base.cc
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc
    ${_repo_root}/src/common/alg_parse.cc
)

# ---------- reduce_scatter_seq64_test: 模板 cost 建模（A/B/C/D 期望值 + HostReduceSlope 门控） ----------
add_executable(reduce_scatter_seq64_test
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_seq64_test.cc
    ${_ut_src}
)

target_compile_options(reduce_scatter_seq64_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(reduce_scatter_seq64_test PRIVATE ${_ut_include_dirs})

_detect_cann_version(reduce_scatter_seq64_test)

target_link_libraries(reduce_scatter_seq64_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
    -ldl
)

run_llt_test(TARGET reduce_scatter_seq64_test TASK_NUM 1)
