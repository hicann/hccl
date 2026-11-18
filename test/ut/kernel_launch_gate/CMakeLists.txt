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
set(_src ${_repo_root}/src)
set(_tpl_aicpu ${_src}/ops/op_common/algorithm/template/aicpu)

# kernel_launch.cc 不在此列：测试文件以源包含方式引入（static 判定函数与匿名命名空间守卫需纳入测试 TU）
set(_gate_closure_src
    ${_src}/common/utils.cc
    ${_src}/common/dev_type.cc
    ${_src}/common/hcomm_dlsym/hccl_dl.cc
    ${_src}/common/hcomm_dlsym/hcomm_primitives_dl.cc
    ${_src}/common/hcomm_dlsym/hccl_device_comm_dl.cc
    ${_src}/common/hcomm_dlsym/hccl_sym_win_dl.cc
    ${_src}/common/hcomm_dlsym/hcomm_diag_dl.cc
    ${_src}/common/hcomm_dlsym/hcomm_device_profiling_dl.cc
    ${_src}/ops/op_common/exec_timeout_manager.cc
    ${_src}/ops/op_common/algorithm/executor/channel/channel.cc
    ${_src}/ops/op_common/algorithm/executor/channel/channel_request.cc
    ${_src}/ops/op_common/algorithm/executor/executor_base.cc
    ${_src}/ops/op_common/algorithm/executor/executor_v2_base.cc
    ${_src}/ops/op_common/algorithm/executor/registry/coll_alg_exec_registry.cc
    ${_src}/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc
    ${_src}/ops/op_common/algorithm/template/template_utils.cc
    ${_tpl_aicpu}/dfx/task_exception_fun.cc
    ${_tpl_aicpu}/task_cache/aicpu_task_cache_key.cc
    ${_tpl_aicpu}/task_cache/aicpu_task_cache_policy.cc
    ${_tpl_aicpu}/task_cache/aicpu_task_cache_utils.cc
    ${_tpl_aicpu}/task_cache/aicpu_task_cache_comm_manager.cc
    ${_src}/ops/op_common/topo_info/topo.cc
    ${_src}/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc
)

set(_gate_include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${_src}/common
    ${_src}/common/hcomm_dlsym
    ${_src}/common/hcomm_dlsym/ccu
    ${_src}/ops
    ${_src}/ops/op_common
    ${_src}/ops/op_common/inc
    ${_src}/ops/op_common/algorithm/executor
    ${_src}/ops/op_common/algorithm/executor/channel
    ${_src}/ops/op_common/algorithm/executor/registry
    ${_src}/ops/op_common/selector
    ${_src}/ops/op_common/selector/inc
    ${_src}/ops/op_common/algorithm/template
    ${_src}/ops/op_common/algorithm/template/registry
    ${_src}/ops/op_common/algorithm/template/aicpu
    ${_src}/ops/op_common/algorithm/template/aicpu/task_cache
    ${_src}/ops/op_common/algorithm/template/wrapper
    ${_src}/ops/op_common/algorithm/template/aiv
    ${_src}/ops/op_common/algorithm/template/ccu
    ${_src}/ops/op_common/topo_info
    ${_src}/ops/op_common/algorithm/topo_match
    ${_src}/ops/send/algorithm/executor
    ${_src}/ops/recv/algorithm/executor
)

file(GLOB _cann_arch_dir LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
set(_cann_ver_num "")
if(_cann_arch_dir)
    list(GET _cann_arch_dir 0 _cann_arch_dir)
    set(_cann_ver_header "${_cann_arch_dir}/include/version/cann_version.h")
    if(EXISTS "${_cann_ver_header}")
        file(STRINGS "${_cann_ver_header}" _cann_ver_line REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
        if(_cann_ver_line)
            string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
                _cann_ver_expr "${_cann_ver_line}")
            math(EXPR _cann_ver_num "${_cann_ver_expr}" OUTPUT_FORMAT DECIMAL)
        endif()
    endif()
endif()

# 目标1：带框架接口桩（取值可控）—— 覆盖判定边界（含 +1 门限）、无线程短路、守卫全部行为
add_executable(kernel_launch_gate_test
    kernel_launch_gate_test.cc
    stub/aicpu_cpu_ctrl_stub.cc
    stub/hcomm_gate_stub.cc
    stub/log_gate_stub.cc
    stub/securec_stub.cc
    ${_gate_closure_src}
)

# 目标2：不链接框架接口桩 —— 弱符号恒 null，专测降级路径与 WARNING 仅一次
add_executable(kernel_launch_gate_null_test
    kernel_launch_gate_test.cc
    stub/hcomm_gate_stub.cc
    stub/log_gate_stub.cc
    stub/securec_stub.cc
    ${_gate_closure_src}
)

foreach(_tgt kernel_launch_gate_test kernel_launch_gate_null_test)
    target_include_directories(${_tgt} PRIVATE ${_gate_include_dirs})
    target_compile_options(${_tgt} PRIVATE -U_FORTIFY_SOURCE)
    target_compile_definitions(${_tgt} PRIVATE AICPU_COMPILE)
    if(_tgt STREQUAL "kernel_launch_gate_test")
        target_compile_definitions(${_tgt} PRIVATE UT_WITH_CPU_CTRL_STUB)
    endif()
    if(_cann_arch_dir)
        target_include_directories(${_tgt} PRIVATE
            ${_cann_arch_dir}/include
            ${_cann_arch_dir}/include/acl
            ${_cann_arch_dir}/include/hccl
            ${_cann_arch_dir}/pkg_inc
            ${_cann_arch_dir}/pkg_inc/base
            ${_cann_arch_dir}/pkg_inc/hccl
            ${_cann_arch_dir}/pkg_inc/hcomm/ccu
        )
    endif()
    if(NOT "${_cann_ver_num}" STREQUAL "")
        target_compile_definitions(${_tgt} PRIVATE CANN_VERSION_NUM=${_cann_ver_num})
    endif()
    target_link_libraries(${_tgt} PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub>
        GTest::gtest_main
        -lpthread
    )
    run_llt_test(TARGET ${_tgt} TASK_NUM 1)
endforeach()
