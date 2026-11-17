# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

# alg_rank: GetAlgRank 错误码路径 UT。用例仅构造对象并调用模板私有方法，
# 不执行通信、资源准备与 KernelRun；通过 -fno-access-control 覆盖 private 方法。
set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

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
    ${_repo_root}/src/ops/op_common/selector/inc
    ${_repo_root}/src/ops/op_common/selector/pub_inc
    ${_repo_root}/src/ops/op_common/topo_info
    ${_repo_root}/src/ops/all_reduce/algorithm/template/aicpu
    ${_repo_root}/src/ops/reduce/algorithm/template/aicpu
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
    ${_repo_root}/src/common/tuner
)

# 版本探测前置（在定义target之前）：缺少或无法解析CANN_VERSION_NUM时跳过本用例，
# 避免宏未定义使预处理器按0处理、静默编译出错误版本分支。
if(NOT ASCEND_CANN_PACKAGE_PATH)
    message(STATUS "[UT] alg_rank skipped: ASCEND_CANN_PACKAGE_PATH is not set.")
    return()
endif()

set(_alg_rank_cann_root "${ASCEND_CANN_PACKAGE_PATH}")
if(IS_DIRECTORY "${_alg_rank_cann_root}/${CMAKE_SYSTEM_PROCESSOR}-linux")
    set(_alg_rank_cann_root "${_alg_rank_cann_root}/${CMAKE_SYSTEM_PROCESSOR}-linux")
endif()

set(_alg_rank_cann_ver_header "${_alg_rank_cann_root}/include/version/cann_version.h")
set(_alg_rank_cann_version_line "")
if(EXISTS "${_alg_rank_cann_ver_header}")
    file(STRINGS "${_alg_rank_cann_ver_header}" _alg_rank_cann_version_line
        REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
endif()
if(_alg_rank_cann_version_line STREQUAL "")
    message(STATUS "[UT] alg_rank skipped: no CANN_VERSION_NUM found in ${_alg_rank_cann_ver_header}.")
    return()
endif()

string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
    _alg_rank_cann_version_expr "${_alg_rank_cann_version_line}")
if(NOT _alg_rank_cann_version_expr MATCHES "^[0-9() \t+*/-]+$")
    message(STATUS "[UT] alg_rank skipped: CANN_VERSION_NUM is not a plain arithmetic "
        "expression, expr = '${_alg_rank_cann_version_expr}'.")
    return()
endif()

math(EXPR _alg_rank_cann_version_num "${_alg_rank_cann_version_expr}" OUTPUT_FORMAT DECIMAL)
message(STATUS "[UT] alg_rank detected CANN_VERSION_NUM = ${_alg_rank_cann_version_num}.")

set(_ut_src
    ${CMAKE_CURRENT_SOURCE_DIR}/ut_stub.cc
    ${_repo_root}/src/common/hcomm_dlsym/hcomm_primitives_dl.cc
    ${_repo_root}/src/common/hcomm_dlsym/hccl_sym_win_dl.cc
    ${_repo_root}/src/common/hcomm_dlsym/hccl_dl.cc
    ${_repo_root}/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_aicpu_reduce_nhr.cc
    ${_repo_root}/src/ops/reduce/algorithm/template/aicpu/reduce_aicpu_reduce_nhr.cc
    ${_repo_root}/src/ops/reduce/algorithm/template/aicpu/reduce_nhr.cc
    ${_repo_root}/src/ops/reduce/algorithm/template/aicpu/reduce_mesh_1D.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/alg_v2_template_base.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/template_utils.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc
    ${_repo_root}/src/ops/op_common/exec_timeout_manager.cc
    ${_repo_root}/src/ops/op_common/selector/cost_model.cc
    ${_repo_root}/src/ops/op_common/selector/algo_name_mapper.cc
    ${_repo_root}/src/ops/op_common/selector/alg_attrs_registry.cc
    ${_repo_root}/src/ops/op_common/selector/auto_selector_base.cc
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc
    ${_repo_root}/src/common/alg_parse.cc
)

add_executable(alg_rank_test
    ${CMAKE_CURRENT_SOURCE_DIR}/alg_rank_test.cc
    ${_ut_src}
)

# 仅对 UT 源文件关闭访问控制，避免为了测试 private 方法改动生产代码。
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/alg_rank_test.cc PROPERTIES COMPILE_OPTIONS "-fno-access-control")

target_compile_options(alg_rank_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(alg_rank_test PRIVATE ${_ut_include_dirs})

target_include_directories(alg_rank_test PRIVATE
    ${_alg_rank_cann_root}/include
    ${_alg_rank_cann_root}/include/hccl
    ${_alg_rank_cann_root}/include/hcomm
    ${_alg_rank_cann_root}/include/hcomm/ccu
    ${_alg_rank_cann_root}/pkg_inc
    ${_alg_rank_cann_root}/pkg_inc/hccl
    ${_alg_rank_cann_root}/pkg_inc/base
    ${_alg_rank_cann_root}/pkg_inc/hcomm/ccu
)

target_compile_definitions(alg_rank_test PRIVATE CANN_VERSION_NUM=${_alg_rank_cann_version_num})

target_link_libraries(alg_rank_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
    -ldl
)

run_llt_test(TARGET alg_rank_test TASK_NUM 1)
