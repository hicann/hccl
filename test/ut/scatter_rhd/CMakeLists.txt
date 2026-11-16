# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

file(GLOB_RECURSE UT_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

# 被测源文件：Scatter RHD 主体步骤生成（消费 template_utils 公共函数族）；
# 注册/选路 UT 链接 executor 注册 TU 及其最小闭包（真注册表 AlgAttrsRegistry/
# CollAlgExecRegistryV2 + 真过滤缝隙 CheckAlgoMatchTopoWithReason），桩经 UT_FILES 拾取
set(SOURCE_FILES
    # alg_parse.cc 置于最前：其 TU 级静态 map 须先于 executor 注册 lambda 构造
    # （REGISTER_ALG_ATTRS 在 main 前经 ParseAlgName 读 ALGO_TYPES 等表）
    ${PROJECT_SOURCE_DIR}/src/common/alg_parse.cc
    ${PROJECT_SOURCE_DIR}/src/ops/scatter/algorithm/template/rhd_scatter_step.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/template_utils.cc
    ${PROJECT_SOURCE_DIR}/src/ops/scatter/algorithm/executor/scatter_sequence_aicpu_executor_3level.cc
    ${PROJECT_SOURCE_DIR}/src/ops/scatter/algorithm/executor/ins_v2_scatter_sequence_executor.cc
    ${PROJECT_SOURCE_DIR}/src/ops/scatter/algorithm/executor/ins_v2_scatter_sequence_executor_3level_dpu.cc
    ${PROJECT_SOURCE_DIR}/src/ops/scatter/algorithm/template/aicpu/ins_temp_scatter_nhr_dpu_inter_node.cc
    ${PROJECT_SOURCE_DIR}/src/ops/scatter/algorithm/template/aicpu/ins_temp_scatter_rhd_dpu_inter_node.cc
    # master e9bd3c0a 新增 NDA 模板：被本 UT 链接的 executor 注册闭包引用
    ${PROJECT_SOURCE_DIR}/src/ops/scatter/algorithm/template/aicpu/ins_temp_scatter_nhr_nda.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/wrapper/dpu_alg_nhr_opt_wrapper.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/wrapper/dpu_alg_data_trans_wrapper.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/registry/alg_v2_template_register.cc
    ${PROJECT_SOURCE_DIR}/src/ops/scatter/algorithm/template/aicpu/ins_temp_scatter_mesh_1D.cc
    ${PROJECT_SOURCE_DIR}/src/ops/scatter/algorithm/template/aicpu/ins_temp_scatter_nhr.cc
    ${PROJECT_SOURCE_DIR}/src/ops/scatter/algorithm/template/aicpu/ins_temp_scatter_rhd.cc
    ${PROJECT_SOURCE_DIR}/src/ops/scatter/algorithm/template/aicpu/aicpu_temp_scatter_mesh_1D_Z_axis_detour.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector/cost_model.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector/algo_name_mapper.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector/alg_attrs_registry.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector/auto_selector_base.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/executor_base.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/alg_v2_template_base.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/exec_timeout_manager.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/executor_v2_base.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/channel/channel.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match/topo_match_two_level.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match/topo_match_three_level.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match/topo_match_base_v2.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/channel/channel_request.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/ccu_alg_template_base.cc
)
# 版本探测前置（在定义任何 target 之前，先例 channel_process_link：STATUS+return() 不阻断其余 UT）
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT ASCEND_HOME_PATH OR NOT IS_DIRECTORY "${ASCEND_HOME_PATH}")
    message(STATUS "[UT] scatter_rhd ASCEND_HOME_PATH is not set or does not exist: "
        "'${ASCEND_HOME_PATH}' — skipping this target (source CANN set_env.sh to enable, see docs/zh/build/build.md)")
    return()
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/x86_64-linux")
else()
    set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/aarch64-linux")
endif()

add_executable(scatter_rhd_test ${UT_FILES} ${SOURCE_FILES})

target_include_directories(scatter_rhd_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/ops/scatter/algorithm/template
    ${PROJECT_SOURCE_DIR}/src/ops/scatter/algorithm/template/aicpu
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/registry
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/wrapper
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/aicpu
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/aicpu/task_cache
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/aiv
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/dpu
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/ccu
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/channel
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/registry
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/inc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector/inc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector/pub_inc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/topo_info
    ${PROJECT_SOURCE_DIR}/src/ops
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/common/tuner
    ${PROJECT_SOURCE_DIR}/src/common/op_graph
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym/ccu
    ${PROJECT_SOURCE_DIR}/include
    "${ASCEND_HOME_PATH}/pkg_inc"
    "${ASCEND_HOME_PATH}/pkg_inc/base"
    "${ASCEND_HOME_PATH}/pkg_inc/hccl"
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/include/hccl"
)

target_compile_options(scatter_rhd_test PRIVATE
    -U_FORTIFY_SOURCE
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
)

# 前向兼容：探测 CANN 版本号，与主构建/ST 保持一致。缺少该宏时 dlsym_common.h
# 中的桩类型块会启用，与 CANN 公共头重复定义导致 conflicting declaration。
set(_ut_cann_ver_header "${ASCEND_HOME_PATH}/include/version/cann_version.h")
if(EXISTS "${_ut_cann_ver_header}")
    file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line
         REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
    if(_ut_cann_ver_line)
        string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
            _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
        math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
        target_compile_definitions(scatter_rhd_test PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
        message(STATUS "[UT] scatter_rhd Detected CANN_VERSION_NUM = ${_UT_CANN_VERSION_NUM}")
    else()
        message(FATAL_ERROR "[UT] Failed to parse CANN_VERSION_NUM from ${_ut_cann_ver_header}: "
            "no '#define CANN_VERSION_NUM' line found.")
    endif()
else()
    message(WARNING "[UT] cann_version.h not found at ${_ut_cann_ver_header}, CANN_VERSION_NUM left undefined.")
endif()

target_link_libraries(scatter_rhd_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
    -lgcov
    -lrt
)

run_llt_test(TARGET scatter_rhd_test TASK_NUM 1)
