# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

# UT_FILES 递归收集本目录用例与 stub/ 下的打桩源文件
file(GLOB_RECURSE UT_FILES CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

# 被测源文件：新算法复用现有执行器与模板，须整体编入执行器 TU 的完整注册闭包
# （含同 TU 另外三个注册引用的 ZAxisDetour/NHR 等模板与 TopoMatchThreeLevel）。
set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/common/alg_parse.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector/alg_attrs_registry.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/executor_v2_base.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/alg_v2_template_base.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match/topo_match_base_v2.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match/topo_match_two_level.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match/topo_match_three_level.cc
    ${PROJECT_SOURCE_DIR}/src/ops/reduce/algorithm/executor/reduce_sequence_executor_aicpu_3level.cc
    ${PROJECT_SOURCE_DIR}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.cc
    ${PROJECT_SOURCE_DIR}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_nhr.cc
    ${PROJECT_SOURCE_DIR}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_mesh_1D.cc
    ${PROJECT_SOURCE_DIR}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_aicpu_reduce_nhr_pcie.cc
    ${PROJECT_SOURCE_DIR}/src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_nhr.cc
    ${PROJECT_SOURCE_DIR}/src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_mesh_1D.cc
    ${PROJECT_SOURCE_DIR}/src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_mesh_1D_Z_axis_detour.cc
    # selector 选路用例编入真实选路实现
    ${PROJECT_SOURCE_DIR}/src/ops/reduce/selector/reduce_auto_selector.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector/selector_registry.cc
)

# 探测 CANN 版本号：缺少该宏时 dlsym_common.h 桩类型块与公共头冲突；探测失败跳过本用例、
# 不阻断其余 UT（STATUS+return，与 common/channel_process_link 约定一致）。
# 根路径取环境变量或 -D 缓存变量，再按平台子目录对齐。
set(_ut_cann_ver_root "$ENV{ASCEND_HOME_PATH}")
if(NOT _ut_cann_ver_root AND ASCEND_HOME_PATH)
    set(_ut_cann_ver_root "${ASCEND_HOME_PATH}")
endif()
if(EXISTS "${_ut_cann_ver_root}/x86_64-linux")
    set(_ut_cann_ver_root "${_ut_cann_ver_root}/x86_64-linux")
elseif(EXISTS "${_ut_cann_ver_root}/aarch64-linux")
    set(_ut_cann_ver_root "${_ut_cann_ver_root}/aarch64-linux")
endif()
set(_ut_cann_ver_header "${_ut_cann_ver_root}/include/version/cann_version.h")
set(_ut_cann_ver_ok FALSE)
if(EXISTS "${_ut_cann_ver_header}")
    file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line
         REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
    if(_ut_cann_ver_line)
        string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
            _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
        string(STRIP "${_UT_CANN_VERSION_EXPR}" _UT_CANN_VERSION_EXPR)
        # 仅放行纯算术表达式进入 math(EXPR) 求值，非法字符 STATUS+return 跳过
        if(NOT _UT_CANN_VERSION_EXPR STREQUAL "" AND NOT _UT_CANN_VERSION_EXPR MATCHES "//"
            AND _UT_CANN_VERSION_EXPR MATCHES "^[0-9() \t+*|&^~<>%-]+$")
            math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
            set(_ut_cann_ver_ok TRUE)
        endif()
    endif()
endif()
if(NOT _ut_cann_ver_ok)
    message(STATUS "[UT] reduce_sequence_mesh_nhr_aicpu_reduce_test skip: CANN_VERSION_NUM "
        "undetectable (header missing/unparsed), see ${_ut_cann_ver_header}.")
    return()
endif()

add_executable(reduce_sequence_mesh_nhr_aicpu_reduce_test ${UT_FILES} ${SOURCE_FILES})

# include 根与版本探测共用同一解析（ENV/-D 缓存变量 + 平台后缀自适应），避免两处不同源
set(ASCEND_HOME_PATH "${_ut_cann_ver_root}")

target_include_directories(reduce_sequence_mesh_nhr_aicpu_reduce_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${PROJECT_SOURCE_DIR}/test/ut/common/topo_match
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym/ccu
    ${PROJECT_SOURCE_DIR}/src/common/op_graph
    ${PROJECT_SOURCE_DIR}/src/ops
    ${PROJECT_SOURCE_DIR}/src/ops/op_common
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/inc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/channel
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/registry
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector/inc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/registry
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/aicpu
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/aicpu/task_cache
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/wrapper
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/aiv
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/dpu
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/ccu
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/topo_info
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/topo_match
    ${PROJECT_SOURCE_DIR}/src/ops/interface_graph_mode
    ${PROJECT_SOURCE_DIR}/src/ops/reduce
    ${PROJECT_SOURCE_DIR}/src/ops/reduce/algorithm/executor
    ${PROJECT_SOURCE_DIR}/src/ops/reduce/selector
    ${PROJECT_SOURCE_DIR}/src/ops/reduce/algorithm/template
    ${PROJECT_SOURCE_DIR}/src/ops/reduce/algorithm/template/aicpu
    ${PROJECT_SOURCE_DIR}/src/ops/reduce/algorithm/template/aiv
    ${PROJECT_SOURCE_DIR}/src/ops/reduce/algorithm/template/ccu
    ${PROJECT_SOURCE_DIR}/src/ops/reduce/algorithm/template/ccu/kernel
    ${PROJECT_SOURCE_DIR}/src/ops/reduce_scatter/algorithm/template
    ${PROJECT_SOURCE_DIR}/src/ops/reduce_scatter/algorithm/template/aicpu
    ${PROJECT_SOURCE_DIR}/src/ops/all_gather/algorithm/template/aicpu
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/base
    ${ASCEND_HOME_PATH}/pkg_inc/hccl
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
    ${ASCEND_HOME_PATH}/include/hcomm
    ${ASCEND_HOME_PATH}/include/hcomm/ccu
)

target_compile_options(reduce_sequence_mesh_nhr_aicpu_reduce_test PRIVATE
    -U_FORTIFY_SOURCE
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
)

target_compile_definitions(reduce_sequence_mesh_nhr_aicpu_reduce_test PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
message(STATUS "[UT] reduce_sequence_mesh_nhr_aicpu_reduce_test Detected CANN_VERSION_NUM = ${_UT_CANN_VERSION_NUM}")
target_link_libraries(reduce_sequence_mesh_nhr_aicpu_reduce_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET reduce_sequence_mesh_nhr_aicpu_reduce_test TASK_NUM 1)
