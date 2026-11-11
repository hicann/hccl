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

# 被测头文件依赖 CANN 版本宏；头文件缺失或版本号无法解析时跳过本用例，不阻断其他 UT。
set(_wb_cann_root "${ASCEND_CANN_PACKAGE_PATH}")
if(IS_DIRECTORY "${_wb_cann_root}/${CMAKE_SYSTEM_PROCESSOR}-linux")
    set(_wb_cann_root "${_wb_cann_root}/${CMAKE_SYSTEM_PROCESSOR}-linux")
endif()
set(_wb_cann_version_header "${_wb_cann_root}/include/version/cann_version.h")
if(NOT EXISTS "${_wb_cann_version_header}")
    message(STATUS "[UT] white_box_guards skipped: ${_wb_cann_version_header} not found.")
    return()
endif()
file(STRINGS "${_wb_cann_version_header}" _wb_cann_version_line
    REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
if(NOT _wb_cann_version_line)
    message(STATUS "[UT] white_box_guards skipped: CANN_VERSION_NUM not found.")
    return()
endif()
string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
    _wb_cann_version_expr "${_wb_cann_version_line}")
if(NOT _wb_cann_version_expr MATCHES "^[0-9() \t+*/-]+$")
    message(STATUS "[UT] white_box_guards skipped: unrecognized CANN_VERSION_NUM expression.")
    return()
endif()
math(EXPR _wb_cann_version_num "${_wb_cann_version_expr}" OUTPUT_FORMAT DECIMAL)

file(GLOB_RECURSE HCCL_HEADERS CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/common/*.h" "${PROJECT_SOURCE_DIR}/src/common/*.hpp"
    "${PROJECT_SOURCE_DIR}/src/ops/*.h" "${PROJECT_SOURCE_DIR}/src/ops/*.hpp")
set(HCCL_INCLUDE_DIRS
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src/common"
    "${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym"
    "${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym/ccu"
    "${PROJECT_SOURCE_DIR}/src/ops/op_common/inc")
foreach(header IN LISTS HCCL_HEADERS)
    get_filename_component(header_dir "${header}" DIRECTORY)
    list(APPEND HCCL_INCLUDE_DIRS "${header_dir}")
endforeach()
list(REMOVE_DUPLICATES HCCL_INCLUDE_DIRS)

set(_wb_cann_include_dirs
    "${_wb_cann_root}/include" "${_wb_cann_root}/include/hccl" "${_wb_cann_root}/include/hcomm"
    "${_wb_cann_root}/include/hcomm/ccu" "${_wb_cann_root}/include/acl"
    "${_wb_cann_root}/include/acl/error_codes" "${_wb_cann_root}/include/mmpa"
    "${_wb_cann_root}/include/mmpa/sub_inc"
    "${_wb_cann_root}/pkg_inc" "${_wb_cann_root}/pkg_inc/base" "${_wb_cann_root}/pkg_inc/hccl"
    "${_wb_cann_root}/pkg_inc/hcomm" "${_wb_cann_root}/pkg_inc/hcomm/ccu"
    "${_wb_cann_root}/pkg_inc/runtime" "${_wb_cann_root}/pkg_inc/runtime/runtime"
    "${_wb_cann_root}/pkg_inc/runtime/runtime/rts" "${_wb_cann_root}/pkg_inc/aicpu"
    "${_wb_cann_root}/pkg_inc/aicpu/common" "${_wb_cann_root}/pkg_inc/toolchain"
    "${_wb_cann_root}/pkg_inc/profiling")

# AlgResourceCtxSerializable / BinaryStream 的防御性校验：越界、截断、尾随数据、流读取失败。
add_executable(alg_param_deserialize_test
    alg_param_deserialize_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../topo_match/stub/log_stub.cc
)
target_include_directories(alg_param_deserialize_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${HCCL_INCLUDE_DIRS})
target_include_directories(alg_param_deserialize_test SYSTEM PRIVATE ${_wb_cann_include_dirs})
target_compile_definitions(alg_param_deserialize_test PRIVATE CANN_VERSION_NUM=${_wb_cann_version_num})
target_compile_options(alg_param_deserialize_test PRIVATE -U_FORTIFY_SOURCE)
target_link_libraries(alg_param_deserialize_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread)
run_llt_test(TARGET alg_param_deserialize_test TASK_NUM 1)

# 图模式资源计算 CalcTaskNum 的 maxDataCountPerLoop 零值守卫与正常路径。
add_executable(graph_mode_calc_test
    graph_mode_calc_test.cc
    stub/param_check_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../topo_match/stub/log_stub.cc
    ${PROJECT_SOURCE_DIR}/src/ops/interface_graph_mode/calc_resource_graph_mode.cc
)
target_compile_definitions(graph_mode_calc_test PRIVATE
    HOST_COMPILE ORION_MODE OPEN_BUILD_PROJECT CANN_VERSION_NUM=${_wb_cann_version_num})
# calc_resource_graph_mode.cc 还包含其他引擎路径，按函数裁剪未调用的依赖。
target_compile_options(graph_mode_calc_test PRIVATE
    -O0 -g -ffunction-sections -fdata-sections -fno-inline -U_FORTIFY_SOURCE)
target_link_options(graph_mode_calc_test PRIVATE -Wl,--gc-sections)
target_include_directories(graph_mode_calc_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/stub"
    ${HCCL_INCLUDE_DIRS})
target_include_directories(graph_mode_calc_test SYSTEM PRIVATE ${_wb_cann_include_dirs})
target_link_libraries(graph_mode_calc_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread)
run_llt_test(TARGET graph_mode_calc_test TASK_NUM 1)

# Send/Recv 执行器零搬运与非法 dataType 守卫。
# ascend_hal 探测前置：未命中时跳过本用例，不阻断其余 UT。
find_library(ASCEND_HAL_LIBRARY ascend_hal HINTS
    "${ASCEND_CANN_PACKAGE_PATH}/devlib/linux/${CMAKE_SYSTEM_PROCESSOR}"
    "${_wb_cann_root}/devlib/linux/${CMAKE_SYSTEM_PROCESSOR}"
    "${_wb_cann_root}/lib64")
if(NOT ASCEND_HAL_LIBRARY)
    message(STATUS "[UT] white_box_guards executor skipped: ascend_hal not found.")
    return()
endif()
get_filename_component(ASCEND_HAL_DIR "${ASCEND_HAL_LIBRARY}" DIRECTORY)

# 链接顺序要求：alg_parse.cc 等基础源在前，执行器源在后。执行器内注册表的静态初始化
# 会回调 alg_parse 的全局表，顺序颠倒会在 main 前因静态初始化顺序而为空指针。
set(_wb_executor_src
    ${PROJECT_SOURCE_DIR}/src/common/alg_parse.cc
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym/hcomm_dlsym.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector/alg_attrs_registry.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector/cost_model.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/selector/cost_table.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/exec_timeout_manager.cc
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym/hccl_sym_win_dl.cc
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym/hccl_host_comm_dl.cc
    ${PROJECT_SOURCE_DIR}/src/common/hcomm_dlsym/hcomm_primitives_dl.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/executor_v2_base.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/alg_v2_template_base.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/registry/alg_v2_template_register.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/wrapper/dpu_alg_data_trans_wrapper.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/channel/channel.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/executor/channel/channel_request.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/template_utils.cc
    ${PROJECT_SOURCE_DIR}/src/ops/send/algorithm/executor/ins_send_executor.cc
    ${PROJECT_SOURCE_DIR}/src/ops/recv/algorithm/executor/ins_recv_executor.cc
    ${PROJECT_SOURCE_DIR}/src/ops/send/algorithm/executor/ins_v2_send_sole_executor.cc
    ${PROJECT_SOURCE_DIR}/src/ops/send/algorithm/template/ins_temp_send_dpu.cc
    ${PROJECT_SOURCE_DIR}/src/ops/recv/algorithm/executor/ins_v2_recv_sole_executor.cc
    ${PROJECT_SOURCE_DIR}/src/ops/recv/algorithm/template/ins_temp_recv_dpu.cc
)
add_executable(executor_guards_test
    executor_guards_test.cc
    ${_wb_executor_src}
    ${CMAKE_CURRENT_SOURCE_DIR}/../topo_match/stub/log_stub.cc
)
target_compile_definitions(executor_guards_test PRIVATE
    HOST_COMPILE ORION_MODE OPEN_BUILD_PROJECT CANN_VERSION_NUM=${_wb_cann_version_num})
# 执行器源包含其他引擎路径，按函数裁剪未调用的依赖。
target_compile_options(executor_guards_test PRIVATE
    -O0 -g -ffunction-sections -fdata-sections -fno-inline -U_FORTIFY_SOURCE)
target_link_options(executor_guards_test PRIVATE -Wl,--gc-sections)
target_include_directories(executor_guards_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${HCCL_INCLUDE_DIRS})
target_include_directories(executor_guards_test SYSTEM PRIVATE ${_wb_cann_include_dirs})
target_link_directories(executor_guards_test PRIVATE "${_wb_cann_root}/lib64")
set_target_properties(executor_guards_test PROPERTIES
    BUILD_RPATH "${_wb_cann_root}/lib64;${ASCEND_HAL_DIR}")
target_link_libraries(executor_guards_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
    hcomm acl_rt c_sec unified_dlog runtime mmpa error_manager "${ASCEND_HAL_LIBRARY}")
run_llt_test(TARGET executor_guards_test TASK_NUM 1)
