# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

set(_includes ${CMAKE_CURRENT_SOURCE_DIR} ${_repo_root}/include ${_repo_root}/src/ops
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/ops/op_common
    ${_repo_root}/src/ops/op_common/selector
    ${_repo_root}/src/ops/op_common/selector/pub_inc
    ${_repo_root}/src/ops/op_common/topo_info
    ${_repo_root}/src/ops/op_common/algorithm/template
    ${_repo_root}/src/ops/op_common/algorithm/template/registry
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper
    ${_repo_root}/src/ops/op_common/algorithm/executor
    ${_repo_root}/src/ops/op_common/algorithm/executor/channel
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry
    ${_repo_root}/src/ops/op_common/algorithm/topo_match
    ${_repo_root}/src/ops/op_common/algorithm/template/aiv
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
    ${_repo_root}/src/common/tuner
    ${_repo_root}/src/ops/all_reduce/selector
    ${_repo_root}/src/ops/all_reduce/algorithm/executor
    ${_repo_root}/src/ops/all_reduce/algorithm/template/aicpu
    ${_repo_root}/src/ops/reduce/selector
    ${_repo_root}/src/ops/reduce/algorithm/executor
    ${_repo_root}/src/ops/reduce/algorithm/template/aicpu
    ${_repo_root}/src/ops/reduce_scatter/selector
    ${_repo_root}/src/ops/reduce_scatter/algorithm/executor
    ${_repo_root}/src/ops/reduce_scatter/algorithm/template/aicpu
)
set(_cann "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux")
list(APPEND _includes ${_cann}/include ${_cann}/include/hccl ${_cann}/include/acl
    ${_cann}/include/hcomm/ccu ${_cann}/pkg_inc ${_cann}/pkg_inc/hccl
    ${_cann}/pkg_inc/base ${_cann}/pkg_inc/hcomm/ccu)
set(_cann_version_header "${_cann}/include/version/cann_version.h")
if(NOT EXISTS "${_cann_version_header}")
    message(FATAL_ERROR "CANN version header not found: ${_cann_version_header}. Check ASCEND_CANN_PACKAGE_PATH.")
endif()
file(STRINGS "${_cann_version_header}" _version_line REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
if(NOT _version_line)
    message(FATAL_ERROR "CANN_VERSION_NUM is missing or empty in ${_cann_version_header}.")
endif()
string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1" _version_expr "${_version_line}")
string(STRIP "${_version_expr}" _version_expr)
if(_version_expr STREQUAL "")
    message(FATAL_ERROR "CANN_VERSION_NUM is missing or empty in ${_cann_version_header}.")
elseif(NOT _version_expr MATCHES "^[0-9a-fA-FxX() \t+*/%<>&|^~-]+$")
    message(FATAL_ERROR "Invalid CANN_VERSION_NUM in ${_cann_version_header}: ${_version_expr}")
endif()
math(EXPR _version "${_version_expr}")

function(configure_inline_reduce_ut target)
    target_include_directories(${target} PRIVATE ${_includes})
    target_compile_definitions(${target} PRIVATE CANN_VERSION_NUM=${_version})
    target_compile_options(${target} PRIVATE -U_FORTIFY_SOURCE -ffunction-sections -fdata-sections -fno-access-control)
    target_link_libraries(${target} PRIVATE $<BUILD_INTERFACE:intf_llt_pub> GTest::gtest_main
        ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so -lpthread)
    target_link_options(${target} PRIVATE -Wl,--gc-sections)
endfunction()

add_library(inline_reduce_registration OBJECT
    ${_repo_root}/src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_sequence_executor.cc
    ${_repo_root}/src/ops/reduce/algorithm/executor/ins_v2_reduce_sequence_executor.cc
    ${_repo_root}/src/ops/reduce_scatter/algorithm/executor/ins_v2_reduce_scatter_sequence_executor.cc)
configure_inline_reduce_ut(inline_reduce_registration)
target_compile_options(inline_reduce_registration PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/registration_stub.h)

add_library(inline_reduce_selector_support OBJECT
    selector_stubs.cc log_stub.cc
    ${_repo_root}/src/common/alg_parse.cc
    ${_repo_root}/src/common/hcomm_dlsym/hccl_dl.cc
    ${_repo_root}/src/common/hcomm_dlsym/hcomm_primitives_dl.cc
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc
    ${_repo_root}/src/ops/op_common/selector/alg_attrs_registry.cc
    ${_repo_root}/src/ops/op_common/selector/selector_registry.cc
    ${_repo_root}/src/ops/op_common/selector/dpu_inline_reduce_selector.cc
    ${_repo_root}/src/ops/op_common/selector/cost_model.cc
    ${_repo_root}/src/ops/op_common/selector/cost_table.cc
    ${_repo_root}/src/ops/op_common/selector/selector_engine.cc
    ${_repo_root}/src/ops/op_common/selector/auto_selector_base.cc
    ${_repo_root}/src/ops/all_reduce/selector/all_reduce_auto_selector.cc
    ${_repo_root}/src/ops/reduce/selector/reduce_auto_selector.cc
    ${_repo_root}/src/ops/reduce_scatter/selector/reduce_scatter_auto_selector.cc)
configure_inline_reduce_ut(inline_reduce_selector_support)

foreach(_part legacy_selector selector_engine)
    add_executable(dpu_${_part}_test ${_part}_test.cc
        $<TARGET_OBJECTS:inline_reduce_selector_support> $<TARGET_OBJECTS:inline_reduce_registration>)
    configure_inline_reduce_ut(dpu_${_part}_test)
    run_llt_test(TARGET dpu_${_part}_test TASK_NUM 1)
endforeach()

add_executable(dpu_wrapper_test dpu_wrapper_test.cc nhr_host_handoff_test.cc wrapper_stubs.cc log_stub.cc
    ${_repo_root}/src/ops/op_common/exec_timeout_manager.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/alg_v2_template_base.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/template_utils.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/registry/alg_v2_template_register.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper/dpu_alg_nhr_opt_wrapper.cc
    ${_repo_root}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_nhr_dpu.cc)
configure_inline_reduce_ut(dpu_wrapper_test)
run_llt_test(TARGET dpu_wrapper_test TASK_NUM 1)
