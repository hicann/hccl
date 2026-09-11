# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

# 公共桩文件
set(_ut_stub_src
    ${CMAKE_CURRENT_SOURCE_DIR}/log_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hcomm_stub.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/template_utils.cc
)

# 公共 include 路径
set(_ut_include_dirs
    ${_repo_root}/experimental/ops/op_common/recursive_executor/inc
    ${_repo_root}/experimental/ops/op_common/recursive_executor
    ${_repo_root}/experimental/ops/op_common/recursive_executor/selector
    ${_repo_root}/experimental/ops/op_common/recursive_executor/executor
    ${_repo_root}/experimental/ops/op_common/recursive_executor/template
    ${_repo_root}/experimental/ops/op_common/recursive_executor/template/comm_planners
    ${_repo_root}/experimental/ops/op_common/recursive_executor/template/aicpu
    ${_repo_root}/experimental/ops/op_common/recursive_executor/topo
    ${_repo_root}/experimental/ops/all_gather
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/ops/op_common/selector
    ${_repo_root}/src/ops/op_common/algorithm/topo_match
    ${_repo_root}/src/ops/op_common/topo_info
    ${_repo_root}/src/ops/op_common/algorithm/template
    ${_repo_root}/src/ops/op_common/algorithm/template/registry
    ${_repo_root}/src/ops/op_common/algorithm/template/aicpu
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper
    ${_repo_root}/src/ops/op_common/algorithm/executor
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry
    ${_repo_root}/src/ops/op_common/algorithm/executor/channel
    ${_repo_root}/src/ops/op_common
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
)

# CANN 版本探测宏
macro(_detect_cann_version _target)
    file(GLOB _cann_arch_dirs LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
    if(_cann_arch_dirs)
        # 按当前平台架构选择匹配的目录
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
        target_include_directories(${_target} PRIVATE
            ${_cann_arch_dir}/include
            ${_cann_arch_dir}/include/hccl
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
                endif()
                break()
            endif()
        endforeach()
    endif()
endmacro()

# ---------- oxc_omnipipe_utils_test: omnipipe_utils 纯计算函数 ----------
set(oxc_omnipipe_utils_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/omnipipe_utils_test.cc
    ${_ut_stub_src}
    ${_repo_root}/experimental/ops/op_common/recursive_executor/executor/omnipipe_utils.cc
)

add_executable(oxc_omnipipe_utils_test ${oxc_omnipipe_utils_src_list})

target_compile_options(oxc_omnipipe_utils_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(oxc_omnipipe_utils_test PRIVATE ${_ut_include_dirs})

_detect_cann_version(oxc_omnipipe_utils_test)

target_link_libraries(oxc_omnipipe_utils_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET oxc_omnipipe_utils_test TASK_NUM 1)

# ---------- oxc_data_ops_test: data_ops 纯计算函数 ----------
set(oxc_data_ops_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/data_ops_test.cc
    ${_ut_stub_src}
    ${_repo_root}/experimental/ops/op_common/recursive_executor/template/data_ops.cc
    ${_repo_root}/experimental/ops/op_common/recursive_executor/executor/omnipipe_utils.cc
)

add_executable(oxc_data_ops_test ${oxc_data_ops_src_list})

target_compile_options(oxc_data_ops_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(oxc_data_ops_test PRIVATE ${_ut_include_dirs})

_detect_cann_version(oxc_data_ops_test)

target_link_libraries(oxc_data_ops_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET oxc_data_ops_test TASK_NUM 1)

# ---------- oxc_comm_planner_test: mesh_comm_planner + nhr_comm_planner ----------
set(oxc_comm_planner_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_planner_test.cc
    ${_ut_stub_src}
    ${_repo_root}/experimental/ops/op_common/recursive_executor/template/comm_planners/mesh_comm_planner.cc
    ${_repo_root}/experimental/ops/op_common/recursive_executor/template/comm_planners/nhr_comm_planner.cc
    ${_repo_root}/experimental/ops/op_common/recursive_executor/template/data_ops.cc
    ${_repo_root}/experimental/ops/op_common/recursive_executor/executor/omnipipe_utils.cc
)

add_executable(oxc_comm_planner_test ${oxc_comm_planner_src_list})

target_compile_options(oxc_comm_planner_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(oxc_comm_planner_test PRIVATE ${_ut_include_dirs})

_detect_cann_version(oxc_comm_planner_test)

target_link_libraries(oxc_comm_planner_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET oxc_comm_planner_test TASK_NUM 1)

# ---------- oxc_algo_desc_test: algo_desc 序列化/反序列化 + AlgSelector 注册 ----------
set(oxc_algo_desc_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/algo_desc_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops_executor_stub.cc
    ${_ut_stub_src}
    ${_repo_root}/experimental/ops/op_common/recursive_executor/algo_desc.cc
    ${_repo_root}/experimental/ops/op_common/recursive_executor/selector/alg_selector.cc
    ${_repo_root}/experimental/ops/op_common/recursive_executor/template/data_ops.cc
)

add_executable(oxc_algo_desc_test ${oxc_algo_desc_src_list})

target_compile_options(oxc_algo_desc_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(oxc_algo_desc_test PRIVATE ${_ut_include_dirs})

_detect_cann_version(oxc_algo_desc_test)

target_link_libraries(oxc_algo_desc_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET oxc_algo_desc_test TASK_NUM 1)
