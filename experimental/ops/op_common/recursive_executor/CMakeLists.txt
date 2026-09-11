# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 构建模式说明（标准插件模式，src 侧零修改）：
# RecursiveExecutor 为 OBJECT 库并入 libhccl.so。OXC executor 通过 REGISTER_EXECUTOR_IMPL 注册到 src 的
# CollAlgExecRegistryV2。当前为骨架阶段，REGISTER_ALG 宏尚未实例化调用，调度未接入。

set(RE_CORE_SRC
    algo_desc.cc
    selector/alg_selector.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../all_gather/all_gather.cc
    topo/topo_match_four_level.cc
    executor/adaptor_executor.cc
    executor/ops_executor.cc
    template/aicpu/aicpu_base_template.cc
    template/aicpu/allgather_mesh_template.cc
    template/aicpu/allgather_nhr_template.cc
    template/comm_planners/mesh_comm_planner.cc
    template/comm_planners/nhr_comm_planner.cc
    template/data_ops.cc
    template/data_transfer.cc
)

set(RE_INCLUDE_LIST
    # src 内部路径
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/common/hcomm_dlsym
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/common/hcomm_dlsym/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/ops/op_common/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/ops/op_common/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/ops/op_common/algorithm/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/ops/op_common/algorithm/executor/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/ops/op_common/algorithm/executor/channel
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/ops/op_common/algorithm/topo_match
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/ops/op_common/topo_info
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/ops/op_common/algorithm/template
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/ops/op_common/algorithm/template/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/ops/op_common/algorithm/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/ops/op_common/algorithm/template/wrapper
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/ops/op_common/algorithm/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/ops/op_common
    # recursive_executor 内部路径
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/template
    ${CMAKE_CURRENT_SOURCE_DIR}/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/template/comm_planners
    ${CMAKE_CURRENT_SOURCE_DIR}/topo
    # CANN 外部头文件路径
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/acl
    ${ASCEND_CANN_PACKAGE_PATH}/include/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/base/
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
    ${ASCEND_CANN_PACKAGE_PATH}/include/dump/
    ${ASCEND_CANN_PACKAGE_PATH}/include/external/
    ${ASCEND_CANN_PACKAGE_PATH}/include/platform/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/mmpa
)

# device 构建分支：注入核心源码到 scatter_aicpu_kernel
if(TARGET scatter_aicpu_kernel AND NOT TARGET hccl)
    target_include_directories(scatter_aicpu_kernel PRIVATE ${RE_INCLUDE_LIST})
    target_compile_definitions(scatter_aicpu_kernel PRIVATE ENABLE_EXPERIMENTAL)
    target_sources(scatter_aicpu_kernel PRIVATE ${RE_CORE_SRC})
    return()
endif()

if(NOT TARGET hccl)
    return()
endif()

add_library(RecursiveExecutor OBJECT ${RE_CORE_SRC})
set_target_properties(RecursiveExecutor PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(RecursiveExecutor PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_include_directories(RecursiveExecutor PUBLIC ${RE_INCLUDE_LIST})
target_link_libraries(RecursiveExecutor PUBLIC hccl_compat)

if(TARGET hccl)
    target_link_libraries(hccl PRIVATE RecursiveExecutor)
    target_compile_definitions(hccl PRIVATE ENABLE_EXPERIMENTAL)
endif()

set(_oxc_cann_version_header "${ASCEND_CANN_PACKAGE_PATH}/include/version/cann_version.h")
if(EXISTS "${_oxc_cann_version_header}")
    file(STRINGS "${_oxc_cann_version_header}" _oxc_cann_ver_line
        REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
    if(_oxc_cann_ver_line)
        string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
            _oxc_cann_expr "${_oxc_cann_ver_line}")
        math(EXPR _oxc_cann_version_num "${_oxc_cann_expr}" OUTPUT_FORMAT DECIMAL)
        message(STATUS "OXC scoped CANN_VERSION_NUM = ${_oxc_cann_version_num}")
        target_compile_definitions(RecursiveExecutor PRIVATE CANN_VERSION_NUM=${_oxc_cann_version_num})
    endif()
endif()
