# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(hccl)

if(BUILD_OPEN_PROJECT)
    message(STATUS "open TOP_DIR = ${TOP_DIR}")
    set(OPS_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)
else()
    message(STATUS "close TOP_DIR = ${TOP_DIR}")
    set(OPS_BASE_DIR ${TOP_DIR}/hccl/)
    set(INSTALL_AICPU_KERNEL_JSON_DIR ${INSTALL_LIBRARY_DIR})
endif()

set(INCLUDE_LIST
    ${OPS_BASE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/ops
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/channel
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/topo
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/aicpu
)

if(BUILD_OPEN_PROJECT)
    list(APPEND INCLUDE_LIST
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/acl

        # hcomm头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl

        # runtime头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/
        ${ASCEND_CANN_PACKAGE_PATH}/include/base/
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
        ${ASCEND_CANN_PACKAGE_PATH}/include/dump/
        ${ASCEND_CANN_PACKAGE_PATH}/include/external/
        ${ASCEND_CANN_PACKAGE_PATH}/include/platform/

        # driver头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/

        # mmpa头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/mmpa/

        # driver头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/

        # 包间接口
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/dump/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace/
    )
else()
    list(APPEND INCLUDE_LIST
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/aicpu/tsd
        ${TOP_DIR}/ace/npuruntime/runtime/platform/inc
        ${TOP_DIR}/inc/driver
        ${TOP_DIR}/inc/external
        ${TOP_DIR}/inc/external/acl
        ${TOP_DIR}/inc/aicpu/
        ${TOP_DIR}/inc/aicpu/aicpu_schedule/aicpu_sharder
        ${TOP_DIR}/abl/qos/qosmng
        ${TOP_DIR}/abl/atrace/inc/utrace
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/inc/aicpu/common
        ${TOP_DIR}/ace/csruntime/inc/tsch
        ${TOP_DIR}/asl/hss/inc
        ${TOP_DIR}/asl/hss/inc/pub
        ${TOP_DIR}/asl/hss/common/hash_table
        ${TOP_DIR}/metadef
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/metadef/inc
        ${TOP_DIR}/metadef/inc/common
        ${TOP_DIR}/metadef/inc/external

        ${TOP_DIR}/open_source/json/include
    )
endif()

add_library(scatter_aicpu_kernel SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/common/utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/adapter_acl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/config_log.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/sal.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/log.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/adapter_error_manager_pub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/alg_env_config.cc


    ${CMAKE_CURRENT_SOURCE_DIR}/ops/aicpu/kernel_launch.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/registry/alg_template_register.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/registry/coll_alg_exec_registry.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/topo/topo.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/channel/channel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/channel/channel_request.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/executor_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/scatter_comm_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/scatter_executor_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/scatter_mesh_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/scatter_ring_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/scatter_single_executor.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/alg_template_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/nhr_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/scatter_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/scatter_nb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/scatter_nhr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/scatter_ring_direct.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/scatter_ring.cc
)

target_include_directories(scatter_aicpu_kernel PRIVATE
    ${INCLUDE_LIST}
)

target_compile_options(scatter_aicpu_kernel PRIVATE
    -g
    -O0
)

target_compile_definitions(scatter_aicpu_kernel PRIVATE
    -DAICPU_COMPILE
)

target_link_libraries(scatter_aicpu_kernel PRIVATE
    -Wl,--no-as-needed
    ccl_kernel
    -Wl,--no-as-needed
)

add_library(hccl SHARED)

if(NOT KERNEL_MODE)
    # 基于ini生成json文件
    SET(HCCL_CMAKE_DIR ${OPS_BASE_DIR}/cmake/)
    message(STATUS "HCCL_CMAKE_DIR = ${HCCL_CMAKE_DIR}")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json
        COMMAND ${HI_PYTHON} ${HCCL_CMAKE_DIR}/scripts/parser_ini.py ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/scatter_aicpu_kernel.ini ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_custom_target(aicpu_kernel_json DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json
        DESTINATION ${INSTALL_AICPU_KERNEL_JSON_DIR} OPTIONAL
    )
    add_dependencies(hccl aicpu_kernel_json)
endif()

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(hccl PRIVATE
        OPEN_BUILD_PROJECT
        $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
    )
else()
    target_compile_definitions(hccl PRIVATE
        $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
    )
endif()

target_include_directories(hccl PRIVATE
    ${INCLUDE_LIST}
)

target_compile_definitions(hccl PRIVATE
    -DHOST_COMPILE
)

target_compile_options(hccl PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    # -O3
    -g -O0
    -std=c++14
    -fstack-protector-all
)

# libhccl
target_link_options(hccl PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    # -s
)

if(BUILD_OPEN_PROJECT)
    target_link_libraries(hccl PRIVATE
        -Wl,--no-as-needed
        hcomm
        c_sec
        unified_dlog
        mmpa
        -Wl,--no-as-needed
    )
else()
    target_link_libraries(hccl PRIVATE
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        -Wl,--no-as-needed
        hcomm
        c_sec
        unified_dlog
        mmpa
        -Wl,--no-as-needed
        ofed_headers
    )
endif()

install(TARGETS hccl
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
)

add_subdirectory(common)
add_subdirectory(ops)

if(DEVICE_MODE AND KERNEL_MODE)
    set(CCL_KERNEL_TAR_DIR ${OPS_BASE_DIR}/build_device/ccl_kernel_tar_pkg/aicpu_kernels_device)
    add_custom_command(
        TARGET scatter_aicpu_kernel
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CCL_KERNEL_TAR_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:scatter_aicpu_kernel> ${CCL_KERNEL_TAR_DIR}
        COMMENT "Copying libscatter_aicpu_kernel.so to ${CCL_KERNEL_TAR_DIR}"
    )

    set(AICPU_CUSTOM_DIR ${OPS_BASE_DIR}/build_device/aicpu_custom/)
    add_custom_command(
        TARGET scatter_aicpu_kernel
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${AICPU_CUSTOM_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:scatter_aicpu_kernel> ${AICPU_CUSTOM_DIR}/libscatter_aicpu_kernel.so
        COMMENT "Copying libscatter_aicpu_kernel.so to ${AICPU_CUSTOM_DIR}/libscatter_aicpu_kernel.so"
    )
endif()