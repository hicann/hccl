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
    ${OPS_BASE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common

    ${CMAKE_CURRENT_SOURCE_DIR}/ops
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/inc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/channel
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/registry

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/selector

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/wrapper
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/aiv
    # ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/hostdpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aiv

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template/aicpu

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/aiv

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aiv

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template/aicpu

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/aiv

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aiv

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/executor

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/executor

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aiv
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
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/mmpa/

        # driver头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/

        # 包间接口
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
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
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/abl/mmpa/inc/mmpa
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
        ${TOP_DIR}/runtime/pkg_inc/base

        ${TOP_DIR}/open_source/json/include

        ${TOP_DIR}/hcomm/inc
        ${TOP_DIR}/hcomm/inc/hccl
        ${TOP_DIR}/hcomm/pkg_inc
        ${TOP_DIR}/hcomm/pkg_inc/hccl
        ${TOP_DIR}/hcomm/pkg_inc/hcomm/ccu
        ${TOP_DIR}/hcomm/include
        ${TOP_DIR}/hcomm/include/hccl
    )
endif()

add_library(scatter_aicpu_kernel SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/common/utils.cc
    # ${CMAKE_CURRENT_SOURCE_DIR}/common/adapter_acl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/config_log.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/sal.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/log.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/adapter_error_manager_pub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/alg_env_config.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/channel/channel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/channel/channel_request.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/registry/coll_alg_exec_registry.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/registry/coll_alg_v2_exec_registry.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/executor_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/executor_v2_base.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/alg_template_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/alg_v2_template_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/template_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/aicpu/kernel_launch.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/aicpu/dfx/task_exception_fun.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/registry/alg_template_register.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc


    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo_match_1d.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo_match_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo_match_2d.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo_match_multilevel.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/scatter_comm_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/scatter_executor_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/scatter_mesh_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/scatter_ring_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/scatter_single_executor.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/nhr_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/scatter_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/scatter_nb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/scatter_nhr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/scatter_ring_direct.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/scatter_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/executor/ins_v2_scatter_sole_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/executor/ins_v2_scatter_parallel_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/aicpu/ins_temp_scatter_mesh_1D.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/aicpu/ins_temp_scatter_nhr.cc


    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sole_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/executor/ins_reduce_scatter_parallel_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_mesh_1D.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_nhr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_mesh_1D_meshchunk.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/executor/ins_v2_broadcast_parallel_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/executor/ins_v2_broadcast_sole_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aicpu/ins_temp_broadcast_mesh_1D_two_shot.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aicpu/ins_temp_broadcast_nhr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_mesh_1d_dpu.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/executor/ins_v2_all_gather_sole_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/executor/ins_v2_all_gather_parallel_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aicpu/ins_temp_all_gather_mesh_1D.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/executor/ins_v2_all_gather_sequence_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr_dpu.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/executor/ins_v2_reduce_sole_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/executor/ins_v2_reduce_parallel_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu/ins_temp_reduce_mesh_1D.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu/ins_temp_reduce_nhr.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/executor/ins_v2_all_gather_v_sole_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template/aicpu/ins_temp_all_gather_v_mesh_1D.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/executor/ins_v2_all_to_all_v_sole_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_sole_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_all_reduce_parallel_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_all_reduce_nhr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot_mesh_chunk.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/executor/ins_send_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/executor/ins_recv_executor.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/executor/ins_v2_reduce_scatter_v_sole_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template/aicpu/ins_temp_reduce_scatter_v_mesh_1D.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/executor/ins_v2_reduce_sole_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/executor/ins_v2_reduce_parallel_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu/ins_temp_reduce_mesh_1D.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu/ins_temp_reduce_nhr.cc
    
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/registry/alg_v2_template_register.cc
)

target_include_directories(scatter_aicpu_kernel PRIVATE
    ${INCLUDE_LIST}
)

target_compile_options(scatter_aicpu_kernel PRIVATE
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O3>
    -fstack-protector-all
)

target_link_options(scatter_aicpu_kernel PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-s>
)

target_compile_definitions(scatter_aicpu_kernel PRIVATE
    -DAICPU_COMPILE
)

target_link_libraries(scatter_aicpu_kernel PRIVATE
    -Wl,--no-as-needed
    ccl_kernel
    -Wl,--no-as-needed
)

add_definitions(-DORION_MODE)

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
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g>
    -std=c++14
    -fstack-protector-all
)

# libhccl
target_link_options(hccl PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-s>
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

install(TARGETS scatter_aicpu_kernel
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