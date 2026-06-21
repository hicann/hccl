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

set(HCCL_CANN_VERSION_NUM 0)

if(BUILD_OPEN_PROJECT)
    message(STATUS "open TOP_DIR = ${TOP_DIR}")
    set(OPS_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

    # 开源仓编译：从 CANN 包的 version 头解析 CANN_VERSION_NUM
    set(_hccl_cann_version_header "${ASCEND_CANN_PACKAGE_PATH}/include/version/cann_version.h")
    message(STATUS "Checking HCCL scoped CANN version header: ${_hccl_cann_version_header}")
    if(EXISTS "${_hccl_cann_version_header}")
        file(STRINGS "${_hccl_cann_version_header}" _hccl_cann_ver_line
            REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
        if(_hccl_cann_ver_line)
            string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
                CANN_EXPR "${_hccl_cann_ver_line}")
            math(EXPR HCCL_CANN_VERSION_NUM "${CANN_EXPR}" OUTPUT_FORMAT DECIMAL)
        endif()
    endif()
else()
    message(STATUS "close TOP_DIR = ${TOP_DIR}")
    set(OPS_BASE_DIR ${TOP_DIR}/hccl/)
    set(INSTALL_AICPU_KERNEL_JSON_DIR ${INSTALL_LIBRARY_DIR})

    # 大工程编译：CANN 包路径不确定，改为从 hccl 自身的 version.cmake
    # (set_cann_package(hccl VERSION "X.Y.Z")) 解析 CANN_VERSION_NUM
    set(_hccl_version_cmake "${OPS_BASE_DIR}/version.cmake")
    message(STATUS "Checking HCCL version.cmake: ${_hccl_version_cmake}")
    if(EXISTS "${_hccl_version_cmake}")
        file(STRINGS "${_hccl_version_cmake}" _hccl_ver_pkg_line
            REGEX "set_cann_package[ \t]*\\([ \t]*hccl[ \t]+VERSION[ \t]+\"[0-9]+\\.[0-9]+\\.[0-9]+\"")
        if(_hccl_ver_pkg_line)
            string(REGEX MATCH "\"([0-9]+)\\.([0-9]+)\\.([0-9]+)\"" _ "${_hccl_ver_pkg_line}")
            if(NOT CMAKE_MATCH_1 STREQUAL "" AND NOT CMAKE_MATCH_2 STREQUAL "" AND NOT CMAKE_MATCH_3 STREQUAL "")
                math(EXPR HCCL_CANN_VERSION_NUM
                    "${CMAKE_MATCH_1} * 10000000 + ${CMAKE_MATCH_2} * 100000 + ${CMAKE_MATCH_3} * 1000")
            endif()
        endif()
    endif()
endif()

if(HCCL_CANN_VERSION_NUM EQUAL 0)
    message(WARNING "HCCL failed to resolve CANN_VERSION_NUM (got 0). "
        "Compatibility branches (e.g. HCCL_CANN_COMPAT_850) will fall back to the 9.0.0+ path.")
endif()

set(HCCL_CANN_COMPAT_850 OFF)
if(HCCL_CANN_VERSION_NUM GREATER 0 AND HCCL_CANN_VERSION_NUM LESS 90000000)
    set(HCCL_CANN_COMPAT_850 ON)
endif()

message(STATUS "HCCL scoped CANN_VERSION_NUM = ${HCCL_CANN_VERSION_NUM}")
message(STATUS "HCCL_CANN_COMPAT_850 = ${HCCL_CANN_COMPAT_850}")

function(hccl_apply_cann_compat target_name)
    if(NOT TARGET ${target_name})
        return()
    endif()
    target_compile_definitions(${target_name} PRIVATE CANN_VERSION_NUM=${HCCL_CANN_VERSION_NUM})
endfunction()

set(INCLUDE_LIST
    ${OPS_BASE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/common/hcomm_dlsym
    ${CMAKE_CURRENT_SOURCE_DIR}/common/hcomm_dlsym/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/common/op_graph

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
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/dpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/interface_graph_mode

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
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/barrier
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/barrier/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/barrier/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/barrier/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/barrier/template/aicpu

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/template/host_nic

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/template/host_nic

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/batch_send_recv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/batch_send_recv/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/batch_send_recv/selector
)

if(BUILD_OPEN_PROJECT)
    list(APPEND INCLUDE_LIST
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/acl

        # hcomm头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/ccu
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm
        ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm/ccu

        # runtime头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/
        ${ASCEND_CANN_PACKAGE_PATH}/include/base/
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
        ${ASCEND_CANN_PACKAGE_PATH}/include/dump/
        ${ASCEND_CANN_PACKAGE_PATH}/include/external/
        ${ASCEND_CANN_PACKAGE_PATH}/include/platform/

        # driver头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/

        # driver头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/

        # 包间接口
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/dump/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace/
    )
    # pkg_inc/hcomm/ccu 仅 9.0.0 CANN 存在，8.5.0 下直接不加入 include 路径
    if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu)
        list(APPEND INCLUDE_LIST ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu)
    endif()
else()
    list(APPEND INCLUDE_LIST
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/aicpu/tsd
        ${TOP_DIR}/inc/driver
        ${TOP_DIR}/inc/external
        ${TOP_DIR}/inc/external/acl
        ${TOP_DIR}/inc/aicpu/
        ${TOP_DIR}/inc/aicpu/aicpu_schedule/aicpu_sharder
        ${TOP_DIR}/abl/qos/qosmng
        ${TOP_DIR}/abl/atrace/inc/utrace
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/abl/libc_sec/include
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
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
        ${TOP_DIR}/runtime/include/dfx
        ${TOP_DIR}/runtime/pkg_inc
        ${TOP_DIR}/runtime/pkg_inc/base

        ${TOP_DIR}/open_source/json/include

        ${TOP_DIR}/hcomm/inc
        ${TOP_DIR}/hcomm/inc/hccl
        ${TOP_DIR}/hcomm/pkg_inc
        ${TOP_DIR}/hcomm/pkg_inc/hccl
        ${TOP_DIR}/hcomm/pkg_inc/hcomm/ccu
        ${TOP_DIR}/hcomm/include
        ${TOP_DIR}/hcomm/include/hccl
        ${TOP_DIR}/hcomm/include/ccu
    )
endif()

if((DEVICE_MODE AND BUILD_OPEN_PROJECT) OR (NOT BUILD_OPEN_PROJECT))
    add_library(scatter_aicpu_kernel SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/common/utils.cc
        # ${CMAKE_CURRENT_SOURCE_DIR}/common/adapter_acl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/config_log.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/sal.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/log.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/adapter_error_manager_pub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/alg_env_config.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/device_compat.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/omnipipe_data_slice_calc.cc
        
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/exec_timeout_manager.cc
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
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/dpu/kernel_launch.cc


        ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo_match_1d.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo_match_base.cc

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
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/executor/ins_v2_reduce_scatter_order_preserved_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_mesh_1D.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_nhr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_mesh_1D_meshchunk.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_aicpu_reduce_nhr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_omnipipe_mesh_1d_dpu.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_omnipipe_mesh_1D.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_omnipipe_nhr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_order_preserved_level1.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/executor/ins_v2_broadcast_parallel_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/executor/ins_v2_broadcast_sole_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aicpu/ins_temp_broadcast_mesh_1D_two_shot.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aicpu/ins_temp_broadcast_nhr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aicpu/ins_temp_allgather_mesh_1D_intra.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aicpu/ins_temp_scatter_mesh_1D_intra.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/executor/ins_v2_all_gather_sole_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/executor/ins_v2_all_gather_parallel_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/executor/ins_v2_all_gather_sequence_executor_aicpu.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aicpu/ins_temp_all_gather_mesh_1D.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aicpu/ins_temp_all_gather_mesh_1D_Z_axis_detour.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aicpu/ins_temp_all_gather_omnipipe_mesh_1D.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aicpu/ins_temp_all_gather_omnipipe_nhr_dpu.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aicpu/ins_temp_all_gather_omnipipe_nhr.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/ops/barrier/template/aicpu/ins_temp_barrier_mesh_1D.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/executor/reduce_sole_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/executor/reduce_parallel_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu/reduce_mesh_1D.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu/reduce_mesh_1D_two_shot.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu/reduce_nhr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu/reduce_aicpu_reduce_nhr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu/ins_temp_gather_mesh_1D_intra.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/executor/ins_v2_all_gather_v_sole_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template/aicpu/ins_temp_all_gather_v_mesh_1D.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/executor/ins_v2_all_to_all_v_sole_executor.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_sole_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_parallel_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_sequence_executor_aicpu.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_two_shot_sole_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_order_preserved_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_all_reduce_nhr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_all_reduce_aicpu_reduce_nhr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot_mesh_chunk.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_reduce_scatter_mesh_1D_intra.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_all_gather_mesh_1D_intra.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/executor/ins_send_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/executor/ins_recv_executor.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/executor/ins_v2_reduce_scatter_v_sole_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template/aicpu/ins_temp_reduce_scatter_v_mesh_1D.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/registry/alg_v2_template_register.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/batch_send_recv/executor/ins_v2_batch_send_recv_executor.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/executor/ins_v2_recv_sole_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/executor/ins_v2_send_sole_executor.cc

    )

    if(NOT HCCL_CANN_COMPAT_850)
        target_sources(scatter_aicpu_kernel PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/executor/ins_send_dpu_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/executor/ins_recv_dpu_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/template/host_nic/ins_temp_send_host_nic_dpu.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/template/host_nic/ins_temp_recv_host_nic_dpu.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/wrapper/dpu_alg_data_trans_wrapper.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo_match_multilevel.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo_match_ubx.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo_match_ubx_1d.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo_match_pcie_mix.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/executor/ins_reduce_scatter_concurrent_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/executor/ins_v2_all_gather_concurrent_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/executor/ins_v2_all_to_all_concurrent_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_concurrent_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/executor/ins_v2_reduce_scatter_omnipipe_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/executor/ins_v2_all_gather_omnipipe_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_omnipipe_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_mesh_1d_dpu.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/executor/ins_v2_broadcast_sequence_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aicpu/ins_temp_allgather_nhr_dpu_inter.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aicpu/ins_temp_scatter_nhr_dpu_inter.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/executor/ins_v2_all_gather_sequence_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr_dpu.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/barrier/executor/ins_v2_barrier_sequence_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/barrier/template/aicpu/ins_temp_barrier_nhr_dpu.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/executor/ins_v2_reduce_sequence_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu/ins_temp_gather_dpu_inter.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/aicpu/ins_temp_dpu_alltoall_mesh.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_sequence_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_reduce_scatter_mesh_1D_dpu_inter.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_all_gather_nhr_dpu_inter.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/template/ins_temp_send_dpu.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/template/ins_temp_recv_dpu.cc
        )
    endif()

    target_include_directories(scatter_aicpu_kernel PRIVATE
        ${INCLUDE_LIST}
    )

    target_compile_options(scatter_aicpu_kernel PRIVATE
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:-O3>
        -fstack-protector-all
        -Werror
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

    hccl_apply_cann_compat(scatter_aicpu_kernel)

    target_link_directories(scatter_aicpu_kernel PRIVATE
      ${ASCEND_CANN_PACKAGE_PATH}/devlib/device
  )

    target_link_libraries(scatter_aicpu_kernel PRIVATE
        -Wl,--no-as-needed
        ccl_kernel
        hccl_kernel_compat
        -Wl,--no-as-needed
    )
    add_dependencies(scatter_aicpu_kernel hccl_kernel_compat)
endif()

add_definitions(-DORION_MODE)

if(STATIC_MODE)
    add_library(hccl STATIC)
    set_target_properties(hccl PROPERTIES
        OUTPUT_NAME "hccl_static"
        POSITION_INDEPENDENT_CODE ON
    )
else()
    add_library(hccl SHARED)
endif()

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
        DESTINATION ${INSTALL_AICPU_KERNEL_JSON_DIR}/config
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )
    add_dependencies(hccl aicpu_kernel_json)
endif()

set(TARGET_NAME hccl)

set(HCCT_TARGET ${TARGET_NAME})

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        OPEN_BUILD_PROJECT
        $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
    )
else()
    target_compile_definitions(${TARGET_NAME} PRIVATE
        $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
    )
endif()

target_include_directories(${TARGET_NAME} PRIVATE
    ${INCLUDE_LIST}
)

target_compile_definitions(${TARGET_NAME} PRIVATE
    -DHOST_COMPILE
)

hccl_apply_cann_compat(hccl)

target_compile_options(hccl PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-O3 -g>
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
    -fstack-protector-all
)

# libhccl
target_link_directories(${TARGET_NAME} PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

if(NOT STATIC_MODE)
    add_dependencies(${TARGET_NAME} hccl_compat)
endif()

if(BUILD_OPEN_PROJECT)
    target_link_libraries(${TARGET_NAME} PRIVATE
        -Wl,--no-as-needed
        hcomm
        hccl_compat
        acl_rt
        c_sec
        unified_dlog
        -Wl,--no-as-needed
    )
else()
    target_link_libraries(${TARGET_NAME} PRIVATE
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:npu_runtime_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        -Wl,--no-as-needed
        hcomm
        hccl_compat
        acl_rt
        c_sec
        unified_dlog
        -Wl,--no-as-needed
        ofed_headers
    )
endif()

if(NOT STATIC_MODE)
    target_link_options(${TARGET_NAME} PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,-z,noexecstack
        $<$<CONFIG:Release>:-s>
    )
endif()

target_link_directories(${TARGET_NAME} PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

if(STATIC_MODE)
    target_link_libraries(hccl PRIVATE
        hcomm
        acl_rt
        c_sec
        unified_dlog
    )
else()
    if(BUILD_OPEN_PROJECT)
        target_link_libraries(hccl PRIVATE
            -Wl,--no-as-needed
            hcomm
            acl_rt
            c_sec
            unified_dlog
            -Wl,--no-as-needed
        )
    else()
        target_link_libraries(hccl PRIVATE
            $<BUILD_INTERFACE:slog_headers>
            $<BUILD_INTERFACE:msprof_headers>
            $<BUILD_INTERFACE:npu_runtime_headers>
            $<BUILD_INTERFACE:mmpa_headers>
            -Wl,--no-as-needed
            hcomm
            acl_rt
            c_sec
            unified_dlog
            -Wl,--no-as-needed
            ofed_headers
        )
    endif()
endif()

if(STATIC_MODE)
    install(TARGETS hccl
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} 
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )
else()
    install(TARGETS hccl
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} 
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )
endif()

if((DEVICE_MODE AND BUILD_OPEN_PROJECT) OR (NOT BUILD_OPEN_PROJECT))
    if(STATIC_MODE)
        install(TARGETS scatter_aicpu_kernel
            LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} 
            ${INSTALL_OPTIONAL}
            COMPONENT hccl
        )
    else()
        install(TARGETS scatter_aicpu_kernel
            LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} 
            ${INSTALL_OPTIONAL}
            COMPONENT hccl
        )
    endif()
endif()

if(BUILD_OPEN_PROJECT)
    set(_op_proto_link_libs
        -Wl,--no-as-needed
        exe_graph
        graph
        graph_base
        register
        -Wl,--as-needed
    )

    set(OP_PROTO_INCLUDE
        ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
    )

    add_library(opgraph_hccl SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/common/log.cc
    )

    target_include_directories(opgraph_hccl PRIVATE
        ${INCLUDE_LIST}
        ${OP_PROTO_INCLUDE}
    )
    target_compile_options(opgraph_hccl PRIVATE
        -fno-common
        -fno-strict-aliasing
        -pipe
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O3 -g>
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
        -fstack-protector-all
        -fvisibility=hidden
    )
    target_link_libraries(opgraph_hccl PRIVATE
        ${_op_proto_link_libs}
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
        -Wl,-Bsymbolic
    )

    target_link_directories(opgraph_hccl PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )

    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/common/op_graph/ops_proto_hccl.h
        DESTINATION ${INSTALL_OPGRAPH_INCLUDE_DIR} 
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )

    install(TARGETS opgraph_hccl
        LIBRARY DESTINATION ${INSTALL_OPGRAPH_LIBRARY_DIR} 
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )

    add_dependencies(hccl opgraph_hccl)
endif()

add_subdirectory(common)
add_subdirectory(ops)


# 定义AIV算子聚合目标
set(AIV_TARGETS
    hccl_aiv_all_gather_op_910_95
    hccl_aiv_all_reduce_op_910_95
    hccl_aiv_all_to_all_v_op_910_95
    hccl_aiv_all_to_all_op_910_95
    hccl_aiv_broadcast_op_910_95
    hccl_aiv_reduce_op_910_95
    hccl_aiv_reduce_scatter_op_910_95
    hccl_aiv_scatter_op_910_95
    hccl_aiv_send_op_910_95
    hccl_aiv_recv_op_910_95
    hccl_ar_superkernel_mesh_1d_oneshot_int8_v2
    hccl_ar_superkernel_mesh_1d_oneshot_int16_v2
    hccl_ar_superkernel_mesh_1d_oneshot_int32_v2
    hccl_ar_superkernel_mesh_1d_oneshot_float16_v2
    hccl_ar_superkernel_mesh_1d_oneshot_float32_v2
    hccl_ar_superkernel_mesh_1d_oneshot_bfloat16_v2
    hccl_ar_superkernel_mesh_1d_twoshot_int8_v2
    hccl_ar_superkernel_mesh_1d_twoshot_int16_v2
    hccl_ar_superkernel_mesh_1d_twoshot_int32_v2
    hccl_ar_superkernel_mesh_1d_twoshot_float16_v2
    hccl_ar_superkernel_mesh_1d_twoshot_float32_v2
    hccl_ar_superkernel_mesh_1d_twoshot_bfloat16_v2
    hccl_rs_superkernel_mesh_1d_int8_v2
    hccl_rs_superkernel_mesh_1d_int16_v2
    hccl_rs_superkernel_mesh_1d_int32_v2
    hccl_rs_superkernel_mesh_1d_float16_v2
    hccl_rs_superkernel_mesh_1d_float32_v2
    hccl_rs_superkernel_mesh_1d_bfloat16_v2
    hccl_ag_superkernel_mesh_1d_int8_v2
    hccl_ag_superkernel_mesh_1d_int16_v2
    hccl_ag_superkernel_mesh_1d_int32_v2
    hccl_ag_superkernel_mesh_1d_float16_v2
    hccl_ag_superkernel_mesh_1d_float32_v2
    hccl_ag_superkernel_mesh_1d_bfloat16_v2
    hccl_ag_superkernel_mesh_1d_uint8_v2
    hccl_ag_superkernel_mesh_1d_uint16_v2
    hccl_ag_superkernel_mesh_1d_uint32_v2
    hccl_ag_superkernel_mesh_1d_int64_v2
    hccl_ag_superkernel_mesh_1d_uint64_v2
    hccl_a2a_superkernel_mesh_1d_int8_v2
    hccl_a2a_superkernel_mesh_1d_int16_v2
    hccl_a2a_superkernel_mesh_1d_int32_v2
    hccl_a2a_superkernel_mesh_1d_float16_v2
    hccl_a2a_superkernel_mesh_1d_float32_v2
    hccl_a2a_superkernel_mesh_1d_bfloat16_v2
    hccl_a2a_superkernel_mesh_1d_uint8_v2
    hccl_a2a_superkernel_mesh_1d_uint16_v2
    hccl_a2a_superkernel_mesh_1d_uint32_v2
    hccl_a2a_superkernel_mesh_1d_int64_v2
    hccl_a2a_superkernel_mesh_1d_uint64_v2
)

add_custom_target(aiv_all_targets)
foreach(target ${AIV_TARGETS})
    if(TARGET ${target})
        add_dependencies(aiv_all_targets ${target})
    endif()
endforeach()

if(DEVICE_MODE AND KERNEL_MODE)
    set(CCL_KERNEL_TAR_DIR ${OPS_BASE_DIR}/build_device/ccl_kernel_tar_pkg/aicpu_kernels_device)
    add_custom_command(
        TARGET scatter_aicpu_kernel
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CCL_KERNEL_TAR_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:scatter_aicpu_kernel> ${CCL_KERNEL_TAR_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:hccl_kernel_compat> ${CCL_KERNEL_TAR_DIR}
        COMMENT "Copying libscatter_aicpu_kernel.so&libhccl_kernel_compat.so to ${CCL_KERNEL_TAR_DIR}"
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

#生成ES API
if(BUILD_OPEN_PROJECT)
    list(APPEND CMAKE_MODULE_PATH "${ASCEND_CANN_PACKAGE_PATH}/include/ge/cmake")
    find_package(GenerateEsPackage REQUIRED)

    add_es_library(
        ES_LINKABLE_AND_ALL_TARGET es_hccl
        OPP_PROTO_TARGET opgraph_hccl
        OUTPUT_PATH ${CMAKE_BINARY_DIR}/es_output
    )

    add_dependencies(hccl es_hccl)

    install(DIRECTORY ${CMAKE_BINARY_DIR}/es_output/include/es_hccl
        DESTINATION ${INSTALL_INCLUDE_DIR}/es/
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )

    install(FILES ${CMAKE_BINARY_DIR}/es_output/lib64/libes_hccl.so
        DESTINATION ${INSTALL_LIBRARY_DIR}
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )
endif()