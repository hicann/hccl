# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

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
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/omnipipe_template_utils.cc


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
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_3level.cc
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
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/executor/ins_v2_all_gather_sequence_executor_3level.cc
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
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/aicpu/ins_temp_ubx_all_to_all_v_mesh_1D.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_sole_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_parallel_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_sequence_executor_aicpu.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_two_shot_sole_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_order_preserved_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_sequence_executor_aicpu_3level.cc
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
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo_match_3_level.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo_match_squeeze_2d.cc
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
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/barrier/executor/ins_v2_barrier_sole_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/barrier/template/aicpu/ins_temp_barrier_nhr_dpu.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/barrier/template/aicpu/ins_temp_barrier_nhr_aicpu.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/executor/ins_v2_reduce_sequence_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu/ins_temp_gather_dpu_inter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/aicpu/ins_temp_dpu_alltoall_mesh.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_sequence_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_reduce_scatter_mesh_1D_dpu_inter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_all_gather_nhr_dpu_inter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/template/ins_temp_send_dpu.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/template/ins_temp_recv_dpu.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/batch_send_recv/template/ins_temp_batch_send_recv_dpu.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/batch_send_recv/executor/ins_v2_batch_send_recv_sole_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/executor/ins_v2_scatter_sequence_executor.cc
 	  	${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/aicpu/ins_temp_scatter_nhr_dpu_inter_node.cc
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

if(NOT HCCL_CANN_COMPAT_850)
    target_link_libraries(scatter_aicpu_kernel PRIVATE
        -Wl,--no-as-needed
        ccl_kernel
        hccl_kernel_compat
        -Wl,--no-as-needed
    )
else()
    target_link_libraries(scatter_aicpu_kernel PRIVATE
        -Wl,--no-as-needed
        hccl_kernel_compat
        -Wl,--no-as-needed
    )
endif()
add_dependencies(scatter_aicpu_kernel hccl_kernel_compat)


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