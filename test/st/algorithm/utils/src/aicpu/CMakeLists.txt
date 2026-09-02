# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(INCLUDE_LIST
    ${HCCL_DEV_BASE}/include
    ${HCCL_DEV_BASE}/src/common
    ${HCCL_DEV_BASE}/src/common/hcomm_dlsym
    ${HCCL_DEV_BASE}/src/common/hcomm_dlsym/ccu

    ${HCCL_DEV_BASE}/src/ops
    ${HCCL_DEV_BASE}/src/ops/op_common
    ${HCCL_DEV_BASE}/src/ops/op_common/inc

    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/executor
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/executor/channel
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/executor/registry

    ${HCCL_DEV_BASE}/src/ops/op_common/selector

    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/registry
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/aicpu/task_cache
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/wrapper
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/aiv
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/ccu

    ${HCCL_DEV_BASE}/src/ops/op_common/topo_info
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/topo_match

    ${HCCL_DEV_BASE}/src/ops/reduce_scatter
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/executor
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/selector
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/template
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/template/aiv
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/template/ccu
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/template/ccu/kernel

    ${HCCL_DEV_BASE}/src/ops/broadcast
    ${HCCL_DEV_BASE}/src/ops/broadcast/algorithm/executor
    ${HCCL_DEV_BASE}/src/ops/broadcast/selector
    ${HCCL_DEV_BASE}/src/ops/broadcast/algorithm/template
    ${HCCL_DEV_BASE}/src/ops/broadcast/algorithm/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/broadcast/algorithm/template/ccu
    ${HCCL_DEV_BASE}/src/ops/broadcast/algorithm/template/ccu/kernel
    ${HCCL_DEV_BASE}/src/ops/broadcast/algorithm/template/aiv

    ${HCCL_DEV_BASE}/src/ops/reduce_scatter_v
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter_v/algorithm/executor
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter_v/selector
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter_v/algorithm/template
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter_v/algorithm/template/aicpu

    ${HCCL_DEV_BASE}/src/ops/scatter
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/template
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/executor
    ${HCCL_DEV_BASE}/src/ops/scatter/selector
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/template/ccu
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/template/ccu/kernel
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/template/aiv

    ${HCCL_DEV_BASE}/src/ops/all_gather
    ${HCCL_DEV_BASE}/src/ops/all_gather/algorithm/executor
    ${HCCL_DEV_BASE}/src/ops/all_gather/selector
    ${HCCL_DEV_BASE}/src/ops/all_gather/algorithm/template
    ${HCCL_DEV_BASE}/src/ops/all_gather/algorithm/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/all_gather/algorithm/template/ccu
    ${HCCL_DEV_BASE}/src/ops/all_gather/algorithm/template/ccu/kernel
    ${HCCL_DEV_BASE}/src/ops/all_gather/algorithm/template/aiv

    ${HCCL_DEV_BASE}/src/ops/all_gather_v
    ${HCCL_DEV_BASE}/src/ops/all_gather_v/algorithm/executor
    ${HCCL_DEV_BASE}/src/ops/all_gather_v/selector
    ${HCCL_DEV_BASE}/src/ops/all_gather_v/algorithm/template
    ${HCCL_DEV_BASE}/src/ops/all_gather_v/algorithm/template/aicpu

    ${HCCL_DEV_BASE}/src/ops/all_to_all_v
    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/algorithm/executor
    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/selector
    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/algorithm/template
    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/algorithm/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/algorithm/template/ccu
    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/algorithm/template/aiv

    ${HCCL_DEV_BASE}/src/ops/all_reduce
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/executor
    ${HCCL_DEV_BASE}/src/ops/all_reduce/selector
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/template
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/template/ccu
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/template/aiv

    ${HCCL_DEV_BASE}/src/ops/send
    ${HCCL_DEV_BASE}/src/ops/send/algorithm/executor
    ${HCCL_DEV_BASE}/src/ops/send/selector
    ${HCCL_DEV_BASE}/src/ops/send/algorithm/template

    ${HCCL_DEV_BASE}/src/ops/recv
    ${HCCL_DEV_BASE}/src/ops/recv/algorithm/executor
    ${HCCL_DEV_BASE}/src/ops/recv/selector
    ${HCCL_DEV_BASE}/src/ops/recv/algorithm/template

    ${HCCL_DEV_BASE}/src/ops/reduce
    ${HCCL_DEV_BASE}/src/ops/reduce/algorithm/executor
    ${HCCL_DEV_BASE}/src/ops/reduce/selector
    ${HCCL_DEV_BASE}/src/ops/reduce/algorithm/template
    ${HCCL_DEV_BASE}/src/ops/reduce/algorithm/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/reduce/algorithm/template/ccu
    ${HCCL_DEV_BASE}/src/ops/reduce/algorithm/template/aiv

    ${HCCL_DEV_BASE}/src/ops/batch_send_recv
    ${HCCL_DEV_BASE}/src/ops/batch_send_recv/algorithm/executor
    ${HCCL_DEV_BASE}/src/ops/batch_send_recv/selector
    ${HCCL_DEV_BASE}/src/common/hcomm_dlsym
)

list(APPEND INCLUDE_LIST
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/acl

    # hcomm头文件
    ${ASCEND_HOME_PATH}/include/hccl
    ${ASCEND_HOME_PATH}/include/hcomm
    ${ASCEND_HOME_PATH}/include/hcomm/ccu

    # runtime头文件
    ${ASCEND_HOME_PATH}/include/
    ${ASCEND_HOME_PATH}/include/base/
    ${ASCEND_HOME_PATH}/include/dump/
    ${ASCEND_HOME_PATH}/include/external/
    ${ASCEND_HOME_PATH}/include/platform/

    # 包间接口
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/hccl
    ${ASCEND_HOME_PATH}/pkg_inc/hcomm/ccu
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/
    ${ASCEND_HOME_PATH}/pkg_inc/aicpu/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/base/
    ${ASCEND_HOME_PATH}/pkg_inc/dump/
    ${ASCEND_HOME_PATH}/pkg_inc/trace/
)

add_library(scatter_aicpu_kernel SHARED
    ${HCCL_DEV_BASE}/src/common/utils.cc
    # ${HCCL_DEV_BASE}/src/common/adapter_acl.cc
    ${HCCL_DEV_BASE}/src/common/config_log.cc
    ${HCCL_DEV_BASE}/src/common/sal.cc
    ${HCCL_DEV_BASE}/src/common/log.cc
    ${HCCL_DEV_BASE}/src/common/dev_type.cc
    ${HCCL_DEV_BASE}/src/common/adapter_error_manager_pub.cc
    ${HCCL_DEV_BASE}/src/common/alg_env_config.cc
    ${HCCL_DEV_BASE}/src/common/hcomm_dlsym/hcomm_primitives_dl.cc

    ${HCCL_DEV_BASE}/src/ops/op_common/exec_timeout_manager.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/executor/channel/channel.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/executor/channel/channel_request.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/executor/registry/coll_alg_exec_registry.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/executor/executor_base.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/executor/executor_v2_base.cc

    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/alg_template_base.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/alg_v2_template_base.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/template_utils.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/aicpu/dfx/task_exception_fun.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/aicpu/task_cache/aicpu_task_cache_comm_manager.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/aicpu/task_cache/aicpu_task_cache_key.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/aicpu/task_cache/aicpu_task_cache_policy.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/aicpu/task_cache/aicpu_task_cache_utils.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/registry/alg_template_register.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/registry/alg_v2_template_register.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/template/wrapper/dpu_alg_data_trans_wrapper.cc

    ${HCCL_DEV_BASE}/src/ops/op_common/topo_info/topo.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/topo_match/topo_match_1d.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/topo_match/topo_match_base.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/topo_match/topo_match_multilevel.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/topo_match/topo_match_ubx.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/algorithm/topo_match/topo_match_pcie_mix.cc

    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/scatter_comm_executor.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/scatter_executor_base.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/scatter_mesh_executor.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/scatter_ring_executor.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/scatter_single_executor.cc

    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/template/nhr_base.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/template/scatter_mesh.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/template/scatter_nb.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/template/scatter_nhr.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/template/scatter_ring_direct.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/template/scatter_ring.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/executor/ins_v2_scatter_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/executor/ins_v2_scatter_parallel_executor.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/executor/scatter_sequence_aicpu_executor_3level.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/template/aicpu/ins_temp_scatter_mesh_1D.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/template/aicpu/ins_temp_scatter_nhr.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algorithm/template/aicpu/aicpu_temp_scatter_mesh_1D_Z_axis_detour.cc

    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/executor/ins_v2_reduce_scatter_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/executor/ins_reduce_scatter_parallel_executor.cc
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/executor/ins_v2_reduce_scatter_sequence_executor.cc
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/executor/ins_v2_reduce_scatter_sequence_executor_aicpu.cc
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/executor/ins_v2_reduce_scatter_sequence_executor_3level.cc
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_mesh_1D.cc
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_nhr.cc
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_mesh_1D_meshchunk.cc
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.cc
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_mesh_1d_dpu.cc

    ${HCCL_DEV_BASE}/src/ops/all_gather/algorithm/executor/ins_v2_all_gather_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_gather/algorithm/executor/ins_v2_all_gather_parallel_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_gather/algorithm/executor/ins_v2_all_gather_sequence_executor_3level.cc
    ${HCCL_DEV_BASE}/src/ops/all_gather/algorithm/executor/ins_v2_all_gather_sequence_executor_aicpu.cc
    ${HCCL_DEV_BASE}/src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_mesh_1D.cc
    ${HCCL_DEV_BASE}/src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_nhr.cc
    ${HCCL_DEV_BASE}/src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_mesh_1D_Z_axis_detour.cc
    ${HCCL_DEV_BASE}/src/ops/all_gather/algorithm/executor/ins_v2_all_gather_sequence_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_nhr_dpu.cc

    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_gather_mesh_1D_intra.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_gather_nhr_dpu_inter.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_reduce_scatter_mesh_1D_dpu_inter.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_reduce_scatter_mesh_1D_intra.cc

    ${HCCL_DEV_BASE}/src/ops/all_gather_v/algorithm/executor/ins_v2_all_gather_v_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_gather_v/algorithm/template/aicpu/ins_temp_all_gather_v_mesh_1D.cc

    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/algorithm/executor/ins_v2_all_to_all_v_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc
    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_dpu_alltoall_mesh.cc

    ${HCCL_DEV_BASE}/src/ops/send/algorithm/executor/ins_v2_send_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/send/algorithm/executor/ins_send_executor.cc
    ${HCCL_DEV_BASE}/src/ops/send/algorithm/template/ins_temp_send_dpu.cc

    ${HCCL_DEV_BASE}/src/ops/recv/algorithm/executor/ins_v2_recv_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/recv/algorithm/executor/ins_recv_executor.cc
    ${HCCL_DEV_BASE}/src/ops/recv/algorithm/template/ins_temp_recv_dpu.cc

    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_parallel_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_sequence_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_sequence_executor_aicpu.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_sequence_executor_aicpu_3level.cc
    
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_two_shot_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_nhr.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_aicpu_reduce_nhr.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot_mesh_chunk.cc

    ${HCCL_DEV_BASE}/src/ops/broadcast/algorithm/executor/ins_v2_broadcast_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/broadcast/algorithm/executor/ins_v2_broadcast_parallel_executor.cc
    ${HCCL_DEV_BASE}/src/ops/broadcast/algorithm/template/aicpu/ins_temp_broadcast_mesh_1D_two_shot.cc
    ${HCCL_DEV_BASE}/src/ops/broadcast/algorithm/template/aicpu/ins_temp_broadcast_nhr.cc

    ${HCCL_DEV_BASE}/src/ops/batch_send_recv/algorithm/executor/ins_v2_batch_send_recv_executor.cc

    ${HCCL_DEV_BASE}/src/ops/reduce/algorithm/executor/reduce_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/reduce/algorithm/executor/reduce_parallel_executor.cc
    ${HCCL_DEV_BASE}/src/ops/reduce/algorithm/executor/reduce_sequence_executor_aicpu_3level.cc
    ${HCCL_DEV_BASE}/src/ops/reduce/algorithm/template/aicpu/reduce_mesh_1D.cc
    ${HCCL_DEV_BASE}/src/ops/reduce/algorithm/template/aicpu/reduce_nhr.cc
    ${HCCL_DEV_BASE}/src/ops/reduce/algorithm/template/aicpu/reduce_aicpu_reduce_nhr.cc
)

target_include_directories(scatter_aicpu_kernel PRIVATE
    ${INCLUDE_LIST}
)

target_compile_options(scatter_aicpu_kernel PRIVATE
    -g
    -O0
    -fstack-protector-all
    -D_GLIBCXX_USE_CXX11_ABI=0
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
    $<BUILD_INTERFACE:intf_llt_pub>
)