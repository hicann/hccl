# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(INCLUDE_LIST
    ${HCCL_DEV_BASE}/src/common
    ${HCCL_DEV_BASE}/src/common

    ${HCCL_DEV_BASE}/src/ops
    ${HCCL_DEV_BASE}/src/ops/op_common
    ${HCCL_DEV_BASE}/src/ops/op_common/inc

    ${HCCL_DEV_BASE}/src/ops/op_common/executor
    ${HCCL_DEV_BASE}/src/ops/op_common/executor/channel
    ${HCCL_DEV_BASE}/src/ops/op_common/executor/registry

    ${HCCL_DEV_BASE}/src/ops/op_common/selector

    ${HCCL_DEV_BASE}/src/ops/op_common/template
    ${HCCL_DEV_BASE}/src/ops/op_common/template/registry
    ${HCCL_DEV_BASE}/src/ops/op_common/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/op_common/template/wrapper
    ${HCCL_DEV_BASE}/src/ops/op_common/template/aiv
    # ${HCCL_DEV_BASE}/src/ops/op_common/template/hostdpu
    ${HCCL_DEV_BASE}/src/ops/op_common/template/ccu
    ${HCCL_DEV_BASE}/src/ops/op_common/template/ccu/kernel

    ${HCCL_DEV_BASE}/src/ops/op_common/topo


    ${HCCL_DEV_BASE}/src/ops/reduce_scatter
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/executor
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/selector
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/template
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/template/aiv
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/template/ccu
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/template/ccu/kernel

    ${HCCL_DEV_BASE}/src/ops/broadcast
    ${HCCL_DEV_BASE}/src/ops/broadcast/executor
    ${HCCL_DEV_BASE}/src/ops/broadcast/selector
    ${HCCL_DEV_BASE}/src/ops/broadcast/template
    ${HCCL_DEV_BASE}/src/ops/broadcast/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/broadcast/template/aiv

    ${HCCL_DEV_BASE}/src/ops/reduce_scatter_v
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter_v/executor
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter_v/selector
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter_v/template
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter_v/template/aicpu

    ${HCCL_DEV_BASE}/src/ops/scatter
    ${HCCL_DEV_BASE}/src/ops/scatter/algo
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/template
    ${HCCL_DEV_BASE}/src/ops/scatter/executor
    ${HCCL_DEV_BASE}/src/ops/scatter/selector
    ${HCCL_DEV_BASE}/src/ops/scatter/template
    ${HCCL_DEV_BASE}/src/ops/scatter/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/scatter/template/aiv

    ${HCCL_DEV_BASE}/src/ops/all_gather
    ${HCCL_DEV_BASE}/src/ops/all_gather/executor
    ${HCCL_DEV_BASE}/src/ops/all_gather/selector
    ${HCCL_DEV_BASE}/src/ops/all_gather/template
    ${HCCL_DEV_BASE}/src/ops/all_gather/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/all_gather/template/aiv

    ${HCCL_DEV_BASE}/src/ops/all_gather_v
    ${HCCL_DEV_BASE}/src/ops/all_gather_v/executor
    ${HCCL_DEV_BASE}/src/ops/all_gather_v/selector
    ${HCCL_DEV_BASE}/src/ops/all_gather_v/template
    ${HCCL_DEV_BASE}/src/ops/all_gather_v/template/aicpu

    ${HCCL_DEV_BASE}/src/ops/all_to_all_v
    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/executor
    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/selector
    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/template
    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/template/aiv

    ${HCCL_DEV_BASE}/src/ops/all_reduce
    ${HCCL_DEV_BASE}/src/ops/all_reduce/executor
    ${HCCL_DEV_BASE}/src/ops/all_reduce/selector
    ${HCCL_DEV_BASE}/src/ops/all_reduce/template
    ${HCCL_DEV_BASE}/src/ops/all_reduce/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/all_reduce/template/aiv

    ${HCCL_DEV_BASE}/src/ops/send
    ${HCCL_DEV_BASE}/src/ops/send/executor

    ${HCCL_DEV_BASE}/src/ops/recv
    ${HCCL_DEV_BASE}/src/ops/recv/executor

    ${HCCL_DEV_BASE}/src/ops/reduce
    ${HCCL_DEV_BASE}/src/ops/reduce/executor
    ${HCCL_DEV_BASE}/src/ops/reduce/selector
    ${HCCL_DEV_BASE}/src/ops/reduce/template
    ${HCCL_DEV_BASE}/src/ops/reduce/template/aicpu
    ${HCCL_DEV_BASE}/src/ops/reduce/template/aiv

)

list(APPEND INCLUDE_LIST
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/acl

    # hcomm头文件
    ${ASCEND_HOME_PATH}/include/hccl

    # runtime头文件
    ${ASCEND_HOME_PATH}/include/
    ${ASCEND_HOME_PATH}/include/base/
    ${ASCEND_HOME_PATH}/include/driver/
    ${ASCEND_HOME_PATH}/include/dump/
    ${ASCEND_HOME_PATH}/include/external/
    ${ASCEND_HOME_PATH}/include/platform/

    # driver头文件
    ${ASCEND_HOME_PATH}/include/driver/

    # mmpa头文件
    ${ASCEND_HOME_PATH}/pkg_inc/mmpa/

    # driver头文件
    ${ASCEND_HOME_PATH}/include/driver/

    # 包间接口
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/hccl
    ${ASCEND_HOME_PATH}/pkg_inc/hcomm/ccu
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/
    ${ASCEND_HOME_PATH}/pkg_inc/aicpu/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime/
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
    ${HCCL_DEV_BASE}/src/common/adapter_error_manager_pub.cc
    ${HCCL_DEV_BASE}/src/common/alg_env_config.cc

    ${HCCL_DEV_BASE}/src/ops/op_common/executor/channel/channel.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/executor/channel/channel_request.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/executor/registry/coll_alg_exec_registry.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/executor/registry/coll_alg_v2_exec_registry.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/executor/executor_base.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/executor/executor_v2_base.cc

    ${HCCL_DEV_BASE}/src/ops/op_common/template/alg_template_base.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/template/alg_v2_template_base.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/template/template_utils.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/template/aicpu/kernel_launch.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/template/aicpu/dfx/task_exception_fun.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/template/registry/alg_template_register.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc

    ${HCCL_DEV_BASE}/src/ops/op_common/topo/topo.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/topo/topo_match_1d.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/topo/topo_match_base.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/topo/topo_match_2d.cc
    ${HCCL_DEV_BASE}/src/ops/op_common/topo/topo_match_multilevel.cc

    ${HCCL_DEV_BASE}/src/ops/scatter/algo/scatter_comm_executor.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/scatter_executor_base.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/scatter_mesh_executor.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/scatter_ring_executor.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/scatter_single_executor.cc

    ${HCCL_DEV_BASE}/src/ops/scatter/algo/template/nhr_base.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/template/scatter_mesh.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/template/scatter_nb.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/template/scatter_nhr.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/template/scatter_ring_direct.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/template/scatter_ring.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/executor/ins_v2_scatter_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/executor/ins_v2_scatter_parallel_executor.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/template/aicpu/ins_temp_scatter_mesh_1D.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/template/aicpu/ins_temp_scatter_nhr.cc

    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/executor/ins_reduce_scatter_parallel_executor.cc
    # ${HCCL_DEV_BASE}/src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor.cc
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_mesh_1D.cc
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_nhr.cc
    ${HCCL_DEV_BASE}/src/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_mesh_1D_meshchunk.cc
    # ${HCCL_DEV_BASE}/src/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_nhr_dpu.cc

    ${HCCL_DEV_BASE}/src/ops/all_gather/executor/ins_v2_all_gather_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_gather/executor/ins_v2_all_gather_parallel_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_gather/template/aicpu/ins_temp_all_gather_mesh_1D.cc
    ${HCCL_DEV_BASE}/src/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr.cc
    ${HCCL_DEV_BASE}/src/ops/all_gather/executor/ins_v2_all_gather_sequence_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr_dpu.cc

    ${HCCL_DEV_BASE}/src/ops/all_gather_v/executor/ins_v2_all_gather_v_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_gather_v/template/aicpu/ins_temp_all_gather_v_mesh_1D.cc

    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/executor/ins_v2_all_to_all_v_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_to_all_v/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc

    ${HCCL_DEV_BASE}/src/ops/send/executor/ins_send_executor.cc
    ${HCCL_DEV_BASE}/src/ops/recv/executor/ins_recv_executor.cc

    ${HCCL_DEV_BASE}/src/ops/all_reduce/executor/ins_v2_all_reduce_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/executor/ins_all_reduce_parallel_executor.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/template/aicpu/ins_temp_all_reduce_nhr.cc
    ${HCCL_DEV_BASE}/src/ops/all_reduce/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot_mesh_chunk.cc
    
    ${HCCL_DEV_BASE}/src/ops/broadcast/executor/ins_v2_broadcast_sole_executor.cc
    ${HCCL_DEV_BASE}/src/ops/broadcast/executor/ins_v2_broadcast_parallel_executor.cc
    ${HCCL_DEV_BASE}/src/ops/broadcast/template/aicpu/ins_temp_broadcast_mesh_1D_two_shot.cc
    ${HCCL_DEV_BASE}/src/ops/broadcast/template/aicpu/ins_temp_broadcast_nhr.cc
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
    -Wl,--no-as-needed
)