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
    ${HCCL_DEV_BASE}/include
    ${HCCL_DEV_BASE}/src/common
    ${HCCL_DEV_BASE}/src/ops
    ${HCCL_DEV_BASE}/src/ops/inc
    ${HCCL_DEV_BASE}/src/ops/channel
    ${HCCL_DEV_BASE}/src/ops/registry
    ${HCCL_DEV_BASE}/src/ops/scatter
    ${HCCL_DEV_BASE}/src/ops/scatter/algo
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/template
    ${HCCL_DEV_BASE}/src/ops/topo
    ${HCCL_DEV_BASE}/src/ops/aicpu
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
    ${HCCL_DEV_BASE}/src/common/adapter_acl.cc
    ${HCCL_DEV_BASE}/src/common/config_log.cc
    ${HCCL_DEV_BASE}/src/common/sal.cc
    ${HCCL_DEV_BASE}/src/common/log.cc
    ${HCCL_DEV_BASE}/src/common/adapter_error_manager_pub.cc
    ${HCCL_DEV_BASE}/src/common/alg_env_config.cc


    ${HCCL_DEV_BASE}/src/ops/aicpu/kernel_launch.cc
    ${HCCL_DEV_BASE}/src/ops/aicpu/dfx/task_exception_fun.cc
    ${HCCL_DEV_BASE}/src/ops/registry/alg_template_register.cc
    ${HCCL_DEV_BASE}/src/ops/registry/coll_alg_exec_registry.cc

    ${HCCL_DEV_BASE}/src/ops/topo/topo.cc
    ${HCCL_DEV_BASE}/src/ops/channel/channel.cc
    ${HCCL_DEV_BASE}/src/ops/channel/channel_request.cc

    ${HCCL_DEV_BASE}/src/ops/executor_base.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/scatter_comm_executor.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/scatter_executor_base.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/scatter_mesh_executor.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/scatter_ring_executor.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/scatter_single_executor.cc

    ${HCCL_DEV_BASE}/src/ops/alg_template_base.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/template/nhr_base.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/template/scatter_mesh.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/template/scatter_nb.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/template/scatter_nhr.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/template/scatter_ring_direct.cc
    ${HCCL_DEV_BASE}/src/ops/scatter/algo/template/scatter_ring.cc
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