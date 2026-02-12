# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_list
    main.cc
    scatter_testcase_a3.cc
    scatter_testcase.cc
    all_reduce_testcase.cc
    all_reduce_parallel_testcase.cc
    alltoall_testcase.cc
    alltoallv_testcase.cc
    alltoallvc_testcase.cc
    all_gather_aicpu_testcase.cc
    all_gather_v_testcase.cc
    reduce_testcase.cc
    send_recv_testcase.cc
    reduce_scatter_v_testcase.cc
    broadcast_testcase.cc
    reduce_scatter_aicpu_testcase.cc
)

add_executable(hccl_checker_ops_stest ${src_list})

if (BUILD_OPEN_PROJECT)
    target_include_directories(hccl_checker_ops_stest PUBLIC
        # 第三方头文件
        ${THIRD_PARTH_PATH}/gtest_shared/include
        ${THIRD_PARTH_PATH}/gtest_shared/include

        # Ascend安装包头文件
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/pkg_inc
        ${ASCEND_HOME_PATH}/pkg_inc/base
        ${ASCEND_HOME_PATH}/hcomm/include/hccl
        ${ASCEND_HOME_PATH}/include/hccl
        ${ASCEND_HOME_PATH}/pkg_inc/hccl
        ${ASCEND_HOME_PATH}/pkg_inc/hcomm/ccu

        # 本仓代码头文件
        ${HCCL_DEV_BASE}/src/common
        ${HCCL_DEV_BASE}/src/ops/op_common/inc

        ${UTILS_DIR}/sim_world
        ${UTILS_DIR}/common/utils
        ${UTILS_DIR}/common/exception
        ${UTILS_DIR}/hccl_verifier
        ${UTILS_DIR}/hccl_verifier/semantics_check
        ${UTILS_DIR}/hccl_verifier/task_graph_generator
        ${UTILS_DIR}/hccl_proxy
        ${UTILS_DIR}/hccl_proxy/communicator
        ${UTILS_DIR}/hccl_proxy/topo_model
    )

    target_link_libraries(hccl_checker_ops_stest PRIVATE
        ${THIRD_PARTH_PATH}/gtest_shared/lib/libgtest.so
        pthread
        hccl-model-proxy
        hccl-verifier
        # 链接hccl库
        ${ASCEND_HOME_PATH}/lib64/libhccl.so
    )

    target_compile_options(hccl_checker_ops_stest PRIVATE
        -std=c++14
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O0
        -g
        -D_GLIBCXX_USE_CXX11_ABI=0
    )
endif()