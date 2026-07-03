# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Default: build all if ST_TASKS is not set or contains "all"
if(NOT DEFINED ST_TASKS)
    set(ST_TASKS "all")
endif()

# Function to create a per-operator ST test executable
function(add_st_test OP_NAME OP_SRC)
    set(target "st_${OP_NAME}_test")
    add_executable(${target} main.cc ${OP_SRC})

    if (BUILD_OPEN_PROJECT)
        target_include_directories(${target} PUBLIC
            # 第三方头文件
            ${THIRD_PARTH_PATH}/gtest_shared/include

            # HCCL仓头文件
            ${HCCL_DEV_BASE}/src/ops/op_common/inc

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
            ${HCCL_DEV_BASE}/src/ops
            ${HCCL_DEV_BASE}/src/ops/op_common
            ${HCCL_DEV_BASE}/src/ops/op_common/inc
            ${HCCL_DEV_BASE}/src/common/hcomm_dlsym

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

        target_link_libraries(${target} PRIVATE
            gtest
            pthread
            hccl-model-proxy
            hccl-verifier
            # 链接hccl库
            ${ASCEND_HOME_PATH}/lib64/libhccl.so
            -ldl
            $<BUILD_INTERFACE:intf_pub>
        )

        target_compile_options(${target} PRIVATE
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

    # 注册CTest测试，工作目录设为构建根目录
    add_test(NAME ${target} COMMAND ${target}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

# 按算子拆分创建独立的测试可执行文件，支持ST_TASKS过滤
if("all" IN_LIST ST_TASKS OR "scatter" IN_LIST ST_TASKS)
    add_st_test(scatter scatter_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "all_reduce" IN_LIST ST_TASKS)
    add_st_test(all_reduce all_reduce_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "all_reduce_parallel" IN_LIST ST_TASKS)
    add_st_test(all_reduce_parallel all_reduce_parallel_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "all_reduce_dpu" IN_LIST ST_TASKS)
    add_st_test(all_reduce_dpu all_reduce_dpu_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "all_reduce_multilevel" IN_LIST ST_TASKS)
    add_st_test(all_reduce_multilevel all_reduce_multilevel_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "all_gather_aicpu" IN_LIST ST_TASKS)
    add_st_test(all_gather_aicpu all_gather_aicpu_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "all_gather_dpu" IN_LIST ST_TASKS)
    add_st_test(all_gather_dpu all_gather_dpu_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "all_gather_v" IN_LIST ST_TASKS)
    add_st_test(all_gather_v all_gather_v_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "all_gather_3level" IN_LIST ST_TASKS)
    add_st_test(all_gather_3level all_gather_3level_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "dpu_sendrecv" IN_LIST ST_TASKS)
    add_st_test(dpu_sendrecv dpu_sendrecv_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "reduce_scatter_aicpu" IN_LIST ST_TASKS)
    add_st_test(reduce_scatter_aicpu reduce_scatter_aicpu_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "reduce_scatter_v" IN_LIST ST_TASKS)
    add_st_test(reduce_scatter_v reduce_scatter_v_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "reduce_scatter" IN_LIST ST_TASKS)
    add_st_test(reduce_scatter reduce_scatter_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "reduce_scatter_3level" IN_LIST ST_TASKS)
    add_st_test(reduce_scatter_3level reduce_scatter_3level_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "reduce" IN_LIST ST_TASKS)
    add_st_test(reduce reduce_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "broadcast_dpu" IN_LIST ST_TASKS)
    add_st_test(broadcast_dpu broadcast_dpu_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "broadcast" IN_LIST ST_TASKS)
    add_st_test(broadcast broadcast_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "alltoall" IN_LIST ST_TASKS)
    add_st_test(alltoall alltoall_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "alltoallv" IN_LIST ST_TASKS)
    add_st_test(alltoallv alltoallv_testcase.cc)
endif()
if("all" IN_LIST ST_TASKS OR "alltoallvc" IN_LIST ST_TASKS)
    add_st_test(alltoallvc alltoallvc_testcase.cc)
endif()
