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
    scatter_testcase.cc
)

add_executable(hccl_checker_ops_stest ${src_list})

if (BUILD_OPEN_PROJECT)
    target_include_directories(hccl_checker_ops_stest PUBLIC
        # 第三方头文件
        ${THIRD_PARTH_PATH}/gtest_shared/include
        ${THIRD_PARTH_PATH}/gtest_shared/include

        # Ascend安装包头文件
        ${ASCEND_HOME_PATH}/x86_64-linux/include
        ${ASCEND_HOME_PATH}/x86_64-linux/pkg_inc
        ${ASCEND_HOME_PATH}/x86_64-linux/pkg_inc/base
        ${ASCEND_HOME_PATH}/x86_64-linux/hcomm/include/hccl
        ${ASCEND_HOME_PATH}/x86_64-linux/include/hccl
        ${ASCEND_HOME_PATH}/x86_64-linux/pkg_inc/hccl
        
        # 后续可能要根据大包安装目录改动
        ${ASCEND_HOME_PATH}/../cann/x86_64-linux/include/hccl

        # 本仓代码头文件
        ${HCCL_DEV_BASE}/src/common

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
        ${ASCEND_HOME_PATH}/x86_64-linux/lib64/libhccl.so
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
else()
    set(HCCL_LIB_DIR ${TOP_DIR}/output/ascend/host/ubuntu20.04/x86_64/lib)
    target_include_directories(hccl_checker_ops_stest PUBLIC
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/sim_world
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/common/utils
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/common/exception
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_verifier
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_verifier/semantics_check
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_verifier/task_graph_generator
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_proxy
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_proxy/communicator
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_proxy/topo_model

        ${TOP_DIR}/hcomm/inc
        ${TOP_DIR}/hcomm/inc/hccl
        ${TOP_DIR}/hcomm/pkg_inc
        ${TOP_DIR}/hcomm/pkg_inc/hccl
        ${TOP_DIR}/hccl/inc
        ${TOP_DIR}/hccl/src/common

        ${TOP_DIR}/ace/npuruntime/inc
        ${TOP_DIR}/ace/npuruntime/inc/runtime
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/abl/slog/inc/toolchain
        ${TOP_DIR}/runtime/include/external
    )

    target_link_libraries(hccl_checker_ops_stest PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub>
        hccl-model-proxy
        hccl-verifier
        # 链接hccl库
        ${HCCL_LIB_DIR}/libhccl.so
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

    run_llt_test(
        TARGET hccl_checker_ops_stest
    )
endif()