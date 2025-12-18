# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_library(hccl-verifier SHARED "")
target_sources(hccl-verifier PRIVATE
    checker.cc
    hccl_verifier.cc
)

add_subdirectory(mem_conflict_check)
add_subdirectory(semantics_check)
add_subdirectory(singletask_check)
add_subdirectory(task_graph_generator)

target_compile_options(hccl-verifier PRIVATE
    -std=c++14
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O0
    -g
    -D_GLIBCXX_USE_CXX11_ABI=0
)

message(STATUS "utils path= ${UTILS_DIR}")

if (BUILD_OPEN_PROJECT)
    target_include_directories(hccl-verifier PRIVATE
        # Ascend安装包头文件
        ${ASCEND_HOME_PATH}/x86_64-linux/include
        ${ASCEND_HOME_PATH}/x86_64-linux/pkg_inc
        ${ASCEND_HOME_PATH}/x86_64-linux/pkg_inc/base
        ${ASCEND_HOME_PATH}/x86_64-linux/pkg_inc/runtime
        ${ASCEND_HOME_PATH}/x86_64-linux/pkg_inc/profiling
        ${ASCEND_HOME_PATH}/x86_64-linux/hcomm/include/hccl
        ${ASCEND_HOME_PATH}/x86_64-linux/include/hccl
        ${ASCEND_HOME_PATH}/x86_64-linux/pkg_inc/hccl

        # hccl-dev头文件
        ${HCCL_DEV_BASE}/src/common

        ${UTILS_DIR}/sim_world
        ${UTILS_DIR}/hccl_proxy
        ${UTILS_DIR}/hccl_verifier
        ${UTILS_DIR}/pub_inc
        ${UTILS_DIR}/common/exception
        ${UTILS_DIR}/common/utils
        ${UTILS_DIR}/hccl_verifier/task_graph_generator
        ${UTILS_DIR}/hccl_verifier/semantics_check
        ${UTILS_DIR}/hccl_verifier/mem_conflict_check
        ${UTILS_DIR}/hccl_verifier/singletask_check
    )

else()
    target_include_directories(hccl-verifier PRIVATE
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/sim_world
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_proxy
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_verifier
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/pub_inc
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/common/exception
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/common/utils
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_verifier/task_graph_generator
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_verifier/semantics_check
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_verifier/mem_conflict_check
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_verifier/singletask_check

        ${TOP_DIR}/hcomm/include
        ${TOP_DIR}/hcomm/include/hccl
        ${TOP_DIR}/hcomm/pkg_inc
        ${TOP_DIR}/hcomm/pkg_inc/hccl
        ${TOP_DIR}/hccl/src/common
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/abl/slog/inc/toolchain  # dlog_pub.h
    )
endif()