# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hccl-model-proxy SHARED "")

add_subdirectory(hccl_verifier)
add_subdirectory(sim_world)
add_subdirectory(hccl_proxy)
add_subdirectory(common)
add_subdirectory(hccl_depends_stub)
add_subdirectory(aicpu)

target_compile_options(hccl-model-proxy PRIVATE
    -std=c++14
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O0
    -g
    -D_GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(hccl-model-proxy PUBLIC
    # 链接hccl_depends_stub
    hcomm
    unified_dlog
    mmpa
    scatter_aicpu_kernel
)

if (BUILD_OPEN_PROJECT)
    target_link_libraries(hccl-model-proxy PUBLIC
        ${ASCEND_HOME_PATH}/lib64/libc_sec.so
    )

    target_include_directories(hccl-model-proxy PRIVATE
        ${UTILS_DIR}/sim_world
        ${UTILS_DIR}/common/exception
        ${UTILS_DIR}/common/utils
        ${UTILS_DIR}/hccl_verifier
        ${UTILS_DIR}/hccl_proxy
        ${UTILS_DIR}/hccl_proxy/communicator
        ${UTILS_DIR}/hccl_proxy/topo_model

        # Ascend安装包头文件
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/pkg_inc
        ${ASCEND_HOME_PATH}/pkg_inc/base
        ${ASCEND_HOME_PATH}/pkg_inc/runtime
        ${ASCEND_HOME_PATH}/pkg_inc/profiling
        ${ASCEND_HOME_PATH}/hcomm/include/hccl
        ${ASCEND_HOME_PATH}/include/hccl
        ${ASCEND_HOME_PATH}/pkg_inc/hccl

        # mmpa头文件
        ${ASCEND_HOME_PATH}/include/mmpa

        # 本仓代码头文件
        ${HCCL_DEV_BASE}/src/common
        ${HCCL_DEV_BASE}/src/ops
        ${HCCL_DEV_BASE}/src/ops/inc
    )

else()
    target_link_libraries(hccl-model-proxy PUBLIC
        c_sec
    )
    target_include_directories(hccl-model-proxy PRIVATE
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/sim_world
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/common/exception
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/common/utils
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_verifier
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_proxy
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_proxy/communicator
        ${TOP_DIR}/hccl/test/st/algorithm/utils/src/hccl_proxy/topo_model

        ${TOP_DIR}/hcomm/include
        ${TOP_DIR}/hcomm/include/hccl
        ${TOP_DIR}/hcomm/pkg_inc
        ${TOP_DIR}/hcomm/pkg_inc/hccl
        ${TOP_DIR}/hccl/src/common

        ${TOP_DIR}/ace/npuruntime/inc
        ${TOP_DIR}/ace/npuruntime/inc/runtime
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/abl/slog/inc/toolchain
        ${TOP_DIR}/runtime/include/external
    )
endif()



