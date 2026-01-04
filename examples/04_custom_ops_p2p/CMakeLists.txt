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
project(hccl_examples_custom_ops_p2p)

set(HCCL_CUSTOM_OPS_INCLUDE_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu

    # CANN Toolkit头文件
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/mmpa
)

# Host 动态链接库
add_library(hccl_custom_p2p SHARED)
target_include_directories(hccl_custom_p2p PRIVATE
    ${HCCL_CUSTOM_OPS_INCLUDE_LIST}
)
target_compile_options(hccl_custom_p2p PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g -O0
    -std=c++14
    -fstack-protector-all
)
target_compile_definitions(hccl_custom_p2p PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    -DHOST_COMPILE
)
target_link_options(hccl_custom_p2p PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)
target_link_libraries(hccl_custom_p2p PRIVATE
    -Wl,--no-as-needed
    hcomm
    c_sec
    unified_dlog
    -Wl,--no-as-needed
)
install(TARGETS hccl_custom_p2p
    LIBRARY DESTINATION ${CUSTOM_OPS_OPP_LIB_PATH}
)
install(FILES
    inc/hccl_custom_p2p.h
    DESTINATION ${CUSTOM_OPS_OPP_INC_PATH}
)

set(AICPU_KERNEL_TAR_FILE ${CMAKE_BINARY_DIR}/aicpu_hccl_custom_p2p.tar.gz)
set(AICPU_KERNEL_JSON_FILE ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu/libp2p_aicpu_kernel.json)
if(KERNEL_MODE)
    # AICPU Kernel 动态链接库
    add_library(p2p_aicpu_kernel SHARED)
    target_include_directories(p2p_aicpu_kernel PRIVATE
        ${HCCL_CUSTOM_OPS_INCLUDE_LIST}
    )
    target_compile_definitions(p2p_aicpu_kernel PRIVATE
        -DAICPU_COMPILE
    )
    target_compile_options(p2p_aicpu_kernel PRIVATE
        -g
        -O0
    )
    target_link_libraries(p2p_aicpu_kernel PRIVATE
        -Wl,--no-as-needed
        ccl_kernel
        -Wl,--no-as-needed
    )
    register_custom_aicpu_target(p2p_aicpu_kernel)

    # 打包：AICPU Kernel tar 包
    package_aicpu_kernel(
        p2p_aicpu_kernel
        ${AICPU_KERNEL_TAR_FILE}
        ${AICPU_KERNEL_JSON_FILE}
    )
    # 签名：添加 AICPU Kernel tar 包的签名
    sign_aicpu_kernel(
        p2p_aicpu_kernel
        ${AICPU_KERNEL_TAR_FILE}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/hccl_custom_p2p_check_cfg.xml
        SIGNED_AICPU_KERNEL_TAR_FILE
    )
else()
    # 安装
    install(FILES
        ${BUILD_DEVICE_DIR}/signatures/aicpu_hccl_custom_p2p.tar.gz
        DESTINATION ${CUSTOM_OPS_OPP_AICPU_KERNEL_PATH}
    )
    install(FILES
        ${AICPU_KERNEL_JSON_FILE}
        DESTINATION ${CUSTOM_OPS_OPP_AICPU_CONFIG_PATH}
    )
endif()

# 添加子目录
add_subdirectory(op_host)
add_subdirectory(op_kernel_aicpu)
