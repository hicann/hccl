# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

set(PRODUCT_SIDE "device")
set(HCCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../" CACHE PATH "Project top directory")
include(${HCCL_DIR}/cmake/fetch_cann_cmake.cmake)
project(hccl_device)

# 初始化 CANN 项目配置
init_cann_project(PREPEND_MODULE_PATH)
add_cann_target_options()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${RULE_LAUNCH})
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${RULE_LAUNCH})

message(STATUS "BUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}")
message(STATUS "ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}")
message(STATUS "CANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}")

# cmake 配置
include(${HCCL_DIR}/cmake/config.cmake)
include(${HCCL_DIR}/cmake/func.cmake)

find_cann_package(msprof MODULE REQUIRED)
find_cann_package(unified_dlog MODULE REQUIRED)
find_cann_package(mmpa MODULE REQUIRED)
find_cann_package(runtime MODULE REQUIRED)
find_cann_package(hcomm MODULE REQUIRED)
find_cann_package(asc_devkit MODULE REQUIRED)

# 添加子目录
add_cann_subdirectories_relative(${HCCL_DIR}
    ${HCCL_DIR}/src
)

# 实验特性: 复用 experimental/ops/CMakeLists.txt 中 if(TARGET scatter_aicpu_kernel) 的注入逻辑
# (sources / includes / ENABLE_EXPERIMENTAL 定义),与 host 侧构建保持单一来源;新增实验算子
# 无需在此同步维护。必须在 add_cann_subdirectories_relative(src) 之后调用,以保证
# scatter_aicpu_kernel 目标已存在;experimental 子目录的 target_sources(hccl ...) 已用
# if(TARGET hccl) 守护,device 构建无 hccl 目标时自动跳过。
if(ENABLE_EXPERIMENTAL)
    add_subdirectory(${HCCL_DIR}/experimental/ops experimental/ops)
endif()

cann_pack_targets_and_files(
    OUTPUT_TARGET "generate_device_aicpu_package"
    OUTPUT "aicpu_hccl.tar.gz"
    TARGETS scatter_aicpu_kernel hccl_kernel_compat
    MANIFEST "bin_hash.cfg"
    TAR_ROOT_DIR "aicpu_kernels_device"
)
add_cann_sign_file(
    INPUT "aicpu_hccl.tar.gz"
    CONFIG "${HCCL_DIR}/scripts/sign.xml"
    RESULT_VAR "aicpu_hccl_sign_file"
    DEPENDS generate_device_aicpu_package
)

# 安装 aicpu_hccl.tar.gz
install(FILES ${aicpu_hccl_sign_file}
    DESTINATION ${INSTALL_AICPU_KERNEL_JSON_DIR}/kernel 
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

# 设置子工程打包
set_cann_subprj_package()