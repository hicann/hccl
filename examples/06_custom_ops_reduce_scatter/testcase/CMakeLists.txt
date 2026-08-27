# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)

project(custom_${OP_NAME}_test LANGUAGES CXX)

add_executable(${PROJECT_NAME} main.cc)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(WARNING "ASCEND_CANN_PACKAGE_PATH not set, defaulting to ${ASCEND_CANN_PACKAGE_PATH}")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../${OP_TYPE}/inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${MPI_CXX_INCLUDE_PATH}
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ascendcl
    hccl
    hccl_custom_reduce_scatter
    ${MPI_LIB}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -std=c++17
)
