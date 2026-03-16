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

find_package(ASC REQUIRED)

project(hccl_examples_custom_ops_allgather LANGUAGES CXX ASC)
set(CMAKE_SYSTEM_PROCESSOR "x86_64")

if (NOT DEFINED CUSTOM_OPS_OPP_LIB_PATH)
    set(CUSTOM_OPS_OPP_LIB_PATH ${CMAKE_INSTALL_PREFIX}/lib)
endif()

if (NOT DEFINED CUSTOM_OPS_OPP_INC_PATH)
    set(CUSTOM_OPS_OPP_INC_PATH ${CMAKE_INSTALL_PREFIX}/include)
endif()

# Install headers
install(FILES
    inc/hccl_custom_allgather.h
    DESTINATION ${CUSTOM_OPS_OPP_INC_PATH}
)

add_subdirectory(op_kernel)
add_subdirectory(op_host)
add_subdirectory(testcase)
