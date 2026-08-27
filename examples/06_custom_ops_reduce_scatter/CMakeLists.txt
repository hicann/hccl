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

message(STATUS "06_custom: Custom ops name: ${CUSTOM_OPS_NAME}")
message(STATUS "06_custom: Custom ops vendor: ${CUSTOM_OPS_VENDOR}")
message(STATUS "06_custom: Custom ops path: ${CUSTOM_OPS_PATH}")

string(REGEX REPLACE "_[^_]*$" "" OP_NAME ${CUSTOM_OPS_NAME})
message(STATUS "06_custom: OP_NAME: ${OP_NAME}")

set(MPI_LIB )
if(CUSTOM_OPS_NAME STREQUAL ${OP_NAME}_ccu)
  set(OP_TYPE ccu)
  add_subdirectory(ccu)
elseif(CUSTOM_OPS_NAME STREQUAL ${OP_NAME}_aiv)
  find_package(MPI REQUIRED)
  set(MPI_LIB MPI::MPI_CXX)
  set(OP_TYPE aiv)
  add_definitions(-DENABLE_MPI)
  add_subdirectory(aiv)
elseif(CUSTOM_OPS_NAME STREQUAL ${OP_NAME}_aicpu)
  set(OP_TYPE aicpu)
  add_subdirectory(aicpu)
else()
  message("Unsupported ops ${CUSTOM_OPS_NAME}")
endif()

message(STATUS "06_custom: OP_TYPE: ${OP_TYPE}")

add_subdirectory(testcase)
