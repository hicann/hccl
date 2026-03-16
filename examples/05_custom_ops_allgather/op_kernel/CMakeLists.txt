# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/launch_kernel_asc.asc)
set(KERNEL_TMP_OBJ ${CMAKE_CURRENT_BINARY_DIR}/hccl_custom_allgather_kernels_tmp.o)
set(KERNEL_OBJ ${CMAKE_CURRENT_BINARY_DIR}/hccl_custom_allgather_kernels.o)

set(CCE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin)
set(CANN_ARCH_BASE ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux)

set(KERNEL_OPERATOR_HINTS
    ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface
    ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
    ${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/interface
    ${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw
    ${ASCEND_CANN_PACKAGE_PATH}/include/tikcpp/tikcfw/interface
    ${ASCEND_CANN_PACKAGE_PATH}/include/tikcpp/tikcfw
    ${CANN_ARCH_BASE}/tikcpp/tikcfw/interface
    ${CANN_ARCH_BASE}/tikcpp/tikcfw
)

find_path(KERNEL_OPERATOR_DIR
    NAMES kernel_operator.h
    HINTS ${KERNEL_OPERATOR_HINTS}
)

if(NOT KERNEL_OPERATOR_DIR)
    message(FATAL_ERROR "kernel_operator.h not found under ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")
endif()

find_program(CCE_COMPILER
    NAMES ccec
    HINTS ${CCE_DIR}
)

find_program(CCE_LLD
    NAMES ld.lld
    HINTS ${CCE_DIR}
)

if(NOT CCE_COMPILER OR NOT CCE_LLD)
    message(FATAL_ERROR "ccec or ld.lld not found under ${CCE_DIR}")
endif()

set(ASC_INCLUDE_FLAGS
    -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
    -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
    -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
    -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
    -I${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw
    -I${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/interface/
    -I${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/impl/
    -I${CANN_ARCH_BASE}/tikcpp/tikcfw
    -I${CANN_ARCH_BASE}/tikcpp/tikcfw/interface/
    -I${CANN_ARCH_BASE}/tikcpp/tikcfw/impl/
    -I${KERNEL_OPERATOR_DIR}
    -I${CMAKE_CURRENT_SOURCE_DIR}/../inc
    -I${ASCEND_CANN_PACKAGE_PATH}/include
    -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
    -I${ASCEND_CANN_PACKAGE_PATH}/include/mmpa
)

if(DEFINED OPS_BASE_DIR)
    list(APPEND ASC_INCLUDE_FLAGS
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
    )
endif()

add_custom_command(
    OUTPUT ${KERNEL_OBJ}
    COMMAND ${CCE_COMPILER} -c -x cce -O2 ${KERNEL_SRC} -o ${KERNEL_TMP_OBJ}
        --cce-aicore-arch=dav-c310-vec
        --cce-aicore-input-parameter-size=28000
        --cce-aicore-only
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=false
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-jump-expand=true
        -std=c++17 -fstack-protector-all
        ${ASC_INCLUDE_FLAGS}
    COMMAND ${CCE_LLD} -m aicorelinux -Ttext=0 ${KERNEL_TMP_OBJ} -static -o ${KERNEL_OBJ}
    DEPENDS ${KERNEL_SRC}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)

add_custom_target(generate_kernel_obj ALL DEPENDS ${KERNEL_OBJ})

install(FILES ${KERNEL_OBJ}
    DESTINATION ${CUSTOM_OPS_OPP_LIB_PATH}
)

add_custom_command(TARGET generate_kernel_obj POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${KERNEL_OBJ}
        ${CMAKE_SOURCE_DIR}/testcase/hccl_custom_allgather_kernels.o
)
