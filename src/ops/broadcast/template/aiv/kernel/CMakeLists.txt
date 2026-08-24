# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(NOT HCCL_CANN_COMPAT_850)
set_source_files_properties(
    aiv_broadcast_op.h PROPERTIES LANGUAGE CXX
)
set(CCE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin)
add_custom_target(hccl_aiv_broadcast_op_910_95 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_broadcast_op_910_95.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_broadcast_op_910_95.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_broadcast_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_broadcast_op_910_95_tmp.o --cce-aicore-arch=dav-c310-vec
        -DASCENDC_DUMP=${ASCENDC_DUMP}
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        COMMAND ${CCE_DIR}/ld.lld -m aicorelinux -Ttext=0 ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_broadcast_op_910_95_tmp.o -static -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_broadcast_op_910_95.o
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)
 
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_broadcast_op_910_95.o 
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

if(HCCL_ENABLE_910_96)
add_custom_target(hccl_aiv_broadcast_op_910_96 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_broadcast_op_910_96.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_broadcast_op_910_96.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_broadcast_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_broadcast_op_910_96_tmp.o --cce-aicore-arch=dav-920r1-vec
        -DASCENDC_DUMP=${ASCENDC_DUMP}
        --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
        -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        COMMAND ${CCE_DIR}/ld.lld -m aicorelinux -Ttext=0 ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_broadcast_op_910_96_tmp.o -static -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_broadcast_op_910_96.o
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_broadcast_op_910_96.o 
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)
endif()

endif()
