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
    aiv_all_gather_op.h PROPERTIES LANGUAGE CXX
)
set(CCE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin)
add_custom_target(hccl_aiv_all_gather_op_910_95 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_gather_op_910_95.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_gather_op_910_95.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_all_gather_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_gather_op_910_95_tmp.o --cce-aicore-arch=dav-c310-vec
        -DASCENDC_DUMP=${ASCENDC_DUMP}
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        COMMAND ${CCE_DIR}/ld.lld -m aicorelinux -Ttext=0 ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_gather_op_910_95_tmp.o -static -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_gather_op_910_95.o
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)
 
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_gather_op_910_95.o 
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

add_custom_target(hccl_ag_superkernel_mesh_1d_int8_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int8_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int8_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int8_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_INT8 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int8
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ag_superkernel_mesh_1d_int16_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int16_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int16_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int16_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_INT16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int16
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ag_superkernel_mesh_1d_int32_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int32_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int32_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int32_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_INT32 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int32
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ag_superkernel_mesh_1d_float16_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float16_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float16_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float16_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_FP16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_float16
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ag_superkernel_mesh_1d_float32_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float32_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float32_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float32_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_FP32 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_float32
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ag_superkernel_mesh_1d_bfloat16_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_bfloat16_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_bfloat16_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_bfloat16_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_BFP16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_bfloat16
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ag_superkernel_mesh_1d_uint8_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint8_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint8_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint8_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_UINT8 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint8
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ag_superkernel_mesh_1d_uint16_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint16_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint16_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint16_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_UINT16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint16
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ag_superkernel_mesh_1d_uint32_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint32_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint32_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint32_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_UINT32 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint32
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ag_superkernel_mesh_1d_int64_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int64_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int64_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int64_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_INT64 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int64
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ag_superkernel_mesh_1d_uint64_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint64_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint64_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint64_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_UINT64 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint64
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int8_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int16_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int32_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float16_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float32_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_bfloat16_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint8_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint16_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint32_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int64_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint64_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

if(HCCL_ENABLE_960)
add_custom_target(hccl_aiv_all_gather_op_960 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_gather_op_960.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_gather_op_960.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_all_gather_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_gather_op_960_tmp.o --cce-aicore-arch=dav-920r1-vec
        -DASCENDC_DUMP=${ASCENDC_DUMP}
        --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
        -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        COMMAND ${CCE_DIR}/ld.lld -m aicorelinux -Ttext=0 ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_gather_op_960_tmp.o -static -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_gather_op_960.o
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_gather_op_960.o 
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

add_custom_target(hccl_ag_superkernel_mesh_1d_int8_v2_960 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int8_v2_960.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int8_v2_960.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int8_v2_960.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-920r1-vec
        --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=920 -D__DAV_920R1__
        -DHCCL_DTYPE_INT8 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int8
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
        -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int8_v2_960.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

add_custom_target(hccl_ag_superkernel_mesh_1d_int16_v2_960 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int16_v2_960.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int16_v2_960.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int16_v2_960.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-920r1-vec
        --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=920 -D__DAV_920R1__
        -DHCCL_DTYPE_INT16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int16
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
        -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int16_v2_960.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

add_custom_target(hccl_ag_superkernel_mesh_1d_int32_v2_960 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int32_v2_960.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int32_v2_960.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int32_v2_960.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-920r1-vec
        --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=920 -D__DAV_920R1__
        -DHCCL_DTYPE_INT32 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int32
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
        -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int32_v2_960.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

add_custom_target(hccl_ag_superkernel_mesh_1d_float16_v2_960 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float16_v2_960.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float16_v2_960.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float16_v2_960.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-920r1-vec
        --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=920 -D__DAV_920R1__
        -DHCCL_DTYPE_FP16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_float16
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
        -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float16_v2_960.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

add_custom_target(hccl_ag_superkernel_mesh_1d_float32_v2_960 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float32_v2_960.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float32_v2_960.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float32_v2_960.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-920r1-vec
        --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=920 -D__DAV_920R1__
        -DHCCL_DTYPE_FP32 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_float32
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
        -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float32_v2_960.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

add_custom_target(hccl_ag_superkernel_mesh_1d_bfloat16_v2_960 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_bfloat16_v2_960.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_bfloat16_v2_960.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_bfloat16_v2_960.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-920r1-vec
        --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=920 -D__DAV_920R1__
        -DHCCL_DTYPE_BFP16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_bfloat16
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
        -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_bfloat16_v2_960.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

add_custom_target(hccl_ag_superkernel_mesh_1d_uint8_v2_960 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint8_v2_960.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint8_v2_960.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint8_v2_960.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-920r1-vec
        --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=920 -D__DAV_920R1__
        -DHCCL_DTYPE_UINT8 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint8
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
        -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint8_v2_960.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

add_custom_target(hccl_ag_superkernel_mesh_1d_uint16_v2_960 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint16_v2_960.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint16_v2_960.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint16_v2_960.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-920r1-vec
        --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=920 -D__DAV_920R1__
        -DHCCL_DTYPE_UINT16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint16
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
        -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint16_v2_960.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

add_custom_target(hccl_ag_superkernel_mesh_1d_uint32_v2_960 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint32_v2_960.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint32_v2_960.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint32_v2_960.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-920r1-vec
        --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=920 -D__DAV_920R1__
        -DHCCL_DTYPE_UINT32 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint32
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
        -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint32_v2_960.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

add_custom_target(hccl_ag_superkernel_mesh_1d_int64_v2_960 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int64_v2_960.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int64_v2_960.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int64_v2_960.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-920r1-vec
        --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=920 -D__DAV_920R1__
        -DHCCL_DTYPE_INT64 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int64
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
        -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int64_v2_960.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

add_custom_target(hccl_ag_superkernel_mesh_1d_uint64_v2_960 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint64_v2_960.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint64_v2_960.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint64_v2_960.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP=${ASCENDC_DUMP} --cce-aicore-arch=dav-920r1-vec
        --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=920 -D__DAV_920R1__
        -DHCCL_DTYPE_UINT64 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint64
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
        -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
        -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/algorithm/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint64_v2_960.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

endif()

endif()
