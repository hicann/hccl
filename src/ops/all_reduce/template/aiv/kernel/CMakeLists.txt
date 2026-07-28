if(NOT HCCL_CANN_COMPAT_850)
set_source_files_properties(
    aiv_all_reduce_op.h PROPERTIES LANGUAGE CXX
)
set(CCE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin)
add_custom_target(hccl_aiv_all_reduce_op_910_95 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_reduce_op_910_95.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_reduce_op_910_95.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_all_reduce_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_reduce_op_910_95_tmp.o --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        COMMAND ${CCE_DIR}/ld.lld -m aicorelinux -Ttext=0 ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_reduce_op_910_95_tmp.o -static -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_reduce_op_910_95.o
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)
 
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_all_reduce_op_910_95.o 
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_int8_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int8_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int8_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int8_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_INT8 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_int8
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_int16_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int16_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int16_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int16_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_INT16 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_int16
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_int32_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int32_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int32_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int32_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_INT32 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_int32
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_float16_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float16_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float16_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float16_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_FP16 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_float16
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_float32_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float32_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float32_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float32_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_FP32 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_float32
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_bfloat16_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_bfloat16_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_bfloat16_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_bfloat16_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_BFP16 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_bfloat16
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_int8_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int8_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int8_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int8_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_INT8 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_int8
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_int16_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int16_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int16_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int16_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_INT16 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_int16
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_int32_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int32_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int32_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int32_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_INT32 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_int32
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_float16_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float16_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float16_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float16_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_FP16 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_float16
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_float32_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float32_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float32_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float32_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_FP32 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_float32
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_bfloat16_v2 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_bfloat16_v2.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_bfloat16_v2.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_bfloat16_v2.o
        -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
        -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
        -DHCCL_DTYPE_BFP16 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_bfloat16
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int8_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int16_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int32_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float16_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float32_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_bfloat16_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int8_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int16_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int32_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float16_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float32_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_bfloat16_v2.o
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

endif()
