if(NOT HCCL_CANN_COMPAT_850)
set_source_files_properties(
    aiv_scatter_op.h PROPERTIES LANGUAGE CXX
)
set(CCE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin)
add_custom_target(hccl_aiv_scatter_op_910_95 ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_scatter_op_910_95.o)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_scatter_op_910_95.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_scatter_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_scatter_op_910_95_tmp.o --cce-aicore-arch=dav-c310-vec
        --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
        "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/
        -I${OPS_BASE_DIR}/src/ops/op_common/template/aiv/aiv_interface/
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
        --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
        COMMAND ${CCE_DIR}/ld.lld -m aicorelinux -Ttext=0 ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_scatter_op_910_95_tmp.o -static -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_scatter_op_910_95.o
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_scatter_op_910_95.o 
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)
endif()
