# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(STATIC_MODE)
    add_library(hccl STATIC)
    set_target_properties(hccl PROPERTIES
        OUTPUT_NAME "hccl_static"
        POSITION_INDEPENDENT_CODE ON
    )
else()
    add_library(hccl SHARED)
endif()

# 基于ini生成json文件
SET(HCCL_CMAKE_DIR ${OPS_BASE_DIR}/cmake/)
message(STATUS "HCCL_CMAKE_DIR = ${HCCL_CMAKE_DIR}")
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json
    COMMAND ${HI_PYTHON}
            ${HCCL_CMAKE_DIR}/scripts/parser_ini.py
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/scatter_aicpu_kernel.ini
            ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_target(aicpu_kernel_json DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json
    DESTINATION ${INSTALL_AICPU_KERNEL_JSON_DIR}/config
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)
add_dependencies(hccl aicpu_kernel_json)

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(hccl PRIVATE
        OPEN_BUILD_PROJECT
        $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
    )
else()
    target_compile_definitions(hccl PRIVATE
        $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
    )
endif()

target_include_directories(hccl PRIVATE
    ${INCLUDE_LIST}
)

target_compile_definitions(hccl PRIVATE
    -DHOST_COMPILE
)

hccl_apply_cann_compat(hccl)

if(HCCL_CANN_COMPAT_850)
    target_compile_definitions(hccl PRIVATE HCCL_CANN_COMPAT_850)
endif()

if(HCCL_ENABLE_960)
    target_compile_definitions(hccl PRIVATE HCCL_ENABLE_960)
endif()

target_compile_options(hccl PRIVATE
    -Werror
    -Wno-error=float-equal
    -Wno-error=maybe-uninitialized
    -Wno-error=unused-but-set-variable
    -Wno-overloaded-virtual
    -Wno-unused-parameter
    -Wno-sign-compare
    -Wno-unused-variable
    -Wno-unused-value
    -fno-common
    -fno-strict-aliasing
    -pipe
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-O3 -g>
    -fstack-protector-all
)

if(NOT STATIC_MODE)
    add_dependencies(hccl hccl_compat)
endif()

if(BUILD_OPEN_PROJECT)
    target_link_libraries(hccl PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:error_manager_headers>
        -Wl,--no-as-needed
        hcomm
        hccl_compat
        acl_rt
        c_sec
        unified_dlog
        -Wl,--no-as-needed
    )
else()
    target_link_libraries(hccl PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:ofed_headers>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:npu_runtime_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        -Wl,--no-as-needed
        hcomm
        hccl_compat
        acl_rt
        c_sec
        unified_dlog
        -Wl,--no-as-needed
    )
endif()

if(NOT STATIC_MODE)
    target_link_options(hccl PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,-z,noexecstack
        $<$<CONFIG:Release>:-s>
    )

    set(HCCL_EXPORT_MAP
        "${CMAKE_CURRENT_LIST_DIR}/hccl.map"
    )

    target_link_options(hccl PRIVATE
        "-Wl,--version-script=${HCCL_EXPORT_MAP}"
    )

    set_property(TARGET hccl APPEND PROPERTY
        LINK_DEPENDS "${HCCL_EXPORT_MAP}"
    )
endif()

if(STATIC_MODE)
    install(TARGETS hccl
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )
else()
    install(TARGETS hccl
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )
endif()

set(_op_proto_link_libs
    -Wl,--no-as-needed
    exe_graph
    graph
    graph_base
    register
    -Wl,--as-needed
)

set(OP_PROTO_INCLUDE
    ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
)

add_library(opgraph_hccl SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/common/log.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/op_graph/hcom_vinputvec_set_pass.cc
)

if(NOT ENABLE_TEST)
    set(OPGRAPH_HCCL_EXPORT_MAP
        "${CMAKE_CURRENT_LIST_DIR}/opgraph_hccl.map"
    )

    target_link_options(opgraph_hccl PRIVATE
        "-Wl,--version-script=${OPGRAPH_HCCL_EXPORT_MAP}"
    )

    set_property(TARGET opgraph_hccl APPEND PROPERTY
        LINK_DEPENDS "${OPGRAPH_HCCL_EXPORT_MAP}"
    )
endif()

target_include_directories(opgraph_hccl PRIVATE
    ${INCLUDE_LIST}
    ${OP_PROTO_INCLUDE}
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/external
)
target_compile_options(opgraph_hccl PRIVATE
    -Wno-error=cpp
    -fno-common
    -fno-strict-aliasing
    -pipe
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-O3 -g>
    -fstack-protector-all
    -fvisibility=hidden
)
target_link_libraries(opgraph_hccl PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:msprof_headers>
    $<BUILD_INTERFACE:mmpa_headers>
    $<BUILD_INTERFACE:runtime_headers>
    $<BUILD_INTERFACE:hcomm_headers>
    $<$<CONFIG:Release>:-s>
    unified_dlog
    ${_op_proto_link_libs}
    -Wl,--whole-archive
    rt2_registry
    -Wl,--no-whole-archive
    ge_common
    -Wl,-Bsymbolic
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)

target_link_directories(opgraph_hccl PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/common/op_graph/ops_proto_hccl.h
    DESTINATION ${INSTALL_OPGRAPH_INCLUDE_DIR} 
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(TARGETS opgraph_hccl
    LIBRARY DESTINATION ${INSTALL_OPGRAPH_LIBRARY_DIR} 
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

add_dependencies(hccl opgraph_hccl)

#生成ES API
list(APPEND CMAKE_MODULE_PATH "${ASCEND_CANN_PACKAGE_PATH}/include/ge/cmake")
find_package(GenerateEsPackage REQUIRED)

add_es_library_and_whl(
    ES_LINKABLE_AND_ALL_TARGET es_hccl
    OPP_PROTO_TARGET opgraph_hccl
    OUTPUT_PATH ${CMAKE_BINARY_DIR}/es_output
)

if(NOT ENABLE_TEST)
    set(ES_HCCL_EXPORT_MAP
        "${CMAKE_CURRENT_LIST_DIR}/es_hccl.map"
    )

    if(NOT TARGET es_hccl_so)
        message(FATAL_ERROR
            "GenerateEsPackage did not create the expected es_hccl_so target"
        )
    endif()

    target_link_options(es_hccl_so PRIVATE
        "-Wl,--version-script=${ES_HCCL_EXPORT_MAP}"
    )

    set_property(TARGET es_hccl_so APPEND PROPERTY
        LINK_DEPENDS "${ES_HCCL_EXPORT_MAP}"
    )
endif()

add_dependencies(hccl es_hccl)

install(DIRECTORY ${CMAKE_BINARY_DIR}/es_output/include/es_hccl
    DESTINATION ${INSTALL_INCLUDE_DIR}/es/
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/es_output/whl/
    DESTINATION ${WHL_INSTALL_DIR}/es_packages/whl
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_BINARY_DIR}/es_output/lib64/libes_hccl.so
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)
