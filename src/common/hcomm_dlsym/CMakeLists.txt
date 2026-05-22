# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(BUILD_OPEN_PROJECT)
    set(INCLUDE_LIST
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}

        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/acl

        # hcomm头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl

        # 包间接口
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    )
else()
    set(INCLUDE_LIST
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/aicpu/tsd
        ${TOP_DIR}/inc/driver
        ${TOP_DIR}/inc/external
        ${TOP_DIR}/inc/external/acl
        ${TOP_DIR}/inc/aicpu/
        ${TOP_DIR}/inc/aicpu/aicpu_schedule/aicpu_sharder
        ${TOP_DIR}/abl/qos/qosmng
        ${TOP_DIR}/abl/atrace/inc/utrace
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/inc/aicpu/common
        ${TOP_DIR}/ace/csruntime/inc/tsch
        ${TOP_DIR}/asl/hss/inc
        ${TOP_DIR}/asl/hss/inc/pub
        ${TOP_DIR}/asl/hss/common/hash_table
        ${TOP_DIR}/metadef
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/metadef/inc
        ${TOP_DIR}/metadef/inc/common
        ${TOP_DIR}/metadef/inc/external
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
        ${TOP_DIR}/runtime/pkg_inc/base

        ${TOP_DIR}/open_source/json/include

        ${TOP_DIR}/hccl/src/common

        ${TOP_DIR}/hcomm/inc
        ${TOP_DIR}/hcomm/inc/hccl
        ${TOP_DIR}/hcomm/pkg_inc
        ${TOP_DIR}/hcomm/pkg_inc/hccl
        ${TOP_DIR}/hcomm/pkg_inc/hcomm/ccu
        ${TOP_DIR}/hcomm/include
        ${TOP_DIR}/hcomm/include/hccl
    )
endif()

if(STATIC_MODE)
    target_sources(${HCCT_TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_rank_graph_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_res_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hcomm_primitives_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hcomm_dlsym.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_inner_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hcomm_host_profiling_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_host_comm_dl.cc
    )
else()
    add_library(hccl_compat SHARED)

    if(BUILD_OPEN_PROJECT)
        target_compile_definitions(hccl_compat PRIVATE
            OPEN_BUILD_PROJECT
            $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
        )
    else()
        target_compile_definitions(hccl_compat PRIVATE
            $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
        )
    endif()

    if(HCCL_CANN_COMPAT_850)
        target_compile_definitions(hccl_compat PRIVATE HCCL_CANN_COMPAT_850)
    endif()

    target_include_directories(hccl_compat PRIVATE
        ${INCLUDE_LIST}
    )

    target_compile_options(hccl_compat PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-g>
        -std=c++14
        -fstack-protector-all
    )

    target_link_options(hccl_compat PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,-z,noexecstack
        $<$<CONFIG:Release>:-s>
    )

if(BUILD_OPEN_PROJECT)
    target_link_libraries(hccl_compat PRIVATE
        -Wl,--no-as-needed
        unified_dlog
        acl_rt
        -Wl,--no-as-needed
    )
else()
    target_link_libraries(hccl_compat PRIVATE
        $<BUILD_INTERFACE:slog_headers>
        -Wl,--no-as-needed
        unified_dlog
        acl_rt
        -Wl,--no-as-needed
    )
endif()

    target_link_directories(hccl_compat PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )

    install(TARGETS hccl_compat
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} 
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )

    target_sources(hccl_compat PRIVATE
        hccl_rank_graph_dl.cc
        hccl_res_dl.cc
        hcomm_primitives_dl.cc
        hcomm_dlsym.cc
        hccl_inner_dl.cc
        hcomm_host_profiling_dl.cc
        hccl_host_comm_dl.cc
    )
endif()

if((DEVICE_MODE AND BUILD_OPEN_PROJECT) OR (NOT BUILD_OPEN_PROJECT))
    add_library(hccl_kernel_compat SHARED
        hcomm_device_dlsym.cc
        hcomm_primitives_dl.cc
        hcomm_diag_dl.cc
        hcomm_device_profiling_dl.cc
        hccl_device_comm_dl.cc
    )

    if(HCCL_CANN_COMPAT_850)
        target_compile_definitions(hccl_kernel_compat PRIVATE HCCL_CANN_COMPAT_850)
    endif()

    target_include_directories(hccl_kernel_compat PRIVATE
        ${INCLUDE_LIST}
    )

    target_compile_options(hccl_kernel_compat PRIVATE
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:-O3>
        -fstack-protector-all
    )

    target_link_options(hccl_kernel_compat PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,-z,noexecstack
        $<$<CONFIG:Release>:-s>
    )

    target_link_directories(hccl_kernel_compat PRIVATE
 	    ${ASCEND_CANN_PACKAGE_PATH}/devlib/device
 	)

    install(TARGETS hccl_kernel_compat
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} 
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )
endif()
