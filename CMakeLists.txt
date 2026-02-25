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
option(BUILD_OPEN_PROJECT "Build open hccl project." ON)
message(STATUS "KERNEL_MODE=${KERNEL_MODE}")
message(STATUS "BUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}")

find_program(CCAHE_PROGRAM ccache)
if(CCAHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCAHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCAHE_PROGRAM})
endif()

if(BUILD_OPEN_PROJECT AND KERNEL_MODE)
    # need to switch
    set(CMAKE_SYSTEM_NAME Linux)
    #Change the path to the absolute path of the cross compilation tool after the toolkit package is decompressed
    # set cross compile toolchain dir
    set(TOOLCHAIN_DIR ${CUSTOM_ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/)
    #Specify cross compiler
    message(STATUS "TOOLCHAIN_DIR=${TOOLCHAIN_DIR}")
    if (NOT "x${TOOLCHAIN_DIR}" STREQUAL "x")
        if (NOT IS_DIRECTORY ${TOOLCHAIN_DIR})
            message(FATAL_ERROR "specify cross compile toolchain directory(${TOOLCHAIN_DIR}) is not exist")
        endif()
    endif()
    set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-g++)
    set(CMAKE_C_COMPILER   ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc)
endif()
project(hccl)

set(CMAKE_CXX_COMPILE_OBJECT
    "<CMAKE_CXX_COMPILER> <DEFINES> -D__FILE__='\"$(notdir $(abspath <SOURCE>))\"' -Wno-builtin-macro-redefined <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>"
)
set(CMAKE_C_COMPILE_OBJECT
    "<CMAKE_C_COMPILER> <DEFINES> -D__FILE__='\"$(notdir $(abspath <SOURCE>))\"' -Wno-builtin-macro-redefined <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>"
)

if(ENABLE_TEST)
    include(cmake/config.cmake)
    include(cmake/intf_pub_gccnative.cmake)
    add_subdirectory(test)

elseif(ENABLE_CUSTOM)
    # 自定义算子打包
    if(NOT CUSTOM_OPS_NAME)
        message(FATAL_ERROR "Custom ops name is not set")
    endif()
    if(NOT CUSTOM_OPS_PATH)
        message(FATAL_ERROR "Custom ops path is not set")
    endif()
    if(NOT CUSTOM_OPS_VENDOR)
        message(FATAL_ERROR "Custom ops vendor is not set")
    endif()

    if(NOT EXISTS ${CUSTOM_OPS_PATH})
        message(FATAL_ERROR "Custom ops path does not exist: ${CUSTOM_OPS_PATH}")
    endif()
    if(NOT EXISTS "${CUSTOM_OPS_PATH}/CMakeLists.txt")
        message(FATAL_ERROR "CMakeLists.txt not found. Please ensure it's a CMake project: ${CUSTOM_OPS_PATH}")
    endif()

    message(STATUS "Custom ops name: ${CUSTOM_OPS_NAME}")
    message(STATUS "Custom ops vendor: ${CUSTOM_OPS_VENDOR}")
    message(STATUS "Custom ops path: ${CUSTOM_OPS_PATH}")

    set(BUILD_DEVICE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build_device)
    set(BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build)
    set(CUSTOM_OPS_OPP_PATH "opp/vendors/${CUSTOM_OPS_VENDOR}")
    set(CUSTOM_OPS_OPP_INC_PATH "${CUSTOM_OPS_OPP_PATH}/include")
    set(CUSTOM_OPS_OPP_LIB_PATH "${CUSTOM_OPS_OPP_PATH}/lib64")
    set(CUSTOM_OPS_OPP_SCRIPTS_PATH "${CUSTOM_OPS_OPP_PATH}/scripts")
    set(CUSTOM_OPS_OPP_AICPU_CONFIG_PATH "${CUSTOM_OPS_OPP_PATH}/aicpu/config")
    set(CUSTOM_OPS_OPP_AICPU_KERNEL_PATH "${CUSTOM_OPS_OPP_PATH}/aicpu/kernel")

    include(cmake/config.cmake)
    include(cmake/func.cmake)
    include(cmake/package.cmake)

    # AICPU Device 构建目标
    add_custom_target(custom_aicpu)
    # CPack 打包
    pack_custom()
    # 添加自定义算子工程
    add_subdirectory(${CUSTOM_OPS_PATH})

elseif(BUILD_OPEN_PROJECT)
    # include(cmake/third_party/json.cmake)
    include(cmake/config.cmake)
    include(cmake/package.cmake)
    pack_built_in()
    add_subdirectory(src)

    if(DEVICE_MODE)
        include(cmake/func.cmake)
        set(CCL_KERNEL_TAR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build_device/ccl_kernel_tar_pkg/)
        if(KERNEL_MODE)
            add_custom_target(generate_device_aicpu_package
                COMMAND cd ${CCL_KERNEL_TAR_DIR} && ls && tar czf ../aicpu_hccl.tar.gz aicpu_kernels_device
                DEPENDS scatter_aicpu_kernel
                COMMENT "Creating aicpu_hccl.tar.gz"
            )
            add_dependencies(generate_device_aicpu_package scatter_aicpu_kernel)
            sign_file(
                INPUT "aicpu_hccl.tar.gz"
                CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/scripts/sign/hccl_check_cfg.xml"
                RESULT_VAR "aicpu_hccl_sign_file"
                DEPENDS generate_device_aicpu_package
            )
            add_dependencies(sign_aicpu_hccl generate_device_aicpu_package)
        endif()
    endif()

    if (NOT DEVICE_MODE)
        message(STATUS "Detected INSTALL_LIBRARY_DIR: ${INSTALL_LIBRARY_DIR}")

        set(BUILD_DEVICE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build_device/)
        add_custom_target(install_tar DEPENDS
            ${BUILD_DEVICE_DIR}/signatures/aicpu_hccl.tar.gz
            ${BUILD_DEVICE_DIR}/aicpu_custom/libscatter_aicpu_kernel.so
        )
        install(FILES ${BUILD_DEVICE_DIR}/signatures/aicpu_hccl.tar.gz
            DESTINATION ${INSTALL_DEVICE_TAR_DIR} OPTIONAL
        )
        install(FILES ${BUILD_DEVICE_DIR}/aicpu_custom/libscatter_aicpu_kernel.so
            DESTINATION ${INSTALL_AICPU_KERNEL_JSON_DIR} OPTIONAL
        )
        add_dependencies(install_tar hccl)
    endif()

    set(HCCL_HEAD
        include/hccl.h
    )

    set(HCCL_JSON
        # hccl json 待补充
    )

    install(FILES
        ${HCCL_HEAD}
        ${HCCL_JSON}
        DESTINATION ${INSTALL_INCLUDE_DIR}/hccl/ OPTIONAL
    )
endif()

# this is a test commit