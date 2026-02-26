# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.
# ----------------------------------------------------------------------------

include_guard(GLOBAL)

unset(gtest_FOUND CACHE)
unset(GTEST_INCLUDE CACHE)
unset(GTEST_STATIC_LIBRARY CACHE)
unset(GTEST_MAIN_STATIC_LIBRARY CACHE)
unset(GMOCK_STATIC_LIBRARY CACHE)
unset(GMOCK_MAIN_STATIC_LIBRARY CACHE)

set(GTEST_FILE "googletest-1.14.0.tar.gz")
set(GTEST_URL "https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/${GTEST_FILE}")
set(GTEST_PKG_PATH ${CANN_3RD_LIB_PATH}/${GTEST_FILE})
set(GTEST_INSTALL_PATH ${CANN_3RD_LIB_PATH}/gtest)

# 查找目录下是否已经安装，避免重复编译安装
message(STATUS "[ThirdParty] GTEST_INSTALL_PATH=${GTEST_INSTALL_PATH}")
find_path(GTEST_INCLUDE
    NAMES gtest/gtest.h
    PATH_SUFFIXES include
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
    PATHS ${GTEST_INSTALL_PATH}
)
find_library(GTEST_STATIC_LIBRARY
    NAMES libgtest.a
    PATH_SUFFIXES lib64
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
    PATHS ${GTEST_INSTALL_PATH}
)
find_library(GTEST_MAIN_STATIC_LIBRARY
    NAMES libgtest_main.a
    PATH_SUFFIXES lib64
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
    PATHS ${GTEST_INSTALL_PATH}
)
find_library(GMOCK_STATIC_LIBRARY
    NAMES libgmock.a
    PATH_SUFFIXES lib64
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
    PATHS ${GTEST_INSTALL_PATH}
)
find_library(GMOCK_MAIN_STATIC_LIBRARY
    NAMES libgmock_main.a
    PATH_SUFFIXES lib64
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
    PATHS ${GTEST_INSTALL_PATH}
)

# 是否全部找到 gtest 的头文件和二进制
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(gtest
    FOUND_VAR
    gtest_FOUND
    REQUIRED_VARS
    GTEST_INCLUDE
    GTEST_STATIC_LIBRARY
    GTEST_MAIN_STATIC_LIBRARY
    GMOCK_STATIC_LIBRARY
    GMOCK_MAIN_STATIC_LIBRARY
)
message(STATUS "[ThirdParty] Found GTest: ${gtest_FOUND}")

if(gtest_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message(STATUS "[ThirdParty] GTest found in ${GTEST_INSTALL_PATH}, and not force rebuild cann third_party")
else()
    if(EXISTS ${GTEST_PKG_PATH})
        # 离线编译场景，优先使用已下载的包
        message(STATUS "[ThirdParty] Found local gtest package: ${GTEST_PKG_PATH}")
        set(GTEST_PROJECT_URL ${GTEST_PKG_PATH})
    else()
        # 下载并编译安装
        message(STATUS "[ThirdParty] Downloading GTest from ${GTEST_URL}")
        set(GTEST_PROJECT_URL ${GTEST_URL})
    endif()

    # 编译选项设置
    set(GTEST_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -O2 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
    set(GTEST_CFLAGS   "-D_GLIBCXX_USE_CXX11_ABI=0 -O2 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")

    set(GTEST_OPTS
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_CXX_FLAGS=${GTEST_CXXFLAGS}
        -DCMAKE_C_FLAGS=${GTEST_CFLAGS}
        -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_PATH}
        -DCMAKE_INSTALL_LIBDIR=lib64
        -DBUILD_TESTING=OFF
        -DBUILD_SHARED_LIBS=OFF
    )

    include(ExternalProject)
    ExternalProject_Add(third_party_gtest
        URL ${GTEST_PROJECT_URL}
        URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
        TLS_VERIFY OFF
        DOWNLOAD_DIR ${CANN_3RD_LIB_PATH}
        DOWNLOAD_NO_PROGRESS TRUE
        CONFIGURE_COMMAND ${CMAKE_COMMAND} ${GTEST_OPTS} <SOURCE_DIR>
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND $(MAKE) install
        EXCLUDE_FROM_ALL TRUE
    )
endif()

# 创建导入的目标
add_library(gtest STATIC IMPORTED)
add_dependencies(gtest third_party_gtest)

add_library(gmock STATIC IMPORTED)
add_dependencies(gmock third_party_gtest)

add_library(gtest_main STATIC IMPORTED)
add_dependencies(gtest_main third_party_gtest)

if(NOT EXISTS ${GTEST_INSTALL_PATH}/include)
    file(MAKE_DIRECTORY "${GTEST_INSTALL_PATH}/include")
endif()

set_target_properties(gtest PROPERTIES
    IMPORTED_LOCATION ${GTEST_INSTALL_PATH}/lib64/libgtest.a
    INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INSTALL_PATH}/include
)

set_target_properties(gmock PROPERTIES
    IMPORTED_LOCATION ${GTEST_INSTALL_PATH}/lib64/libgmock.a
    INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INSTALL_PATH}/include
)

set_target_properties(gtest_main PROPERTIES
    IMPORTED_LOCATION ${GTEST_INSTALL_PATH}/lib64/libgtest_main.a
    INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INSTALL_PATH}/include
)
