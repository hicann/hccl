# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(MAKESELF_NAME "makeself")
set(MAKESELF_FILE "makeself-release-2.5.0-patch1.tar.gz")
set(MAKESELF_URL "https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/${MAKESELF_FILE}")
set(MAKESELF_PKG_PATH ${CANN_3RD_LIB_PATH}/${MAKESELF_FILE})
set(MAKESELF_SRC_PATH ${CANN_3RD_LIB_PATH}/makeself)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# 默认配置的makeself还是不存在则下载
if (NOT EXISTS "${MAKESELF_SRC_PATH}/makeself-header.sh" OR
    NOT EXISTS "${MAKESELF_SRC_PATH}/makeself.sh")
    if(EXISTS ${MAKESELF_PKG_PATH})
        # 离线编译场景，优先使用已下载的包
        message(STATUS "[ThirdParty] Found local makeself package: ${MAKESELF_PKG_PATH}")
        set(MAKESELF_PROJECT_URL ${MAKESELF_PKG_PATH})
    else()
        # 下载并编译安装
        message(STATUS "[ThirdParty] Downloading makeself from ${MAKESELF_URL}")
        set(MAKESELF_PROJECT_URL ${MAKESELF_URL})
    endif()

    include(FetchContent)
    FetchContent_Declare(
        ${MAKESELF_NAME}
        URL ${MAKESELF_PROJECT_URL}
        URL_HASH SHA256=bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a
        DOWNLOAD_NO_PROGRESS TRUE
        DOWNLOAD_DIR ${CANN_3RD_LIB_PATH}
        SOURCE_DIR ${MAKESELF_SRC_PATH}  # 直接解压到此目录
    )
    FetchContent_MakeAvailable(${MAKESELF_NAME})
    execute_process(
        COMMAND chmod 700 "${MAKESELF_SRC_PATH}/makeself.sh"
        COMMAND chmod 700 "${MAKESELF_SRC_PATH}/makeself-header.sh"
        RESULT_VARIABLE CHMOD_RESULT
        ERROR_VARIABLE CHMOD_ERROR
    )
endif()
