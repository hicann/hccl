# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 设置 makeself 路径
set(MAKESELF_EXE ${CPACK_CMAKE_BINARY_DIR}/makeself/makeself.sh)
set(MAKESELF_HEADER_EXE ${CPACK_CMAKE_BINARY_DIR}/makeself/makeself-header.sh)
if(NOT MAKESELF_EXE)
    message(FATAL_ERROR "makeself not found! Install it with: sudo apt install makeself")
endif()

# 创建临时安装目录
set(STAGING_DIR "${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/makeself_staging")
file(MAKE_DIRECTORY "${STAGING_DIR}")

# 执行安装到临时目录
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${CPACK_CMAKE_BINARY_DIR}" --prefix "${STAGING_DIR}"
    RESULT_VARIABLE INSTALL_RESULT
    ERROR_VARIABLE  INSTALL_ERROR
)
if(NOT INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR "Installation to staging directory failed: ${INSTALL_ERROR}")
endif()

# makeself打包
set(package_path "${STAGING_DIR}")
set(package_name "cann-hccl_custom_${CPACK_CUSTOM_OPS_NAME}_linux-${CPACK_ARCH}.run")
set(package_label "CANN_HCCL_RUN_PACKAGE")
set(makeself_param 
    --pigz
    --complevel 4
    --nomd5
    --sha256
    --nooverwrite
    --chown
    --tar-format gnu
    --tar-extra
    --numeric-owner
    --tar-quietly)

execute_process(COMMAND bash ${MAKESELF_EXE}
        --header ${MAKESELF_HEADER_EXE}
        --help-header ${CPACK_CUSTOM_OPS_OPP_SCRIPTS_PATH}/help.info
        ${makeself_param}
        ${package_path}         # 打包的源目录
        ${package_name}         # 生成的归档文件名
        ${package_label}        # 标签
        ${CPACK_CUSTOM_OPS_OPP_SCRIPTS_PATH}/install.sh   # 启动脚本
        WORKING_DIRECTORY ${STAGING_DIR}
        RESULT_VARIABLE EXEC_RESULT
        ERROR_VARIABLE  EXEC_ERROR
)

if(NOT EXEC_RESULT EQUAL 0)
    message(FATAL_ERROR "makeself packaging failed: ${EXEC_ERROR}")
endif()

execute_process(
    COMMAND mkdir -p ${CPACK_PACKAGE_DIRECTORY}
    COMMAND mv ${STAGING_DIR}/${package_name} ${CPACK_PACKAGE_DIRECTORY}/
    COMMAND echo "Move ${STAGING_DIR}/${package_name} to ${CPACK_PACKAGE_DIRECTORY}/"
    WORKING_DIRECTORY ${STAGING_DIR}
)
