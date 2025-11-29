#!/bin/bash
# Perform version check for compiler package
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

############### 全局变量定义 ###############
CURPATH=$(dirname $(readlink -f "$0"))   # 当前路径
COMMON_FUNC="${CURPATH}/common_func.inc" # 公共
DEP_INFO_FILE="/etc/ascend_install.info"

if [ ! "$1" ]; then
    version_info_file="$(dirname ${CURPATH})/version.info"
    DEP_PKG_NAME="driver"
    driver_install_path_param="$(grep -iw driver_install_path_param $DEP_INFO_FILE | cut -d"=" -f2-)"
    DEP_PKG_VER_FILE="${driver_install_path_param}/${DEP_PKG_NAME}/version.info"
else
    version_info_file="$1"
    DEP_PKG_NAME="$2"     # 依赖包名
    DEP_PKG_VER_FILE="$3" # 依赖包路径
fi

# 写日志
log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    echo "[Compiler] [$cur_date] $*"
}

main() {
    if [ ! -f "$version_info_file" ]; then
        log "[WARNING]: file $version_info_file not exists!"
        return 0
    fi

    if [ ! -f "$DEP_PKG_VER_FILE" ]; then
        log "[WARNING]: file $DEP_PKG_VER_FILE not exists!"
        return 0
    fi

    . "${COMMON_FUNC}"
    check_pkg_ver_deps "${version_info_file}" "${DEP_PKG_NAME}" "${DEP_PKG_VER_FILE}"

    if [ "${ver_check_status}" = "SUCC" ]; then
        log "[INFO]: Check version matched!"
        return 0
    else
        log "[WARNING]: Check version does not matched!"
        return 1
    fi
}

main
