#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# The code snippet comes from Huawei's open-source Ascend project.
# Copyright 2020-2021 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# ----------------------------------------------------------------------------

target_install_path=/usr/local/Ascend/cann/opp
source_path=$PWD

INSTALL="y"
UNINSTALL="n"
QUIET="n"

while true
do
    case $1 in
    --quiet)
        QUIET="y"
        shift
    ;;
    --install)
        INSTALL="y"
        shift
    ;;
    --uninstall)
        UNINSTALL="y"
        shift
    ;;
    --install-path=*)
        INSTALL_PATH=$(echo $1 | cut -d"=" -f2-)
        INSTALL_PATH=${INSTALL_PATH%*/}
        shift
    ;;
    --*)
        shift
    ;;
    *)
        break
    ;;
    esac
done

log() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[hccl_custom_ops] [$cur_date] "$1
}

# 安装路径优先级：--install-path > ASCEND_CUSTOM_OPP_PATH > ASCEND_OPP_PATH
if [ -n "${INSTALL_PATH}" ]; then
    if [[ ! "${INSTALL_PATH}" = /* ]]; then
        log "[ERROR] use absolute path for --install-path argument"
        exit 1
    fi
    if [ ! -d ${INSTALL_PATH} ]; then
        mkdir -p ${INSTALL_PATH} >> /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "[ERROR] create ${INSTALL_PATH} failed"
            exit 1
        fi
    fi
    target_install_path=${INSTALL_PATH}
elif [ -n "${ASCEND_CUSTOM_OPP_PATH}" ]; then
    if [ ! -d ${ASCEND_CUSTOM_OPP_PATH} ]; then
        mkdir -p ${ASCEND_CUSTOM_OPP_PATH} >> /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "[ERROR] create ${ASCEND_CUSTOM_OPP_PATH} failed"
            exit 1
        fi
    fi
    target_install_path=${ASCEND_CUSTOM_OPP_PATH}
else
    if [ "x${ASCEND_OPP_PATH}" == "x" ]; then
        log "[ERROR] env ASCEND_OPP_PATH no exist"
        exit 1
    fi
    target_install_path=$(dirname "${ASCEND_OPP_PATH}")
fi

if [ ! -d $target_install_path ];then
    log "[ERROR] $target_install_path no exist"
    exit 1
fi

if [ ! -x $target_install_path ] || [ ! -w $target_install_path ] || [ ! -r $target_install_path ];then
    log "[WARNING] The directory $target_install_path does not have sufficient permissions. \
    Please check and modify the folder permissions (e.g., using chmod), \
    or use the --install-path option to specify an installation path and \
    change the environment variable ASCEND_CUSTOM_OPP_PATH to the specified path."
fi

# 卸载
uninstall() {
    log "[INFO] Uninstalling existing files..."

    if [ ! -d "$source_path" ]; then
        log "[ERROR] Source directory does not exist"
        exit 1
    fi

    while IFS= read -r -d '' item; do
        rel_path="${item#$source_path/}"
        target_item="$target_install_path/$rel_path"

        # 删除文件
        if [ -f "$target_item" ]; then
            rm -f "$target_item" && log "[Info] Removed file: ${target_item}"
        # 删除空目录
        elif [ -d "$target_item" ] && [ -z "$(ls -A "$target_item" 2>/dev/null)" ]; then
            rmdir "$target_item" 2>/dev/null && log "[Info] Removed empty directory: ${target_item}"
        fi
    done < <(find "$source_path" -depth -print0)

    log "[INFO] Uninstalled successfully"
}

# 安装函数
install() {
    log "[INFO] Installing files..."
    log "[INFO] Installation directory: ${target_install_path}"

    # 检查源目录
    if [ ! -d "$source_path" ]; then
        log "[ERROR] Source directory does not exist"
        exit 1
    fi

    # 检查是否有冲突文件
    conflict_files=()
    while IFS= read -r -d '' file; do
        rel_path="${file#$source_path/}"
        if [ -e "$target_install_path/$rel_path" ]; then
            conflict_files+=("$rel_path")
        fi
    done < <(find "$source_path" -type f -print0)

    # 如果有冲突文件且非安静模式，提示用户进行卸载
    if [ ${#conflict_files[@]} -gt 0 ] && [ "$QUIET" != "y" ]; then
        echo "Files already exist in the target directory. Do you want to uninstall them first? (y/N):"
        read -r user_input
        if [ "$user_input" = "y" ] || [ "$user_input" = "Y" ]; then
            uninstall
        else
            log "[INFO] Installation cancelled by user"
            exit 0
        fi
    elif [ ${#conflict_files[@]} -gt 0 ] && [ "$QUIET" = "y" ]; then
        # 安静模式下自动卸载
        uninstall
    fi

    # 创建目标目录
    mkdir -p "$target_install_path"

    # 复制所有文件
    cp -rf "$source_path"/* "$target_install_path"/ 2>/dev/null
    if [ $? -ne 0 ]; then
        log "[ERROR] Failed to copy files"
        exit 1
    fi

    log "[INFO] Installation successfully"
}

# 主函数
main() {
    # 卸载
    if [ "$UNINSTALL" = "y" ]; then
        uninstall
        exit 0
    fi

    # 安装
    if [ "$INSTALL" = "y" ]; then
        install
        exit 0
    fi

    log "[ERROR] No operation specified. Use --install or --uninstall"
    exit 1
}

# 主函数
main
exit 0
