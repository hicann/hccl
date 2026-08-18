#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

while true; do
    case "$1" in
    --install-path=*)
        install_path=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --version-dir=*)
        version_dir=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --latest-dir=*)
        latest_dir=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    -*)
        shift
        ;;
    *)
        break
        ;;
    esac
done

get_arch_name() {
    local pkg_dir="$1"
    local scene_file="$pkg_dir/scene.info"
    grep '^arch=' $scene_file | cut -d"=" -f2
}

remove_stub_softlink() {
    local ref_dir="$1"
    if [ ! -d "$ref_dir" ]; then
        return
    fi
    local stub_dir="$2"
    if [ ! -d "$stub_dir" ]; then
        return
    fi
    local pwdbak="$(pwd)"
    cd $stub_dir && chmod u+w . && ls -1 "$ref_dir" | xargs --no-run-if-empty rm -rf
    cd $pwdbak
}

recreate_common_stub_softlink() {
    local arch_name="$1"
    local stub_dir="$2"
    if [ ! -d "$stub_dir" ]; then
        return
    fi
    local pwdbak="$(pwd)"
    cd $stub_dir && [ -L "$arch_name/libgraph.so" ] && chmod u+w . && ln -s "$arch_name/libgraph.so" libgraph.so
    cd $pwdbak
}

remove_softlink() {
    rm -rf $WHL_SOFTLINK_INSTALL_DIR_PATH/$1 > /dev/null 2>&1
}

remove_empty_dir() {
    local _path="${1}"
    if [ -d "${_path}" ]; then
        local is_empty=$(ls "${_path}" | wc -l)
        if [ "$is_empty" -eq 0 ]; then
            prev_path=$(dirname "${_path}")
            chmod +w "${prev_path}" > /dev/null 2>&1
            rm -rf "${_path}" > /dev/null 2>&1
        fi
    fi
}

python_dir_chmod_set() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        return
    fi
    chmod u+w "$dir" > /dev/null 2>&1
}

WHL_SOFTLINK_INSTALL_DIR_PATH="$install_path/$latest_dir/python/site-packages"

python_dir_chmod_set "$WHL_SOFTLINK_INSTALL_DIR_PATH"

remove_softlink "hccl"
remove_softlink "hccl-*.dist-info"

remove_empty_dir "$WHL_SOFTLINK_INSTALL_DIR_PATH"
remove_empty_dir "$install_path/$latest_dir/python"
