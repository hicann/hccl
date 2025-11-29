#!/bin/bash
# Perform custom create softlink script for hccl package
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

curpath=$(dirname $(readlink -f "$0"))
common_func_path="${curpath}/common_func.inc"

. "${common_func_path}"

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

create_stub_softlink() {
    local stub_dir="$1"
    if [ ! -d "$stub_dir" ]; then
        return
    fi
    local arch_name="$2"
    local pwdbak="$(pwd)"
    cd $stub_dir && [ -d "$arch_name" ] && for so_file in $(find "$arch_name" -type f -o -type l); do
        ln -sf "$so_file" "$(basename $so_file)"
    done
    cd $pwdbak
}

create_auto_tune_soft_link() {
    local _src_dir="$1"
    local _dst_dir="$2"
    local _package_name="$3"
    local _package_egg_name="$4"
    if [ -d "${_src_dir}/${_package_name}" ]; then
        mkdir -p "${_dst_dir}/${_package_egg_name}/${_package_name}"
        ln -srfn "$_src_dir/${_package_name}" "${_dst_dir}/${_package_egg_name}/${_package_name}/${_package_name}"
        ln -srfn "$_src_dir/${_package_name}" "${_dst_dir}/${_package_egg_name}/${_package_name}/auto_tune_main"
        ln -srfn "$_src_dir/${_package_name}" "${_dst_dir}/auto_tune_main"
    fi
}

create_rl_soft_link() {
    local _src_dir="$1"
    local _dst_dir="$2"
    local _package_name="$3"
    local _package_egg_name="$4"
    if [ -d "${_src_dir}/${_package_name}" ]; then
        mkdir -p "${_dst_dir}/${_package_egg_name}"
        ln -srfn "$_src_dir/${_package_name}" "${_dst_dir}/${_package_egg_name}/${_package_name}"
    fi
}

python_dir_chmod_set() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        return
    fi
    if [ $(id -u) -eq 0 ]; then
        chmod 755 "$dir" > /dev/null 2>&1
    else
        chmod 750 "$dir" > /dev/null 2>&1
    fi
}

python_dir_chmod_reset() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        return
    fi
    chmod u+w "$dir" > /dev/null 2>&1
}

WHL_INSTALL_DIR_PATH="$install_path/$version_dir/python/site-packages"
WHL_SOFTLINK_INSTALL_DIR_PATH="$install_path/$latest_dir/python/site-packages"

mkdir -p "$WHL_SOFTLINK_INSTALL_DIR_PATH"
python_dir_chmod_reset "$WHL_SOFTLINK_INSTALL_DIR_PATH"

create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "hccl"
create_softlink_if_exists "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "hccl-*.dist-info"

python_dir_chmod_set "$WHL_SOFTLINK_INSTALL_DIR_PATH"
python_dir_chmod_set "$(dirname $WHL_SOFTLINK_INSTALL_DIR_PATH)"
