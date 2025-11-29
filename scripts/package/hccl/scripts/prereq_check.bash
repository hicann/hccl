#!/bin/bash
# Perform pre-check for hccl package
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

MIN_PIP_VERSION=19
PYTHON_VERSION=3.7.5

IS_QUIET=y
if [ "x$1" = "x--no-quiet" ]; then
    IS_QUIET=n
fi

log() {
    local content=$(echo "$@" | cut -d" " -f2-)
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    echo -e "[Hccl] [$cur_date] [$1]: $content"
}

input_check() {
    [ "${IS_QUIET}" = y ] && return
    log "INFO" "\033[32mDo you want to continue? [y/n]\033[0m"
    while true; do
        read yn
        if [ "$yn" = n ]; then
            exit 1
        elif [ "$yn" = y ]; then
            break
        else
            log "ERROR" "ERR_NO:0x0002;ERR_DES:input error, please input again!"
        fi
    done
}

version_lt() {
    test $(echo "$@" | tr " " "\n" | sort -rV | head -n 1) != "$1"
}

check_py_module_depends() {
    local module_name="$1"
    local module_version="$2"
    local depends_message="$3"
    local required_message=$(eval echo "$4")
    pip3 show "$module_name" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log "WARNING" "\033[33mpython module '$module_name' is not installed$required_message$depends_message\033[0m"
    elif [ "x$module_version" != "x" ]; then
        local version=$(pip3 show "$module_name" 2> /dev/null | grep -iw ^version | cut -d' ' -f2)
        if [ "x$version" = "x" ]; then
            log "WARNING" "\033[33mcannot get module '$module_name' version\033[0m"
        elif version_lt "$version" "$module_version"; then
            log "WARNING" "\033[33mpython module '$module_name' version ${version} is lower than ${module_version}$required_message\033[0m"
        fi
    fi
}

which pip3 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    pip3_version="$(pip3 --version 2>/dev/null | head -n 1)"
    pip3_version=$(expr "$pip3_version" : 'pip \([0-9]\+\(\.[0-9]\+\)\+\)')
    if [ "x$pip3_version" = "x" ]; then
        log "WARNING" "\033[33mcannot get pip3 version\033[0m"
        input_check
    elif version_lt "$pip3_version" "$MIN_PIP_VERSION"; then
        log "WARNING" "\033[33mpip3 version ${pip3_version} is lower than ${MIN_PIP_VERSION}.x.x\033[0m"
        input_check
    fi
else
    log "WARNING" "\033[33mpip3 is not found.\033[0m"
fi

curpath="$(dirname ${BASH_SOURCE:-$0})"
install_dir="$(realpath $curpath/..)"
common_interface=$(realpath $install_dir/script*/common_interface.bash)
if [ -f "$common_interface" ]; then
    . "$common_interface"
    py_version_check
fi
