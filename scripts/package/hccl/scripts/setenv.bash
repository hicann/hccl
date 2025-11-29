#!/bin/bash
# Perform setenv for hccl package
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

curpath="$(dirname ${BASH_SOURCE:-$0})"
curfile="$(realpath ${BASH_SOURCE:-$0})"
param_mult_ver=$1

get_install_param() {
    local _key="$1"
    local _file="$2"
    local _param=""

    if [ ! -f "${_file}" ]; then
        exit 1
    fi
    local install_info_key_array="Hccl_Install_Type Hccl_Feature_Type Hccl_UserName Hccl_UserGroup Hccl_Install_Path_Param Hccl_Arch_Linux_Path Hccl_Hetero_Arch_Flag"
    for key_param in ${install_info_key_array}; do
        if [ "${key_param}" = "${_key}" ]; then
            _param=$(grep -i "${_key}=" "${_file}" | cut --only-delimited -d"=" -f2-)
            break
        fi
    done
    echo "${_param}"
}

get_install_dir() {
    local install_info="$curpath/../ascend_install.info"
    local hetero_arch=$(get_install_param "Hccl_Hetero_Arch_Flag" "${install_info}")
    if [ "$param_mult_ver" = "multi_version" ]; then
        if [ "$hetero_arch" = "y" ]; then
            echo "$(realpath $curpath/../../../../../cann)"
        else
            echo "$(realpath $curpath/../../../cann)"
        fi
    else
        echo "$(realpath $curpath/..)"
    fi
}

INSTALL_DIR="$(get_install_dir)"
lib_path="${INSTALL_DIR}/python/site-packages/"
if [ -d "${lib_path}" ]; then
    python_path="${PYTHONPATH}"
    num=$(echo ":${python_path}:" | grep ":${lib_path}:" | wc -l)
    if [ "${num}" -eq 0 ]; then
        if [ "-${python_path}" = "-" ]; then
            export PYTHONPATH="${lib_path}"
        else
            export PYTHONPATH="${lib_path}:${python_path}"
        fi
    fi
fi

library_path="${INSTALL_DIR}/lib64"
if [ -d "${library_path}" ]; then
    ld_library_path="${LD_LIBRARY_PATH}"
    num=$(echo ":${ld_library_path}:" | grep ":${library_path}:" | wc -l)
    if [ "${num}" -eq 0 ]; then
        if [ "-${ld_library_path}" = "-" ]; then
            export LD_LIBRARY_PATH="${library_path}:${library_path}/plugin/nnengine:${library_path}/stub"
        else
            export LD_LIBRARY_PATH="${library_path}:${library_path}/plugin/nnengine:${ld_library_path}:${library_path}/stub"
        fi
    fi
fi

custom_path_file="$INSTALL_DIR/../conf/path.cfg"
common_interface="$INSTALL_DIR/script/common_interface.bash"
owner=$(stat -c %U "$curfile")
if [ $(id -u) -ne 0 ] && [ "$owner" != "$(whoami)" ] && [ -f "$custom_path_file" ] && [ -f "$common_interface" ]; then
    . "$common_interface"
    mk_custom_path "$custom_path_file"
    for dir_name in "conf" "data"; do
        dst_dir="$(grep -w "$dir_name" "$custom_path_file" | cut --only-delimited -d"=" -f2-)"
        eval "dst_dir=$dst_dir"
        if [ -d "$INSTALL_DIR/$dir_name" ] && [ -d "$dst_dir" ]; then
            chmod -R u+w $dst_dir/* > /dev/null 2>&1
            cp -rfL $INSTALL_DIR/$dir_name/* "$dst_dir"
        fi
    done
fi
