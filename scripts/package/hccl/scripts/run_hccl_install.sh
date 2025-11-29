#!/bin/bash
# Perform install for hccl package
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

username="$(id -un)"
usergroup="$(id -gn)"
is_quiet=n
pylocal=n
in_install_for_all=n
setenv_flag=n
docker_root=""
sourcedir="$PWD/hccl"
curpath=$(dirname $(readlink -f "$0"))
common_func_path="${curpath}/common_func.inc"
pkg_version_path="${curpath}/../version.info"
chip_type="all"
feature_type="all"

. "${common_func_path}"

if [ "$1" ]; then
    input_install_dir="${2}"
    common_parse_type="${3}"
    is_quiet="${4}"
    pylocal="${5}"
    setenv_flag="${6}"
    docker_root="${7}"
    in_install_for_all="${8}"
fi

if [ "x${docker_root}" != "x" ]; then
    common_parse_dir="${docker_root}${input_install_dir}"
else
    common_parse_dir="${input_install_dir}"
fi

get_version "pkg_version" "$pkg_version_path"
is_multi_version_pkg "pkg_is_multi_version" "$pkg_version_path"
if [ "$pkg_is_multi_version" = "true" ] && [ "$hetero_arch" != "y" ]; then
    get_version_dir "pkg_version_dir" "$pkg_version_path"
    common_parse_dir="$common_parse_dir/$pkg_version_dir"
fi

if [ $(id -u) -ne 0 ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi
logfile="${log_dir}/ascend_install.log"

get_install_param() {
    local _key="$1"
    local _file="$2"
    local _param=""

    if [ ! -f "${_file}" ]; then
        exit 1
    fi
    local install_info_key_array="Hccl_Install_Type Hccl_Chip_Type Hccl_Feature_Type Hccl_UserName Hccl_UserGroup Hccl_Install_Path_Param Hccl_Arch_Linux_Path Hccl_Hetero_Arch_Flag"
    for key_param in ${install_info_key_array}; do
        if [ "${key_param}" = "${_key}" ]; then
            _param=$(grep -i "${_key}=" "${_file}" | cut -d"=" -f2-)
            break
        fi
    done
    echo "${_param}"
}

install_info="${common_parse_dir}/share/info/hccl/ascend_install.info"
if [ -f "$install_info" ]; then
    chip_type=$(get_install_param "Hccl_Chip_Type" "${install_info}")
    feature_type=$(get_install_param "Hccl_Feature_Type" "${install_info}")
    hetero_arch=$(get_install_param "Hccl_Hetero_Arch_Flag" "${install_info}")
fi

# 写日志
log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    local log_type="$1"
    shift
    if [ "$log_type" = "INFO" -o "$log_type" = "WARNING" -o "$log_type" = "ERROR" ]; then
        echo -e "[Hccl] [$cur_date] [$log_type]: $*"
    else
        echo "[Hccl] [$cur_date] [$log_type]: $*" 1> /dev/null
    fi
    echo "[Hccl] [$cur_date] [$log_type]: $*" >> "$logfile"
}

# 静默模式日志打印
new_echo() {
    local log_type="$1"
    local log_msg="$2"
    if [ "${is_quiet}" = "n" ]; then
        echo "${log_type}" "${log_msg}" 1> /dev/null
    fi
}

output_progress() {
    new_echo "INFO" "install upgradePercentage:$1%"
    log "INFO" "install upgradePercentage:$1%"
}

create_latest_linux_softlink() {
    if [ "$pkg_is_multi_version" = "true" ] && [ "$hetero_arch" = "y" ]; then
        local linux_path="$(realpath $common_parse_dir/..)"
        local arch_path="$(basename $linux_path)"
        local latest_path="$(realpath $linux_path/../..)/latest"
        if [ -d "$latest_path" ]; then
            if [ ! -e "$latest_path/$arch_path" ] || [ -L "$latest_path/$arch_path" ]; then
                ln -srfn "$linux_path" "$latest_path"
            fi
        fi
    fi
}

##########################################################################
log "INFO" "step into run_hccl_install.sh ......"
log "INFO" "install target dir $common_parse_dir, type $common_parse_type."

if [ ! -d "$common_parse_dir" ]; then
    log "ERROR" "ERR_NO:0x0001;ERR_DES:path $common_parse_dir is not exist."
    exit 1
fi

new_install() {
    if [ ! -d "${sourcedir}" ]; then
        log "INFO" "no need to install hccl files."
        return 0
    fi
    output_progress 10

    local setenv_option=""
    if [ "${setenv_flag}" = y ]; then
        setenv_option="--setenv"
    fi

    # 执行安装
    custom_options="--custom-options=--common-parse-dir=$common_parse_dir,--logfile=$logfile,--stage=install,--quiet=$is_quiet,--pylocal=$pylocal,--hetero-arch=$hetero_arch"
    sh "$curpath/install_common_parser.sh" --package="hccl" --install --username="$username" --usergroup="$usergroup" --set-cann-uninstall \
        --version=$pkg_version --version-dir=$pkg_version_dir --use-share-info \
        $setenv_option $in_install_for_all --docker-root="$docker_root" --chip="$chip_type" --feature="$feature_type" \
        $custom_options "$common_parse_type" "$input_install_dir" "$curpath/filelist.csv"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0085;ERR_DES:failed to install package."
        return 1
    fi

    create_latest_linux_softlink
    return 0
}

new_install
if [ $? -ne 0 ]; then
    exit 1
fi

output_progress 100
exit 0
