#!/bin/bash
# Perform custom_install script for hccl package
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

sourcedir="$PWD/hccl"
curpath=$(dirname $(readlink -f "$0"))
common_func_path="${curpath}/common_func.inc"
unset PYTHONPATH

. "${common_func_path}"

common_parse_dir=""
logfile=""
stage=""
is_quiet="n"
pylocal="n"
hetero_arch="n"

while true; do
    case "$1" in
    --install-path=*)
        pkg_install_path=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --common-parse-dir=*)
        common_parse_dir=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --version-dir=*)
        pkg_version_dir=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --logfile=*)
        logfile=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --stage=*)
        stage=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --quiet=*)
        is_quiet=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --pylocal=*)
        pylocal=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --hetero-arch=*)
        hetero_arch=$(echo "$1" | cut -d"=" -f2)
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

hccl_install_package() {
    local _package="$1"
    local _pythonlocalpath="$2"
    log "INFO" "install python module package in ${_package}"
    if [ -f "$_package" ]; then
        if [ "$pylocal" = "y" ]; then
            pip3 install --disable-pip-version-check --upgrade --no-deps --force-reinstall "${_package}" -t "${_pythonlocalpath}" 1> /dev/null
        else
            if [ $(id -u) -ne 0 ]; then
                pip3 install --disable-pip-version-check --upgrade --no-deps --force-reinstall "${_package}" --user 1> /dev/null
            else
                pip3 install --disable-pip-version-check --upgrade --no-deps --force-reinstall "${_package}" 1> /dev/null
            fi
        fi
        local ret=$?
        if [ $ret -ne 0 ]; then
            log "WARNING" "install ${_package} failed, error code: $ret."
            exit 1
        else
            log "INFO" "install ${_package} successfully!"
        fi
    else
        log "ERROR" "ERR_NO:0x0080;ERR_DES:install ${_package} faied, can not find the matched package for this platform."
        exit 1
    fi
}

create_auto_tune_soft_link() {
    local _src_dir="$1"
    local _dst_dir="$2"
    local _package_name="$3"
    local _package_egg_name="$4"
    if [ -d "${_src_dir}/${_package_name}" ]; then
        mkdir -p "${_dst_dir}/${_package_egg_name}/${_package_name}"
        ln -sr "$_src_dir/${_package_name}" "${_dst_dir}/${_package_egg_name}/${_package_name}/${_package_name}"
        ln -sr "$_src_dir/${_package_name}" "${_dst_dir}/${_package_egg_name}/${_package_name}/auto_tune_main"
        ln -sr "$_src_dir/${_package_name}" "${_dst_dir}/auto_tune_main"
    fi
}

create_rl_soft_link() {
    local _src_dir="$1"
    local _dst_dir="$2"
    local _package_name="$3"
    local _package_egg_name="$4"
    if [ -d "${_src_dir}/${_package_name}" ]; then
        mkdir -p "${_dst_dir}/${_package_egg_name}"
        ln -sr "$_src_dir/${_package_name}" "${_dst_dir}/${_package_egg_name}/${_package_name}"
    fi
}

# 在指定路径下搜索文件
__find_files_in_paths() {
    local paths="$1"
    local out_varname="$2"
    local result=""
    local total_result=""

    for path in ${paths}; do
        if [ "${path}" != "" ] && [ -d "${path}" ]; then
            result="$(find -L "${path}" -type f -print0 | tr '\0' ';')"
            total_result="${total_result}${result}"
        fi
    done
    eval "${out_varname}=\"${total_result%%;}\""
}

search_user_assets() {
    local install_path="$1"
    local default_install_path="$2"
    local out_varname="$3"

    __find_files_in_paths "${install_path}" "$out_varname"
    if [ "$(eval echo \$$out_varname)" != "" ]; then
        last_valid_user_assets_dir="${install_path}"
        last_valid_user_assets_dir_real="$(realpath ${install_path})"
        return 0
    fi

    __find_files_in_paths "${default_install_path}" "$out_varname"
    if [ "$(eval echo \$$out_varname)" != "" ]; then
        last_valid_user_assets_dir="${default_install_path}"
        last_valid_user_assets_dir_real="$(realpath ${default_install_path})"
        return 0
    fi

    eval "${out_varname}=''"
    return 1
}

get_newname() {
    local fname="$1"
    local bname="$(basename "$fname")"
    while true; do
        log "INFO" "\033[32mplease input a new name for file '$bname' (input q to quit, input l to list files in target directory):\033[0m"
        read newname
        if [ "$newname" = "q" ]; then
            newname=""
            return
        elif [ "$newname" = "l" ]; then
            newname=""
            echo "# ls $(dirname $fname)" && ls "$(dirname $fname)"
        elif [ "x$newname" != "x" ]; then
            local matched=$(expr "$newname" : '\([a-zA-Z0-9_\-\.]\+\)')
            if [ "$newname" = "$matched" ]; then
                return
            else
                log "INFO" "\033[32mfile name '$newname' is invalid, a valid file name can only includes a-zA-Z0-9_-.\033[0m"
            fi
        fi
    done
}

get_input() {
    local fname="$1"
    if [ "x$all_yes_flag" = "x" ]; then
        all_yes_flag=n
    fi
    overwrite=y
    [ "${is_quiet}" = y ] && return
    [ "$all_yes_flag" = y ] && return
    while true; do
        log "INFO" "\033[32mfile '$fname' already exists, do you want to overwrite/rename it? [y(yes)/n(no)/r(rename)/a(yes to all)]\033[0m"
        read yn
        if [ "$yn" = y -o "$yn" = yes ]; then
            overwrite=y
            return
        elif [ "$yn" = n -o "$yn" = no ]; then
            overwrite=n
            return
        elif [ "$yn" = r -o "$yn" = rename ]; then
            overwrite=r
            get_newname "$fname"
            [ "x$newname" != "x" ] && return
        elif [ "$yn" = a -o "$yn" = all ]; then
            overwrite=y
            all_yes_flag=y
            return
        else
            log "INFO" "\033[32munknown input, please input again!\033[0m"
        fi
    done
}

migrate_user_assets() {
    local pkg_name="$1"
    local data_dir="data/fusion_strategy/custom"

    local install_path="$common_parse_dir/$pkg_name/$data_dir"

    local default_install_path="/usr/local/Ascend/$pkg_name/$data_dir"
    if [ $(id -u) -ne 0 ]; then
        default_install_path="${HOME}/Ascend/$pkg_name/$data_dir"
    fi

    if [ "x$last_valid_user_assets_dir_real" != "x" ]; then
        if [ "x$install_path" != "x" ] && [ "$(realpath "$install_path" 2>&1)" = "$last_valid_user_assets_dir_real" ]; then
            log "INFO" "'$install_path' and '$last_valid_user_assets_dir' are the same path, no need to migrate user assets."
            return
        fi
        if [ "x$default_install_path" != "x" ] && [ "$(realpath "$default_install_path" 2>&1)" = "$last_valid_user_assets_dir_real" ]; then
            log "INFO" "'$default_install_path' and '$last_valid_user_assets_dir' are the same path, no need to migrate user assets."
            return
        fi
    fi

    search_user_assets "$install_path" "$default_install_path" path_list
    if [ "x$path_list" != "x" ]; then
        OLD_IFS="$IFS"
        IFS=';'
        for src in $path_list
        do
            dst=${src##*$pkg_name/}
            dst="$common_parse_dir/share/info/hccl/$dst"

            if [ "$dst" = "$(realpath "$src")" ]; then
                log "INFO" "file '$src' and '$dst' are the same file, no need copy."
            else
                log "INFO" "migrating file '$src' ..."
                if [ -e "$dst" ]; then
                    get_input "$dst"
                    if [ "$overwrite" = y ]; then
                        log "INFO" "cp -rpf $src $dst"
                        mkdir -p $(dirname "$dst") && cp -rpf "$src" "$dst"
                    elif [ "$overwrite" = r ]; then
                        dst="$(dirname "$dst")/$newname"
                        log "INFO" "cp -rpf $src $dst"
                        mkdir -p $(dirname "$dst") && cp -rpf "$src" "$dst"
                    fi
                else
                    log "INFO" "cp -rpf $src $dst"
                    mkdir -p $(dirname "$dst") && cp -rpf "$src" "$dst"
                fi
            fi
        done
        IFS="$OLD_IFS"
    fi
}

clear_kernel_cache_dir() {
    local dir_atc_data="$HOME/atc_data"
    local dir_kernel_caches="$(ls -d $dir_atc_data/kernel_cache* 2> /dev/null)"
    if [ -z "$dir_kernel_caches" ]; then
        return
    fi
    if [ -w "$dir_atc_data" ]; then
        for dir_cache in $dir_kernel_caches; do
            [ ! -d "$dir_cache" ] && continue
            [ -n "$dir_cache" ] && rm -rf "$dir_cache" > /dev/null 2>&1
            if [ -d "$dir_cache" ]; then
                log "WARNING" "failed to delete directory '$dir_cache'"
            else
                log "INFO" "directory '$dir_cache' was deleted."
            fi
        done
    else
        log "WARNING" "current user do not have permission to delete kernel_cache* directories in '$dir_atc_data'."
    fi
}

custom_install() {
    if [ -z "$common_parse_dir/share/info/hccl" ]; then
        log "ERROR" "ERR_NO:0x0001;ERR_DES:hccl directory is empty"
        exit 1
    fi

    if [ "$hetero_arch" != "y" ]; then
        local arch_name="$(get_arch_name $common_parse_dir/share/info/hccl)"
    else
        local arch_name="$(get_arch_name $common_parse_dir/share/info/hccl)"
    fi

    if [ "$hetero_arch" != "y" ]; then
        if [ "x$stage" = "xinstall" ]; then
            log "INFO" "hccl do migrate user assets."
            migrate_user_assets atc
            migrate_user_assets fwkacllib
            if [ $? -ne 0 ]; then
                log "WARNING" "failed to copy custom directories."
                return 1
            fi
        fi
    fi

    return 0
}

custom_install
if [ $? -ne 0 ]; then
    exit 1
fi

clear_kernel_cache_dir
exit 0
