#!/bin/bash
# Perform uninstall for hccl package
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

############### 全局变量定义 ###############
HOST="127.0.0.1"
RUN_USERNAME="$(id -un)"      # 当前执行用户
DEFAULT_ROOT_DIR="/usr/local/Ascend"  # 根用户默认安装路径
DEFAULT_NORMAL_DIR="${HOME}/Ascend"   # 普通用户默认安装路径

curpath="$(dirname $(readlink -f $0))" # 脚本目录
common_func_path="${curpath}/common_func.inc"
hccl_func_path="${curpath}/hccl_func.sh"
UNINSTALL_SHELL="${curpath}/run_hccl_uninstall.sh" # 卸载脚本路径
install_path_param="$(dirname "${curpath}")"

SCENE_INFO_FILE="$install_path_param/scene.info"
version_info_file="$install_path_param/version.info"
ASCEND_INSTALL_INFO_FILE="$install_path_param/ascend_install.info"
ASCEND_INSTALL_INFO_OLD_FILE="/etc/ascend_install.info"

RUN_CMD="uninstall"        # 执行程序命令
RUN_CMD_TYPE="Uninstall"   # 执行程序命令类型
IS_QUIET="n"               # 静默模式默认为否

. "${common_func_path}"
. "${hccl_func_path}"

# 执行程序等级
case "${RUN_CMD_TYPE}" in
    Install)
        LEVEL="SUGGESTION"
        ;;
    Upgrade)
        LEVEL="MINOR"
        ;;
    Uninstall)
        LEVEL="MAJOR"
        ;;
    *)
        LEVEL="UNKNOWN"
        ;;
esac

if [ "$1" ]; then
    RUN_FILE_NAME="$(expr substr $1 5 $(expr ${#1} - 4))"
fi

# 判断当前用户身份, 指定日志和安装路径
if [ $(id -u) -ne 0 ]; then
    LOG_DIR="${HOME}/var/log/ascend_seclog"  # 普通用户日志存放目录
    DEFAULT_INSTALL_PATH="${HOME}/Ascend"    # 普通用户默认安装路径
else
    LOG_DIR="/var/log/ascend_seclog"          # 根用户日志存放目录
    DEFAULT_INSTALL_PATH="/usr/local/Ascend"  # 根用户默认安装路径
fi

logfile="${LOG_DIR}/ascend_install.log"        # 安装日志文件路径
OPERATION_LOG_FILE="${LOG_DIR}/operation.log"  # 操作日志路径

############### 日志函数 ###############
# 过程日志打印
log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    local log_type="$1"
    local log_msg="$2"
    local log_format="[Hccl] [$cur_date] [$log_type]: $log_msg"
    if [ "$log_type" = "INFO" ]; then
        echo "$log_format"
    elif [ "$log_type" = "WARNING" ]; then
        echo "$log_format"
    elif [ "$log_type" = "ERROR" ]; then
        echo "$log_format"
    elif [ "$log_type" = "DEBUG" ]; then
        echo "$log_format" 1> /dev/null
    fi
    echo "$log_format" >> "$logfile"
}

# 静默模式日志打印
new_echo() {
    local log_type="$1"
    local log_msg="$2"
    if [ "${is_quiet}" = "n" ]; then
        echo "${log_type}" "${log_msg}" 1> /dev/null
    fi
}

# 开始执行前打印开始信息
start_log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    new_echo "INFO" "Start time:${cur_date}"
    log "INFO" "Start time:${cur_date}"
    log "INFO" "LogFile:${logfile}"
    log "INFO" "InputParams:--${RUN_CMD}"
}

# 退出时打印结束日志
exit_log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    new_echo "INFO" "End time:${cur_date}"
    log "INFO" "End time:${cur_date}"
    exit "$1"
}

# 打印操作日志
log_operation() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    if [ ! -f "${OPERATION_LOG_FILE}" ]; then
        touch "${OPERATION_LOG_FILE}"
        chmod 640 "${OPERATION_LOG_FILE}"
    fi
    echo "${RUN_CMD_TYPE} ${LEVEL} ${RUN_USERNAME} ${cur_date} ${HOST} hccl $2 installmode=${RUN_CMD}; cmdlist=--${RUN_CMD}" >> "${OPERATION_LOG_FILE}"
}

############### 错误函数 ###############
# 不支持的参数
err_no0x0004() {
    log "ERROR" "ERR_NO:0x0004;ERR_DES: Unrecognized parameters: $1"
    exit_log 1
}

# 文件没有找到
err_no0x0080() {
    log "ERROR" "ERR_NO:0x0080;ERR_DES:This file or directory does not exist, $1"
    exit_log 1
}

# 用户权限不足
err_no0x0093() {
    log "ERROR" "ERR_NO:0x0093;ERR_DES:Permission denied, $1"
    exit_log 1
}

############### 环境适配函数 ###############
chmod_start() {
    chmod -R 750 "$install_path_param"
}

############### 检验函数 ###############
# 用户权限认证
user_auth() {
    local dir_user_id=$(stat -c "%u" "$install_path_param")
    local run_user_id=$(id -u)
    if [ "${run_user_id}" -ne 0 ]; then
        if [ "${run_user_id}" -ne "${dir_user_id}" ]; then
            err_no0x0093 "Current user is not supported to ${RUN_CMD} the hccl package"
        fi
    fi
}

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
            _param=$(grep -i "${_key}=" "${_file}" | cut -d"=" -f2-)
            break
        fi
    done
    echo "${_param}"
}

hetero_arch=$(get_install_param "Hccl_Hetero_Arch_Flag" "${ASCEND_INSTALL_INFO_FILE}")
export hetero_arch
get_version_dir "version_dir" "$version_info_file"

get_pkg_install_path_param() {
    if [ -n "$version_dir" ] && [ "$hetero_arch" != "y" ]; then
        realpath "$install_path_param/../../../.."
    else
        realpath "$install_path_param/../../.."
    fi
}

get_install_top_path() {
    local pkg_install_path_param="$(get_pkg_install_path_param)"
    if [ "$hetero_arch" = "y" ]; then
        if [ -n "$version_dir" ]; then
            realpath "$pkg_install_path_param/../../../.."
        else
            realpath "$pkg_install_path_param/../../.."
        fi
    else
        realpath "$pkg_install_path_param/.."
    fi
}

save_user_files_to_log() {
    if [ "$1" = "$install_path_param" ] && [ -s "$1" ]; then
        local filenum=$(ls -lR "$1"|grep "^-"|wc -l)
        local dirnum=$(ls -lR "$1"|grep "^d"|wc -l)
        local totalnum=$(expr "${filenum}" + "${dirnum}")
        if [ "$totalnum" -eq 2 ]; then
            if [ -f "${version_info_file}" ] && [ -f "${ASCEND_INSTALL_INFO_FILE}" ]; then
                return 0
            fi
        fi
        if [ "$totalnum" -eq 1 ]; then
            if [ -f "${version_info_file}" ] || [ -f "${ASCEND_INSTALL_INFO_FILE}" ]; then
                return 0
            fi
        fi
        log "INFO" "Some files generated by user are not cleared, if necessary, manually clear them, get details in $logfile"
    fi
    if [ -s "$1" ]; then
        for file in $(ls -a "$1"); do
            if [ -d "$1/$file" -a ! -L "$1/$file" ]; then
                if [ "$file" != '.' ] && [ "$file" != '..' ]; then
                    echo "$1/$file" >> "$logfile"
                    save_user_files_to_log "$1/$file"
                fi
            else
                echo "$1/$file" >> "$logfile"
            fi
        done
    fi
}

############### 执行函数 ###############
uninstall_run() {
    user_auth
    chmod_start
    local num=0
    local operation="${RUN_CMD_TYPE}"
    local hccl_install_path_param="$(get_pkg_install_path_param)"
    local install_top_path="$(get_install_top_path)"

    if [ -f "$ASCEND_INSTALL_INFO_FILE" ]; then
        local hccl_install_type=$(get_install_param "Hccl_Install_Type" "${ASCEND_INSTALL_INFO_FILE}")
    elif [ -f "${ASCEND_INSTALL_INFO_OLD_FILE}" ]; then
        num=$(grep -c -i hccl_install_path_param "${ASCEND_INSTALL_INFO_OLD_FILE}")
        if [ "${num}" != "0" ]; then
            local hccl_install_type="$(grep -iw hccl_install_type "$ASCEND_INSTALL_INFO_OLD_FILE" | cut -d"=" -f2-)"
        fi
    else
        err_no0x0080 "please complete ${ASCEND_INSTALL_INFO_FILE} or ${ASCEND_INSTALL_INFO_OLD_FILE}"
    fi
    if [ $? -eq 0 ]; then
        log "INFO" "${RUN_CMD} ${hccl_install_path_param} ${hccl_install_type}"

        bash "${UNINSTALL_SHELL}" "${RUN_CMD}" "${hccl_install_path_param}" "${hccl_install_type}" "${IS_QUIET}" "n" "" "y"
        if [ $? -eq 0 ]; then
            rm -f "${ASCEND_INSTALL_INFO_FILE}"
            rm -f "${version_info_file}"
            if [ $? -eq 0 ] && [ -f "${ASCEND_INSTALL_INFO_OLD_FILE}" ] && [ -w "${ASCEND_INSTALL_INFO_OLD_FILE}" ] && [ "${num}" != "0" ]; then
                sed -i '/hccl_install_path_param=/Id' "${ASCEND_INSTALL_INFO_OLD_FILE}"
                sed -i '/hccl_install_type=/Id' "${ASCEND_INSTALL_INFO_OLD_FILE}"
            fi
            remove_dir_recursive "$install_top_path" "$install_path_param"
            new_echo "INFO" "Hccl package uninstalled successfully! Uninstallation takes effect immediately."
            log "INFO" "Hccl package uninstalled successfully! Uninstallation takes effect immediately."
            log_operation "${operation}" "succeeded"
            save_user_files_to_log "$install_path_param"
            save_user_files_to_log "$(dirname $install_path_param)/atc"
            save_user_files_to_log "$(dirname $install_path_param)/fwkacllib"
        else
            log "WARNING" "${operation}" "failed"
            log_operation "${operation}" "failed"
            exit_log 1
        fi
    fi
    return $?
}

############### 程序执行 ###############
while true
do
    case "$1" in
    --quiet)
        IS_QUIET="y"
        shift
        ;;
    --hetero-arch)
        in_hetero_arch="y"
        shift
        ;;
    *)
        if [ "x$1" != "x" ]; then
            err_no0x0004 "$1 . Only support '--quiet' and '--hetero-arch'."
        fi
        break
        ;;
    esac
done

if [ "$hetero_arch" != "y" ]; then
    arch_path="$(dirname "$install_path_param")/$arch_scripts_path_hetero/share/info/hccl/script/uninstall.sh"
    ret=0
    if [ -f "$arch_path" ]; then
        bash "$arch_path"
        ret=$?
    elif [ "$in_hetero_arch" = "y" ]; then
        log "WARNING" "no hetero arch hccl package installed!"
    fi
    if [ "$in_hetero_arch" = "y" ]; then
        exit $ret
    fi
fi

# 输出执行开始日志
start_log

# 验证此目录是否为空
is_dir_empty "$install_path_param"
if [ $? -ne 0 ]; then
    # 执行卸载
    uninstall_run
else
    # 报错
    err_no0x0080 "runfile is not installed on this device, uninstall failed"
fi

exit_log 0
