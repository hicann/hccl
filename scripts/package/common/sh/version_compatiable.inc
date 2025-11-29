#!/bin/sh
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

# 创建列表
_make_list() {
    local _outvar="$1"
    eval "${_outvar}=\"\""
}

# 列表添加元素
_append_list() {
    local _outvar="$1"
    local _item="$2"

    local _value
    eval "_value=\"\${${_outvar}}\""

    if [ "${_value}" = "" ]; then
        _value="${_item}"
    else
        _value="${_value} ${_item}"
    fi

    eval "${_outvar}=\"${_value}\""
}

# 获取需求包列表。
_get_required_packages() {
    local _outvar="$1"
    local _version_info_path="$2"
    local _packages

    if [ ! -f "${_version_info_path}" ]; then
        eval "${_outvar}=\"\""
        return 1
    fi

    _packages=$(awk '
        function join(items, len, sep,  result, i) {
            if (len >= 1) {
                result = items[1]
                for (i = 2; i <= len; i++) {
                    result = sprintf("%s%s%s", result, sep, items[i])
                }
                return result
            }
            return ""
        }

        BEGIN {
            len_packages = 0
        }
        /^required_package_.+_version=/ {
            split($0, line_tokens, "=")
            len_require_tokens = split(line_tokens[1], require_tokens, "_")
            len_package_names = 0
            for (i = 3; i < len_require_tokens; i++) {
                len_package_names++
                package_names[len_package_names] = require_tokens[i]
            }
            package_name = join(package_names, len_package_names, "_")

            len_packages += 1
            packages[len_packages] = package_name
        }
        END {
            if (len_packages >= 1) {
                printf("%s", packages[1])
                for (i = 2; i <= len_packages; i++) {
                    printf(" %s", packages[i])
                }
            }
        }
    ' "${_version_info_path}")
    eval "${_outvar}=\"${_packages}\""
}

# 获取需求包信息。
_get_required_package_info() {
    local _outvar="$1"
    local _version_info_path="$2"
    local _pkg_name="$3"
    local _required_pkg_name
    local _required_value

    if [ ! -f "${_version_info_path}" ]; then
        eval "${_outvar}=\"\""
        return 1
    fi

    _required_pkg_name="required_package_${_pkg_name}_version"
    _required_value="$(grep "^${_required_pkg_name}=" "${_version_info_path}" | cut -d= -f2-)"
    # _required_value取得的值带有双引号，eval时不可以在外侧再添加双引号

    eval "${_outvar}=${_required_value}"
}

# 检查版本与需求版本兼容性。
# 兼容返回0，不兼容返回1
_check_version_required() {
    local version="$1"
    local require="$2"
    local script_dir="$3"
    local result

    result=$(awk -f "${script_dir}/check_version_required.awk" -v version="${version}" -v all_required="${require}")
    if [ "${result}" = "T" ]; then
        return 0
    fi

    return 1
}

# 检查包版本兼容性。
_check_package_version_compatiable() {
    local version_info_path="$1"
    local pkg_version_info_path="$2"
    local package="$3"
    local script_dir="$4"

    local pkg_version
    local required

    # get pkg_version
    get_package_version "pkg_version" "${pkg_version_info_path}"
    # get required
    _get_required_package_info "required" "${version_info_path}" "${package}"
    # check pkg_version and required
    _check_version_required "${pkg_version}" "${required}" "${script_dir}"
    if [ $? -eq 0 ]; then
        comm_log "INFO" "${package} is installed, current version: ${pkg_version}, version check successfully."
        return 0
    fi

    comm_log "ERROR" "${package} is installed, current version: ${pkg_version}, required version is ${required}, version check failed."
    return 1
}


# 检查版本兼容性。
_check_version_compatiable() {
    local version_info_path="$1"
    local install_path="$2"
    local script_dir="$3"

    local check_result=0
    local failed_list
    local required_packages
    local package
    local self_version
    local pkg_version_info_path
    local pkg_installed
    local driver_install_path

    _get_required_packages "required_packages" "${version_info_path}"
    get_package_version "self_version" "${version_info_path}"

    _make_list "failed_list"

    for package in ${required_packages}; do
        if [ "${package}" = "driver" ]; then
            get_driver_install_path "driver_install_path" "${install_path}"
            if [ "${driver_install_path}" != "" ]; then
                pkg_installed="true"
            else
                pkg_installed="false"
            fi
            pkg_version_info_path="${driver_install_path}/${package}/version.info"
        else
            does_pkg_installed "${install_path}" "${package}" "pkg_installed"
            pkg_version_info_path="${install_path}/${package}/version.info"
        fi
        if [ "${pkg_installed}" = "true" ] && [ -f "${pkg_version_info_path}" ]; then
            _check_package_version_compatiable "${version_info_path}" "${pkg_version_info_path}" "${package}" "${script_dir}"
            if [ $? -ne 0 ]; then
                check_result=1
                _append_list "failed_list" "${package}"
            fi
        else
            comm_log "INFO" "${package} is not installed, does not need to check."
        fi
    done

    if [ ${check_result} -ne 0 ]; then
        comm_log "ERROR" "version check failed, please uninstall incompatible packages or install compatible packages. [$(echo ${failed_list} | sed 's/ /, /g')]"
    fi

    return ${check_result}
}