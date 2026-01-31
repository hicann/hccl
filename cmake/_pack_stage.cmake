# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# _pack_stage.cmake
# Unified staging script: copies files + optionally generates SHA256 manifest.
#
# Inputs:
#   _STAGING_DIR    : required, absolute path
#   _ITEMS          : required, ;-list of file paths
#   _MANIFEST_FILE  : optional, if set -> generate manifest at this path

cmake_minimum_required(VERSION 3.16)

if(NOT _STAGING_DIR)
    message(FATAL_ERROR "_STAGING_DIR not set")
endif()
if(NOT _ITEMS)
    message(FATAL_ERROR "_ITEMS not set")
endif()

# Ensure staging dir exists (defensive)
file(MAKE_DIRECTORY "${_STAGING_DIR}")

# Clear manifest if requested
if(_MANIFEST_FILE)
    file(WRITE "${_MANIFEST_FILE}" "")
endif()

set(failed 0)
foreach(full_path IN LISTS _ITEMS)
    string(STRIP "${full_path}" full_path)
    if("${full_path}" MATCHES "^\\$")
        message(FATAL_ERROR "Generator expression not expanded: '${full_path}'")
    endif()

    if(NOT EXISTS "${full_path}")
        message(WARNING "Skip missing: '${full_path}'")
        math(EXPR failed "${failed} + 1")
        continue()
    endif()

    get_filename_component(basename "${full_path}" NAME)
    if("${basename}" STREQUAL "")
        message(WARNING "Skip empty basename: '${full_path}'")
        math(EXPR failed "${failed} + 1")
        continue()
    endif()

    # Copy flat
    file(COPY "${full_path}" DESTINATION "${_STAGING_DIR}")

    # Optionally compute & append SHA256
    if(_MANIFEST_FILE)
        file(SHA256 "${full_path}" sha256)
        file(APPEND "${_MANIFEST_FILE}" "${basename}=${sha256}\n")
        message(STATUS "${basename} -> ${sha256}")
    else()
        message(STATUS "${basename}")
    endif()
endforeach()

if(failed GREATER 0)
    message(FATAL_ERROR "${failed} file(s) missing")
endif()

if(_MANIFEST_FILE)
    message(STATUS "Manifest: ${_MANIFEST_FILE}")
endif()