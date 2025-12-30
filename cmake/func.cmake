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

# =============================================================================
# Function: pack_targets_and_files
#
# Packs targets and/or files into a flat tar.gz archive (no directory structure).
# Optionally generates a SHA256 manifest file and includes it in the archive.
#
# Usage:
#   pack_targets_and_files(
#       OUTPUT <output.tar.gz>           # e.g., "cann-tsch-compat.tar.gz"
#       [TARGETS target1 [target2 ...]]
#       [FILES file1 [file2 ...]]
#       [MANIFEST <manifest_filename>]   # e.g., "aicpu_compat_bin_hash.cfg"
#   )
#
# Examples:
#   # With manifest
#   pack_targets_and_files(
#       OUTPUT cann-tsch-compat.tar.gz
#       TARGETS app server
#       FILES "LICENSE" "config/default.json"
#       MANIFEST "aicpu_compat_bin_hash.cfg"
#   )
#
#   # Without manifest
#   pack_targets_and_files(
#       OUTPUT cann-tsch-compat.tar.gz
#       TARGETS app
#       FILES "README.md"
#   )
# =============================================================================
set(_FUNC_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
function(pack_targets_and_files)
    cmake_parse_arguments(ARG
        ""
        "OUTPUT;MANIFEST;OUTPUT_TARGET" 
        "TARGETS;FILES"
        ${ARGN}
    )

    # --- Validation ---
    if(NOT ARG_OUTPUT)
        message(FATAL_ERROR "[pack_targets_and_files] OUTPUT is required")
    endif()

    if(NOT IS_ABSOLUTE "${ARG_OUTPUT}")
        set(ARG_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${ARG_OUTPUT}")
    endif()

    if(NOT ARG_OUTPUT_TARGET)
        message(FATAL_ERROR "[pack_targets_and_files] OUTPUT_TARGET is required")
    endif()

    # Generate safe target name
    get_filename_component(tar_basename "${ARG_OUTPUT}" NAME_WE)
    string(MAKE_C_IDENTIFIER "pack_${tar_basename}" safe_name)
    set(staging_dir "${CMAKE_CURRENT_BINARY_DIR}/_${safe_name}_stage")

    # --- Collect all source items (as generator expressions) ---
    set(src_items "")
    foreach(tgt IN LISTS ARG_TARGETS)
        if(NOT TARGET ${tgt})
            message(FATAL_ERROR "[pack_targets_and_files] Target '${tgt}' does not exist")
        endif()
        list(APPEND src_items "$<TARGET_FILE:${tgt}>")
    endforeach()
    list(APPEND src_items ${ARG_FILES})

    if(NOT src_items)
        message(FATAL_ERROR "[pack_targets_and_files] No targets or files specified to pack")
    endif()

    set(manifest_arg "")
    if(ARG_MANIFEST)
        if("${ARG_MANIFEST}" STREQUAL "")
            message(FATAL_ERROR "[pack] MANIFEST filename cannot be empty")
        endif()
        if(IS_ABSOLUTE "${ARG_MANIFEST}")
            message(FATAL_ERROR "[pack] MANIFEST must be relative (e.g., 'sha256sums.cfg')")
        endif()
        set(manifest_arg -D_MANIFEST_FILE=${staging_dir}/${ARG_MANIFEST})
    endif()

    add_custom_command(
        OUTPUT ${staging_dir}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${staging_dir}"
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${ARG_OUTPUT}"
        COMMAND ${CMAKE_COMMAND}
            -D _STAGING_DIR=${staging_dir}
            ${manifest_arg}
            -D "_ITEMS=$<JOIN:${src_items},;>"
            -P "${_FUNC_CMAKE_DIR}/_pack_stage.cmake"
        COMMAND ${CMAKE_COMMAND} -E tar "czf" "${ARG_OUTPUT}" .
        WORKING_DIRECTORY ${staging_dir}
        DEPENDS ${ARG_TARGETS} ${staging_dir}
        COMMENT "Packing with ${ARG_OUTPUT}"
        VERBATIM
    )

    add_custom_target(${ARG_OUTPUT_TARGET} ALL DEPENDS "${ARG_OUTPUT}")
endfunction()

# =============================================================================
# Function: sign_file
#
# Signs a file and places signature in a standard directory.
#
# Usage:
#   sign_file(
#       INPUT <input_file>
#       CONFIG <sign_config>
#       RESULT_VAR <output_var>     # ← returns generated sig path
#       [DEPENDS ...]
#   )
#
# =============================================================================
function(sign_file)
    cmake_parse_arguments(
        ARG
        ""
        "INPUT;CONFIG;RESULT_VAR"
        "DEPENDS"
        ${ARGN}
    )

    # --- Validation ---
    if(DEFINED CUSTOM_SIGN_SCRIPT AND NOT CUSTOM_SIGN_SCRIPT STREQUAL "")
        set(SIGN_SCRIPT ${CUSTOM_SIGN_SCRIPT})
    else()
        set(SIGN_SCRIPT)
    endif()

    if(ENABLE_SIGN)
        set(sign_flag "true")
    else()
        set(sign_flag "false")
    endif()

    foreach(var INPUT CONFIG RESULT_VAR)
        if(NOT ARG_${var})
            message(FATAL_ERROR "[sign_file] Missing required: ${var}")
        endif()
    endforeach()

    if(NOT EXISTS "${ARG_CONFIG}")
        message(FATAL_ERROR "[sign_file] Sign config not found: ${ARG_CONFIG}")
    endif()

    # Normalize input
    if(NOT IS_ABSOLUTE "${ARG_INPUT}")
        set(ARG_INPUT "${CMAKE_CURRENT_BINARY_DIR}/${ARG_INPUT}")
    endif()

    # Auto output path: ${CMAKE_CURRENT_BINARY_DIR}/signatures
    set(signatures_dir "${CMAKE_CURRENT_BINARY_DIR}/signatures")
    get_filename_component(input_name "${ARG_INPUT}" NAME)
    set(output_sig "${signatures_dir}/${input_name}")

    if(EXISTS "${SIGN_SCRIPT}")
        get_filename_component(EXT ${SIGN_SCRIPT} EXT)  # 获取文件扩展名

        if(${EXT} STREQUAL ".sh")
            set(sign_cmd bash ${SIGN_SCRIPT} ${output_sig} ${ARG_CONFIG} ${sign_flag})
        elseif(${EXT} STREQUAL ".py")
            set(root_dir ${CMAKE_SOURCE_DIR})
            message(STATUS "Detected +++VERSION_INFO: ${VERSION_INFO}")
            set(sign_cmd python3 ${root_dir}/scripts/sign/add_header_sign.py ${signatures_dir} ${sign_flag} --bios_check_cfg=${ARG_CONFIG} --sign_script=${SIGN_SCRIPT} --version=${VERSION_INFO})
        endif()
    else()
        set(sign_cmd )
    endif()

    # Ensure dir exists
    file(MAKE_DIRECTORY "${signatures_dir}")

    # Target name
    get_filename_component(sign_basename "${ARG_INPUT}" NAME_WE)
    string(MAKE_C_IDENTIFIER "${sign_basename}" safe_name)
    set(sign_target "sign_${safe_name}")

    add_custom_command(
        OUTPUT "${output_sig}"
        COMMAND ${CMAKE_COMMAND} -E make_directory ${signatures_dir}
        COMMAND ${CMAKE_COMMAND} -E copy ${ARG_INPUT} ${output_sig}
        COMMAND ${sign_cmd}
        DEPENDS "${ARG_INPUT}" "${SIGN_SCRIPT}" ${ARG_DEPENDS} ${ARG_CONFIG}
        COMMENT "Signing: ${ARG_INPUT} -> ${output_sig}"
        VERBATIM
    )

    add_custom_target(${sign_target} ALL DEPENDS "${output_sig}")

    # Return path via RESULT_VAR
    if(ARG_RESULT_VAR)
        set(${ARG_RESULT_VAR} "${output_sig}" PARENT_SCOPE)
    endif()
endfunction()

# =============================================================================
# Function: package_aicpu_kernel
#
# Packages an AICPU kernel into a tar archive and installs it along with its
# JSON configuration file.#
#
# Usage:
#   package_aicpu_kernel(
#       <target_name>          # CMake target name of the AICPU kernel
#       <tar_directory>        # Directory where files are staged before packaging
#       <tar_output_file>      # Full path of the output tar.gz archive
#       <json_config_file>     # Path to the AICPU kernel JSON configuration file
#   )
#
# =============================================================================
function(package_aicpu_kernel TARGET_NAME TAR_FILE JSON_FILE)
    if(NOT TARGET ${TARGET_NAME})
        message(FATAL_ERROR 
            "[package_aicpu_kernel] Target '${TARGET_NAME}' does not exist. "
            "Ensure the target is defined before calling this function.")
    endif()

    set(TAR_DIR ${CMAKE_BINARY_DIR}/aicpu_kernels_device)

    # POST_BUILD 阶段进行打包
    add_custom_command(
        TARGET ${TARGET_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TAR_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> ${TAR_DIR}
        COMMAND cd ${CMAKE_BINARY_DIR} && tar czf ${TAR_FILE} aicpu_kernels_device
        COMMENT "[package_aicpu_kernel] Created package: ${TAR_FILE}"
    )
endfunction()

# =============================================================================
# Function: sign_aicpu_kernel
#
# Signs an AICPU kernel package using a custom signing script. Creates a signed
# version of the input tar package and returns its path.
#
# Usage:
#   sign_aicpu_kernel(
#       <target_name>           # CMake target to associate signing with
#       <input_tar_file>        # Input tar package to sign
#       <config_file>           # Signing configuration file
#       <output_tar_file_var>   # Output variable for signed package path
#   )
#
# =============================================================================
function(sign_aicpu_kernel TARGET_NAME TAR_FILE CONFIG_FILE OUTPUT_TAR_FILE_VAR)
    if(NOT TARGET ${TARGET_NAME})
        message(FATAL_ERROR
            "[sign_aicpu_kernel] Target '${TARGET_NAME}' does not exist. "
            "Ensure the target is defined before calling this function.")
    endif()

    if(DEFINED CUSTOM_SIGN_SCRIPT AND NOT CUSTOM_SIGN_SCRIPT STREQUAL "")
        set(SIGN_SCRIPT ${CUSTOM_SIGN_SCRIPT})
    else()
        set(SIGN_SCRIPT)
    endif()

    if(ENABLE_SIGN)
        set(SIGN_FLAG "true")
    else()
        set(SIGN_FLAG "false")
    endif()

    if(NOT EXISTS "${CONFIG_FILE}")
        message(FATAL_ERROR "[sign_aicpu_kernel] Sign config file not found: ${CONFIG_FILE}")
    endif()

    if(NOT IS_ABSOLUTE "${TAR_FILE}")
        set(TAR_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TAR_FILE}")
        message(STATUS "[sign_aicpu_kernel] TAR_FILE = ${TAR_FILE}")
    endif()

    get_filename_component(TAR_FILE_NAME "${TAR_FILE}" NAME)  # 获取文件名
    set(OUTPUT_TAR_DIR "${CMAKE_CURRENT_BINARY_DIR}/signatures")
    set(OUTPUT_TAR_FILE "${OUTPUT_TAR_DIR}/${TAR_FILE_NAME}")

    if(EXISTS "${SIGN_SCRIPT}")
        get_filename_component(SCRIPT_EXT ${SIGN_SCRIPT} EXT)  # 获取文件扩展名

        if(${SCRIPT_EXT} STREQUAL ".sh")
            set(SIGN_CMD bash ${SIGN_SCRIPT} ${OUTPUT_TAR_FILE} ${CONFIG_FILE} ${SIGN_FLAG})
        elseif(${SCRIPT_EXT} STREQUAL ".py")
            set(SIGN_CMD
                python3 ${CMAKE_SOURCE_DIR}/scripts/sign/add_header_sign.py
                ${OUTPUT_TAR_DIR} ${SIGN_FLAG}
                --bios_check_cfg=${CONFIG_FILE}
                --sign_script=${SIGN_SCRIPT}
                --version=${VERSION_INFO}
            )
        else()
            message(WARNING "[sign_aicpu_kernel] Unsupported script type: ${SCRIPT_EXT}")
        endif()
    else()
        message(WARNING "[sign_aicpu_kernel] Sign script not found: ${SIGN_SCRIPT}")
        set(SIGN_CMD)
    endif()

    # POST_BUILD 阶段进行签名
    add_custom_command(
        TARGET ${TARGET_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_TAR_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy ${TAR_FILE} ${OUTPUT_TAR_DIR}
        COMMAND ${SIGN_CMD}
        DEPENDS ${TAR_FILE} ${SIGN_SCRIPT} ${CONFIG_FILE}
        COMMENT "[sign_aicpu_kernel] Signing package: ${OUTPUT_TAR_DIR}"
        VERBATIM
    )

    # 签名包路径
    set(${OUTPUT_TAR_FILE_VAR} ${OUTPUT_TAR_FILE} PARENT_SCOPE)
endfunction()
