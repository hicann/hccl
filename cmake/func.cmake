# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# =============================================================================
# Function: register_custom_aicpu_target
#
# Register a target for custom_aicpu build target.
#
# Usage:
#   register_custom_aicpu_target(
#       <target_name>           # CMake target to register
#   )
#
# =============================================================================
function(register_custom_aicpu_target TARGET_NAME)
    add_dependencies(custom_aicpu ${TARGET_NAME})
endfunction()

# 检查构建依赖
function(check_pkg_build_deps pkg_name)
    execute_process(
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_build_dependencies.py "$ENV{ASCEND_HOME_PATH}" ${CANN_VERSION_${pkg_name}_BUILD_DEPS}
        RESULT_VARIABLE result
    )
    if(result)
        message(FATAL_ERROR "Check ${pkg_name} build dependencies failed!")
    endif()
endfunction()



# 添加生成version.info的目标
# 目标名格式为：version_${包名}_info
function(add_version_info_targets)
    if(ENABLE_DEVICE)
        set(HOST_ONLY "false")
    else()
        set(HOST_ONLY "true")
    endif()
    
    foreach(pkg_name ${CANN_VERSION_PACKAGES})
        add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/version.${pkg_name}.info
            COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_version_info.py --output ${CMAKE_BINARY_DIR}/version.${pkg_name}.info
                    "${CANN_VERSION_${pkg_name}_VERSION}" ${CANN_VERSION_${pkg_name}_RUN_DEPS}
            COMMAND ${CMAKE_COMMAND} -E echo "host_only=${HOST_ONLY}" >> ${CMAKE_BINARY_DIR}/version.${pkg_name}.info
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/version.cmake ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_version_info.py
            VERBATIM
        )
        add_custom_target(version_${pkg_name}_info ALL DEPENDS ${CMAKE_BINARY_DIR}/version.${pkg_name}.info)
    endforeach()
endfunction()