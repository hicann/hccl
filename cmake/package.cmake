# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#### CPACK to package run #####

# download makeself package
add_cann_third_party(makeself-fetch)

function(pack_custom)
  message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      message(STATUS "Detected architecture: x86_64")
      set(ARCH x86_64)
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
      message(STATUS "Detected architecture: ARM64")
      set(ARCH aarch64)
  else ()
      message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif ()

  install(DIRECTORY ${CMAKE_SOURCE_DIR}/scripts/custom/
      DESTINATION ${CUSTOM_OPS_OPP_SCRIPTS_PATH}
      FILE_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 文件权限
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
      DIRECTORY_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 目录权限
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
      COMPONENT hccl
  )

  # ============= CPack =============
  set(CPACK_PACKAGE_NAME "cann-hccl-${CUSTOM_OPS_NAME}-${CUSTOM_OPS_VENDOR}")
  set(CUSTOM_PACKAGE_VERSION "")
  set(_version_cmake "${CMAKE_SOURCE_DIR}/version.cmake")
  if(EXISTS "${_version_cmake}")
      file(STRINGS "${_version_cmake}" _ver_pkg_line
          REGEX "set_cann_package[ \t]*\\([ \t]*hccl[ \t]+VERSION[ \t]+\"[0-9]+\\.[0-9]+\\.[0-9]+\"")
      if(_ver_pkg_line)
          string(REGEX MATCH "\"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _ "${_ver_pkg_line}")
          if(CMAKE_MATCH_1)
              set(CUSTOM_PACKAGE_VERSION "${CMAKE_MATCH_1}")
          endif()
      endif()
  endif()
  set(CPACK_PACKAGE_VERSION "${CUSTOM_PACKAGE_VERSION}")
  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CUSTOM_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}")

  set(CPACK_INSTALL_PREFIX "/")

  set(CPACK_CMAKE_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
  set(CPACK_CMAKE_BINARY_DIR "${CMAKE_BINARY_DIR}")
  set(CPACK_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
  set(CPACK_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  set(CPACK_ARCH "${ARCH}")
  set(CPACK_SET_DESTDIR ON)
  set(CPACK_GENERATOR External)
  set(CPACK_EXTERNAL_PACKAGE_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/makeself_custom.cmake")
  set(CPACK_EXTERNAL_ENABLE_STAGING true)
  set(CPACK_PACKAGE_DIRECTORY "${CMAKE_INSTALL_PREFIX}")
  set(CPACK_3RD_LIB_PATH "${CANN_3RD_LIB_PATH}")

  set(CPACK_CUSTOM_OPS_NAME "${CUSTOM_OPS_NAME}")
  set(CPACK_CUSTOM_OPS_PATH "${CUSTOM_OPS_PATH}")
  set(CPACK_CUSTOM_OPS_VENDOR "${CUSTOM_OPS_VENDOR}")
  set(CPACK_CUSTOM_OPS_OPP_SCRIPTS_PATH "${CUSTOM_OPS_OPP_SCRIPTS_PATH}")

  include(CPack)
endfunction()

function(pack_built_in)
  #### built-in package ####
  message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      message(STATUS "Detected architecture: x86_64")
      set(ARCH x86_64)
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
      message(STATUS "Detected architecture: ARM64")
      set(ARCH aarch64)
  else ()
      message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif ()

  set(script_prefix ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/hccl/scripts)
  install(DIRECTORY ${script_prefix}/
      DESTINATION share/info/hccl/script
      FILE_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 文件权限
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
      DIRECTORY_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 目录权限
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
      COMPONENT hccl
  )

  set(SCRIPTS_FILES
      ${CANN_CMAKE_DIR}/scripts/install/check_version_required.awk
      ${CANN_CMAKE_DIR}/scripts/install/common_func.inc
      ${CANN_CMAKE_DIR}/scripts/install/common_interface.sh
      ${CANN_CMAKE_DIR}/scripts/install/common_interface.csh
      ${CANN_CMAKE_DIR}/scripts/install/common_interface.fish
      ${CANN_CMAKE_DIR}/scripts/install/version_compatiable.inc
      ${CANN_CMAKE_DIR}/scripts/package/merge_binary_info_config.py
  )

  install(FILES ${SCRIPTS_FILES}
      DESTINATION share/info/hccl/script
      COMPONENT hccl
  )
  set(COMMON_FILES
      ${CANN_CMAKE_DIR}/scripts/install/install_common_parser.sh
      ${CANN_CMAKE_DIR}/scripts/install/common_func_v2.inc
      ${CANN_CMAKE_DIR}/scripts/install/common_installer.inc
      ${CANN_CMAKE_DIR}/scripts/install/script_operator.inc
      ${CANN_CMAKE_DIR}/scripts/install/version_cfg.inc
  )

  set(PACKAGE_FILES
      ${COMMON_FILES}
      ${CANN_CMAKE_DIR}/scripts/install/multi_version.inc
  )
  set(CONF_FILES
      ${CANN_CMAKE_DIR}/scripts/package/cfg/path.cfg
  )

  install(FILES ${CMAKE_BINARY_DIR}/version.hccl.info
      DESTINATION share/info/hccl
      RENAME version.info
      ${INSTALL_OPTIONAL}
      COMPONENT hccl
  )
  install(FILES ${CONF_FILES}
      DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/conf
      COMPONENT hccl
  )
  install(FILES ${PACKAGE_FILES}
      DESTINATION share/info/hccl/script
      COMPONENT hccl
  )
  string(FIND "${ASCEND_COMPUTE_UNIT}" ";" SEMICOLON_INDEX)
  if (SEMICOLON_INDEX GREATER -1)
      # 截取分号前的字串
      math(EXPR SUBSTRING_LENGTH "${SEMICOLON_INDEX}")
      string(SUBSTRING "${ASCEND_COMPUTE_UNIT}" 0 "${SUBSTRING_LENGTH}" compute_unit)
  else()
      # 没有分号取全部内容
      set(compute_unit "${ASCEND_COMPUTE_UNIT}")
  endif()

  message(STATUS "current compute_unit is: ${compute_unit}")

  # ============= CPack =============
  if (NOT ENABLE_COV AND NOT ENABLE_UT)
    set_cann_cpack_config(hccl ENABLE_DEVICE ${ENABLE_DEVICE} PACKAGE_TYPE "${PACKAGE_TYPE}")
  endif()
endfunction()
