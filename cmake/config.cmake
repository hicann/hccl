set(DEFAULT_BUILD_TYPE "Release")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the build type: Release/Debug" FORCE)
endif()

if(CUSTOM_ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH  ${CUSTOM_ASCEND_CANN_PACKAGE_PATH})
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH  $ENV{ASCEND_HOME_PATH})
elseif(DEFINED ENV{ASCEND_OPP_PATH})
    get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_OPP_PATH}/.." ABSOLUTE)
else()
    set(ASCEND_CANN_PACKAGE_PATH  "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(ASCEND_MOCKCPP_PACKAGE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# if (NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}")
#     message(FATAL_ERROR "${ASCEND_CANN_PACKAGE_PATH} does not exist, please install the cann package and set environment variables.")
# endif()

# if (NOT EXISTS "${THIRD_PARTY_NLOHMANN_PATH}")
#     message(FATAL_ERROR "${THIRD_PARTY_NLOHMANN_PATH} does not exist, please check the setting of THIRD_PARTY_NLOHMANN_PATH.")
# endif()

set(ASCEND_SDK_PACKAGE_PATH "${ASCEND_CANN_PACKAGE_PATH}")
if (NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}/opensdk")
    # 设置社区包sdk安装位置
    set(ASCEND_SDK_PACKAGE_PATH "${ASCEND_CANN_PACKAGE_PATH}/../../latest")
endif()

#execute_process(COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/check_version_compatiable.sh
#                             ${ASCEND_CANN_PACKAGE_PATH}
#                             hccl
#                             ${CMAKE_CURRENT_SOURCE_DIR}/version.info
#    RESULT_VARIABLE result
#    OUTPUT_STRIP_TRAILING_WHITESPACE
#    OUTPUT_VARIABLE CANN_VERSION
#    )

#if (result)
#    message(FATAL_ERROR "${CANN_VERSION}")
#else()
#     string(TOLOWER ${CANN_VERSION} CANN_VERSION)
#endif()

if (CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX     "${CMAKE_CURRENT_SOURCE_DIR}/output"  CACHE STRING "path for install()" FORCE)
endif ()

set(HI_PYTHON                     "python3"                       CACHE   STRING   "python executor")

message(STATUS "config.cmake KERNEL_MODE=${KERNEL_MODE} BUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}")
if(BUILD_OPEN_PROJECT AND KERNEL_MODE)
    set(PRODUCT_SIDE                  device)
else()
    set(PRODUCT_SIDE                  host)
endif()
set(INSTALL_LIBRARY_DIR hccl/lib64)
set(INSTALL_INCLUDE_DIR hccl/include)
set(INSTALL_AICPU_KERNEL_JSON_DIR hccl/built-in/data/op/aicpu)
set(INSTALL_DEVICE_TAR_DIR hccl/Ascend/aicpu)

if (ENABLE_TEST)
    set(CMAKE_SKIP_RPATH FALSE)
else ()
    set(CMAKE_SKIP_RPATH TRUE)
endif ()
