set(DEFAULT_BUILD_TYPE "Release")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the build type: Release/Debug" FORCE)
endif()

function(generate_stub_with_output_name STUB STUB_OUTPUT_NAME)
    if(EXISTS ${DOWNLOAD_LIB_DIR}/lib${STUB_OUTPUT_NAME}.so)
        add_library(${STUB} SHARED IMPORTED GLOBAL)
        set_target_properties(${STUB} PROPERTIES
            IMPORTED_LOCATION "${DOWNLOAD_LIB_DIR}/lib${STUB_OUTPUT_NAME}.so"
            INTERFACE_LINK_OPTIONS "-Wl,-rpath-link=${DOWNLOAD_LIB_DIR}"
        )
        message(STATUS "Imported library lib${STUB_OUTPUT_NAME}.so")
    else()
        string(FIND ${STUB_OUTPUT_NAME} "::" temp)
        if (temp EQUAL "-1")
            set(target_plain_name ${STUB_OUTPUT_NAME})
        else()
            string(REPLACE "::" ";" temp_list ${STUB_OUTPUT_NAME})
            list(GET temp_list 1 target_plain_name)
        endif()

        if (NOT TARGET ${target_plain_name}_stub_tmp)
            add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stub/${target_plain_name}.c
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/stub
                COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/stub/${target_plain_name}.c)
            add_library(${target_plain_name}_stub_tmp SHARED ${CMAKE_CURRENT_BINARY_DIR}/stub/${target_plain_name}.c)
            set_target_properties(${target_plain_name}_stub_tmp PROPERTIES
                WINDOWS_EXPORT_ALL_SYMBOLS TRUE
                LIBRARY_OUTPUT_NAME ${target_plain_name}
                RUNTIME_OUTPUT_NAME ${target_plain_name}
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/stub
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/stub)
        endif()

        add_library(${STUB} SHARED IMPORTED GLOBAL)
        if (UNIX)
            set_target_properties(${STUB} PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/stub/lib${target_plain_name}.so")
        endif()
        if (WIN32)
            set_target_properties(${STUB} PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/stub/${target_plain_name}.dll"
                IMPORTED_IMPLIB "${CMAKE_CURRENT_BINARY_DIR}/stub/${target_plain_name}.lib")
        endif()
        add_dependencies(${STUB} ${target_plain_name}_stub_tmp)

        message(STATUS "Stub library lib${STUB_OUTPUT_NAME}.so")
    endif()
endfunction()

function(generate_stub STUB)
    if(DEFINED STUB_OUTPUT_NAME_${STUB})
        set(STUB_OUTPUT_NAME ${STUB_OUTPUT_NAME_${STUB}})
    else()
        set(STUB_OUTPUT_NAME ${STUB})
    endif()

    generate_stub_with_output_name(${STUB} ${STUB_OUTPUT_NAME})

    if(DEFINED STUB_LINK_LIBRARIES_${STUB})
        foreach(LIB ${STUB_LINK_LIBRARIES_${STUB}})
            if(TARGET ${LIB})
                target_link_libraries(${STUB} INTERFACE ${LIB})
            endif()
        endforeach()
    endif()
endfunction(generate_stub)

set(STUBS
    ccl_kernel
)
foreach(STUB ${STUBS})
    if(NOT TARGET ${STUB})
        generate_stub(${STUB})
    endif()
endforeach()

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
