# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(STATIC_MODE)
    add_library(hccl STATIC)
    set_target_properties(hccl PROPERTIES
        OUTPUT_NAME "hccl_static"
        POSITION_INDEPENDENT_CODE ON
    )
else()
    add_library(hccl SHARED)
endif()

# 基于ini生成json文件
SET(HCCL_CMAKE_DIR ${OPS_BASE_DIR}/cmake/)
message(STATUS "HCCL_CMAKE_DIR = ${HCCL_CMAKE_DIR}")
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libhccl_aicpu_kernel.json
    COMMAND ${HI_PYTHON}
            ${HCCL_CMAKE_DIR}/scripts/parser_ini.py
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/hccl_aicpu_kernel.ini
            ${CMAKE_CURRENT_BINARY_DIR}/libhccl_aicpu_kernel.json
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_target(aicpu_kernel_json DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libhccl_aicpu_kernel.json)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libhccl_aicpu_kernel.json
    DESTINATION ${INSTALL_AICPU_KERNEL_JSON_DIR}/config
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)
add_dependencies(hccl aicpu_kernel_json)

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(hccl PRIVATE
        OPEN_BUILD_PROJECT
        $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
    )
else()
    target_compile_definitions(hccl PRIVATE
        $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
    )
endif()

target_include_directories(hccl PRIVATE
    ${INCLUDE_LIST}
)

target_compile_definitions(hccl PRIVATE
    -DHOST_COMPILE
)

hccl_apply_cann_compat(hccl)

if(HCCL_CANN_COMPAT_850)
    target_compile_definitions(hccl PRIVATE HCCL_CANN_COMPAT_850)
endif()

target_compile_options(hccl PRIVATE
    -Werror
    -Wno-error=float-equal
    -Wno-error=maybe-uninitialized
    -Wno-error=unused-but-set-variable
    -Wno-overloaded-virtual
    -Wno-unused-parameter
    -Wno-sign-compare
    -Wno-unused-variable
    -Wno-unused-value
    -fno-common
    -fno-strict-aliasing
    -pipe
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-O3 -g>
    -fstack-protector-all
)

if(NOT STATIC_MODE)
    add_dependencies(hccl hccl_compat)
endif()

if(BUILD_OPEN_PROJECT)
    target_link_libraries(hccl PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:error_manager_headers>
        -Wl,--no-as-needed
        hcomm
        hccl_compat
        acl_rt
        c_sec
        unified_dlog
        -Wl,--no-as-needed
    )
else()
    target_link_libraries(hccl PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:ofed_headers>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:npu_runtime_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        -Wl,--no-as-needed
        hcomm
        hccl_compat
        acl_rt
        c_sec
        unified_dlog
        -Wl,--no-as-needed
    )
endif()

if(NOT STATIC_MODE)
    target_link_options(hccl PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,-z,nodelete
        -Wl,-z,noexecstack
        $<$<CONFIG:Release>:-s>
    )

    set(HCCL_EXPORT_MAP
        "${CMAKE_CURRENT_LIST_DIR}/hccl.map"
    )

    target_link_options(hccl PRIVATE
        "-Wl,--version-script=${HCCL_EXPORT_MAP}"
    )

    set_property(TARGET hccl APPEND PROPERTY
        LINK_DEPENDS "${HCCL_EXPORT_MAP}"
    )
endif()

if(STATIC_MODE)
    install(TARGETS hccl
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )
else()
    install(TARGETS hccl
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )
endif()

set(_op_proto_link_libs
    -Wl,--no-as-needed
    exe_graph
    graph
    graph_base
    register
    -Wl,--as-needed
)

set(OP_PROTO_INCLUDE
    ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
)

# 通过编译期特性探测判断 GE 是否已开放 HCCL SuperKernel GenTask 接口
include(CheckCXXSourceCompiles)
set(_hccl_sk_probe_saved_includes "${CMAKE_REQUIRED_INCLUDES}")
set(_hccl_sk_probe_saved_flags "${CMAKE_REQUIRED_FLAGS}")
set(_hccl_sk_probe_saved_try_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_REQUIRED_INCLUDES
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
)
set(CMAKE_REQUIRED_FLAGS "-std=c++17")
# 仅做编译探测，不参与链接，避免被 GE 库的传递依赖（protobuf/platform 等）干扰
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
unset(HCCL_HAS_GE_SUPERKERNEL_GENTASK CACHE)
check_cxx_source_compiles([=[
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "exe_graph/runtime/task_def_builder.h"
#include "graph/arg_desc_info.h"
#include "graph/ge_version_check.h"
#include "register/op_impl_registry.h"

// 等价于 hccl_gen_task_superkernel.cc 的 MakeSuperKernelReg：构造后按值返回，
// 同时覆盖 OpImplRegisterV2 的拷贝/移动性（生产代码 return reg 依赖）
static std::optional<gert::OpImplRegisterV2> MakeRegProbe(const char* opType)
{
    gert::OpImplRegisterV2 reg(opType);
    reg.GenerateTask(nullptr);
    return reg;
}

int main()
{
    // 弱符号版本探测（老 toolkit 上解析为空指针，调用前判空，见 cann/ge!5881）
    if (&ge::IsVersionSupported != nullptr) {
        (void)ge::IsVersionSupported("9.3.0");
    }
    // 运行期条件注册等价于 IMPL_OP(...).GenerateTask(...) 的静态注册
    const auto reg = MakeRegProbe("HcomAllReduce");
    (void)reg;
    // TaskDefBuilder 调用链：即 hccl_gen_task_superkernel.cc 的 TaskDef 构建/序列化形状
    gert::TaskDefBuilder task;
    task.SetStreamId(1U).SetType(0U);
    task.GetKernelBuilder().SetBlockDim(1U).GetContextBuilder().SetArgsFormat(ge::AscendString("{i0*}"));
    const size_t size = task.ByteSizeLong();
    std::vector<uint8_t> buffer(size, 0U);
    (void)task.SerializeToArray(buffer.data(), size);
    // args format 的构建/序列化（superkernel 语义标记所在）
    std::vector<ge::ArgDescInfo> args;
    args.emplace_back(ge::ArgDescInfo(ge::ArgDescType::kIrInput, 0));
    args.insert(args.begin(), ge::ArgDescInfo::CreateHiddenInput(ge::HiddenInputSubType::kHcclSuperKernel));
    const ge::AscendString fmt = ge::ArgsFormatSerializer::Serialize(args);
    (void)fmt;
    return 0;
}
]=] HCCL_HAS_GE_SUPERKERNEL_GENTASK)
set(CMAKE_REQUIRED_INCLUDES "${_hccl_sk_probe_saved_includes}")
set(CMAKE_REQUIRED_FLAGS "${_hccl_sk_probe_saved_flags}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_hccl_sk_probe_saved_try_type}")

set(HCCL_SUPERKERNEL_GENTASK_ENABLED OFF)
if(HCCL_HAS_GE_SUPERKERNEL_GENTASK)
    # 接口存在即启用，不依赖 CANN 版本号
    set(HCCL_SUPERKERNEL_GENTASK_ENABLED ON)
    message(STATUS "HCCL SuperKernel GenTask enabled (GE superkernel GenTask interface found).")
else()
    # 探测不到接口：降级（不编入 superkernel gen_task），不中断构建。
    # 运行期另有弱符号版本守卫（IsGeSuperKernelGenTaskSupported）：老 toolkit 上不注册回调，
    # 由 toolkit 自带的旧 superkernel 实现接管兜底（cann/ge!5881 兼容方案）。
    # 注意：接口是否随 CANN 包发布与版本号不严格绑定。
    message(WARNING
        "HCCL SuperKernel GenTask is disabled: GE superkernel GenTask interface "
        "not found in this CANN package (CANN_VERSION_NUM=${HCCL_CANN_VERSION_NUM}). "
        "Build proceeds without superkernel gen_task. "
        "If superkernel gen_task is expected, please check whether the CANN package "
        "provides the GE superkernel GenTask interface.")
endif()

set(_opgraph_hccl_superkernel_libs)

add_library(opgraph_hccl SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/common/log.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/op_graph/hcom_vinputvec_set_pass.cc
)

if(HCCL_SUPERKERNEL_GENTASK_ENABLED)
    # nlohmann/json：CANN 框架自动下载并提供 nlohmann=ascend_nlohmann 命名空间重映射
    # 防御：cann-cmake 未提供 add_cann_third_party 或 json 目标未创建时降级，不中断构建
    if(COMMAND add_cann_third_party)
        add_cann_third_party(json)
    endif()
    if(NOT TARGET json)
        message(WARNING
            "HCCL SuperKernel GenTask is disabled: nlohmann/json third-party target not available. "
            "Build proceeds without superkernel gen_task.")
    else()
        target_sources(opgraph_hccl PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/common/op_graph/hccl_gen_task_superkernel.cc
        )
        list(APPEND _opgraph_hccl_superkernel_libs platform json)
    endif()
endif()

if(NOT ENABLE_TEST)
    set(OPGRAPH_HCCL_EXPORT_MAP
        "${CMAKE_CURRENT_LIST_DIR}/opgraph_hccl.map"
    )

    target_link_options(opgraph_hccl PRIVATE
        "-Wl,--version-script=${OPGRAPH_HCCL_EXPORT_MAP}"
    )

    set_property(TARGET opgraph_hccl APPEND PROPERTY
        LINK_DEPENDS "${OPGRAPH_HCCL_EXPORT_MAP}"
    )
endif()

target_include_directories(opgraph_hccl PRIVATE
    ${INCLUDE_LIST}
    ${OP_PROTO_INCLUDE}
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/external
)
target_compile_options(opgraph_hccl PRIVATE
    -Wno-error=cpp
    -fno-common
    -fno-strict-aliasing
    -pipe
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-O3 -g>
    -fstack-protector-all
    -fvisibility=hidden
)
target_link_libraries(opgraph_hccl PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:msprof_headers>
    $<BUILD_INTERFACE:mmpa_headers>
    $<BUILD_INTERFACE:runtime_headers>
    $<BUILD_INTERFACE:hcomm_headers>
    $<$<CONFIG:Release>:-s>
    unified_dlog
    ${_op_proto_link_libs}
    -Wl,--whole-archive
    rt2_registry
    -Wl,--no-whole-archive
    ge_common
    ${_opgraph_hccl_superkernel_libs}
    -Wl,-Bsymbolic
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)

target_link_directories(opgraph_hccl PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/common/op_graph/ops_proto_hccl.h
    DESTINATION ${INSTALL_OPGRAPH_INCLUDE_DIR} 
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(TARGETS opgraph_hccl
    LIBRARY DESTINATION ${INSTALL_OPGRAPH_LIBRARY_DIR} 
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

add_dependencies(hccl opgraph_hccl)

#生成ES API
list(APPEND CMAKE_MODULE_PATH "${ASCEND_CANN_PACKAGE_PATH}/include/ge/cmake")
find_package(GenerateEsPackage REQUIRED)

add_es_library_and_whl(
    ES_LINKABLE_AND_ALL_TARGET es_hccl
    OPP_PROTO_TARGET opgraph_hccl
    OUTPUT_PATH ${CMAKE_BINARY_DIR}/es_output
)

if(NOT ENABLE_TEST)
    set(ES_HCCL_EXPORT_MAP
        "${CMAKE_CURRENT_LIST_DIR}/es_hccl.map"
    )

    if(NOT TARGET es_hccl_so)
        message(FATAL_ERROR
            "GenerateEsPackage did not create the expected es_hccl_so target"
        )
    endif()

    target_link_options(es_hccl_so PRIVATE
        "-Wl,--version-script=${ES_HCCL_EXPORT_MAP}"
    )

    set_property(TARGET es_hccl_so APPEND PROPERTY
        LINK_DEPENDS "${ES_HCCL_EXPORT_MAP}"
    )
endif()

add_dependencies(hccl es_hccl)

install(DIRECTORY ${CMAKE_BINARY_DIR}/es_output/include/es_hccl
    DESTINATION ${INSTALL_INCLUDE_DIR}/es/
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/es_output/whl/
    DESTINATION ${WHL_INSTALL_DIR}/es_packages/whl
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

install(FILES ${CMAKE_BINARY_DIR}/es_output/lib64/libes_hccl.so
    DESTINATION ${INSTALL_LIBRARY_DIR}
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)
