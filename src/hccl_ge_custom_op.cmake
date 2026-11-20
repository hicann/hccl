# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 自定义算子模块 (GE EagerExecuteOp 适配层, HCCL 算子的 GE 自定义算子适配):
# 源/包含/链接直接编入 opgraph_hccl, libopgraph_hccl.so 即最终产物。
# 公共模块源在此挂载; 各算子源由 src/ops/<op>/op_graph/CMakeLists.txt 以
# if(HCCL_GE_CUSTOP) 追加。toolkit 能力不足时由下方特性探测整体跳过
# (变量不置位, 算子追加随之跳过)。
#
# ---- toolkit 特性探测 (门控判据, 编译探测范式同 superkernel GenTask 探测) ----
# 本模块依赖以下自定义算子框架接口: GetContextValue (执行上下文交互参数)、
# RequestAttachedStream 带 flags 参数 (辅流创建标志)、rtGetModelByStream/rtModelAddHcclStream
# (外部流注册)、五参 CustomOpCreatorRegister 构造器 (接管注册)。
# 判据为"本模块的真实调用能否在 toolkit 头文件上编过"——探测体即调用代码本身,
# 对换行/参数改名/格式化/注释全部免疫, 只有接口语义真正影响编译时才判缺失;
# 接口与 toolkit 版本号不一一对应 (两个方向均有反例), 故不用 CANN_VERSION_NUM。
# STATIC_LIBRARY 只编译不链接, 避免被 GE 库传递依赖 (protobuf/platform) 干扰;
# 探测结果 cache 变量每次强制清空重探。
include(CheckCXXSourceCompiles)
set(_CUSTOP_CTX_HDR "${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph/runtime/eager_op_execution_context.h")
set(_CUSTOP_COP_HDR "${ASCEND_CANN_PACKAGE_PATH}/include/graph/custom_op.h")
find_file(_CUSTOP_RT_HDR rt_external_model.h
    PATHS ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime ${ASCEND_CANN_PACKAGE_PATH}/include/runtime
    NO_DEFAULT_PATH)

# toolkit 原地升级 (头文件变化) 触发 cmake 重跑重探, 避免探测结果过期
foreach(_dep "${_CUSTOP_CTX_HDR}" "${_CUSTOP_COP_HDR}" "${_CUSTOP_RT_HDR}")
    if(EXISTS "${_dep}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_dep}")
    endif()
endforeach()

set(_CUSTOP_PROBE_SAVED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
set(_CUSTOP_PROBE_SAVED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
set(_CUSTOP_PROBE_SAVED_TRY_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_REQUIRED_INCLUDES
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
)
set(CMAKE_REQUIRED_FLAGS "-std=c++17")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
unset(HCCL_GE_HAS_FRAMEWORK CACHE)
check_cxx_source_compiles([=[
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "graph/custom_op.h"
#include "runtime/rt_external_model.h"

static ge::BaseCustomOp* ProbeCreator()
{
    return nullptr;
}

int main()
{
    gert::EagerOpExecutionContext* ctx = nullptr;
    uint64_t value = 0;
    (void)ctx->GetContextValue(gert::ContextParam::kStaticModel, value);
    (void)ctx->RequestAttachedStream(ge::AscendString("probe"), 0U);
    rtModel_t model = nullptr;
    (void)rtGetModelByStream(nullptr, &model);
    (void)rtModelAddHcclStream(nullptr, model);
    ge::CustomOpCreatorRegister registerProbe(
        ge::AscendString("probe"), ge::OpBackend::kDevice, ge::OpRegistrationPriority::kBottom,
        ge::OpEngine::kHccl, ProbeCreator);
    (void)registerProbe;
    return 0;
}
]=] HCCL_GE_HAS_FRAMEWORK)
set(CMAKE_REQUIRED_INCLUDES "${_CUSTOP_PROBE_SAVED_INCLUDES}")
set(CMAKE_REQUIRED_FLAGS "${_CUSTOP_PROBE_SAVED_FLAGS}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_CUSTOP_PROBE_SAVED_TRY_TYPE}")

set(_CUSTOP_MISSING "")
if(NOT HCCL_GE_HAS_FRAMEWORK)
    list(APPEND _CUSTOP_MISSING
        "GetContextValue" "RequestAttachedStream(flags)" "rtGetModelByStream" "rtModelAddHcclStream"
        "CustomOpCreatorRegister(5-param)")
endif()

# GraphMode 链接依赖与接口探测同口径: 缺 liblowering.so 时并入缺失清单优雅跳过,
# 而非把 NOTFOUND 空值带进 target_link_libraries 到链接期才报含混错误
# PATHS 显式定位 (repo 惯例: ${ASCEND_CANN_PACKAGE_PATH}/lib64,
# 参照 src/hccl.cmake 的 target_link_directories), 不依赖调用方 CMAKE_PREFIX_PATH 命中;
# NO_DEFAULT_PATH 与上方 find_file 同口径 — 只认 CANN 目录, toolkit 缺库时不得命中
# 系统路径的同名 liblowering.so (实验证实: 无此 flag 时 CMAKE_PREFIX_PATH 下的
# 同名库会被误命中, 门控误放行链到错库)
find_library(_CANN_LOWERING_LIB
    NAMES liblowering.so
    PATHS ${ASCEND_CANN_PACKAGE_PATH}/lib64
    NO_DEFAULT_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)
if(NOT _CANN_LOWERING_LIB)
    list(APPEND _CUSTOP_MISSING "liblowering.so")
endif()

if(_CUSTOP_MISSING)
    string(REPLACE ";" ", " _CUSTOP_MISSING_STR "${_CUSTOP_MISSING}")
    message(STATUS "hccl_ge_custom_op: skip, toolkit 缺少自定义算子框架接口 [${_CUSTOP_MISSING_STR}]")
    set(HCCL_GE_CUSTOP OFF)
    return()
endif()

find_package(graph QUIET)
if(NOT graph_FOUND)
    list(APPEND _CUSTOP_MISSING "graph package")
    message(STATUS "hccl_ge_custom_op: skip, graph package not found")
    # 跳过前复位 OFF, 否则算子源仍按 HCCL_GE_CUSTOP=ON 挂载而公共模块未挂载 → 编译失败
    set(HCCL_GE_CUSTOP OFF)
    return()
endif()
# 必须用裸 set 而非 PARENT_SCOPE: include() 不创建新作用域, 本文件命令在引用方 (src/) 作用域内
# 执行 — 裸 set 恰好落在 src/ 作用域, 后续 add_subdirectory(ops) 的算子 CMakeLists 按目录继承可见;
# PARENT_SCOPE 会写到顶层作用域 (add_subdirectory(src) 时已拷贝快照), 算子源将静默不编译
set(HCCL_GE_CUSTOP ON)

target_sources(opgraph_hccl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common/op_graph/hccl_custom_op.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/op_graph/hccl_op_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/op_graph/hccl_op_v_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/op_graph/hccl_rt_utils.cc
)
target_include_directories(opgraph_hccl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common/op_graph
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    # 仓库根 include/: 本仓自有的 hccl.h 源头 (src/ 同款无前缀引入)。
    # 不得依赖 ${ASCEND_CANN_PACKAGE_PATH}/include 下的 hccl/hccl.h — 那是本仓安装后的产物,
    # 编译本仓时引用会造成自依赖/版本漂移; hccl.h 内部的 <hccl/hccl_types.h> 等仍经 CANN 路径解析。
    ${CMAKE_SOURCE_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
)

target_link_libraries(opgraph_hccl PRIVATE
    hccl
    hcomm
    unified_dlog
    c_sec
    graph
    ${_CANN_LOWERING_LIB}
)

