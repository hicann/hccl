# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 该 UT 依赖 GE 对外接口 TaskDefBuilder / OpImplRegisterV2 / kHcclSuperKernel / ge_version_check，
# 接口不存在时跳过。注意：以下编译探测代码与 src/hccl.cmake 保持逐字一致，修改时请同步另一处。
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

if(NOT HCCL_HAS_GE_SUPERKERNEL_GENTASK)
    message(STATUS "[UT] hccl_gen_task_superkernel skipped: GE superkernel GenTask interface not available.")
    return()
endif()

# 与 src/hccl.cmake 保持一致：cann-cmake 未提供 add_cann_third_party 或未创建 json 目标时跳过本 UT
if(COMMAND add_cann_third_party)
    add_cann_third_party(json)
endif()
if(NOT TARGET json)
    message(STATUS "[UT] hccl_gen_task_superkernel skipped: nlohmann/json third-party target not available.")
    return()
endif()

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

add_executable(hccl_gen_task_superkernel_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_hccl_gen_task_superkernel.cc
    ${_repo_root}/src/common/op_graph/hccl_gen_task_superkernel.cc
    ${_repo_root}/src/common/log.cc
)

target_include_directories(hccl_gen_task_superkernel_test PRIVATE
    ${_repo_root}/src/common
    ${_repo_root}/src/common/op_graph
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
    ${ASCEND_CANN_PACKAGE_PATH}/include/external
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
)

target_compile_options(hccl_gen_task_superkernel_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_link_directories(hccl_gen_task_superkernel_test PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(hccl_gen_task_superkernel_test PRIVATE
    intf_llt_pub
    GTest::gtest_main
    -lpthread
    ${CMAKE_DL_LIBS}
    json
    graph
    graph_base
    register
    opp_registry
    exe_graph
    platform
    ge_common
    unified_dlog
)

target_link_options(hccl_gen_task_superkernel_test PRIVATE
    "-Wl,-rpath-link,${ASCEND_CANN_PACKAGE_PATH}/lib64"
    "-Wl,-rpath,${ASCEND_CANN_PACKAGE_PATH}/lib64"
)

run_llt_test(TARGET hccl_gen_task_superkernel_test)
