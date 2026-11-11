# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# AllToAllV/VC Pairwise 选路修复的生产注册直连 UT：
# wrap/ins_v2_all_to_all_v_attrs_wrap.cc 以 #include 生产源文件的方式执行
# REGISTER_ALG_ATTRS 注册（屏蔽 REGISTER_EXEC_V2），断言 AlgAttrsRegistry 中
# topoCustomCheck/topoPriorityCheck 的真实生产行为，src/ 生产代码零改动、零复刻。

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

# 链接顺序约束：.init_array 按目标文件链接顺序执行。alg_parse.cc 的维度映射表
# （EngineTypes / OpTypes / ExecutorTypes / AlgoTypes）原为 TU 级 static const，
# 存在跨 TU 静态初始化顺序未定义风险（REGISTER_ALG_ATTRS 静态注册经 ParseAlgName
# -> GetAlgoExecutors 读取这些表）。现已改为 Meyers singleton（函数内 static），
# 初始化时机由首次调用决定，不再依赖链接顺序；此处生产源文件排在 wrap 之前仅为保持
# 既有结构，不构成正确性约束。

# ASCEND_HOME_PATH 与版本探测前置到 target 定义之前：校验不通过时 return() 跳过本目标，
# 避免遗留半成品 target 阻断全量构建（对齐 channel_process_link / ccu_resource_expansion）。
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/x86_64-linux")
else()
    set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/aarch64-linux")
endif()

# 前向兼容：探测 CANN 版本号。缺少该宏时 dlsym_common.h 中的桩类型块会启用，
# 与 CANN 公共头重复定义导致 conflicting declaration，因此版本不可用时跳过本目标。
set(_ut_cann_ver_header "${ASCEND_HOME_PATH}/include/version/cann_version.h")
set(_ut_cann_ver_line "")
if(EXISTS "${_ut_cann_ver_header}")
    file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line
         REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
endif()
if(_ut_cann_ver_line STREQUAL "")
    message(STATUS "[UT] alg_attrs skipped: no CANN_VERSION_NUM found in ${_ut_cann_ver_header}.")
    return()
endif()
string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
    _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
# math(EXPR) 支持纯数字与+-*/()算术表达式，但宏引用等其他形式会 FATAL_ERROR，
# 先校验字符集再计算（对齐 channel_process_link / ccu_resource_expansion）。
if(NOT _UT_CANN_VERSION_EXPR MATCHES "^[0-9() \t+*/-]+$")
    message(STATUS "[UT] alg_attrs skipped: CANN_VERSION_NUM expression '${_UT_CANN_VERSION_EXPR}' "
        "is not pure arithmetic.")
    return()
endif()
math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
message(STATUS "[UT] alg_attrs Detected CANN_VERSION_NUM = ${_UT_CANN_VERSION_NUM}")

add_executable(pairwise_alg_attrs_test
    # 生产源码：attrs 注册表 + 算法名解析（真实实现）
    ${_repo_root}/src/common/alg_parse.cc
    ${_repo_root}/src/ops/op_common/selector/alg_attrs_registry.cc
    # 既有 UT 桩：日志 / securec
    ${_repo_root}/test/ut/common/topo_match/stub/log_stub.cc
    ${_repo_root}/test/ut/selector_engine/stub/securec_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/pairwise_alg_attrs_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/pairwise_alg_attrs_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap/ins_v2_all_to_all_v_attrs_wrap.cc
)

target_include_directories(pairwise_alg_attrs_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/ops/op_common
    ${_repo_root}/src/ops/op_common/selector
    ${_repo_root}/src/ops/op_common/topo_info
    ${_repo_root}/src/ops/op_common/algorithm/executor
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry
    ${_repo_root}/src/ops/op_common/algorithm/executor/channel
    ${_repo_root}/src/ops/op_common/algorithm/topo_match
    ${_repo_root}/src/ops/op_common/algorithm/template
    ${_repo_root}/src/ops/op_common/algorithm/template/registry
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper
    ${_repo_root}/src/ops/op_common/algorithm/template/aiv
    ${_repo_root}/src/ops/op_common/algorithm/template/ccu
    ${_repo_root}/src/ops/all_to_all_v/algorithm/executor
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/aicpu
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/aiv
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/ccu
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/base
    ${ASCEND_HOME_PATH}/pkg_inc/hccl
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
    ${ASCEND_HOME_PATH}/include/hcomm
    ${ASCEND_HOME_PATH}/include/hcomm/ccu
    ${ASCEND_HOME_PATH}/pkg_inc/hcomm
    ${ASCEND_HOME_PATH}/pkg_inc/hcomm/ccu
)

target_compile_options(pairwise_alg_attrs_test PRIVATE
    -U_FORTIFY_SOURCE
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
)

target_compile_definitions(pairwise_alg_attrs_test PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})

target_link_libraries(pairwise_alg_attrs_test PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    GTest::gtest
    GTest::gtest_main
    -lpthread
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -lgcov
    -lrt
    -lpthread
)

run_llt_test(TARGET pairwise_alg_attrs_test TASK_NUM 1)
