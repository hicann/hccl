# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

# 公共桩文件
set(_ut_stub_src
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/log_stub.cc
)

# 公共 include 路径（主库 INCLUDE_LIST 的 CCU 头链子集 + 仓根 include/）
set(_ut_include_dirs
    ${_repo_root}/include
    ${_repo_root}/src/ops/all_to_all_v
    ${_repo_root}/src/ops/all_to_all_v/selector
    ${_repo_root}/src/ops/all_to_all_v/algorithm/executor
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/aicpu
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/aicpu/hier
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/aiv
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/ccu
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/ccu/hier
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/ccu/kernel
    ${_repo_root}/src/ops
    ${_repo_root}/src/ops/op_common
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/ops/op_common/algorithm/template
    ${_repo_root}/src/ops/op_common/algorithm/template/aicpu
    ${_repo_root}/src/ops/op_common/algorithm/template/aiv
    ${_repo_root}/src/ops/op_common/algorithm/template/dpu
    ${_repo_root}/src/ops/op_common/algorithm/template/ccu
    ${_repo_root}/src/ops/op_common/algorithm/template/registry
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper
    ${_repo_root}/src/ops/op_common/algorithm/executor
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry
    ${_repo_root}/src/ops/op_common/algorithm/executor/channel
    ${_repo_root}/src/ops/op_common/algorithm/topo_match
    ${_repo_root}/src/ops/op_common/selector
    ${_repo_root}/src/ops/op_common/selector/pub_inc
    ${_repo_root}/src/ops/op_common/topo_info
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
)

# CANN 版本探测宏（CCU 分支代码需 CANN_VERSION_NUM >= 9.0.0 才编译）
# 探测不到仅 WARNING：CCU UT 目标已按版本条件创建，探测失败走 else 优雅跳过，
# 不产生"未定义宏→断言失配"的远根因编译错；既有目标不依赖版本宏，不应被探测失败阻断。
# 解析失败（头存在但无宏行）保 FATAL——头损坏属环境异常，与 topo_match UT 一致
# 探测与应用拆分：探测单次无条件调用（先于一切版本门），应用逐目标注入编译定义
macro(_detect_cann_version)
    file(GLOB _cann_arch_dirs LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
    if(NOT _cann_arch_dirs)
        message(WARNING "[UT] no arch dir under ASCEND_CANN_PACKAGE_PATH"
            "[${ASCEND_CANN_PACKAGE_PATH}], cannot detect CANN_VERSION_NUM, skip version-sensitive targets.")
    endif()
    set(_ut_cann_version_found FALSE)
    foreach(_arch_dir ${_cann_arch_dirs})
        set(_ut_cann_ver_header "${_arch_dir}/include/version/cann_version.h")
        if(EXISTS "${_ut_cann_ver_header}")
            file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line
                 REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
            if(_ut_cann_ver_line)
                string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
                    _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
                math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
                message(STATUS "[UT] Detected CANN_VERSION_NUM = ${_UT_CANN_VERSION_NUM}")
                set(_ut_cann_version_found TRUE)
            else()
                message(FATAL_ERROR "[UT] Failed to parse CANN_VERSION_NUM from ${_ut_cann_ver_header}.")
            endif()
            break()
        endif()
    endforeach()
    if(NOT _ut_cann_version_found)
        message(WARNING "[UT] cann_version.h not found under any arch dir"
            " in ${ASCEND_CANN_PACKAGE_PATH}, skip version-sensitive targets.")
    endif()
endmacro()

macro(_apply_cann_version _target)
    if(_UT_CANN_VERSION_NUM)
        target_compile_definitions(${_target}
            PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
    endif()
endmacro()

# 文件级单次探测，先于一切版本门——消除跨目标副作用依赖
_detect_cann_version()

# CCU 二级 AllToAll stage 真实源（注册表/hier stage/基类链/kernel 基座，两目标共用）
set(_ccu_a2a_stage_src
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/ccu/hier/ccu_alltoall_stage_template_registry.cc
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/ccu/hier/ccu_temp_all_to_all_hier_last_stage.cc
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/ccu/hier/ccu_temp_all_to_all_hier_first_stage.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/ccu_alg_template_base.cc
    ${_repo_root}/src/ops/op_common/ccu_log.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/template_utils.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/ccu/ccu_kernel_alg_base.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/ccu/ccu_kernel_utils.cc
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/ccu/kernel/ccu_kernel_all_to_all_hier_last_stage.cc
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/ccu/kernel/ccu_kernel_all_to_all_hier_first_stage.cc
)

# 序列执行器额外真实源（executor + InsCollAlgBase + 双注册表 + 模板基类 + topo matcher）
set(_ccu_a2a_executor_src
    ${_repo_root}/src/ops/all_to_all_v/algorithm/executor/ins_v2_all_to_all_sequence_executor.cc
    ${_repo_root}/src/ops/all_to_all_v/algorithm/template/aicpu/hier/alltoall_stage_template_registry.cc
    ${_repo_root}/src/ops/op_common/algorithm/executor/executor_v2_base.cc
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/alg_v2_template_base.cc
    ${_repo_root}/src/ops/op_common/algorithm/topo_match/topo_match_two_level.cc
    ${_repo_root}/src/ops/op_common/algorithm/topo_match/topo_match_base_v2.cc
)

# 公共链接
set(_ut_link_libs
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

# CANN 包 include 目录（按当前平台架构匹配）
macro(_apply_cann_include _target)
    file(GLOB _cann_arch_dirs LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
    if(_cann_arch_dirs)
        set(_cann_arch_dir "")
        foreach(_dir ${_cann_arch_dirs})
            if(_dir MATCHES "/${CMAKE_SYSTEM_PROCESSOR}-linux$")
                set(_cann_arch_dir ${_dir})
                break()
            endif()
        endforeach()
        if(NOT _cann_arch_dir)
            list(GET _cann_arch_dirs 0 _cann_arch_dir)
        endif()
        target_include_directories(${_target} PRIVATE
            ${_cann_arch_dir}/include
            ${_cann_arch_dir}/include/hccl
            ${_cann_arch_dir}/include/hcomm/ccu
            ${_cann_arch_dir}/pkg_inc
            ${_cann_arch_dir}/pkg_inc/hccl
            ${_cann_arch_dir}/pkg_inc/base
            ${_cann_arch_dir}/pkg_inc/hcomm/ccu
        )
    endif()
endmacro()

# ---------- alltoall_hier_test：偏移公式与选路逻辑镜像（保留既有目标） ----------
set(ut_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoall_hier_test.cc
)

add_executable(alltoall_hier_test ${ut_src_list})

target_compile_options(alltoall_hier_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(alltoall_hier_test PRIVATE ${_ut_include_dirs})

    _apply_cann_include(alltoall_hier_test)
    _apply_cann_version(alltoall_hier_test)

target_link_libraries(alltoall_hier_test PRIVATE ${_ut_link_libs})

run_llt_test(TARGET alltoall_hier_test TASK_NUM 1)

# ---------- ccu_a2a_stage_registry_test：注册表契约 + SetStageRole 角色 ----------
# CCU UT 目标按版本条件创建：CANN < 9.0.0 环境编译必然失败（hier 源需 CCU 分支），优雅跳过
if(_UT_CANN_VERSION_NUM GREATER_EQUAL 90000000)
    add_executable(ccu_a2a_stage_registry_test
        ${CMAKE_CURRENT_SOURCE_DIR}/ccu_a2a_stage_registry_test.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/stub/ccu_stub.cc
        ${_ut_stub_src}
        ${_ccu_a2a_stage_src}
    )

    target_compile_options(ccu_a2a_stage_registry_test PRIVATE
        -U_FORTIFY_SOURCE
        -fno-access-control
    )

    target_include_directories(ccu_a2a_stage_registry_test PRIVATE ${_ut_include_dirs})

    _apply_cann_include(ccu_a2a_stage_registry_test)
    _apply_cann_version(ccu_a2a_stage_registry_test)

    target_link_libraries(ccu_a2a_stage_registry_test PRIVATE ${_ut_link_libs})

    run_llt_test(TARGET ccu_a2a_stage_registry_test TASK_NUM 1)
else()
    message(STATUS "[UT] CANN_VERSION_NUM[${_UT_CANN_VERSION_NUM}] < 9.0.0, skip ccu_a2a_stage_registry_test.")
endif()

# ---------- ccu_a2a_sequence_executor_test：执行器算法名契约与 stage 全链创建 ----------
# -fno-access-control：访问 protected/private 成员（topo_match UT 已有先例）
if(_UT_CANN_VERSION_NUM GREATER_EQUAL 90000000)
    add_executable(ccu_a2a_sequence_executor_test
        ${CMAKE_CURRENT_SOURCE_DIR}/ccu_a2a_sequence_executor_test.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/stub/ccu_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/stub/executor_stub.cc
        ${_ut_stub_src}
        ${_ccu_a2a_stage_src}
        ${_ccu_a2a_executor_src}
    )

    target_compile_options(ccu_a2a_sequence_executor_test PRIVATE
        -U_FORTIFY_SOURCE
        -fno-access-control
    )

    target_include_directories(ccu_a2a_sequence_executor_test PRIVATE ${_ut_include_dirs})

    _apply_cann_include(ccu_a2a_sequence_executor_test)
    _apply_cann_version(ccu_a2a_sequence_executor_test)

    target_link_libraries(ccu_a2a_sequence_executor_test PRIVATE ${_ut_link_libs})

    run_llt_test(TARGET ccu_a2a_sequence_executor_test TASK_NUM 1)
else()
    message(STATUS "[UT] CANN_VERSION_NUM[${_UT_CANN_VERSION_NUM}] < 9.0.0, skip ccu_a2a_sequence_executor_test.")
endif()
