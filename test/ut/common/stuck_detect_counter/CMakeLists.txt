# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

# 被测代码：src/ops/op_common/stuck_detect_counter_helper.h（header-only）
# 依赖（hccl_aiv_utils / hccl_host_comm_dl / hccl_res_dl / hcomm_dlsym / log）均以 stub 提供，
# 其中通信域状态回调依赖函数内 static once_flag，无法在同一进程内切换场景，
# 因此按“正常 / 回调不支持 / 回调注册失败”拆成三个可执行文件分别验证。
set(STUCK_UT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/stuck_detect_counter_helper_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/stuck_stub.cc
)

function(add_stuck_detect_ut target_name)
    add_executable(${target_name} ${STUCK_UT_SOURCES})
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/stub
        ${PROJECT_SOURCE_DIR}/src/ops/op_common
    )
    target_compile_options(${target_name} PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage -U_FORTIFY_SOURCE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
    )
    if(ARGC GREATER 1)
        target_compile_definitions(${target_name} PRIVATE ${ARGV1})
    endif()
    target_link_libraries(${target_name} PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub>
        GTest::gtest
        GTest::gtest_main
        -lpthread
        -lgcov
    )
    run_llt_test(TARGET ${target_name} TASK_NUM 1)
endfunction()

add_stuck_detect_ut(stuck_detect_counter_helper_test)
add_stuck_detect_ut(stuck_detect_counter_helper_degraded_unsupported_test UT_SCENARIO_UNSUPPORTED)
add_stuck_detect_ut(stuck_detect_counter_helper_degraded_regfail_test UT_SCENARIO_REGFAIL)

# AICPU 卡死检测计数：head/tail 下发（被测头文件为 header-only）
add_executable(aicpu_stuck_counter_test
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_stuck_counter_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/stuck_stub.cc
)
target_include_directories(aicpu_stuck_counter_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${PROJECT_SOURCE_DIR}/src/ops/op_common
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/aicpu
)
target_compile_options(aicpu_stuck_counter_test PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage -U_FORTIFY_SOURCE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
)
target_link_libraries(aicpu_stuck_counter_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest
    GTest::gtest_main
    -lpthread
    -lgcov
)
run_llt_test(TARGET aicpu_stuck_counter_test TASK_NUM 1)

# CCU 卡死检测计数：LocalReduce + EventWait 下发（被测头文件为 header-only）
add_executable(ccu_stuck_counter_test
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_stuck_counter_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/ccu_stub.cc
)
target_include_directories(ccu_stuck_counter_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${PROJECT_SOURCE_DIR}/src/ops/op_common
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/ccu
)
target_compile_options(ccu_stuck_counter_test PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage -U_FORTIFY_SOURCE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
)
target_link_libraries(ccu_stuck_counter_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest
    GTest::gtest_main
    -lpthread
    -lgcov
)
run_llt_test(TARGET ccu_stuck_counter_test TASK_NUM 1)

# AIV 卡死检测计数：FillAivStuckCounter 参数映射（被测头文件为 header-only）
add_executable(aiv_stuck_counter_test
    ${CMAKE_CURRENT_SOURCE_DIR}/aiv_stuck_counter_test.cc
)
target_include_directories(aiv_stuck_counter_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${PROJECT_SOURCE_DIR}/src/ops/op_common
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/aiv
)
target_compile_options(aiv_stuck_counter_test PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage -U_FORTIFY_SOURCE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
)
target_link_libraries(aiv_stuck_counter_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest
    GTest::gtest_main
    -lpthread
    -lgcov
)
run_llt_test(TARGET aiv_stuck_counter_test TASK_NUM 1)
