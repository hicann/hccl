# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(report_dir "${OUTPUT_PATH}/report/ut")
# 定义add_run_command函数
function(add_run_command TARGET_NAME TASK_NUM)
    add_custom_command(
        TARGET ${TARGET_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CANN_3RD_LIB_PATH}/gtest/lib/ ASAN_OPTIONS=detect_leaks=0 timeout -s SIGKILL ${LLT_KILL_TIME}s ./${TARGET_NAME} --gtest_output=xml:${report_dir}/${TARGET_NAME}.xml
        COMMAND echo "Task number: ${TASK_NUM} timeout=${LLT_KILL_TIME}"
        COMMENT "Run ops${TARGET_NAME} with task number ${TASK_NUM} ASAN(${ENABLE_ASAN})"
        DEPENDS ${TARGET_NAME}
    )
endfunction()
 	 
# 定义run_llt_test函数
function(run_llt_test)
    cmake_parse_arguments(RUN_LLT_TEST "" "TARGET;TASK_NUM" "" ${ARGN})

    message(STATUS "RUN_LLT_TEST_TARGET=${RUN_LLT_TEST_TARGET} ${RUN_LLT_TEST_TASK_NUM}.")
    if(RUN_LLT_TEST_TARGET AND RUN_LLT_TEST_TASK_NUM)
        add_run_command(${RUN_LLT_TEST_TARGET} ${RUN_LLT_TEST_TASK_NUM})
    endif()
endfunction()