# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# batch_send_recv_aiv_core_test: 在 CPU 桩上执行生产 kernel（aiv_batch_send_recv_mesh_1D.h），覆盖
# "tag 只由轮次决定"（不读不写 65MB 块的共享计数器、值域与其它 AIV 算子隔离、半区随奇偶切换）、
# 收发两侧 flag 编址一致、数据经 sender bucket 的端到端落点、Producer/Consumer 分核完备性、
# itemNum 钳制、自收发路径、屏障只覆盖活跃对端，以及消费即清。
# AscendC 设备 API 由 stub/ 提供；aiv_defines.h 与 kernel 头直接用生产文件，保证常量与编码取真值。
add_executable(batch_send_recv_aiv_core_test batch_send_recv_aiv_core_test.cc)
target_compile_features(batch_send_recv_aiv_core_test PRIVATE cxx_std_17)
target_include_directories(batch_send_recv_aiv_core_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/aiv
    ${PROJECT_SOURCE_DIR}/src/ops/batch_send_recv/algorithm/template/aiv/kernel
)
target_link_libraries(batch_send_recv_aiv_core_test PRIVATE $<BUILD_INTERFACE:intf_llt_pub> GTest::gtest_main)
run_llt_test(TARGET batch_send_recv_aiv_core_test TASK_NUM 1)
