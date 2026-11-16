# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# batch_send_recv_host: BSR AIV 的 host 侧 CPU 级覆盖，被测对象是生产的模板与 executor 源文件本身。
# 覆盖 tag 只由轮次决定（值域与其它 AIV 算子隔离、越界回绕后仍留在本值域）、收发两侧对同一 pair 切出
# 一致的 (peer, sliceIdx, byteCount)、对端分桶与 recv 落自己桶、自收发的目的地址来自配对 recv 项、
# 零任务 rank 不下发/不推进轮次、CalcRes 只为本次真正通信的对端建链，以及算法注册的 rank/拓扑准入门槛。
#
# stub/ 只补 CANN 基础类型（hccl_types.h、acl/acl_rt.h）与若干 host 侧基类，hccl_aiv_utils.h 等仍走
# 生产头，好让 ExtraArgsBsr 的 sizeof/offset 断言在 UT 里真跑一遍。
#
# 被测 .cc 若新增了生产头的直接 include，记得在 stub/ 里补同名桩，或把该头所在目录加进下面的
# include 列表：本目标的 include 路径是白名单（含 stub 优先级最高），src/CMakeLists.txt 的
# INCLUDE_LIST 对它不生效，生产侧能编过不代表这里能编过。
#
# 注意：executor 是模板类且成员定义在 .cc 里，由测试文件直接 #include 引入；不要把
# ins_v2_aiv_batch_send_recv_sole_executor.cc 加进源列表，否则符号重复定义。
add_executable(batch_send_recv_host_test
    batch_send_recv_host_test.cc
    ${PROJECT_SOURCE_DIR}/src/ops/batch_send_recv/algorithm/template/aiv/aiv_temp_batch_send_recv_mesh_1D.cc
)
target_compile_features(batch_send_recv_host_test PRIVATE cxx_std_17)
target_include_directories(batch_send_recv_host_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template/aiv
    ${PROJECT_SOURCE_DIR}/src/ops/batch_send_recv/algorithm/template/aiv
    ${PROJECT_SOURCE_DIR}/src/ops/batch_send_recv/algorithm/executor
)
target_link_libraries(batch_send_recv_host_test PRIVATE $<BUILD_INTERFACE:intf_llt_pub> GTest::gtest_main)
run_llt_test(TARGET batch_send_recv_host_test TASK_NUM 1)
