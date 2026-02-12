# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

target_sources(hccl-verifier PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/task_check_op_semantics.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/allgather_semantics_checker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/allgather_v_semantics_checker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/allreduce_semantics_checker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_v_semantics_checker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_semantics_checker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all2all_semantics_checker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all2allv_semantics_checker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_semantics_checker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_semantics_checker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/send_recv_semantics_checker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/scatter_semantics_checker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/batchsendrecv_semantics_checker.cc
)