# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(HCCL_TF_PLUGIN_SRCS
    hccl_all_gather_plugin.cc
    hccl_all_reduce_plugin.cc
    hccl_all_to_all_v_c_plugin.cc
    hccl_all_to_all_v_plugin.cc
    hccl_broadcast_plugin.cc
    hccl_receive_plugin.cc
    hccl_reduce_plugin.cc
    hccl_reduce_scatter_plugin.cc
    hccl_send_plugin.cc
    hvd_all_gather_plugin.cc
    hvd_all_reduce_plugin.cc
    hvd_broadcast_plugin.cc
)

add_library(ops_proto_init STATIC ops_proto_init.cc)
set_target_properties(ops_proto_init PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ops_proto_init PRIVATE
    ${INCLUDE_LIST}
    ${ASCEND_CANN_PACKAGE_PATH}/include
)
target_compile_options(ops_proto_init PRIVATE -O0 -Dgoogle=ascend_private -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(ops_proto_init PRIVATE
    graph
    graph_base
    register
)

add_library(optf_plugin_hccl SHARED ${HCCL_TF_PLUGIN_SRCS})

target_include_directories(optf_plugin_hccl PRIVATE
    ${INCLUDE_LIST}
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
)

target_compile_options(optf_plugin_hccl PRIVATE -Dgoogle=ascend_private -D_GLIBCXX_USE_CXX11_ABI=0)

target_link_libraries(optf_plugin_hccl PRIVATE
    -Wl,--whole-archive
    ops_proto_init
    -Wl,--no-whole-archive
    graph
    graph_base
    register
    unified_dlog
    runtime_headers
)

install(TARGETS optf_plugin_hccl
    LIBRARY DESTINATION opp/built-in/framework/tensorflow
    COMPONENT hccl
)
