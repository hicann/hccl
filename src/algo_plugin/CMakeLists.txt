# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# HcclAlgoPluginMgr —— 随HCCL主库libhccl.so一同编译
# 写法与同级 src/ops/op_common/CMakeLists.txt 保持一致：直接向父作用域已创建的 hccl 目标
# target_sources，不新增独立的 add_library。

set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_algo_plugin_mgr.cc
)

if(TARGET hccl)
    target_sources(hccl PRIVATE
        ${src_list}
    )
    target_link_libraries(hccl PRIVATE dl)
endif()
