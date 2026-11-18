# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hcomm SHARED hccl_depends_stub.cc)
target_compile_options(hcomm PRIVATE
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)

add_library(unified_dlog SHARED hccl_depends_stub.cc)
target_compile_options(unified_dlog PRIVATE
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)

add_library(acl_rt SHARED hccl_depends_stub.cc)
target_compile_options(acl_rt PRIVATE
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)

# 门控返回 HCCL_E_NOT_SUPPORT 需 CANN 头文件（三桩目标编译同一源文件，路径随 ASCEND_HOME_PATH 与依赖成对）
if (ASCEND_HOME_PATH)
    target_include_directories(hcomm PRIVATE ${ASCEND_HOME_PATH}/include)
    target_include_directories(unified_dlog PRIVATE ${ASCEND_HOME_PATH}/include)
    target_include_directories(acl_rt PRIVATE ${ASCEND_HOME_PATH}/include)
endif()