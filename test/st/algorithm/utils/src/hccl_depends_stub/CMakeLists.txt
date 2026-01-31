# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hcomm SHARED hccl_depends_stub.cc)
target_compile_options(hcomm PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)

add_library(unified_dlog SHARED hccl_depends_stub.cc)
target_compile_options(unified_dlog PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)

add_library(mmpa SHARED hccl_depends_stub.cc)
target_compile_options(mmpa PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)