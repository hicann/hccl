# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hccl_kernel_compat SHARED
    hccl_dl.cc
    hcomm_device_dlsym.cc
    hcomm_primitives_dl.cc
    hcomm_diag_dl.cc
    hcomm_device_profiling_dl.cc
    hccl_device_comm_dl.cc
)

hccl_apply_cann_compat(hccl_kernel_compat)

target_include_directories(hccl_kernel_compat PRIVATE
    ${INCLUDE_LIST}
)

target_compile_options(hccl_kernel_compat PRIVATE
    -Wno-unused-parameter
    -Wno-missing-field-initializers
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O3>
    -fstack-protector-all
)

target_link_libraries(hccl_kernel_compat PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:runtime_headers>
    $<BUILD_INTERFACE:hcomm_headers>
    unified_dlog
)

target_link_options(hccl_kernel_compat PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-s>
)
