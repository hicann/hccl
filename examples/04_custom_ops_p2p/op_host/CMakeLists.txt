# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set_source_files_properties(
    ${HCCL_CUSTOM_OPS_ROOT}/op_kernel_aicpu/aicpu_kernel_asc.aicpu
    PROPERTIES LANGUAGE AICPU
)

set(op_host_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/send.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/recv.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/load_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/launch_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/launch_kernel_asc.asc

    ${HCCL_CUSTOM_OPS_ROOT}/op_kernel_aicpu/aicpu_kernel_asc.aicpu
)

target_sources(hccl_custom_p2p PRIVATE
    ${op_host_src_list}
)
