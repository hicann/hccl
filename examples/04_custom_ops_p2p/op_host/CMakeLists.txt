# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(op_host_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/send.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/recv.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/exec_op.cc
)

target_sources(hccl_custom_p2p PRIVATE
    ${op_host_src_list}
)

if(KERNEL_MODE)
    target_sources(p2p_aicpu_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/exec_op.cc
    )
endif()
