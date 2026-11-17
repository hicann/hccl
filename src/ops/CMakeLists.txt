# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# hcomm_inner_def.h 引用 "hcomm/hcomm_common.h"，需要 adv_api 目录在 include 路径中
if(EXISTS "${_CANN_ASC_KERNEL_INCLUDE_DIR}/include/adv_api/hcomm")
    set(HCCL_URMA_EXTRA_INCLUDE "-I${_CANN_ASC_KERNEL_INCLUDE_DIR}/include/adv_api")
else()
    set(HCCL_URMA_EXTRA_INCLUDE "")
endif()

add_subdirectory(scatter)
add_subdirectory(op_common)
add_subdirectory(reduce_scatter)
add_subdirectory(all_gather)
add_subdirectory(all_to_all_v)
add_subdirectory(broadcast)
add_subdirectory(all_reduce)
add_subdirectory(send)
add_subdirectory(recv)
add_subdirectory(reduce_scatter_v)
add_subdirectory(all_gather_v)
add_subdirectory(reduce)
add_subdirectory(batch_send_recv)
add_subdirectory(interface_graph_mode)
add_subdirectory(barrier)