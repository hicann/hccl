# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_subdirectory(aicpu)
add_subdirectory(aiv)
add_subdirectory(registry)
add_subdirectory(wrapper)
add_subdirectory(dpu)
add_subdirectory(ccu)

set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/alg_template_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alg_v2_template_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/aiv_alg_template_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/template_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/aiv_alg_template_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_alg_template_base.cc
)

target_sources(hccl PRIVATE
    ${src_list}
)