# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_aiv_utils.cc
)

# 仅在静态库模式下，给 hccl_aiv_utils.cc 注入 HCCL_STATIC_MODE 宏，
# 用于切换到从内嵌符号加载 AIV kernel 的实现，作用域局限于该单文件。
if(STATIC_MODE)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_aiv_utils.cc
        PROPERTIES COMPILE_DEFINITIONS "HCCL_STATIC_MODE"
    )
endif()

if(TARGET hccl)
    target_sources(hccl PRIVATE
        ${src_list}
    )
endif()
