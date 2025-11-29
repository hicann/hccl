# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH
    ${ASCEND_3RD_LIB_PATH}/json 
    ${ASCEND_3RD_LIB_PATH}/gtest_shared/lib64
    ${ASCEND_INSTALL_PATH}
)

# find_package(GTest REQUIRED)

set(COMMON_INC_PATH
    ${ASCEND_3RD_LIB_PATH}/gtest_shared/include
)

# UT
if(ENABLE_UT)
    add_subdirectory(ut)
endif()

# ST
if(ENABLE_ST)
    add_subdirectory(st)
endif()
