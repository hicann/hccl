# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(BUILD_OPEN_PROJECT)
    set(INCLUDE_LIST
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ccu
    )
else()
    set(INCLUDE_LIST
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/aicpu/tsd
        ${TOP_DIR}/inc/driver
        ${TOP_DIR}/inc/external
        ${TOP_DIR}/inc/external/acl
        ${TOP_DIR}/inc/aicpu/
        ${TOP_DIR}/inc/aicpu/aicpu_schedule/aicpu_sharder
        ${TOP_DIR}/abl/qos/qosmng
        ${TOP_DIR}/abl/atrace/inc/utrace
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/inc/aicpu/common
        ${TOP_DIR}/ace/csruntime/inc/tsch
        ${TOP_DIR}/asl/hss/inc
        ${TOP_DIR}/asl/hss/inc/pub
        ${TOP_DIR}/asl/hss/common/hash_table
        ${TOP_DIR}/metadef
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/metadef/inc
        ${TOP_DIR}/metadef/inc/common
        ${TOP_DIR}/metadef/inc/external
        ${TOP_DIR}/runtime/include/dfx
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
        ${TOP_DIR}/runtime/pkg_inc/base

        ${TOP_DIR}/open_source/json/include

        ${TOP_DIR}/hccl/src/common

        ${TOP_DIR}/hcomm/inc
        ${TOP_DIR}/hcomm/inc/hccl
        ${TOP_DIR}/hcomm/pkg_inc
        ${TOP_DIR}/hcomm/pkg_inc/hccl
        ${TOP_DIR}/hcomm/pkg_inc/legacy
        ${TOP_DIR}/hcomm/pkg_inc/legacy/hccl
        ${TOP_DIR}/hcomm/pkg_inc/hcomm/ccu
        ${TOP_DIR}/hcomm/include
        ${TOP_DIR}/hcomm/include/hccl
    )
endif()

if(PRODUCT_SIDE STREQUAL "host")
    include(hccl_compat.cmake)
else()
    include(hccl_kernel_compat.cmake)
endif()
