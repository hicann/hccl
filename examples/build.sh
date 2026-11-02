#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e
set -o pipefail

CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
BUILD_DIR=${CURRENT_DIR}/../build/

# 创建构建目录并截断历史日志，避免历史残留记录干扰失败判定
mkdir -p "${BUILD_DIR}"
: > "${BUILD_DIR}/build.log"

# 并行编译参数：外部未设置JOB_NUM时默认按CPU核数并行编译
JOB_NUM="${JOB_NUM:--j$(nproc)}"

CB_TEST_DIRS=("02_collectives/01_allreduce")

fail_cnt=0

for cb_test_dir in "${CB_TEST_DIRS[@]}"; do
    dir="${CURRENT_DIR}/${cb_test_dir}"

    # 进入子目录
    if ! cd "$dir"; then
        echo "Failed to enter directory: $dir" | tee -a "${BUILD_DIR}/build.log"
        fail_cnt=$((fail_cnt + 1))
        continue
    fi

    # 检查是否存在Makefile
    if [ -f Makefile ]; then
        echo "Processing directory: $dir" | tee -a "${BUILD_DIR}/build.log"
        # 执行make和make test，将输出记录到build.log，失败时以非零退出码退出脚本
        if make ${JOB_NUM} 2>&1 | tee -a "${BUILD_DIR}/build.log"; then
            echo "Make Success" | tee -a "${BUILD_DIR}/build.log"
        else
            echo "Make Failure" | tee -a "${BUILD_DIR}/build.log"
            echo "Processing directory: $dir .. make failed" | tee -a "${BUILD_DIR}/build.log"
            exit 1
        fi

        if make test 2>&1 | tee -a "${BUILD_DIR}/build.log"; then
            echo "Make test Success" | tee -a "${BUILD_DIR}/build.log"
        else
            echo "Make test Failure" | tee -a "${BUILD_DIR}/build.log"
            echo "Processing directory: $dir .. make test failed" | tee -a "${BUILD_DIR}/build.log"
            exit 1
        fi
    else
        echo "No Makefile found in directory: $dir" | tee -a "${BUILD_DIR}/build.log"
        fail_cnt=$((fail_cnt + 1))
    fi
done

# 存在无法进入或无Makefile的目录时视为构建失败，避免未执行任何构建却返回成功
if [ ${fail_cnt} -gt 0 ]; then
    echo "Failed directories: ${fail_cnt}" | tee -a "${BUILD_DIR}/build.log"
    exit 1
fi
