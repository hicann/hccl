# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e

CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
BUILD_DIR=${CURRENT_DIR}/../build/

CB_TEST_DIRS=("02_collectives/01_allreduce")

for cb_test_dir in "${CB_TEST_DIRS[@]}"; do
    dir="${CURRENT_DIR}/${cb_test_dir}"

    # 进入子目录
    if ! cd "$dir"; then
        echo "Failed to enter directory: $dir" | tee -a ${BUILD_DIR}/build.log
        continue
    fi

    # 检查是否存在Makefile
    if [ -f Makefile ]; then
        echo "Processing directory: $dir" | tee -a ${BUILD_DIR}/build.log
        # 执行make和make test，并将输出记录到build.log
        make ${JOB_NUM} && echo "Make Success" || echo "Make Failure" | tee -a ${BUILD_DIR}/build.log
        if grep -q "Make Failure" ${BUILD_DIR}/build.log; then
            echo "Processing directory: $dir .. make failed" | tee -a ${BUILD_DIR}/build.log
            break
        fi

        make test && echo "Make test Success" || echo "Make test Failure" | tee -a ${BUILD_DIR}/build.log
        if grep -q "Make test Failure" ${BUILD_DIR}/build.log; then
            echo "Processing directory: $dir .. make test failed" | tee -a ${BUILD_DIR}/build.log
            break
        fi
    else
        echo "No Makefile found in directory: $dir" | tee -a ${BUILD_DIR}/build.log
    fi
done
