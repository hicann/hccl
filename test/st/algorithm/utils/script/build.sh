# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

#!/bin/bash
set -e
trap 'echo "❌ Error occurred in build.sh at line $LINENO"; exit 1' ERR

# 获取shell脚本目录并获取TOP_DIR(work_code)目录
SHELL_DIR=$(cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
TOP_DIR="$SHELL_DIR"
for ((i = 0; i < 6; i++)); do
    TOP_DIR=$(dirname "$TOP_DIR")
done
echo $TOP_DIR

# 进入work_code目录创建tmp目录并进去
cd $TOP_DIR
mkdir -p ./tmp && cd ./tmp/ && rm -rf ../tmp/* && rm -rf ../output/ascend/*

# 执行HCCL编译
echo "HCCL build start..."
cmake ../cmake/superbuild/ -DHOST_PACKAGE="ops_hccl" -DCMAKE_BUILD_TYPE=Debug -DPRODUCT=ascend && TARGETS=hccl make host -j64
echo "HCCL build finish..."
exit 0