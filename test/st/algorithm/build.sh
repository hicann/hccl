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

# 获取shell脚本目录作为根目录
SHELL_DIR=$(cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

# 创建build编译目录
cd $SHELL_DIR
mkdir -p ./build && cd ./build/ && rm -rf ../build/*

# 编译用例工程，配置执行条件并执行
cmake .. -DBUILD_OPEN_PROJECT=ON && make -j8
LIBRARY_DIR="${SHELL_DIR}/build/utils/src/hccl_depends_stub:"
export LD_LIBRARY_PATH=${LIBRARY_DIR}${LD_LIBRARY_PATH} && ${SHELL_DIR}/build/testcase/hccl_checker_ops_stest

exit 0